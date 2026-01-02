# video_worker.py - Video processing worker thread
import threading
import time
import logging
import traceback
import cv2
import numpy as np
from PyQt6 import QtCore
from ultralytics import YOLO

import config
import vision_utils as vu
from deepsort_tracker import MultiCameraTracker
from crowd_analytics import CrowdFlowAnalyzer
from camera_config import CameraConfig

logger = logging.getLogger('CrowdAnalysis')

class VideoWorker(QtCore.QObject):
    """Worker thread for video processing"""
    
    topdown_signal = QtCore.pyqtSignal(np.ndarray)
    frame_updated = QtCore.pyqtSignal(str, np.ndarray)
    status_signal = QtCore.pyqtSignal(str)
    fps_signal = QtCore.pyqtSignal(float)
    finished = QtCore.pyqtSignal()
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.info("Initializing VideoWorker")
        self._running = False
        self.paused = False
        self.pause_condition = threading.Condition()
        
        # Dictionary to store camera configurations
        self.cameras = {}
        
        # World coordinates for calibration
        self.world_pts = np.array([
            [0, 0], 
            [0, config.WORLD_H], 
            [config.WORLD_W, config.WORLD_H], 
            [config.WORLD_W, 0]
        ], np.float32)
        
        self.output_w = int(config.WORLD_W * config.SCALE) + config.MARGIN * 2
        self.output_h = int(config.WORLD_H * config.SCALE) + config.MARGIN * 2
        
        # Transformation matrix for top-down view
        self.S = np.array([
            [config.SCALE, 0, config.MARGIN],
            [0, -config.SCALE, config.WORLD_H * config.SCALE + config.MARGIN],
            [0, 0, 1]
        ], np.float32)
        
        # Create blank dark background instead of loading image
        self.bg_photo = np.zeros((self.output_h, self.output_w, 3), np.uint8)
        self.bg_photo[:] = (30, 30, 30)  # Dark gray background
        self.bg_faint = self.bg_photo.copy()
        logger.info("Using blank background for top-down view")
        
        self.tracker = MultiCameraTracker()
        
        # Initialize tracker dictionaries if they don't exist
        if not hasattr(self.tracker, 'original_dims'):
            self.tracker.original_dims = {}
        if not hasattr(self.tracker, 'deep_sort_trackers'):
            self.tracker.deep_sort_trackers = {}
        
        self.crowd_analyzer = CrowdFlowAnalyzer(config.WORLD_W, config.WORLD_H, config.HEATMAP_CELL_SIZE)
        self.yolo_model = None
        self.conf = 0.3
        self.frame_count = 0
        self.show_trajectories = True
        self.show_flow = True
        self.show_heatmap = True
        self.show_predictions = True
        self.show_legend = True
        logger.info("VideoWorker initialized successfully")

    def add_camera(self, camera_id, video_path=None, calibration_points=None):
        """Add a new camera to the system"""
        try:
            logger.info(f"Adding camera: {camera_id}")
            camera = CameraConfig(camera_id, video_path, calibration_points)
            
            # Compute homography if calibration points are provided
            if calibration_points is not None:
                logger.debug(f"Computing homography for {camera_id}")
                camera.homography, _ = cv2.findHomography(
                    np.array(calibration_points, np.float32), 
                    self.world_pts
                )
                logger.debug(f"Homography computed successfully for {camera_id}")
            
            self.cameras[camera_id] = camera
            
            # Initialize tracker for this camera
            if not hasattr(self.tracker, 'original_dims'):
                self.tracker.original_dims = {}
            self.tracker.original_dims[camera_id] = None
            
            logger.info(f"Camera {camera_id} added successfully and registered with tracker")
            return camera
        except Exception as e:
            logger.error(f"Error adding camera {camera_id}: {e}", exc_info=True)
            raise

    def remove_camera(self, camera_id):
        """Remove a camera from the system"""
        try:
            if camera_id in self.cameras:
                camera = self.cameras[camera_id]
                if camera.capture is not None:
                    camera.capture.release()
                del self.cameras[camera_id]
                
                # Clean up tracker state for this camera
                if hasattr(self.tracker, 'original_dims') and camera_id in self.tracker.original_dims:
                    del self.tracker.original_dims[camera_id]
                
                logger.info(f"Camera {camera_id} removed")
        except Exception as e:
            logger.error(f"Error removing camera {camera_id}: {e}", exc_info=True)

    def update_camera_video(self, camera_id, video_path):
        """Update video path for a camera"""
        try:
            if camera_id in self.cameras:
                self.cameras[camera_id].video_path = video_path
                logger.info(f"Updated video for {camera_id}: {video_path}")
        except Exception as e:
            logger.error(f"Error updating video for {camera_id}: {e}", exc_info=True)

    def update_camera_calibration(self, camera_id, calibration_points):
        """Update calibration points for a camera"""
        try:
            if camera_id in self.cameras:
                self.cameras[camera_id].calibration_points = calibration_points
                self.cameras[camera_id].homography, _ = cv2.findHomography(
                    np.array(calibration_points, np.float32), 
                    self.world_pts
                )
                logger.info(f"Updated calibration for {camera_id}")
        except Exception as e:
            logger.error(f"Error updating calibration for {camera_id}: {e}", exc_info=True)

    def start_processing(self, model_path, conf):
        """Start processing all configured cameras"""
        try:
            logger.info("=" * 60)
            logger.info("STARTING PROCESSING")
            logger.info("=" * 60)
            
            if not self.cameras:
                error_msg = "No cameras configured"
                logger.error(error_msg)
                self.error_signal.emit(error_msg)
                return
            
            # Check if all cameras have video paths and calibration
            missing_config = []
            for cam_id, camera in self.cameras.items():
                logger.info(f"Checking camera {cam_id}...")
                if not camera.video_path:
                    missing_config.append(f"{cam_id}: missing video")
                    logger.warning(f"  - Missing video path")
                if camera.homography is None:
                    missing_config.append(f"{cam_id}: missing calibration")
                    logger.warning(f"  - Missing calibration")
                else:
                    logger.info(f"  - Video: {camera.video_path}")
                    logger.info(f"  - Homography: OK")
            
            if missing_config:
                error_msg = "Missing configuration:\n" + "\n".join(missing_config)
                logger.error(error_msg)
                self.error_signal.emit(error_msg)
                return
            
            self.conf = conf
            logger.info(f"Confidence threshold: {conf}")
            
            # Load YOLO model
            logger.info(f"Loading YOLO model: {model_path}")
            self.yolo_model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
            
            # Initialize video captures for all cameras
            for cam_id, camera in self.cameras.items():
                logger.info(f"Opening video capture for {cam_id}: {camera.video_path}")
                camera.capture = cv2.VideoCapture(camera.video_path)
                
                if not camera.capture.isOpened():
                    raise ValueError(f"Could not open video for camera {cam_id}: {camera.video_path}")
                
                # Log video properties
                fps = camera.capture.get(cv2.CAP_PROP_FPS)
                width = int(camera.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(camera.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(camera.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                logger.info(f"  - FPS: {fps}, Size: {width}x{height}, Frames: {frame_count}")
                
                # Single camera mode: use full frame area (no calibration needed)
                if len(self.cameras) == 1:
                    logger.info(f"Single camera mode detected - using full frame for {cam_id}")
                    full_frame_points = [
                        [0, height],           # Bottom-left
                        [0, 0],                # Top-left
                        [width, 0],            # Top-right
                        [width, height]        # Bottom-right
                    ]
                    camera.calibration_points = full_frame_points
                    camera.homography, _ = cv2.findHomography(
                        np.array(full_frame_points, np.float32), 
                        self.world_pts
                    )
                    logger.info(f"  - Auto-calibrated to full frame: {width}x{height}")
            
            self._running = True
            self.frame_count = 0
            logger.info("Starting processing thread...")
            threading.Thread(target=self.run, daemon=True).start()
            logger.info("Processing thread started")
            
        except Exception as e:
            error_msg = f"Error starting processing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_signal.emit(error_msg)

    def stop_processing(self):
        logger.info("Stopping processing...")
        self._running = False
        QtCore.QThread.msleep(100)
        self._cleanup_resources()
        self._reset_tracking_state()
    
    def _reset_tracking_state(self):
        """Reset all tracking and analytics state"""
        logger.info("Resetting tracking state...")
        try:
            # Reinitialize tracker
            self.tracker = MultiCameraTracker()
            if not hasattr(self.tracker, 'original_dims'):
                self.tracker.original_dims = {}
            if not hasattr(self.tracker, 'deep_sort_trackers'):
                self.tracker.deep_sort_trackers = {}
            
            # Re-register existing cameras with the new tracker
            for cam_id in self.cameras.keys():
                self.tracker.original_dims[cam_id] = None
                logger.debug(f"Re-registered {cam_id} with fresh tracker")
            
            # Reinitialize crowd analyzer
            self.crowd_analyzer = CrowdFlowAnalyzer(
                config.WORLD_W, 
                config.WORLD_H, 
                config.HEATMAP_CELL_SIZE
            )
            
            logger.info("Tracking state reset complete")
        except Exception as e:
            logger.error(f"Error resetting tracking state: {e}", exc_info=True)

    def _cleanup_resources(self):
        """Safely release all video capture resources"""
        logger.info("Cleaning up resources...")
        for cam_id, camera in self.cameras.items():
            if camera.capture is not None:
                try:
                    camera.capture.release()
                    logger.debug(f"Released capture for {cam_id}")
                except Exception as e:
                    logger.error(f"Error releasing capture for {cam_id}: {e}")
                camera.capture = None

    def pause(self):
        with self.pause_condition:
            self.paused = True
            logger.info("Processing paused")

    def resume(self):
        with self.pause_condition:
            self.paused = False
            self.pause_condition.notify_all()
            logger.info("Processing resumed")

    def _merge_multiple_camera_tracks(self, camera_tracks_dict):
        """Merge tracks from multiple cameras iteratively"""
        try:
            track_lists = [tracks for tracks in camera_tracks_dict.values() if tracks]
            
            if len(track_lists) == 0:
                return []
            elif len(track_lists) == 1:
                # Ensure single camera tracks have global_id
                for track in track_lists[0]:
                    if not hasattr(track, 'global_id'):
                        track.global_id = track.local_id
                return track_lists[0]
            elif len(track_lists) == 2:
                merged = self.tracker.merge_camera_tracks(track_lists[0], track_lists[1])
                # Ensure all merged tracks have global_id
                for track in merged:
                    if not hasattr(track, 'global_id'):
                        track.global_id = track.local_id
                return merged
            else:
                # For 3+ cameras, merge iteratively
                merged = self.tracker.merge_camera_tracks(track_lists[0], track_lists[1])
                for i in range(2, len(track_lists)):
                    merged = self.tracker.merge_camera_tracks(merged, track_lists[i])
                # Ensure all merged tracks have global_id
                for track in merged:
                    if not hasattr(track, 'global_id'):
                        track.global_id = track.local_id
                return merged
        except Exception as e:
            logger.error(f"Error merging tracks: {e}", exc_info=True)
            # Return tracks with global_id set as fallback
            all_tracks = []
            for tracks in camera_tracks_dict.values():
                for track in tracks:
                    if not hasattr(track, 'global_id'):
                        track.global_id = track.local_id
                    all_tracks.append(track)
            return all_tracks

    def run(self):
        """Main processing loop"""
        logger.info("Processing loop started")
        last_time = time.time()
        frames_processed = 0
        
        GRID_W, GRID_H = config.GRID_W, config.GRID_H
        cell_w = config.WORLD_W / GRID_W
        cell_h = config.WORLD_H / GRID_H

        try:
            while self._running:
                t0 = time.time()

                # Pause handling
                with self.pause_condition:
                    while self.paused:
                        self.pause_condition.wait()

                # Read frames from all cameras
                active_cameras = 0
                camera_tracks = {}
                
                for cam_id, camera in self.cameras.items():
                    ret, frame = False, None
                    if camera.capture:
                        ret, frame = camera.capture.read()
                    
                    if ret and frame is not None:
                        active_cameras += 1
                        camera_tracks[cam_id] = self._process_camera_frame(
                            cam_id, camera, frame, cell_w, cell_h
                        )
                    else:
                        camera_tracks[cam_id] = []
                        if self.frame_count == 0:
                            logger.warning(f"Could not read frame from {cam_id}")

                # If no cameras have frames, we're done
                if active_cameras == 0:
                    logger.info("No more frames from any camera")
                    self.status_signal.emit("Reached end of all videos.")
                    break

                self.frame_count += 1
                
                if self.frame_count % 30 == 0:
                    logger.debug(f"Processed frame {self.frame_count}")
                
                # Merge tracks from all cameras
                all_tracks = self._merge_multiple_camera_tracks(camera_tracks)
                
                # Ensure all tracks have global_id before updating crowd analyzer
                for track in all_tracks:
                    if not hasattr(track, 'global_id'):
                        logger.warning(f"Track from {track.camera_id} missing global_id, assigning local_id")
                        track.global_id = track.local_id
                
                self.crowd_analyzer.update(all_tracks)
                
                # Generate top-down visualization
                topdown = self._generate_topdown_view(all_tracks, cell_w, cell_h)
                self.topdown_signal.emit(topdown)
                
                # FPS calculation
                frames_processed += 1
                t1 = time.time()
                elapsed = t1 - last_time
                if elapsed >= 1.0:
                    self.fps_signal.emit(frames_processed / elapsed)
                    last_time = t1
                    frames_processed = 0
                
                # Loop timing
                loop_time = time.time() - t0
                sleep_time = max(0.001, 1.0 / 30.0 - loop_time)
                time.sleep(sleep_time)

        except Exception as e:
            error_msg = f"Fatal error in processing loop: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_signal.emit(error_msg)
        
        finally:
            logger.info("Processing loop ended")
            self._cleanup_resources()
            self.finished.emit()

    def _process_camera_frame(self, cam_id, camera, frame, cell_w, cell_h):
        """Process a single camera frame"""
        tracks = []
        
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                logger.warning(f"{cam_id}: Empty frame received")
                return tracks
            
            # Validate homography
            if camera.homography is None:
                logger.error(f"{cam_id}: Missing homography matrix")
                return tracks
            
            # Ensure tracker knows about this camera (safety check)
            if not hasattr(self.tracker, 'original_dims'):
                self.tracker.original_dims = {}
            if cam_id not in self.tracker.original_dims:
                logger.info(f"Late initialization of tracker for {cam_id}")
                self.tracker.original_dims[cam_id] = None
            
            # Run YOLO detection
            results = self.yolo_model.track(
                frame, 
                conf=self.conf, 
                classes=[0],
                persist=True,
                verbose=False
            )
            
            # Validate results
            if results is None or len(results) == 0:
                return tracks
            
            if not hasattr(results[0], 'boxes') or results[0].boxes is None:
                return tracks
            
            if len(results[0].boxes) > 0:
                tracks = self.tracker.update_tracks(
                    results[0].boxes, frame, cam_id, 
                    camera.homography, (config.WORLD_W, config.WORLD_H)
                )
            
            # Annotate frame
            annotated = frame.copy()
            
            for track in tracks:
                try:
                    ltrb = track.ltrb
                    x1, y1, x2, y2 = map(int, ltrb)
                    
                    # Bounds checking
                    h, w = annotated.shape[:2]
                    x1, y1 = max(0, min(x1, w-1)), max(0, min(y1, h-1))
                    x2, y2 = max(0, min(x2, w-1)), max(0, min(y2, h-1))
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), camera.color, 2)
                    cv2.putText(
                        annotated, f"ID:{track.local_id}", (x1, max(15, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, camera.color, 2
                    )
                    
                    # Draw foot position marker (yellow circle at bottom center)
                    foot_x = int((x1 + x2) / 2)
                    foot_y = y2
                    cv2.circle(annotated, (foot_x, foot_y), 5, (0, 255, 255), -1)
                    
                except Exception as track_error:
                    logger.debug(f"Error drawing track: {track_error}")
                    continue
            
            # Draw grid
            try:
                H_inv = np.linalg.inv(camera.homography)
                annotated = vu.draw_world_grid_on_camera(
                    annotated, H_inv, config.GRID_W, config.GRID_H, 
                    cell_w, cell_h, config.WORLD_W, config.WORLD_H, 
                    camera.color
                )
            except np.linalg.LinAlgError as e:
                logger.error(f"{cam_id}: Homography inversion failed: {e}")
            
            self.frame_updated.emit(cam_id, annotated)
            
        except Exception as e:
            error_msg = f"Error processing {cam_id}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.status_signal.emit(f"Error in {cam_id}: {str(e)}")
        
        return tracks

    def _generate_topdown_view(self, all_tracks, cell_w, cell_h):
        """Generate the top-down view with all visualizations"""
        try:
            topdown = self.bg_faint.copy()
            
            if self.show_heatmap:
                topdown = self._draw_heatmap(topdown)
            
            self._draw_grid(topdown, cell_w, cell_h)
            
            if self.show_flow:
                self._draw_flow_vectors(topdown)
            
            self._draw_tracks(topdown, all_tracks)
            self._draw_statistics(topdown)
            
            if self.show_legend:
                self._draw_legend(topdown)
            
            return topdown
        except Exception as e:
            logger.error(f"Error generating topdown view: {e}", exc_info=True)
            return self.bg_faint.copy()

    def _draw_heatmap(self, topdown):
        """Draw heatmap overlay"""
        try:
            heatmap = self.crowd_analyzer.get_density_heatmap(smooth_sigma=2.5)
            heatmap_colored = np.zeros(
                (self.crowd_analyzer.hmap_h, self.crowd_analyzer.hmap_w, 3), 
                dtype=np.uint8
            )
            
            for hy in range(self.crowd_analyzer.hmap_h):
                for hx in range(self.crowd_analyzer.hmap_w):
                    value = heatmap[hy, hx]
                    if value > 0:
                        if value < 0.33:
                            r = 0
                            g = int(255 * (value / 0.33))
                            b = int(255 * (1 - value / 0.33))
                        elif value < 0.66:
                            r = int(255 * ((value - 0.33) / 0.33))
                            g = 255
                            b = 0
                        else:
                            r = 255
                            g = int(255 * (1 - (value - 0.66) / 0.34))
                            b = 0
                        
                        heatmap_colored[hy, hx] = [b, g, r]
            
            heatmap_resized = cv2.resize(
                heatmap_colored,
                (int(config.WORLD_W * config.SCALE), int(config.WORLD_H * config.SCALE)),
                interpolation=cv2.INTER_LINEAR
            )
            
            y_start = config.MARGIN
            y_end = config.MARGIN + int(config.WORLD_H * config.SCALE)
            x_start = config.MARGIN
            x_end = config.MARGIN + int(config.WORLD_W * config.SCALE)
            
            mask = (heatmap_resized.sum(axis=2) > 0).astype(np.uint8) * 255
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            roi = topdown[y_start:y_end, x_start:x_end]
            blended = cv2.addWeighted(roi, 0.6, heatmap_resized, 0.4, 0)
            topdown[y_start:y_end, x_start:x_end] = np.where(mask > 0, blended, roi)
            
            return topdown
        except Exception as e:
            logger.error(f"Error drawing heatmap: {e}")
            return topdown

    def _draw_grid(self, topdown, cell_w, cell_h):
        """Draw coordinate grid"""
        try:
            for i in range(config.GRID_W + 1):
                x = int(config.MARGIN + i * cell_w * config.SCALE)
                cv2.line(topdown, (x, config.MARGIN), 
                        (x, self.output_h - config.MARGIN), (80, 80, 80), 1)
            for j in range(config.GRID_H + 1):
                y = int(self.output_h - config.MARGIN - j * cell_h * config.SCALE)
                cv2.line(topdown, (config.MARGIN, y), 
                        (self.output_w - config.MARGIN, y), (80, 80, 80), 1)
            
            vu.draw_axis_labels(topdown, config.GRID_W, config.GRID_H, cell_w, cell_h,
                               config.WORLD_W, config.WORLD_H, config.SCALE, config.MARGIN)
        except Exception as e:
            logger.error(f"Error drawing grid: {e}")

    def _draw_flow_vectors(self, topdown):
        """Draw crowd flow vectors"""
        try:
            flow_vectors = self.crowd_analyzer.get_flow_vectors()
            for vec in flow_vectors:
                px, py = vu.world_to_topdown(vec['x'], vec['y'], self.S)
                arrow_scale = min(vec['magnitude'] * 300, 50)
                end_x = int(px + vec['vx'] * arrow_scale)
                end_y = int(py - vec['vy'] * arrow_scale)
                cv2.arrowedLine(topdown, (px, py), (end_x, end_y), 
                              (0, 255, 255), 2, tipLength=0.3)
        except Exception as e:
            logger.error(f"Error drawing flow vectors: {e}")

    def _draw_tracks(self, topdown, all_tracks):
        """Draw tracked people on top-down view"""
        try:
            for track in all_tracks:
                # Ensure track has global_id (safety check)
                if not hasattr(track, 'global_id'):
                    logger.warning(f"Track missing global_id, using local_id: {track.local_id}")
                    track.global_id = track.local_id
                
                wx, wy = track.world_x, track.world_y
                px, py = vu.world_to_topdown(wx, wy, self.S)
                
                if hasattr(track, 'merged_from'):
                    color = (255, 0, 255)  # Magenta for merged
                    thickness = 3
                else:
                    camera = self.cameras.get(track.camera_id)
                    color = camera.color if camera else (255, 255, 255)
                    thickness = 2
                
                cv2.circle(topdown, (px, py), 8, color, thickness)
                cv2.circle(topdown, (px, py), 10, (255, 255, 255), 1)
                
                # Draw ID with white outline for better visibility
                cv2.putText(topdown, str(track.global_id), (px+12, py-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(topdown, str(track.global_id), (px+12, py-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
                # Draw trajectory
                if self.show_trajectories:
                    trajectory = self.crowd_analyzer.get_trajectory(track.global_id)
                    if len(trajectory) > 1:
                        for i in range(len(trajectory) - 1):
                            pt1 = vu.world_to_topdown(trajectory[i][0], trajectory[i][1], self.S)
                            pt2 = vu.world_to_topdown(trajectory[i+1][0], trajectory[i+1][1], self.S)
                            alpha = i / len(trajectory)
                            fade_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
                            cv2.line(topdown, pt1, pt2, fade_color, 2)
                
                # Draw prediction
                if self.show_predictions:
                    pred_pos = self.crowd_analyzer.predict_position(
                        track.global_id, 
                        config.PREDICTION_HORIZON if hasattr(config, 'PREDICTION_HORIZON') else 1.0
                    )
                    if pred_pos:
                        pred_px, pred_py = vu.world_to_topdown(pred_pos[0], pred_pos[1], self.S)
                        # Blue circle for predicted position
                        cv2.circle(topdown, (pred_px, pred_py), 6, (0, 200, 255), 1)
                        # Line connecting current to predicted
                        cv2.line(topdown, (px, py), (pred_px, pred_py), (0, 200, 255), 1)
                        
        except Exception as e:
            logger.error(f"Error drawing tracks: {e}")

    def _draw_statistics(self, topdown):
        """Draw statistics overlay"""
        try:
            stats = self.crowd_analyzer.get_statistics()
            info_text = [
                f"Frame: {self.frame_count}",
                f"Cameras: {len(self.cameras)}",
                f"People: {stats['current_count']}",
                f"Total: {stats['total_unique']}",
                f"Heatmap: {self.crowd_analyzer.heatmap_min_people}-{self.crowd_analyzer.heatmap_max_people}"
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(topdown, (8, y_offset + i*25 - 18), 
                             (tw + 12, y_offset + i*25 + 5), (0, 0, 0), -1)
                # White outline
                cv2.putText(topdown, text, (10, y_offset + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                # Green fill
                cv2.putText(topdown, text, (10, y_offset + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        except Exception as e:
            logger.error(f"Error drawing statistics: {e}")
    
    def _draw_legend(self, topdown):
        """Draw color-coded legend"""
        try:
            legend_y = self.output_h - 100
            legend_x = 10
            
            # Background box
            cv2.rectangle(topdown, (legend_x - 2, legend_y - 5), 
                         (legend_x + 200, self.output_h - 8), (0, 0, 0), -1)
            
            # Title
            cv2.putText(topdown, "Legend:", (legend_x + 5, legend_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Get list of cameras for legend
            camera_list = list(self.cameras.items())
            
            # Draw legend entries for each camera
            for idx, (cam_id, camera) in enumerate(camera_list[:3]):  # Limit to 3 for space
                y_pos = legend_y + 30 + (idx * 20)
                cv2.circle(topdown, (legend_x + 10, y_pos), 6, camera.color, 2)
                cv2.putText(topdown, cam_id, (legend_x + 25, y_pos + 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Merged tracks legend
            if len(camera_list) > 1:
                y_pos = legend_y + 30 + (min(len(camera_list), 3) * 20)
                cv2.circle(topdown, (legend_x + 10, y_pos), 6, (255, 0, 255), 3)
                cv2.putText(topdown, "Merged", (legend_x + 25, y_pos + 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            
        except Exception as e:
            logger.error(f"Error drawing legend: {e}")