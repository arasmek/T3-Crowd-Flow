# UI.py - Modular version with comprehensive error logging and auto-load
import sys
import os
import threading
import time
import cv2
import numpy as np
import logging
import traceback
from datetime import datetime
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO

import config
import vision_utils as vu
from deepsort_tracker import MultiCameraTracker
from crowd_analytics import CrowdFlowAnalyzer

# Get script directory for auto-loading videos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# ========== LOGGING SETUP ==========
log_filename = f"crowd_analysis_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('CrowdAnalysis')
logger.info(f"Logging initialized. Log file: {log_filename}")

# -----------------------------
# Helper: Convert OpenCV BGR → QImage
# -----------------------------
def cv2_to_qimage(bgr_img):
    try:
        if bgr_img is None:
            return None
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    except Exception as e:
        logger.error(f"Error converting image: {e}")
        return None

# ==============================
# Camera Configuration Class
# ==============================
class CameraConfig:
    """Stores configuration for a single camera"""
    def __init__(self, camera_id, video_path=None, calibration_points=None):
        self.camera_id = camera_id
        self.video_path = video_path
        self.calibration_points = calibration_points
        self.capture = None
        self.homography = None
        self.color = self._generate_color()
        logger.info(f"Created camera config: {camera_id}")
    
    def _generate_color(self):
        """Generate a unique color for this camera based on its ID"""
        colors = [
            (0, 255, 0),      # Green
            (255, 100, 0),    # Orange
            (0, 100, 255),    # Blue
            (255, 0, 255),    # Magenta
            (255, 255, 0),    # Yellow
            (0, 255, 255),    # Cyan
            (255, 128, 128),  # Pink
            (128, 255, 128),  # Light Green
        ]
        color_idx = hash(self.camera_id) % len(colors)
        return colors[color_idx]

# ==============================
# VideoWorker for threading
# ==============================
class VideoWorker(QtCore.QObject):
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
        
        # Load background with error handling
        try:
            logger.info(f"Loading background image: {config.TOPDOWN_REF}")
            self.bg_photo = cv2.imread(config.TOPDOWN_REF)
            if self.bg_photo is None:
                raise ValueError("Could not load background image")
            logger.info(f"Background loaded: shape={self.bg_photo.shape}")
        except Exception as e:
            logger.warning(f"Could not load background image: {e}")
            self.bg_photo = np.zeros((self.output_h, self.output_w, 3), np.uint8)
        
        self.bg_faint = vu.make_faint_background(self.bg_photo, alpha=0.18)
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
            # The tracker needs to know about the camera before processing frames
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
                persist=True,  # Changed from False
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
                    color = (255, 0, 255)
                    thickness = 3
                else:
                    camera = self.cameras.get(track.camera_id)
                    color = camera.color if camera else (255, 255, 255)
                    thickness = 2
                
                cv2.circle(topdown, (px, py), 8, color, thickness)
                cv2.circle(topdown, (px, py), 10, (255, 255, 255), 1)
                
                cv2.putText(topdown, str(track.global_id), (px+12, py-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                if self.show_trajectories:
                    trajectory = self.crowd_analyzer.get_trajectory(track.global_id)
                    if len(trajectory) > 1:
                        for i in range(len(trajectory) - 1):
                            pt1 = vu.world_to_topdown(trajectory[i][0], trajectory[i][1], self.S)
                            pt2 = vu.world_to_topdown(trajectory[i+1][0], trajectory[i+1][1], self.S)
                            alpha = i / len(trajectory)
                            fade_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
                            cv2.line(topdown, pt1, pt2, fade_color, 2)
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
                f"Total: {stats['total_unique']}"
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(topdown, (8, y_offset + i*25 - 18), 
                             (tw + 12, y_offset + i*25 + 5), (0, 0, 0), -1)
                cv2.putText(topdown, text, (10, y_offset + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        except Exception as e:
            logger.error(f"Error drawing statistics: {e}")

# ==============================
# Camera Widget
# ==============================
class CameraWidget(QtWidgets.QWidget):
    """Widget representing a single camera feed"""
    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.label = QtWidgets.QLabel(f"Camera {camera_id}")
        self.label.setFixedSize(360, 240)
        self.label.setStyleSheet("background:black; color:white;")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        
        self.info_label = QtWidgets.QLabel("No video loaded")
        self.info_label.setStyleSheet("color: gray; font-size: 10px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        # First row of controls
        control_layout1 = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load Video")
        self.btn_remove = QtWidgets.QPushButton("Remove")
        control_layout1.addWidget(self.btn_load)
        control_layout1.addWidget(self.btn_remove)
        layout.addLayout(control_layout1)
        
        # Second row of controls
        control_layout2 = QtWidgets.QHBoxLayout()
        self.btn_calibrate = QtWidgets.QPushButton("Calibrate")
        self.btn_upload_calib = QtWidgets.QPushButton("Upload Calib Img")
        control_layout2.addWidget(self.btn_calibrate)
        control_layout2.addWidget(self.btn_upload_calib)
        layout.addLayout(control_layout2)

# ==============================
# Calibration Window
# ==============================
class CalibrationWindow(QtWidgets.QDialog):
    """Simple calibration window for selecting 4 points"""
    def __init__(self, camera_id, source_path, current_points, is_image=False, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.source_path = source_path
        self.is_image = is_image
        self.points = current_points.copy() if current_points else []
        self.frame = None
        self.display_frame = None
        
        self.setWindowTitle(f"Calibrate {camera_id}")
        self.resize(900, 700)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Instructions
        instructions = QtWidgets.QLabel(
            "Click 4 points in this order:\n"
            "1. Bottom-Left → 2. Top-Left → 3. Top-Right → 4. Bottom-Right\n"
            "(These should form the area you want to map to the world coordinates)"
        )
        instructions.setStyleSheet("background: yellow; padding: 10px; font-weight: bold;")
        layout.addWidget(instructions)
        
        # Image display
        self.image_label = QtWidgets.QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("background: black;")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.mousePressEvent = self.on_image_click
        layout.addWidget(self.image_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("Reset Points")
        self.btn_reset.clicked.connect(self.reset_points)
        self.btn_save = QtWidgets.QPushButton("Save Calibration")
        self.btn_save.clicked.connect(self.accept)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        
        button_layout.addWidget(self.btn_reset)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_save)
        button_layout.addWidget(self.btn_cancel)
        layout.addLayout(button_layout)
        
        # Load frame
        self.load_frame()
    
    def load_frame(self):
        """Load frame from video or image"""
        if self.is_image:
            # Load from image file
            frame = cv2.imread(self.source_path)
            if frame is not None:
                self.frame = frame
                self.update_display()
            else:
                QtWidgets.QMessageBox.critical(self, "Error", "Could not load calibration image")
                self.reject()
        else:
            # Load first frame from video
            cap = cv2.VideoCapture(self.source_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                self.frame = frame
                self.update_display()
            else:
                QtWidgets.QMessageBox.critical(self, "Error", "Could not load video frame")
                self.reject()
    
    def update_display(self):
        """Update the display with current points"""
        if self.frame is None:
            return
        
        display = self.frame.copy()
        
        # Draw existing points
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        labels = ["1:BL", "2:TL", "3:TR", "4:BR"]
        
        for i, pt in enumerate(self.points):
            cv2.circle(display, tuple(pt), 10, colors[i], -1)
            cv2.putText(display, labels[i], (pt[0]+15, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2)
        
        # Draw lines between points
        if len(self.points) >= 2:
            for i in range(len(self.points)-1):
                cv2.line(display, tuple(self.points[i]), tuple(self.points[i+1]), 
                        (255, 255, 255), 2)
        if len(self.points) == 4:
            cv2.line(display, tuple(self.points[3]), tuple(self.points[0]), 
                    (255, 255, 255), 2)
        
        # Convert to QImage
        qimg = cv2_to_qimage(display)
        if qimg:
            self.image_label.setPixmap(
                QPixmap.fromImage(qimg).scaled(
                    self.image_label.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
            )
    
    def on_image_click(self, event):
        """Handle mouse click on image"""
        if len(self.points) >= 4:
            QtWidgets.QMessageBox.information(
                self, "Info", 
                "4 points already selected. Click 'Reset Points' to start over."
            )
            return
        
        # Get click position relative to the actual image
        label_rect = self.image_label.rect()
        pixmap = self.image_label.pixmap()
        
        if pixmap is None:
            return
        
        # Calculate scaling
        pixmap_rect = pixmap.rect()
        x_scale = self.frame.shape[1] / pixmap_rect.width()
        y_scale = self.frame.shape[0] / pixmap_rect.height()
        
        # Calculate offset (image is centered in label)
        x_offset = (label_rect.width() - pixmap_rect.width()) // 2
        y_offset = (label_rect.height() - pixmap_rect.height()) // 2
        
        # Convert click position to image coordinates
        click_x = int((event.pos().x() - x_offset) * x_scale)
        click_y = int((event.pos().y() - y_offset) * y_scale)
        
        # Validate click is within image bounds
        if 0 <= click_x < self.frame.shape[1] and 0 <= click_y < self.frame.shape[0]:
            self.points.append([click_x, click_y])
            logger.info(f"Point {len(self.points)} selected: ({click_x}, {click_y})")
            self.update_display()
    
    def reset_points(self):
        """Reset all points"""
        self.points = []
        self.update_display()
    
    def get_points(self):
        """Get the calibration points"""
        return self.points if len(self.points) == 4 else None

# ==============================
# MainWindow
# ==============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing MainWindow")
        self.setWindowTitle(f"Multi-Camera Crowd Analysis - Log: {log_filename}")
        self.resize(1600, 900)
        
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        
        # Camera management section
        camera_mgmt_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(camera_mgmt_layout)
        
        camera_mgmt_layout.addWidget(QtWidgets.QLabel("Cameras:"))
        self.btn_add_camera = QtWidgets.QPushButton("+ Add Camera")
        self.btn_add_camera.clicked.connect(self.on_add_camera)
        camera_mgmt_layout.addWidget(self.btn_add_camera)
        camera_mgmt_layout.addStretch()
        
        # Scrollable area for camera feeds
        self.camera_scroll = QtWidgets.QScrollArea()
        self.camera_scroll.setWidgetResizable(True)
        self.camera_scroll.setMinimumHeight(300)
        
        self.camera_container = QtWidgets.QWidget()
        self.camera_layout = QtWidgets.QHBoxLayout(self.camera_container)
        self.camera_scroll.setWidget(self.camera_container)
        main_layout.addWidget(self.camera_scroll)
        
        # Top-down display
        topdown_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(topdown_layout)
        
        self.labelTD = QtWidgets.QLabel("Top-Down Heatmap View")
        self.labelTD.setFixedSize(800, 600)
        self.labelTD.setStyleSheet("background:black; color:white;")
        self.labelTD.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        topdown_layout.addWidget(self.labelTD)
        topdown_layout.addStretch()

        # Model and confidence controls
        controls_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(controls_layout)
        
        controls_layout.addWidget(QtWidgets.QLabel("Model:"))
        self.model_path_edit = QtWidgets.QLineEdit(config.YOLO_MODEL_PATH)
        controls_layout.addWidget(self.model_path_edit)
        
        controls_layout.addWidget(QtWidgets.QLabel("Confidence:"))
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(40)
        self.conf_slider.setMaximumWidth(150)
        self.conf_label = QtWidgets.QLabel("0.40")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v/100:.2f}"))
        controls_layout.addWidget(self.conf_slider)
        controls_layout.addWidget(self.conf_label)

        # Control buttons
        buttons_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(buttons_layout)
        
        self.btn_start = QtWidgets.QPushButton("Start Processing")
        self.btn_start.clicked.connect(self.on_start)
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_pause.setEnabled(False)
        self.btn_resume = QtWidgets.QPushButton("Resume")
        self.btn_resume.clicked.connect(self.on_resume)
        self.btn_resume.setEnabled(False)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        
        self.btn_toggle_trails = QtWidgets.QPushButton("Toggle Trails")
        self.btn_toggle_trails.clicked.connect(self.on_toggle_trails)
        self.btn_toggle_flow = QtWidgets.QPushButton("Toggle Flow")
        self.btn_toggle_flow.clicked.connect(self.on_toggle_flow)
        self.btn_toggle_heatmap = QtWidgets.QPushButton("Toggle Heatmap")
        self.btn_toggle_heatmap.clicked.connect(self.on_toggle_heatmap)

        buttons_layout.addWidget(self.btn_start)
        buttons_layout.addWidget(self.btn_pause)
        buttons_layout.addWidget(self.btn_resume)
        buttons_layout.addWidget(self.btn_stop)
        buttons_layout.addWidget(self.btn_toggle_trails)
        buttons_layout.addWidget(self.btn_toggle_flow)
        buttons_layout.addWidget(self.btn_toggle_heatmap)
        buttons_layout.addStretch()

        self.status = QtWidgets.QLabel(f"Ready. Logging to: {log_filename}")
        main_layout.addWidget(self.status)

        # Worker
        self.worker = VideoWorker()
        self.worker.frame_updated.connect(self.on_frame_update)
        self.worker.topdown_signal.connect(self.update_topdown)
        self.worker.status_signal.connect(self.update_status)
        self.worker.error_signal.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        
        self.camera_widgets = {}
        self.camera_counter = 0
        self.calib_images = {}  # Store paths to calibration images
        
        # Different calibration points for each camera
        # Format: [bottom-left, top-left, top-right, bottom-right]
        self.camera_calibrations = {
            "camA.mp4": [[289, 577], [689, 156], [1102, 174], [1236, 680]],
            "camB.mp4": [[150, 600], [500, 200], [950, 200], [1200, 600]],  # Adjust these!
            "camC.mp4": [[289, 577], [689, 156], [1102, 174], [1236, 680]],  # Adjust these!
            "camD.mp4": [[289, 577], [689, 156], [1102, 174], [1236, 680]],  # Adjust these!
        }
        
        # Default calibration for manually added cameras
        self.default_calibration = [[289, 577], [689, 156], [1102, 174], [1236, 680]]
        
        logger.info("MainWindow initialized")
        
        # Auto-load videos from data/ directory
        self._auto_load_videos()
    
    def _auto_load_videos(self):
        """Automatically load camA.mp4 and camB.mp4 from data/ directory"""
        try:
            video_files = [
                ("camA.mp4", "camA_calib.jpg", "Camera A"),
                ("camB.mp4", "camB_calib.jpg", "Camera B"),
                ("camC.mp4", "camC_calib.jpg", "Camera C"),
                ("camD.mp4", "camD_calib.jpg", "Camera D")
            ]
            
            loaded_count = 0
            for video_file, calib_image, display_name in video_files:
                video_path = os.path.join(DATA_DIR, video_file)
                calib_image_path = os.path.join(DATA_DIR, calib_image)
                
                if os.path.exists(video_path):
                    logger.info(f"Auto-loading: {video_path}")
                    
                    # Check if calibration image exists
                    calibration = None
                    calib_source = "default"
                    
                    if os.path.exists(calib_image_path):
                        logger.info(f"Found calibration image: {calib_image_path}")
                        # We'll use the calibration image but need to get points first
                        # For now, use default and mark for image calibration
                        calibration = self.default_calibration
                        calib_source = "image"
                    else:
                        # Use predefined calibration if available
                        calibration = self.camera_calibrations.get(video_file, self.default_calibration)
                        calib_source = "predefined"
                    
                    logger.info(f"Using {calib_source} calibration for {video_file}")
                    
                    # Add camera
                    camera_id = f"CAM_{self.camera_counter}"
                    self.camera_counter += 1
                    
                    widget = CameraWidget(camera_id, self)
                    widget.btn_load.clicked.connect(lambda checked, cid=camera_id: self.on_load_video(cid))
                    widget.btn_calibrate.clicked.connect(lambda checked, cid=camera_id: self.on_calibrate_camera(cid))
                    widget.btn_upload_calib.clicked.connect(lambda checked, cid=camera_id: self.on_upload_calibration_image(cid))
                    widget.btn_remove.clicked.connect(lambda checked, cid=camera_id: self.on_remove_camera(cid))
                    
                    self.camera_layout.addWidget(widget)
                    self.camera_widgets[camera_id] = widget
                    
                    # Add to worker with camera-specific calibration
                    self.worker.add_camera(camera_id, calibration_points=calibration)
                    
                    # Load video
                    self.worker.update_camera_video(camera_id, video_path)
                    
                    # If calibration image exists, open calibration dialog
                    if calib_source == "image":
                        widget.info_label.setText(f"Loaded: {video_file}\nCalibration image available - click Calibrate")
                        # Store the calibration image path for later use
                        self.calib_images[camera_id] = calib_image_path
                    else:
                        widget.info_label.setText(f"Loaded: {video_file}")
                    
                    loaded_count += 1
                    logger.info(f"Auto-loaded {video_file} as {camera_id}")
            
            if loaded_count > 0:
                self.update_status(f"Auto-loaded {loaded_count} video(s) from data/ directory")
                logger.info(f"Auto-loading complete: {loaded_count} videos loaded")
                
                # Show calibration prompt if calibration images were found
                if self.calib_images:
                    QtWidgets.QMessageBox.information(
                        self, "Calibration Images Found",
                        f"Found calibration images for {len(self.calib_images)} camera(s).\n"
                        "Click 'Calibrate' button on each camera to set calibration points."
                    )
            else:
                logger.info("No videos found in data/ directory for auto-loading")
                
        except Exception as e:
            logger.error(f"Error during auto-load: {e}", exc_info=True)
            self.update_status("Failed to auto-load videos - check log")

    def on_add_camera(self):
        try:
            camera_id = f"CAM_{self.camera_counter}"
            self.camera_counter += 1
            
            widget = CameraWidget(camera_id, self)
            widget.btn_load.clicked.connect(lambda: self.on_load_video(camera_id))
            widget.btn_calibrate.clicked.connect(lambda: self.on_calibrate_camera(camera_id))
            widget.btn_upload_calib.clicked.connect(lambda: self.on_upload_calibration_image(camera_id))
            widget.btn_remove.clicked.connect(lambda: self.on_remove_camera(camera_id))
            
            self.camera_layout.addWidget(widget)
            self.camera_widgets[camera_id] = widget
            
            self.worker.add_camera(camera_id, calibration_points=self.default_calibration)
            
            self.update_status(f"Added {camera_id} - Load a video to configure")
        except Exception as e:
            logger.error(f"Error adding camera: {e}", exc_info=True)
            self.on_error(f"Error adding camera: {str(e)}")

    def on_remove_camera(self, camera_id):
        try:
            if camera_id in self.camera_widgets:
                widget = self.camera_widgets[camera_id]
                self.camera_layout.removeWidget(widget)
                widget.deleteLater()
                del self.camera_widgets[camera_id]
                
                self.worker.remove_camera(camera_id)
                self.update_status(f"Removed {camera_id}")
        except Exception as e:
            logger.error(f"Error removing camera: {e}", exc_info=True)

    def on_load_video(self, camera_id):
        try:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, f"Select video for {camera_id}", 
                "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )
            if path:
                self.worker.update_camera_video(camera_id, path)
                
                if camera_id in self.camera_widgets:
                    filename = path.split('/')[-1]
                    self.camera_widgets[camera_id].info_label.setText(f"Loaded: {filename}")
                
                self.update_status(f"{camera_id}: {filename}")
        except Exception as e:
            logger.error(f"Error loading video: {e}", exc_info=True)
    
    def on_calibrate_camera(self, camera_id):
        """Open calibration window for a camera"""
        try:
            if camera_id not in self.worker.cameras:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Camera {camera_id} not found")
                return
            
            camera = self.worker.cameras[camera_id]
            
            # Check if we have a stored calibration image for this camera
            source_path = None
            is_image = False
            
            if camera_id in self.calib_images:
                # Use calibration image
                source_path = self.calib_images[camera_id]
                is_image = True
                logger.info(f"Using calibration image: {source_path}")
            elif camera.video_path:
                # Use video first frame
                source_path = camera.video_path
                is_image = False
                logger.info(f"Using video first frame: {source_path}")
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", 
                    f"Please load a video or upload a calibration image for {camera_id} first"
                )
                return
            
            # Get current calibration points
            current_points = camera.calibration_points if camera.calibration_points else []
            
            # Open calibration dialog
            dialog = CalibrationWindow(camera_id, source_path, current_points, is_image, self)
            
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                new_points = dialog.get_points()
                
                if new_points and len(new_points) == 4:
                    # Update calibration
                    self.worker.update_camera_calibration(camera_id, new_points)
                    logger.info(f"Updated calibration for {camera_id}: {new_points}")
                    self.update_status(f"Calibration updated for {camera_id}")
                    
                    # Update widget info
                    if camera_id in self.camera_widgets:
                        widget = self.camera_widgets[camera_id]
                        current_text = widget.info_label.text().split('\n')[0]  # Keep first line
                        widget.info_label.setText(f"{current_text}\nCalibrated ✓")
                else:
                    QtWidgets.QMessageBox.warning(
                        self, "Warning",
                        "Please select exactly 4 points to calibrate"
                    )
        except Exception as e:
            logger.error(f"Error calibrating camera: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Error", f"Calibration failed: {str(e)}")
    
    def on_upload_calibration_image(self, camera_id):
        """Upload a calibration image for a camera"""
        try:
            if camera_id not in self.worker.cameras:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Camera {camera_id} not found")
                return
            
            # Open file dialog to select calibration image
            image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, f"Select calibration image for {camera_id}",
                "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
            )
            
            if not image_path:
                return
            
            # Verify image can be loaded
            test_img = cv2.imread(image_path)
            if test_img is None:
                QtWidgets.QMessageBox.critical(
                    self, "Error",
                    "Could not load the selected image"
                )
                return
            
            # Store the calibration image path
            self.calib_images[camera_id] = image_path
            
            logger.info(f"Uploaded calibration image for {camera_id}: {image_path}")
            
            # Update widget info
            if camera_id in self.camera_widgets:
                widget = self.camera_widgets[camera_id]
                current_text = widget.info_label.text().split('\n')[0]  # Keep first line
                filename = os.path.basename(image_path)
                widget.info_label.setText(f"{current_text}\nCalib image: {filename}")
            
            self.update_status(f"Calibration image uploaded for {camera_id}")
            
            # Ask if user wants to calibrate now
            reply = QtWidgets.QMessageBox.question(
                self, "Calibrate Now?",
                f"Calibration image uploaded for {camera_id}.\n\nDo you want to set calibration points now?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )
            
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                self.on_calibrate_camera(camera_id)
                
        except Exception as e:
            logger.error(f"Error uploading calibration image: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Error", f"Upload failed: {str(e)}")

    def on_start(self):
        try:
            if not self.worker.cameras:
                self.update_status("Please add at least one camera.")
                return
            
            # Reset tracking state before starting
            self.worker._reset_tracking_state()
            
            model_path = self.model_path_edit.text()
            conf = self.conf_slider.value() / 100.0
            self.worker.start_processing(model_path, conf)
            
            self.btn_start.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.btn_add_camera.setEnabled(False)
            
            for widget in self.camera_widgets.values():
                widget.btn_load.setEnabled(False)
                widget.btn_calibrate.setEnabled(False)
                widget.btn_upload_calib.setEnabled(False)
                widget.btn_remove.setEnabled(False)
            
            self.update_status(f"Processing {len(self.worker.cameras)} camera(s)...")
        except Exception as e:
            logger.error(f"Error starting: {e}", exc_info=True)

    def on_pause(self):
        self.worker.pause()
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        self.update_status("Paused")

    def on_resume(self):
        self.worker.resume()
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.update_status("Resumed")

    def on_stop(self):
        self.worker.stop_processing()
        self.update_status("Stopping...")
    
    def on_toggle_trails(self):
        self.worker.show_trajectories = not self.worker.show_trajectories
        status = "ON" if self.worker.show_trajectories else "OFF"
        self.update_status(f"Trajectories: {status}")
    
    def on_toggle_flow(self):
        self.worker.show_flow = not self.worker.show_flow
        status = "ON" if self.worker.show_flow else "OFF"
        self.update_status(f"Flow vectors: {status}")
    
    def on_toggle_heatmap(self):
        self.worker.show_heatmap = not self.worker.show_heatmap
        status = "ON" if self.worker.show_heatmap else "OFF"
        self.update_status(f"Heatmap: {status}")

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_add_camera.setEnabled(True)
        
        for widget in self.camera_widgets.values():
            widget.btn_load.setEnabled(True)
            widget.btn_calibrate.setEnabled(True)
            widget.btn_upload_calib.setEnabled(True)
            widget.btn_remove.setEnabled(True)
        
        self.update_status("Processing finished")

    def on_error(self, error_msg):
        logger.error(f"GUI Error: {error_msg}")
        QtWidgets.QMessageBox.critical(self, "Error", error_msg)
        self.update_status(f"Error: {error_msg}")
        self.on_finished()

    def on_frame_update(self, cam_id, frame):
        try:
            if cam_id in self.camera_widgets:
                qimg = cv2_to_qimage(frame)
                if qimg is None:
                    return
                
                widget = self.camera_widgets[cam_id]
                widget.label.setPixmap(
                    QPixmap.fromImage(qimg).scaled(
                        widget.label.size(),
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation
                    )
                )
        except Exception as e:
            logger.error(f"Error updating frame for {cam_id}: {e}")

    def update_topdown(self, img):
        try:
            qimg = cv2_to_qimage(img)
            if qimg is None:
                return
            
            self.labelTD.setPixmap(
                QPixmap.fromImage(qimg).scaled(
                    self.labelTD.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
            )
        except Exception as e:
            logger.error(f"Error updating topdown: {e}")

    def update_status(self, text):
        self.status.setText(text)

    def closeEvent(self, event):
        if self.worker._running:
            reply = QtWidgets.QMessageBox.question(
                self, 'Confirm Exit',
                'Processing is still running. Exit?',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )
            
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                self.worker.stop_processing()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    try:
        logger.info("="*60)
        logger.info("APPLICATION STARTING")
        logger.info("="*60)
        app = QtWidgets.QApplication(sys.argv)
        win = MainWindow()
        win.show()
        logger.info("GUI shown successfully")
        sys.exit(app.exec())
    except Exception as e:
        logger.critical("Fatal error starting application", exc_info=True)
        print(f"\n\nFATAL ERROR: {e}")
        print(f"Check log file: {log_filename}")
        raise