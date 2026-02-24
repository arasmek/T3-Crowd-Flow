import numpy as np
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import config
import logging

logger = logging.getLogger('CrowdAnalysis')

class MultiCameraTracker:  

    @staticmethod
    def to_numpy(x):
        if hasattr(x, "cpu"):
            return x.cpu().numpy()
        return x

    def __init__(self):
        # FIXED: Use dictionary to store trackers dynamically
        self.trackers = {}
        
        # Global track ID mapping for merging
        self.global_id_counter = 0
        self.world_position_history = {}
        self.merged_ids = {}
        
        # Store original frame dimensions for coordinate scaling
        self.original_dims = {}
        
        logger.info("MultiCameraTracker initialized with dynamic camera support")
    
    def _get_or_create_tracker(self, camera_id):
        if camera_id not in self.trackers:
            logger.info(f"Creating new DeepSort tracker for {camera_id}")
            self.trackers[camera_id] = DeepSort(
                max_age=config.MAX_AGE,
                n_init=config.N_INIT,
                max_iou_distance=config.MAX_IOU_DISTANCE,
                max_cosine_distance=config.MAX_COSINE_DISTANCE,
                nn_budget=config.NN_BUDGET,
                embedder="mobilenet",
                half=True,
                embedder_gpu=torch.cuda.is_available()
            )
            logger.info(f"DeepSort tracker created for {camera_id}")
        return self.trackers[camera_id]
    
    def resize_frame_for_inference(self, frame):
        h_orig, w_orig = frame.shape[:2]
        
        if config.MAINTAIN_ASPECT_RATIO:
            # Calculate scale to fit within target dimensions
            scale = min(config.INFERENCE_WIDTH / w_orig, 
                       config.INFERENCE_HEIGHT / h_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
        else:
            new_w = config.INFERENCE_WIDTH
            new_h = config.INFERENCE_HEIGHT
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scale_x = w_orig / new_w
        scale_y = h_orig / new_h
        
        return resized, scale_x, scale_y
    
    def scale_detections(self, boxes, scale_x, scale_y):
        scaled_boxes = []
        for box in boxes:
            xyxy = self.to_numpy(box.xyxy[0])
            conf = float(self.to_numpy(box.conf[0]))

            x1, y1, x2, y2 = xyxy
            
            # Scale back to original coordinates
            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y
            
            scaled_boxes.append({
                'xyxy': [x1_scaled, y1_scaled, x2_scaled, y2_scaled],
                'conf': conf
            })
        
        return scaled_boxes
    
    def update_tracks(self, detections, frame, camera_id, homography, world_bounds):
        """Update tracks for a specific camera - FIXED VERSION"""
        # Get or create tracker for this camera
        tracker = self._get_or_create_tracker(camera_id)
        
        # Store original dimensions if not yet stored
        if camera_id not in self.original_dims:
            self.original_dims[camera_id] = frame.shape[:2]
            logger.debug(f"Stored dimensions for {camera_id}: {frame.shape[:2]}")
        
        # Convert YOLO detections to DeepSORT format
        det_list = []
        for box in detections:
            xyxy = self.to_numpy(box.xyxy[0])
            conf = float(self.to_numpy(box.conf[0]))

            x1, y1, x2, y2 = xyxy

            det_list.append((
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                conf,
                0
            ))
        
        if len(det_list) > 0:
            logger.debug(f"{camera_id}: Processing {len(det_list)} detections")
        
        # Update tracker with original-resolution frame for embedding
        tracks = tracker.update_tracks(det_list, frame=frame)
        
        # Process tracks and filter to grid bounds
        processed_tracks = []
        world_w, world_h = world_bounds
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            ltrb = track.to_ltrb()
            
            # Calculate foot point (bottom center)
            foot_x = (ltrb[0] + ltrb[2]) / 2
            foot_y = ltrb[3]
            
            # Project to world coordinates
            try:
                pts = np.array([[[foot_x, foot_y]]], dtype=np.float32)
                world_pt = cv2.perspectiveTransform(pts, homography)[0, 0]
                
                wx, wy = float(world_pt[0]), float(world_pt[1])
            except Exception as e:
                logger.error(f"Error projecting point to world coords: {e}")
                continue

            # Filter to world bounds
            if not (0 <= wx <= world_w and 0 <= wy <= world_h):
                continue

            local_id = f"{camera_id}_{track.track_id}"
            
            # Add attributes
            track.world_x = wx
            track.world_y = wy
            track.camera_id = camera_id
            track.local_id = local_id
            track.ltrb = ltrb
            
            # Store position history for merging
            if local_id not in self.world_position_history:
                self.world_position_history[local_id] = []
            self.world_position_history[local_id].append((wx, wy))
            if len(self.world_position_history[local_id]) > 10:
                self.world_position_history[local_id].pop(0)
            
            processed_tracks.append(track)
        
        logger.debug(f"{camera_id}: {len(processed_tracks)} tracks confirmed and in bounds")
        return processed_tracks
    
    def merge_camera_tracks(self, tracks_A, tracks_B):
        all_tracks = []
        matched_B = set()
        
        # Match tracks between cameras based on world position
        for track_A in tracks_A:
            best_match = None
            min_dist = float('inf')
            
            for i, track_B in enumerate(tracks_B):
                if i in matched_B:
                    continue

                dist = np.sqrt(
                    (track_A.world_x - track_B.world_x)**2 + 
                    (track_A.world_y - track_B.world_y)**2
                )
                
                # Find closest match within threshold (0.5 meters)
                if dist < min_dist and dist < 0.5:
                    min_dist = dist
                    best_match = i
            
            if best_match is not None:
                # Average the positions for better accuracy
                track_B = tracks_B[best_match]
                track_A.world_x = (track_A.world_x + track_B.world_x) / 2
                track_A.world_y = (track_A.world_y + track_B.world_y) / 2
                matched_B.add(best_match)
                
                # Use camera A's ID as primary but mark as merged
                track_A.merged_from = [track_A.local_id, track_B.local_id]
                
                logger.debug(f"Merged {track_A.local_id} with {track_B.local_id} (dist: {min_dist:.2f}m)")
            
            # Assign global ID
            if track_A.local_id not in self.merged_ids:
                self.merged_ids[track_A.local_id] = self.global_id_counter
                self.global_id_counter += 1
            track_A.global_id = self.merged_ids[track_A.local_id]
            
            all_tracks.append(track_A)
        
        # Add unmatched tracks from camera B
        for i, track_B in enumerate(tracks_B):
            if i not in matched_B:
                if track_B.local_id not in self.merged_ids:
                    self.merged_ids[track_B.local_id] = self.global_id_counter
                    self.global_id_counter += 1
                track_B.global_id = self.merged_ids[track_B.local_id]
                all_tracks.append(track_B)
        
        logger.debug(f"Merge result: {len(all_tracks)} total tracks ({len(matched_B)} merged)")
        return all_tracks