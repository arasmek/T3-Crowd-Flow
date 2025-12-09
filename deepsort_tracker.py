# deepsort_tracker.py
import numpy as np
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import config

class MultiCameraTracker:
    """Wrapper for DeepSORT tracking across multiple cameras."""
    
    def __init__(self):
        # Initialize separate trackers for each camera
        self.tracker_A = DeepSort(
            max_age=config.MAX_AGE,
            n_init=config.N_INIT,
            max_iou_distance=config.MAX_IOU_DISTANCE,
            max_cosine_distance=config.MAX_COSINE_DISTANCE,
            nn_budget=config.NN_BUDGET,
            embedder="mobilenet",
            half=True,
            embedder_gpu=torch.cuda.is_available()
        )
        
        self.tracker_B = DeepSort(
            max_age=config.MAX_AGE,
            n_init=config.N_INIT,
            max_iou_distance=config.MAX_IOU_DISTANCE,
            max_cosine_distance=config.MAX_COSINE_DISTANCE,
            nn_budget=config.NN_BUDGET,
            embedder="mobilenet",
            half=True,
            embedder_gpu=torch.cuda.is_available()
        )
        
        # Global track ID mapping for merging
        self.global_id_counter = 0
        self.world_position_history = {}  # track_id -> recent world positions
        self.merged_ids = {}  # local_id -> global_id mapping
    
    def update_tracks(self, detections, frame, camera_id, homography, world_bounds):
        """
        Update tracks for a specific camera.
        
        Args:
            detections: YOLO detection results
            frame: Current frame image
            camera_id: 'A' or 'B'
            homography: Homography matrix for this camera
            world_bounds: (world_w, world_h) tuple
            
        Returns:
            List of Track objects with world coordinates (only those in bounds)
        """
        tracker = self.tracker_A if camera_id == 'A' else self.tracker_B
        
        # Convert YOLO detections to DeepSORT format
        det_list = []
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            det_list.append((
                [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                float(conf),
                0  # class: person
            ))
        
        # Update tracker
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
            pts = np.array([[[foot_x, foot_y]]], dtype=np.float32)
            world_pt = cv2.perspectiveTransform(pts, homography)[0, 0]
            
            wx, wy = float(world_pt[0]), float(world_pt[1])
            
            # FILTER: Only keep tracks within world bounds
            if not (0 <= wx <= world_w and 0 <= wy <= world_h):
                continue
            
            # Create unique camera-specific ID
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
            # Keep only last 10 positions
            if len(self.world_position_history[local_id]) > 10:
                self.world_position_history[local_id].pop(0)
            
            processed_tracks.append(track)
        
        return processed_tracks
    
    def merge_camera_tracks(self, tracks_A, tracks_B):
        """
        Intelligently merge tracks from both cameras.
        Same person seen by both cameras gets one global ID.
        """
        all_tracks = []
        matched_B = set()
        
        # First pass: Match tracks between cameras based on world position
        for track_A in tracks_A:
            best_match = None
            min_dist = float('inf')
            
            for i, track_B in enumerate(tracks_B):
                if i in matched_B:
                    continue
                
                # Calculate distance in world coordinates
                dist = np.sqrt(
                    (track_A.world_x - track_B.world_x)**2 + 
                    (track_A.world_y - track_B.world_y)**2
                )
                
                # Find closest match within threshold
                if dist < min_dist and dist < 0.5:  # 50cm threshold
                    min_dist = dist
                    best_match = i
            
            if best_match is not None:
                # Merge: Average the positions for better accuracy
                track_B = tracks_B[best_match]
                track_A.world_x = (track_A.world_x + track_B.world_x) / 2
                track_A.world_y = (track_A.world_y + track_B.world_y) / 2
                matched_B.add(best_match)
                
                # Use camera A's ID as primary but mark as merged
                track_A.merged_from = [track_A.local_id, track_B.local_id]
            
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
        
        return all_tracks