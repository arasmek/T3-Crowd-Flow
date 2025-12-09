# crowd_analytics.py
import numpy as np
import cv2
from collections import defaultdict, deque
from scipy.ndimage import gaussian_filter
import config

class CrowdFlowAnalyzer:
    def __init__(self, world_w, world_h, heatmap_cell_size=0.2):
        self.world_w = world_w
        self.world_h = world_h
        self.cell_size = heatmap_cell_size
        
        # Heatmap dimensions
        self.hmap_w = int(np.ceil(world_w / self.cell_size))
        self.hmap_h = int(np.ceil(world_h / self.cell_size))
        
        # Tracking data
        self.trajectories = defaultdict(lambda: deque(maxlen=config.TRAJECTORY_HISTORY))
        self.velocities = {}
        self.velocity_history = defaultdict(lambda: deque(maxlen=config.VELOCITY_SMOOTHING_FRAMES))
        self.last_positions = {}
        self.last_update_time = {}
        
        # Flow field only (heatmap removed)
        self.flow_field_x = np.zeros((self.hmap_h, self.hmap_w), dtype=np.float32)
        self.flow_field_y = np.zeros((self.hmap_h, self.hmap_w), dtype=np.float32)
        self.flow_count = np.zeros((self.hmap_h, self.hmap_w), dtype=np.float32)
        
        # Statistics
        self.current_count = 0
        self.total_count = set()
        self.frame_number = 0
        
    def world_to_heatmap(self, wx, wy):
        """Convert world coordinates to heatmap cell indices."""
        hx = int(wx / self.cell_size)
        hy = int(wy / self.cell_size)
        # Clamp to valid range
        hx = max(0, min(self.hmap_w - 1, hx))
        hy = max(0, min(self.hmap_h - 1, hy))
        return hx, hy
    
    def update(self, tracks):
        """Update analyzer with new tracking data."""
        self.frame_number += 1
        
        # Decay flow fields
        self.flow_field_x *= config.HEATMAP_DECAY
        self.flow_field_y *= config.HEATMAP_DECAY
        self.flow_count *= config.HEATMAP_DECAY
        
        self.current_count = len(tracks)
        active_ids = set()
        
        for track in tracks:
            track_id = track.global_id
            wx, wy = track.world_x, track.world_y
            
            active_ids.add(track_id)
            self.total_count.add(track_id)
            
            # Update trajectory
            self.trajectories[track_id].append((wx, wy))
            
            # Calculate velocity
            if track_id in self.last_positions:
                last_x, last_y = self.last_positions[track_id]
                vx = wx - last_x
                vy = wy - last_y
                self.velocities[track_id] = (vx, vy)
                
                # Update flow field (movement direction)
                hx, hy = self.world_to_heatmap(wx, wy)
                self.flow_field_x[hy, hx] += vx
                self.flow_field_y[hy, hx] += vy
                self.flow_count[hy, hx] += 1
            
            self.last_positions[track_id] = (wx, wy)
            self.last_update_time[track_id] = self.frame_number
        
        # Clean up old tracks (not seen for 30 frames)
        ids_to_remove = []
        for tid in list(self.trajectories.keys()):
            if tid not in active_ids:
                if self.frame_number - self.last_update_time.get(tid, 0) > 30:
                    ids_to_remove.append(tid)
        
        for tid in ids_to_remove:
            self.trajectories.pop(tid, None)
            self.velocities.pop(tid, None)
            self.last_positions.pop(tid, None)
            self.last_update_time.pop(tid, None)
    
    def get_trajectory(self, track_id):
        """Get trajectory for a specific track."""
        return list(self.trajectories.get(track_id, []))
    
    def get_velocity(self, track_id):
        """Get velocity vector for a track."""
        return self.velocities.get(track_id, (0, 0))
    
    def get_direction(self, track_id):
        """Calculate dominant direction for a track's trajectory."""
        traj = self.get_trajectory(track_id)
        if len(traj) < config.MIN_TRAJECTORY_LENGTH:
            return None
        
        # Calculate overall displacement
        start = traj[0]
        end = traj[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 0.01:  # Too small movement
            return None
        
        # Calculate angle in degrees
        angle = np.arctan2(dy, dx) * 180 / np.pi
        speed = distance / len(traj)
        
        return {'angle': angle, 'speed': speed, 'dx': dx, 'dy': dy}
    
    def predict_position(self, track_id, frames_ahead):
        """Predict future position based on current velocity."""
        if track_id not in self.last_positions or track_id not in self.velocities:
            return None
        
        wx, wy = self.last_positions[track_id]
        vx, vy = self.velocities[track_id]
        
        pred_x = wx + vx * frames_ahead
        pred_y = wy + vy * frames_ahead
        
        # Clamp to world bounds
        pred_x = max(0, min(self.world_w, pred_x))
        pred_y = max(0, min(self.world_h, pred_y))
        
        return (pred_x, pred_y)
    
    def get_flow_vectors(self):
        """Get average flow direction for each cell."""
        vectors = []
        
        for hy in range(self.hmap_h):
            for hx in range(self.hmap_w):
                if self.flow_count[hy, hx] > 0.5:
                    # Average flow in this cell
                    avg_vx = self.flow_field_x[hy, hx] / self.flow_count[hy, hx]
                    avg_vy = self.flow_field_y[hy, hx] / self.flow_count[hy, hx]
                    
                    magnitude = np.sqrt(avg_vx**2 + avg_vy**2)
                    if magnitude > 0.01:  # Only if significant movement
                        wx = (hx + 0.5) * self.cell_size
                        wy = (hy + 0.5) * self.cell_size
                        
                        vectors.append({
                            'x': wx,
                            'y': wy,
                            'vx': avg_vx,
                            'vy': avg_vy,
                            'magnitude': magnitude
                        })
        
        return vectors
    
    def get_statistics(self):
        """Get current statistics."""
        return {
            'current_count': self.current_count,
            'total_unique': len(self.total_count)
        }