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
        
        # Flow field
        self.flow_field_x = np.zeros((self.hmap_h, self.hmap_w), dtype=np.float32)
        self.flow_field_y = np.zeros((self.hmap_h, self.hmap_w), dtype=np.float32)
        self.flow_count = np.zeros((self.hmap_h, self.hmap_w), dtype=np.float32)
        
        # Density heatmap for congestion
        self.density_map = np.zeros((self.hmap_h, self.hmap_w), dtype=np.float32)
        
        # Heatmap configuration - UPDATED DEBUG VALUES
        self.heatmap_min_people = 1  # Changed from 2 to 1
        self.heatmap_max_people = 4  # Changed from 10 to 4
        
        # Statistics
        self.current_count = 0
        self.total_count = set()
        self.frame_number = 0
        
        print(f"[CrowdFlowAnalyzer] Initialized heatmap: {self.hmap_w}x{self.hmap_h} cells")
        print(f"[CrowdFlowAnalyzer] World bounds: {self.world_w}x{self.world_h}")
        print(f"[CrowdFlowAnalyzer] Cell size: {self.cell_size}")
        
    def world_to_heatmap(self, wx, wy):
        hx = int(wx / self.cell_size)
        hy = int((self.world_h - wy) / self.cell_size)  # ← FLIP Y

        hx = max(0, min(self.hmap_w - 1, hx))
        hy = max(0, min(self.hmap_h - 1, hy))
        return hx, hy
    
    def update(self, tracks):
        self.frame_number += 1

        # Decay flow field
        self.flow_field_x *= config.HEATMAP_DECAY
        self.flow_field_y *= config.HEATMAP_DECAY
        self.flow_count *= config.HEATMAP_DECAY
        
        # Decay density map
        self.density_map *= config.HEATMAP_DECAY
        
        self.current_count = len(tracks)
        active_ids = set()
        
        for track in tracks:
            track_id = track.global_id
            wx, wy = track.world_x, track.world_y
            
            active_ids.add(track_id)
            self.total_count.add(track_id)
            self.trajectories[track_id].append((wx, wy))

            # Update density heatmap - FIX: Use correct coordinate system
            hx, hy = self.world_to_heatmap(wx, wy)
            
            # DEBUG: Print first few updates to verify coordinates
            if self.frame_number < 3:
                print(f"[DEBUG] Track {track_id}: world({wx:.2f}, {wy:.2f}) -> heatmap({hx}, {hy})")
            
            self.density_map[hy, hx] += 1.0

            if track_id in self.last_positions:
                last_x, last_y = self.last_positions[track_id]
                vx = wx - last_x
                vy = wy - last_y
                self.velocities[track_id] = (vx, vy)

                self.flow_field_x[hy, hx] += vx
                self.flow_field_y[hy, hx] += vy
                self.flow_count[hy, hx] += 1
            
            self.last_positions[track_id] = (wx, wy)
            self.last_update_time[track_id] = self.frame_number

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
        return list(self.trajectories.get(track_id, []))
    
    def get_velocity(self, track_id):
        return self.velocities.get(track_id, (0, 0))
    
    def get_direction(self, track_id):
        traj = self.get_trajectory(track_id)
        if len(traj) < config.MIN_TRAJECTORY_LENGTH:
            return None
        
        # Calculate overall displacement
        start = traj[0]
        end = traj[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 0.01:
            return None
        
        # Calculate angle in degrees
        angle = np.arctan2(dy, dx) * 180 / np.pi
        speed = distance / len(traj)
        
        return {'angle': angle, 'speed': speed, 'dx': dx, 'dy': dy}
    
    def predict_position(self, track_id, frames_ahead):
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
        vectors = []
        
        for hy in range(self.hmap_h):
            for hx in range(self.hmap_w):
                if self.flow_count[hy, hx] > 0.5:
                    avg_vx = self.flow_field_x[hy, hx] / self.flow_count[hy, hx]
                    avg_vy = self.flow_field_y[hy, hx] / self.flow_count[hy, hx]
                    
                    magnitude = np.sqrt(avg_vx**2 + avg_vy**2)
                    if magnitude > 0.01:
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
    
    def get_density_heatmap(self, smooth_sigma=2.0):
        """
        Returns a smoothed density heatmap for visualization.
        
        Args:
            smooth_sigma: Gaussian smoothing parameter for smooth heatmap
            
        Returns:
            Smoothed density map normalized between 0 and 1
        """
        # Apply Gaussian filter for smooth heatmap
        smoothed = gaussian_filter(self.density_map, sigma=smooth_sigma)
        
        # Normalize based on configurable min/max people
        # Values below min_people will be 0, at max_people will be 1
        normalized = np.clip(
            (smoothed - self.heatmap_min_people) / (self.heatmap_max_people - self.heatmap_min_people),
            0, 1
        )
        
        return normalized
    
    def set_heatmap_range(self, min_people, max_people):
        """Set the range for heatmap visualization"""
        self.heatmap_min_people = min_people
        self.heatmap_max_people = max_people
    
    def get_congestion_zones(self, threshold=0.7):
        """
        Identify high-density congestion zones.
        
        Args:
            threshold: Density threshold (0-1) to classify as congested
            
        Returns:
            List of congestion zone centers in world coordinates
        """
        heatmap = self.get_density_heatmap()
        congestion_zones = []
        
        for hy in range(self.hmap_h):
            for hx in range(self.hmap_w):
                if heatmap[hy, hx] >= threshold:
                    wx = (hx + 0.5) * self.cell_size
                    wy = (hy + 0.5) * self.cell_size
                    congestion_zones.append({
                        'x': wx,
                        'y': wy,
                        'density': heatmap[hy, hx]
                    })
        
        return congestion_zones
    
    def get_statistics(self):
        return {
            'current_count': self.current_count,
            'total_unique': len(self.total_count)
        }