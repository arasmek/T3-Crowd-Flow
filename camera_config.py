# camera_config.py - Camera configuration class
import logging

logger = logging.getLogger('CrowdAnalysis')

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