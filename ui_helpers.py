# ui_helpers.py - Helper functions for UI
import logging
import cv2
from PyQt6.QtGui import QImage

logger = logging.getLogger('CrowdAnalysis')

def cv2_to_qimage(bgr_img):
    """Convert OpenCV BGR image to QImage"""
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