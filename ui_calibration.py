# ui_calibration.py - Calibration window for camera setup
import logging
import cv2
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPixmap
from ui_helpers import cv2_to_qimage

logger = logging.getLogger('CrowdAnalysis')

class CalibrationWindow(QtWidgets.QDialog):
    """Interactive calibration window for selecting 4 corner points"""
    
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
        
        self._setup_ui()
        self.load_frame()
    
    def _setup_ui(self):
        """Setup the calibration UI"""
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