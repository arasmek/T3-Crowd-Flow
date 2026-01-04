# ui_calibration.py - Enhanced calibration window with edge/corner detection
import logging
import cv2
import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPixmap
from ui_helpers import cv2_to_qimage

logger = logging.getLogger('CrowdAnalysis')

class CalibrationWindow(QtWidgets.QDialog):
    """Interactive calibration window with automatic edge/corner detection"""
    
    def __init__(self, camera_id, source_path, current_points, is_image=False, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.source_path = source_path
        self.is_image = is_image
        self.points = current_points.copy() if current_points else []
        self.frame = None
        self.frame_gray = None
        self.display_frame = None
        
        # Detection mode
        self.detection_mode = "auto"  # "auto" or "manual"
        self.detected_corners = []
        self.detected_lines = []
        self.detected_intersections = []
        self.snap_radius = 30  # Pixels to snap to detected features
        
        self.setWindowTitle(f"Calibrate {camera_id}")
        self.resize(1000, 800)
        
        self._setup_ui()
        self.load_frame()
    
    def _setup_ui(self):
        """Setup the calibration UI"""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Instructions
        self.instructions = QtWidgets.QLabel()
        self.update_instructions()
        self.instructions.setStyleSheet("background: gray; padding: 10px; font-weight: bold;")
        self.instructions.setWordWrap(True)
        layout.addWidget(self.instructions)
        
        # Mode selection
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Detection Mode:"))
        
        self.radio_auto = QtWidgets.QRadioButton("Auto-Detect (Recommended)")
        self.radio_auto.setChecked(True)
        self.radio_auto.toggled.connect(self.on_mode_changed)
        
        self.radio_manual = QtWidgets.QRadioButton("Manual Selection")
        self.radio_manual.toggled.connect(self.on_mode_changed)
        
        mode_layout.addWidget(self.radio_auto)
        mode_layout.addWidget(self.radio_manual)
        mode_layout.addStretch()
        
        # Detection settings button
        self.btn_detect_settings = QtWidgets.QPushButton("Detection Settings")
        self.btn_detect_settings.clicked.connect(self.show_detection_settings)
        mode_layout.addWidget(self.btn_detect_settings)
        
        layout.addLayout(mode_layout)
        
        # Detection info
        self.detection_info = QtWidgets.QLabel()
        self.detection_info.setStyleSheet("color: blue; font-style: italic;")
        layout.addWidget(self.detection_info)
        
        # Image display
        self.image_label = QtWidgets.QLabel()
        self.image_label.setMinimumSize(900, 650)
        self.image_label.setStyleSheet("background: black;")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.mousePressEvent = self.on_image_click
        layout.addWidget(self.image_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.btn_detect = QtWidgets.QPushButton("Run Detection")
        self.btn_detect.clicked.connect(self.run_detection)
        button_layout.addWidget(self.btn_detect)
        
        self.btn_reset = QtWidgets.QPushButton("Reset Points")
        self.btn_reset.clicked.connect(self.reset_points)
        button_layout.addWidget(self.btn_reset)
        
        button_layout.addStretch()
        
        self.btn_save = QtWidgets.QPushButton("Save Calibration")
        self.btn_save.clicked.connect(self.accept)
        button_layout.addWidget(self.btn_save)
        
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        layout.addLayout(button_layout)
    
    def update_instructions(self):
        """Update instruction text based on mode"""
        if self.detection_mode == "auto":
            text = (
                "AUTO-DETECT MODE: Click 'Run Detection' to find edges and corners automatically.\n"
                "Then click near detected features (shown in green) to select calibration points.\n"
                "Order: 1. Bottom-Left > 2. Top-Left > 3. Top-Right > 4. Bottom-Right"
            )
        else:
            text = (
                "MANUAL MODE: Click 4 points in this order:\n"
                "1. Bottom-Left > 2. Top-Left > 3. Top-Right > 4. Bottom-Right\n"
                "(These should form the area you want to map to the world coordinates)"
            )
        self.instructions.setText(text)
    
    def on_mode_changed(self):
        """Handle mode change"""
        self.detection_mode = "auto" if self.radio_auto.isChecked() else "manual"
        self.update_instructions()
        self.update_display()
        logger.info(f"Calibration mode changed to: {self.detection_mode}")
    
    def load_frame(self):
        """Load frame from video or image"""
        if self.is_image:
            frame = cv2.imread(self.source_path)
            if frame is not None:
                self.frame = frame
                self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Auto-run detection for convenience
                self.run_detection()
            else:
                QtWidgets.QMessageBox.critical(self, "Error", "Could not load calibration image")
                self.reject()
        else:
            cap = cv2.VideoCapture(self.source_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                self.frame = frame
                self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Auto-run detection for convenience
                self.run_detection()
            else:
                QtWidgets.QMessageBox.critical(self, "Error", "Could not load video frame")
                self.reject()
    
    def run_detection(self):
        """Run edge and corner detection"""
        if self.frame_gray is None:
            return
        
        logger.info("Running edge and corner detection...")
        self.detection_info.setText("Detecting edges and corners...")
        QtWidgets.QApplication.processEvents()
        
        try:
            # 1. Detect corners using Shi-Tomasi
            self.detect_corners()
            
            # 2. Detect lines using Hough Transform
            self.detect_lines()
            
            # 3. Find line intersections
            self.find_intersections()
            
            # Update info
            info_text = (
                f"Detected: {len(self.detected_corners)} corners, "
                f"{len(self.detected_lines)} lines, "
                f"{len(self.detected_intersections)} intersections"
            )
            self.detection_info.setText(info_text)
            logger.info(info_text)
            
            # Update display
            self.update_display()
            
        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            self.detection_info.setText(f"Detection failed: {str(e)}")
    
    def detect_corners(self):
        """Detect corner points using Shi-Tomasi corner detection"""
        # Parameters for corner detection
        max_corners = 100
        quality_level = 0.01
        min_distance = 30
        
        corners = cv2.goodFeaturesToTrack(
            self.frame_gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )
        
        if corners is not None:
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(
                self.frame_gray, corners, (5, 5), (-1, -1), criteria
            )
            
            self.detected_corners = corners.reshape(-1, 2).tolist()
            logger.info(f"Detected {len(self.detected_corners)} corners")
        else:
            self.detected_corners = []
            logger.warning("No corners detected")
    
    def detect_lines(self):
        """Detect straight lines using Hough Transform"""
        # Edge detection
        edges = cv2.Canny(self.frame_gray, 50, 150, apertureSize=3)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=50,
            maxLineGap=10
        )
        
        if lines is not None:
            self.detected_lines = lines.reshape(-1, 4).tolist()
            logger.info(f"Detected {len(self.detected_lines)} lines")
        else:
            self.detected_lines = []
            logger.warning("No lines detected")
    
    def find_intersections(self):
        """Find intersections of detected lines"""
        self.detected_intersections = []
        
        if len(self.detected_lines) < 2:
            return
        
        # Convert lines to (a, b, c) format: ax + by + c = 0
        def line_to_abc(x1, y1, x2, y2):
            a = y2 - y1
            b = x1 - x2
            c = x2*y1 - x1*y2
            return (a, b, c)
        
        # Find intersection of two lines
        def line_intersection(line1, line2):
            a1, b1, c1 = line_to_abc(*line1)
            a2, b2, c2 = line_to_abc(*line2)
            
            det = a1*b2 - a2*b1
            if abs(det) < 1e-6:  # Lines are parallel
                return None
            
            x = (b1*c2 - b2*c1) / det
            y = (a2*c1 - a1*c2) / det
            
            # Check if intersection is within image bounds
            h, w = self.frame_gray.shape
            if 0 <= x < w and 0 <= y < h:
                return [x, y]
            return None
        
        # Find all intersections
        for i, line1 in enumerate(self.detected_lines):
            for line2 in self.detected_lines[i+1:]:
                intersection = line_intersection(line1, line2)
                if intersection:
                    # Check if intersection is far enough from existing ones
                    is_unique = True
                    for existing in self.detected_intersections:
                        dist = np.sqrt((intersection[0]-existing[0])**2 + 
                                     (intersection[1]-existing[1])**2)
                        if dist < 20:  # Minimum distance between intersections
                            is_unique = False
                            break
                    
                    if is_unique:
                        self.detected_intersections.append(intersection)
        
        logger.info(f"Found {len(self.detected_intersections)} intersections")
    
    def find_nearest_feature(self, x, y):
        """Find nearest detected feature to clicked point"""
        min_dist = float('inf')
        nearest_point = None
        
        # Check all detected features
        all_features = (
            self.detected_corners + 
            self.detected_intersections
        )
        
        for feature in all_features:
            fx, fy = feature[0], feature[1] if len(feature) > 1 else feature[0]
            dist = np.sqrt((x - fx)**2 + (y - fy)**2)
            
            if dist < min_dist and dist < self.snap_radius:
                min_dist = dist
                nearest_point = [int(fx), int(fy)]
        
        return nearest_point
    
    def update_display(self):
        """Update the display with current points and detected features"""
        if self.frame is None:
            return
        
        display = self.frame.copy()
        
        # Draw detected features if in auto mode
        if self.detection_mode == "auto":
            # Draw corners (small green circles)
            for corner in self.detected_corners:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(display, (x, y), 3, (0, 255, 0), -1)
            
            # Draw lines (thin cyan lines)
            for line in self.detected_lines:
                x1, y1, x2, y2 = map(int, line)
                cv2.line(display, (x1, y1), (x2, y2), (255, 255, 0), 1)
            
            # Draw intersections (larger green circles)
            for intersection in self.detected_intersections:
                x, y = int(intersection[0]), int(intersection[1])
                cv2.circle(display, (x, y), 6, (0, 255, 0), 2)
        
        # Draw selected calibration points
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        labels = ["1:BL", "2:TL", "3:TR", "4:BR"]
        
        for i, pt in enumerate(self.points):
            cv2.circle(display, tuple(pt), 10, colors[i], -1)
            cv2.circle(display, tuple(pt), 12, (255, 255, 255), 2)
            cv2.putText(display, labels[i], (pt[0]+15, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2)
        
        # Draw lines between selected points
        if len(self.points) >= 2:
            for i in range(len(self.points)-1):
                cv2.line(display, tuple(self.points[i]), tuple(self.points[i+1]), 
                        (255, 255, 255), 3)
        if len(self.points) == 4:
            cv2.line(display, tuple(self.points[3]), tuple(self.points[0]), 
                    (255, 255, 255), 3)
        
        # Convert to QImage and display
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
        if not (0 <= click_x < self.frame.shape[1] and 0 <= click_y < self.frame.shape[0]):
            return
        
        # In auto mode, snap to nearest detected feature
        if self.detection_mode == "auto":
            nearest = self.find_nearest_feature(click_x, click_y)
            if nearest:
                self.points.append(nearest)
                logger.info(f"Point {len(self.points)} snapped to detected feature: {nearest}")
            else:
                # No feature nearby, use clicked point
                self.points.append([click_x, click_y])
                logger.info(f"Point {len(self.points)} selected (no feature nearby): ({click_x}, {click_y})")
        else:
            # Manual mode - use exact clicked point
            self.points.append([click_x, click_y])
            logger.info(f"Point {len(self.points)} selected: ({click_x}, {click_y})")
        
        self.update_display()
    
    def reset_points(self):
        """Reset all selected points"""
        self.points = []
        self.update_display()
        logger.info("Calibration points reset")
    
    def show_detection_settings(self):
        """Show dialog for adjusting detection parameters"""
        dialog = DetectionSettingsDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Re-run detection with new settings
            self.run_detection()
    
    def get_points(self):
        """Get the calibration points"""
        return self.points if len(self.points) == 4 else None


class DetectionSettingsDialog(QtWidgets.QDialog):
    """Dialog for adjusting edge/corner detection parameters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        
        self.setWindowTitle("Detection Settings")
        self.resize(400, 300)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Corner detection settings
        layout.addWidget(QtWidgets.QLabel("<b>Corner Detection:</b>"))
        
        corner_layout = QtWidgets.QFormLayout()
        
        self.corner_quality = QtWidgets.QDoubleSpinBox()
        self.corner_quality.setRange(0.001, 0.1)
        self.corner_quality.setSingleStep(0.001)
        self.corner_quality.setValue(0.01)
        self.corner_quality.setDecimals(3)
        corner_layout.addRow("Quality Level:", self.corner_quality)
        
        self.corner_distance = QtWidgets.QSpinBox()
        self.corner_distance.setRange(10, 100)
        self.corner_distance.setValue(30)
        corner_layout.addRow("Min Distance:", self.corner_distance)
        
        layout.addLayout(corner_layout)
        
        # Line detection settings
        layout.addWidget(QtWidgets.QLabel("<b>Line Detection:</b>"))
        
        line_layout = QtWidgets.QFormLayout()
        
        self.line_threshold = QtWidgets.QSpinBox()
        self.line_threshold.setRange(50, 200)
        self.line_threshold.setValue(100)
        line_layout.addRow("Threshold:", self.line_threshold)
        
        self.line_min_length = QtWidgets.QSpinBox()
        self.line_min_length.setRange(20, 200)
        self.line_min_length.setValue(50)
        line_layout.addRow("Min Length:", self.line_min_length)
        
        layout.addLayout(line_layout)
        
        # Snap radius
        layout.addWidget(QtWidgets.QLabel("<b>Snapping:</b>"))
        
        snap_layout = QtWidgets.QFormLayout()
        self.snap_radius = QtWidgets.QSpinBox()
        self.snap_radius.setRange(10, 100)
        self.snap_radius.setValue(30)
        snap_layout.addRow("Snap Radius:", self.snap_radius)
        
        layout.addLayout(snap_layout)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.clicked.connect(self.apply_settings)
        
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(btn_apply)
        button_layout.addWidget(btn_cancel)
        
        layout.addLayout(button_layout)
    
    def apply_settings(self):
        """Apply settings and close dialog"""
        # Update parent window's detection parameters
        if hasattr(self.parent_window, 'snap_radius'):
            self.parent_window.snap_radius = self.snap_radius.value()
        
        self.accept()