# ui_camera_widget.py - Camera widget for individual camera feeds
from PyQt6 import QtCore, QtWidgets

class CameraWidget(QtWidgets.QWidget):
    """Widget representing a single camera feed"""
    
    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the camera widget UI"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Video display label
        self.label = QtWidgets.QLabel(f"Camera {self.camera_id}")
        self.label.setFixedSize(360, 240)
        self.label.setStyleSheet("background:black; color:white;")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        
        # Info label
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
        
        # Second row of controls (calibration)
        control_layout2 = QtWidgets.QHBoxLayout()
        self.btn_calibrate = QtWidgets.QPushButton("Calibrate")
        self.btn_upload_calib = QtWidgets.QPushButton("Upload Calib Img")
        control_layout2.addWidget(self.btn_calibrate)
        control_layout2.addWidget(self.btn_upload_calib)
        layout.addLayout(control_layout2)