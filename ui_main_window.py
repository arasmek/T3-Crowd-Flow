# ui_main_window.py - Main application window
import os
import logging
import cv2
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPixmap

import config
from ui_helpers import cv2_to_qimage
from ui_camera_widget import CameraWidget
from ui_calibration import CalibrationWindow
from video_worker import VideoWorker

logger = logging.getLogger('CrowdAnalysis')

# Get script directory for auto-loading videos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

class MainWindow(QtWidgets.QMainWindow):
    """Main application window for multi-camera crowd analysis"""
    
    def __init__(self, log_filename):
        super().__init__()
        logger.info("Initializing MainWindow")
        self.log_filename = log_filename
        self.setWindowTitle(f"Multi-Camera Crowd Analysis - Log: {log_filename}")
        self.resize(1600, 900)
        
        self.camera_widgets = {}
        self.camera_counter = 0
        self.calib_images = {}  # Store paths to calibration images
        
        # Different calibration points for each camera
        # Format: [bottom-left, top-left, top-right, bottom-right]
        self.camera_calibrations = {
            "camA.mp4": [[289, 577], [689, 156], [1102, 174], [1236, 680]],
            "camB.mp4": [[14, 691], [126, 233], [477, 207], [801, 544]],
            "camC.mp4": [[289, 577], [689, 156], [1102, 174], [1236, 680]],
            "camD.mp4": [[289, 577], [689, 156], [1102, 174], [1236, 680]],
        }
        
        # Default calibration for manually added cameras
        self.default_calibration = [[289, 577], [689, 156], [1102, 174], [1236, 680]]
        
        self._setup_ui()
        self._setup_worker()
        
        logger.info("MainWindow initialized")
        
        # Auto-load videos from data/ directory
        self._auto_load_videos()
    
    def _setup_ui(self):
        """Setup the main UI"""
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

        self.status = QtWidgets.QLabel(f"Ready. Logging to: {self.log_filename}")
        main_layout.addWidget(self.status)
    
    def _setup_worker(self):
        """Setup the video worker"""
        self.worker = VideoWorker()
        self.worker.frame_updated.connect(self.on_frame_update)
        self.worker.topdown_signal.connect(self.update_topdown)
        self.worker.status_signal.connect(self.update_status)
        self.worker.error_signal.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
    
    def _auto_load_videos(self):
        """Automatically load videos from data/ directory"""
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
                    
                    # Display thumbnail
                    self._display_video_thumbnail(camera_id, video_path)
                    
                    # If calibration image exists, store it
                    if calib_source == "image":
                        widget.info_label.setText(f"Loaded: {video_file}\nCalibration image available - click Calibrate")
                        self.calib_images[camera_id] = calib_image_path
                    else:
                        widget.info_label.setText(f"Loaded: {video_file}")
                    
                    loaded_count += 1
                    logger.info(f"Auto-loaded {video_file} as {camera_id}")
            
            if loaded_count > 0:
                status_msg = f"Auto-loaded {loaded_count} video(s) from data/ directory"
                if loaded_count == 1:
                    status_msg += " (Single camera - full frame will be used)"
                self.update_status(status_msg)
                logger.info(f"Auto-loading complete: {loaded_count} videos loaded")
                
                # Update calibration button visibility
                self._update_calibration_buttons()
                
                # Show calibration prompt if calibration images were found and multiple cameras
                if self.calib_images and loaded_count > 1:
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
    
    def _display_video_thumbnail(self, camera_id, video_path):
        """Extract first frame from video and display as thumbnail"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                # Convert to QImage and display
                qimg = cv2_to_qimage(frame)
                if qimg and camera_id in self.camera_widgets:
                    widget = self.camera_widgets[camera_id]
                    widget.label.setPixmap(
                        QPixmap.fromImage(qimg).scaled(
                            widget.label.size(),
                            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                            QtCore.Qt.TransformationMode.SmoothTransformation
                        )
                    )
                    logger.info(f"Displayed thumbnail for {camera_id}")
            else:
                logger.warning(f"Could not extract thumbnail from {video_path}")
                
        except Exception as e:
            logger.error(f"Error displaying thumbnail for {camera_id}: {e}", exc_info=True)
    
    def _update_calibration_buttons(self):
        """Update calibration button visibility based on camera count"""
        is_single_camera = len(self.worker.cameras) == 1
        
        for cam_id, widget in self.camera_widgets.items():
            # Hide/disable calibration buttons in single camera mode
            widget.btn_calibrate.setVisible(not is_single_camera)
            widget.btn_upload_calib.setVisible(not is_single_camera)
            
            # Update info text for single camera
            if is_single_camera and cam_id in self.worker.cameras:
                camera = self.worker.cameras[cam_id]
                if camera.video_path:
                    filename = os.path.basename(camera.video_path)
                    # Keep calibration info if it exists, otherwise show single camera mode
                    current_text = widget.info_label.text()
                    if "Calibration image available" not in current_text:
                        widget.info_label.setText(
                            f"Loaded: {filename}\n"
                            "Single camera mode - full frame will be used"
                        )

    def on_add_camera(self):
        """Add a new camera"""
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
            
            self._update_calibration_buttons()
            self.update_status(f"Added {camera_id} - Load a video to configure")
        except Exception as e:
            logger.error(f"Error adding camera: {e}", exc_info=True)
            self.on_error(f"Error adding camera: {str(e)}")

    def on_remove_camera(self, camera_id):
        """Remove a camera"""
        try:
            if camera_id in self.camera_widgets:
                widget = self.camera_widgets[camera_id]
                self.camera_layout.removeWidget(widget)
                widget.deleteLater()
                del self.camera_widgets[camera_id]
                
                self.worker.remove_camera(camera_id)
                self._update_calibration_buttons()
                self.update_status(f"Removed {camera_id}")
        except Exception as e:
            logger.error(f"Error removing camera: {e}", exc_info=True)

    def on_load_video(self, camera_id):
        """Load video for a camera"""
        try:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, f"Select video for {camera_id}", 
                "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )
            if path:
                self.worker.update_camera_video(camera_id, path)
                
                if camera_id in self.camera_widgets:
                    filename = os.path.basename(path)
                    self.camera_widgets[camera_id].info_label.setText(f"Loaded: {filename}")
                    
                    # Extract and display thumbnail
                    self._display_video_thumbnail(camera_id, path)
                
                # Update calibration buttons after loading
                self._update_calibration_buttons()
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
        """Start processing"""
        try:
            if not self.worker.cameras:
                self.update_status("Please add at least one camera.")
                return
            
            # Inform user about single camera mode
            if len(self.worker.cameras) == 1:
                self.update_status("Single camera mode - using full frame area")
                logger.info("Starting in single camera mode")
            
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
            
            cam_count = len(self.worker.cameras)
            mode = "single camera" if cam_count == 1 else f"{cam_count} cameras"
            self.update_status(f"Processing ({mode})...")
        except Exception as e:
            logger.error(f"Error starting: {e}", exc_info=True)

    def on_pause(self):
        """Pause processing"""
        self.worker.pause()
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        self.update_status("Paused")

    def on_resume(self):
        """Resume processing"""
        self.worker.resume()
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.update_status("Resumed")

    def on_stop(self):
        """Stop processing"""
        self.worker.stop_processing()
        self.update_status("Stopping...")
    
    def on_toggle_trails(self):
        """Toggle trajectory display"""
        self.worker.show_trajectories = not self.worker.show_trajectories
        status = "ON" if self.worker.show_trajectories else "OFF"
        self.update_status(f"Trajectories: {status}")
    
    def on_toggle_flow(self):
        """Toggle flow vectors display"""
        self.worker.show_flow = not self.worker.show_flow
        status = "ON" if self.worker.show_flow else "OFF"
        self.update_status(f"Flow vectors: {status}")
    
    def on_toggle_heatmap(self):
        """Toggle heatmap display"""
        self.worker.show_heatmap = not self.worker.show_heatmap
        status = "ON" if self.worker.show_heatmap else "OFF"
        self.update_status(f"Heatmap: {status}")

    def on_finished(self):
        """Handle processing finished"""
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
        """Handle error"""
        logger.error(f"GUI Error: {error_msg}")
        QtWidgets.QMessageBox.critical(self, "Error", error_msg)
        self.update_status(f"Error: {error_msg}")
        self.on_finished()

    def on_frame_update(self, cam_id, frame):
        """Update camera frame display"""
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
        """Update top-down view"""
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
        """Update status bar"""
        self.status.setText(text)

    def closeEvent(self, event):
        """Handle window close event"""
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