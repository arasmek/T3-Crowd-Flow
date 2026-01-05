import os
import logging
import cv2
import numpy as np
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

class StreamInputDialog(QtWidgets.QDialog):
    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.setWindowTitle(f"Load Stream for {camera_id}")
        self.resize(550, 300)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Instructions
        instructions = QtWidgets.QLabel(
            "Enter a stream URL or select a local webcam/virtual camera.\n\n"
            "Supported formats:\n"
            "• Webcam: 0, 1, 2, ... (device index)\n"
            "• RTMP: rtmp://server/stream\n"
            "• RTSP: rtsp://server/stream\n"
            "• HTTP/HLS: http://server/stream.m3u8\n"
            "• YouTube: Use yt-dlp to get direct stream URL (see guide)\n"
            "• OBS Virtual Camera: Use device index (usually 0 or 1)"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Stream type selector
        type_layout = QtWidgets.QHBoxLayout()
        type_layout.addWidget(QtWidgets.QLabel("Stream Type:"))
        self.stream_type = QtWidgets.QComboBox()
        self.stream_type.addItems([
            "Custom URL",
            "Webcam/Virtual Camera (0)",
            "Webcam/Virtual Camera (1)",
            "Webcam/Virtual Camera (2)",
            "Webcam/Virtual Camera (3)"
        ])
        self.stream_type.currentIndexChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.stream_type)
        layout.addLayout(type_layout)
        
        # URL/Index input
        url_layout = QtWidgets.QHBoxLayout()
        url_layout.addWidget(QtWidgets.QLabel("Stream URL:"))
        self.url_input = QtWidgets.QLineEdit()
        self.url_input.setPlaceholderText("rtmp://localhost/live/stream or rtsp://... or http://...")
        url_layout.addWidget(self.url_input)
        layout.addLayout(url_layout)
        
        # YouTube helper button
        youtube_layout = QtWidgets.QHBoxLayout()
        self.btn_youtube_help = QtWidgets.QPushButton("YouTube Stream Help")
        self.btn_youtube_help.clicked.connect(self._show_youtube_help)
        youtube_layout.addWidget(self.btn_youtube_help)
        youtube_layout.addStretch()
        layout.addLayout(youtube_layout)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_test = QtWidgets.QPushButton("Test Connection")
        self.btn_test.clicked.connect(self._test_connection)
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        
        button_layout.addWidget(self.btn_test)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_ok)
        button_layout.addWidget(self.btn_cancel)
        layout.addLayout(button_layout)
    
    def _on_type_changed(self, index):
        """Handle stream type change"""
        if index == 0:  # Custom URL
            self.url_input.setEnabled(True)
            self.url_input.setText("")
            self.url_input.setPlaceholderText("rtmp://localhost/live/stream or rtsp://... or http://...")
        else:  # Webcam
            webcam_index = index - 1
            self.url_input.setEnabled(False)
            self.url_input.setText(str(webcam_index))
    
    def _show_youtube_help(self):
        """Show help for YouTube livestreams"""
        help_text = """
YouTube Livestream Setup:

1. Install yt-dlp (YouTube downloader):
   pip install yt-dlp

2. Get the direct stream URL:
   yt-dlp -g "YOUTUBE_LIVESTREAM_URL"
   
   Example:
   yt-dlp -g "https://www.youtube.com/watch?v=jfKfPfyJRdk"
   
3. Copy the resulting URL (starts with https://)

4. Paste it into the "Stream URL" field

Note: The URL expires after some time, so you may need to 
regenerate it if the stream stops working.
"""
        QtWidgets.QMessageBox.information(self, "YouTube Stream Help", help_text)
    
    def _test_connection(self):
        stream_url = self.get_stream_url()
        
        if not stream_url:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please enter a stream URL")
            return
        
        try:
            # Show progress
            progress = QtWidgets.QProgressDialog("Testing connection...", None, 0, 0, self)
            progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
            progress.show()
            QtWidgets.QApplication.processEvents()
            
            # Try to open the stream
            cap = cv2.VideoCapture(stream_url if not stream_url.isdigit() else int(stream_url))
            
            if cap.isOpened():
                # Try to read a frame with timeout
                ret, frame = cap.read()
                cap.release()
                progress.close()
                
                if ret and frame is not None:
                    QtWidgets.QMessageBox.information(
                        self, "Success", 
                        f"Successfully connected to stream!\n"
                        f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
                    )
                else:
                    QtWidgets.QMessageBox.warning(
                        self, "Warning",
                        "Connected to stream but couldn't read frame.\n"
                        "Stream might not be ready yet or URL expired."
                    )
            else:
                progress.close()
                QtWidgets.QMessageBox.critical(
                    self, "Error",
                    "Failed to connect to stream.\n"
                    "Please check the URL and try again.\n\n"
                    "For YouTube streams, make sure you're using the direct stream URL\n"
                    "obtained from yt-dlp (click 'YouTube Stream Help')"
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"Error testing stream:\n{str(e)}"
            )
    
    def get_stream_url(self):
        return self.url_input.text().strip()

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, log_filename):
        super().__init__()
        logger.info("Initializing MainWindow")
        self.log_filename = log_filename
        self.setWindowTitle(f"Multi-Camera Crowd Analysis - Log: {log_filename}")
        self.resize(1800, 900)
        
        self.camera_widgets = {}
        self.camera_counter = 0
        self.calib_images = {}
        
        self.world_pts = np.array([
            [0, 0], 
            [0, config.WORLD_H], 
            [config.WORLD_W, config.WORLD_H], 
            [config.WORLD_W, 0]
        ], np.float32)
        
        # Default calibration for manually added cameras
        self.default_calibration = [[289, 577], [689, 156], [1102, 174], [1236, 680]]
        
        self._setup_ui()
        self._setup_worker()
        
        logger.info("MainWindow initialized")
    
    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        
        # Main content area - horizontal split
        content_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # LEFT SIDE: Top-down display
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.labelTD = QtWidgets.QLabel("Top-Down Heatmap View")
        self.labelTD.setMinimumSize(900, 700)
        self.labelTD.setStyleSheet("background:black; color:white; border: 2px solid #444;")
        self.labelTD.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.labelTD)
        
        content_layout.addWidget(left_widget, stretch=2)
        
        # Camera feeds only
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        # Camera management header
        camera_header = QtWidgets.QHBoxLayout()
        camera_header.addWidget(QtWidgets.QLabel("<b>Camera Feeds</b>"))
        self.btn_add_camera = QtWidgets.QPushButton("+ Add Camera")
        self.btn_add_camera.clicked.connect(self.on_add_camera)
        camera_header.addWidget(self.btn_add_camera)
        right_layout.addLayout(camera_header)
        
        # Scrollable area for camera feeds (VERTICAL)
        self.camera_scroll = QtWidgets.QScrollArea()
        self.camera_scroll.setWidgetResizable(True)
        self.camera_scroll.setMinimumWidth(400)
        self.camera_scroll.setStyleSheet("QScrollArea { border: 1px solid #ccc; }")
        
        self.camera_container = QtWidgets.QWidget()
        self.camera_layout = QtWidgets.QVBoxLayout(self.camera_container)
        self.camera_layout.addStretch()
        self.camera_scroll.setWidget(self.camera_container)
        right_layout.addWidget(self.camera_scroll, stretch=1)
        
        content_layout.addWidget(right_widget, stretch=1)
        
        # Processing settings and controls
        bottom_controls = QtWidgets.QHBoxLayout()
        main_layout.addLayout(bottom_controls)
        
        # Model and confidence controls
        model_layout = QtWidgets.QHBoxLayout()
        model_layout.addWidget(QtWidgets.QLabel("Model:"))
        self.model_path_edit = QtWidgets.QLineEdit(config.YOLO_MODEL_PATH)
        self.model_path_edit.setMaximumWidth(300)
        model_layout.addWidget(self.model_path_edit)
        
        model_layout.addWidget(QtWidgets.QLabel("Confidence:"))
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(40)
        self.conf_slider.setMaximumWidth(150)
        self.conf_label = QtWidgets.QLabel("0.40")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v/100:.2f}"))
        model_layout.addWidget(self.conf_slider)
        model_layout.addWidget(self.conf_label)
        bottom_controls.addLayout(model_layout)
        
        bottom_controls.addStretch()
        
        # Control buttons
        buttons_layout = QtWidgets.QHBoxLayout()
        
        self.btn_start = QtWidgets.QPushButton("Start")
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
        
        buttons_layout.addWidget(self.btn_start)
        buttons_layout.addWidget(self.btn_pause)
        buttons_layout.addWidget(self.btn_resume)
        buttons_layout.addWidget(self.btn_stop)
        
        self.btn_toggle_trails = QtWidgets.QPushButton("Trails")
        self.btn_toggle_trails.setCheckable(True)
        self.btn_toggle_trails.setChecked(True)
        self.btn_toggle_trails.clicked.connect(self.on_toggle_trails)
        
        self.btn_toggle_flow = QtWidgets.QPushButton("Flow")
        self.btn_toggle_flow.setCheckable(True)
        self.btn_toggle_flow.setChecked(True)
        self.btn_toggle_flow.clicked.connect(self.on_toggle_flow)
        
        self.btn_toggle_heatmap = QtWidgets.QPushButton("Heatmap")
        self.btn_toggle_heatmap.setCheckable(True)
        self.btn_toggle_heatmap.setChecked(True)
        self.btn_toggle_heatmap.clicked.connect(self.on_toggle_heatmap)
        
        buttons_layout.addWidget(self.btn_toggle_trails)
        buttons_layout.addWidget(self.btn_toggle_flow)
        buttons_layout.addWidget(self.btn_toggle_heatmap)
        
        bottom_controls.addLayout(buttons_layout)
        
        # Status bar at bottom
        self.status = QtWidgets.QLabel(f"Ready. Logging to: {self.log_filename}")
        self.status.setStyleSheet("padding: 5px; background: #333; color: #fff; border-top: 1px solid #555;")
        main_layout.addWidget(self.status)
    
    def _setup_worker(self):
        self.worker = VideoWorker()
        self.worker.frame_updated.connect(self.on_frame_update)
        self.worker.topdown_signal.connect(self.update_topdown)
        self.worker.status_signal.connect(self.update_status)
        self.worker.error_signal.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
    
    def _display_video_thumbnail(self, camera_id, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
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
        is_single_camera = len(self.worker.cameras) == 1
        
        for cam_id, widget in self.camera_widgets.items():
            widget.btn_calibrate.setVisible(not is_single_camera)
            widget.btn_upload_calib.setVisible(not is_single_camera)
            
            if is_single_camera and cam_id in self.worker.cameras:
                camera = self.worker.cameras[cam_id]
                if camera.video_path:
                    filename = os.path.basename(camera.video_path)
                    current_text = widget.info_label.text()
                    if "Calibration image available" not in current_text:
                        widget.info_label.setText(
                            f"Loaded: {filename}\n"
                            "Single camera mode - full frame will be used"
                        )

    def on_add_camera(self):
        try:
            camera_id = f"CAM_{self.camera_counter}"
            self.camera_counter += 1
            
            widget = CameraWidget(camera_id, self)
            widget.btn_load.clicked.connect(lambda: self.on_load_video(camera_id))
            widget.btn_load_stream.clicked.connect(lambda: self.on_load_stream(camera_id))
            widget.btn_calibrate.clicked.connect(lambda: self.on_calibrate_camera(camera_id))
            widget.btn_upload_calib.clicked.connect(lambda: self.on_upload_calibration_image(camera_id))
            widget.btn_remove.clicked.connect(lambda: self.on_remove_camera(camera_id))
            
            self.camera_layout.insertWidget(self.camera_layout.count() - 1, widget)
            self.camera_widgets[camera_id] = widget
            
            self.worker.add_camera(camera_id, calibration_points=self.default_calibration)
            
            self._update_calibration_buttons()
            self.update_status(f"Added {camera_id} - Load a video to configure")
        except Exception as e:
            logger.error(f"Error adding camera: {e}", exc_info=True)
            self.on_error(f"Error adding camera: {str(e)}")

    def on_remove_camera(self, camera_id):
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
                    self._display_video_thumbnail(camera_id, path)
                
                self._update_calibration_buttons()
                self.update_status(f"{camera_id}: {filename}")
        except Exception as e:
            logger.error(f"Error loading video: {e}", exc_info=True)

    def on_load_stream(self, camera_id):
        try:
            dialog = StreamInputDialog(camera_id, self)
            
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                stream_url = dialog.get_stream_url()
                
                if stream_url:
                    self.worker.update_camera_video(camera_id, stream_url)
                    
                    if camera_id in self.camera_widgets:
                        if stream_url.isdigit():
                            display_name = f"Webcam {stream_url}"
                        elif stream_url.startswith('rtmp://'):
                            display_name = "RTMP Stream"
                        elif stream_url.startswith('rtsp://'):
                            display_name = "RTSP Stream"
                        elif stream_url.startswith('http://') or stream_url.startswith('https://'):
                            display_name = "HTTP/HLS Stream"
                        else:
                            display_name = "Live Stream"
                        
                        self.camera_widgets[camera_id].info_label.setText(
                            f"Stream: {display_name}\n{stream_url[:50]}..."
                        )
                        self._display_video_thumbnail(camera_id, stream_url)
                    
                    self._update_calibration_buttons()
                    self.update_status(f"{camera_id}: Stream connected")
        except Exception as e:
            logger.error(f"Error loading stream: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load stream: {str(e)}")
    
    def on_calibrate_camera(self, camera_id):
        try:
            if camera_id not in self.worker.cameras:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Camera {camera_id} not found")
                return
            
            camera = self.worker.cameras[camera_id]
            
            source_path = None
            is_image = False
            
            if camera_id in self.calib_images:
                source_path = self.calib_images[camera_id]
                is_image = True
                logger.info(f"Using calibration image: {source_path}")
            elif camera.video_path:
                source_path = camera.video_path
                is_image = False
                logger.info(f"Using video first frame: {source_path}")
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", 
                    f"Please load a video or upload a calibration image for {camera_id} first"
                )
                return
            
            current_points = camera.calibration_points if camera.calibration_points else []
            
            dialog = CalibrationWindow(camera_id, source_path, current_points, is_image, self)
            
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                new_points = dialog.get_points()
                
                if new_points and len(new_points) == 4:
                    # Compute homography using the same world points
                    logger.info(f"Calibration points for {camera_id}: {new_points}")
                    logger.info(f"World points: {self.world_pts.tolist()}")
                    
                    # Compute homography exactly like main.py does
                    H, status = cv2.findHomography(
                        np.array(new_points, np.float32), 
                        self.world_pts
                    )
                    
                    if H is not None:
                        logger.info(f"Homography computed successfully for {camera_id}")
                        logger.debug(f"Homography matrix:\n{H}")
                        
                        # Update BOTH calibration points AND homography
                        camera.calibration_points = new_points
                        camera.homography = H
                        
                        self.update_status(f"Calibration updated for {camera_id}")
                        
                        if camera_id in self.camera_widgets:
                            widget = self.camera_widgets[camera_id]
                            current_text = widget.info_label.text().split('\n')[0]
                            widget.info_label.setText(f"{current_text}\nCalibrated ✓")
                    else:
                        logger.error(f"Failed to compute homography for {camera_id}")
                        QtWidgets.QMessageBox.critical(
                            self, "Error",
                            "Failed to compute homography from selected points"
                        )
                else:
                    QtWidgets.QMessageBox.warning(
                        self, "Warning",
                        "Please select exactly 4 points to calibrate"
                    )
        except Exception as e:
            logger.error(f"Error calibrating camera: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Error", f"Calibration failed: {str(e)}")
    
    def on_upload_calibration_image(self, camera_id):
        try:
            if camera_id not in self.worker.cameras:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Camera {camera_id} not found")
                return
            
            image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, f"Select calibration image for {camera_id}",
                "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
            )
            
            if not image_path:
                return
            
            test_img = cv2.imread(image_path)
            if test_img is None:
                QtWidgets.QMessageBox.critical(
                    self, "Error",
                    "Could not load the selected image"
                )
                return
            
            self.calib_images[camera_id] = image_path
            
            logger.info(f"Uploaded calibration image for {camera_id}: {image_path}")
            
            if camera_id in self.camera_widgets:
                widget = self.camera_widgets[camera_id]
                current_text = widget.info_label.text().split('\n')[0]
                filename = os.path.basename(image_path)
                widget.info_label.setText(f"{current_text}\nCalib image: {filename}")
            
            self.update_status(f"Calibration image uploaded for {camera_id}")
            
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
        try:
            if not self.worker.cameras:
                self.update_status("Please add at least one camera.")
                return
            
            if len(self.worker.cameras) == 1:
                self.update_status("Single camera mode - using full frame area")
                logger.info("Starting in single camera mode")
            
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
        self.worker.pause()
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        self.update_status("Paused")

    def on_resume(self):
        self.worker.resume()
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.update_status("Resumed")

    def on_stop(self):
        self.worker.stop_processing()
        self.update_status("Stopping...")
    
    def on_toggle_trails(self):
        self.worker.show_trajectories = not self.worker.show_trajectories
        status = "ON" if self.worker.show_trajectories else "OFF"
        self.update_status(f"Trajectories: {status}")
    
    def on_toggle_flow(self):
        self.worker.show_flow = not self.worker.show_flow
        status = "ON" if self.worker.show_flow else "OFF"
        self.update_status(f"Flow vectors: {status}")
    
    def on_toggle_heatmap(self):
        self.worker.show_heatmap = not self.worker.show_heatmap
        status = "ON" if self.worker.show_heatmap else "OFF"
        self.update_status(f"Heatmap: {status}")

    def on_finished(self):
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
        logger.error(f"GUI Error: {error_msg}")
        QtWidgets.QMessageBox.critical(self, "Error", error_msg)
        self.update_status(f"Error: {error_msg}")
        self.on_finished()

    def on_frame_update(self, cam_id, frame):
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
        self.status.setText(text)

    def closeEvent(self, event):
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