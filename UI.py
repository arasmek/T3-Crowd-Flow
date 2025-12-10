# UI.py - Fixed version
import sys
import threading
import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO

import config
import vision_utils as vu
from deepsort_tracker import MultiCameraTracker
from crowd_analytics import CrowdFlowAnalyzer

# -----------------------------
# Helper: Convert OpenCV BGR → QImage
# -----------------------------
def cv2_to_qimage(bgr_img):
    if bgr_img is None:
        return None
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

# -----------------------------
# Clickable QLabel
# -----------------------------
class ClickableLabel(QtWidgets.QLabel):
    mouse_clicked = QtCore.pyqtSignal(int, int)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = ev.position()
            self.mouse_clicked.emit(int(pos.x()), int(pos.y()))
        super().mousePressEvent(ev)

# ==============================
# Calibration Dialog
# ==============================
class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, frame_bgr: np.ndarray, title="Select 4 points", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.points = []

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        layout = QtWidgets.QVBoxLayout(self)

        self.label = ClickableLabel()
        self.label.setPixmap(QPixmap.fromImage(qimg))
        self.label.mouse_clicked.connect(self.on_click)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.label)

        self.info = QtWidgets.QLabel("Click 4 points (order doesn't matter, will be saved).")
        layout.addWidget(self.info)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        self.setMinimumSize(min(w + 40, 1600), min(h + 100, 1000))

    def on_click(self, x, y):
        if len(self.points) >= 4:
            return
        self.points.append((int(x), int(y)))
        self.info.setText(f"{len(self.points)}/4 points selected")
        pix = self.label.pixmap().copy()
        painter = QtGui.QPainter(pix)
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
        pen.setWidth(10)
        painter.setPen(pen)
        painter.drawPoint(x, y)
        painter.end()
        self.label.setPixmap(pix)
        if len(self.points) == 4:
            self.btn_ok.setEnabled(True)

# ==============================
# VideoWorker for threading
# ==============================
class VideoWorker(QtCore.QObject):
    topdown_signal = QtCore.pyqtSignal(np.ndarray)
    frame_updated = QtCore.pyqtSignal(str, np.ndarray)  # camera 'A'/'B'
    status_signal = QtCore.pyqtSignal(str)
    fps_signal = QtCore.pyqtSignal(float)
    finished = QtCore.pyqtSignal()
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.paused = False
        self.pause_condition = threading.Condition()
        self.capA = None
        self.capB = None
        self.H_A = np.eye(3)
        self.H_B = np.eye(3)
        self.output_w = int(config.WORLD_W * config.SCALE) + config.MARGIN * 2
        self.output_h = int(config.WORLD_H * config.SCALE) + config.MARGIN * 2
        
        # Transformation matrix for top-down view
        self.S = np.array([
            [config.SCALE, 0, config.MARGIN],
            [0, -config.SCALE, config.WORLD_H * config.SCALE + config.MARGIN],
            [0, 0, 1]
        ], np.float32)
        
        # Load background with error handling
        try:
            self.bg_photo = cv2.imread(config.TOPDOWN_REF)
            if self.bg_photo is None:
                raise ValueError("Could not load background image")
        except Exception as e:
            print(f"Warning: Could not load background image: {e}")
            self.bg_photo = np.zeros((self.output_h, self.output_w, 3), np.uint8)
        
        self.bg_faint = vu.make_faint_background(self.bg_photo, alpha=0.18)
        self.tracker = MultiCameraTracker()
        self.crowd_analyzer = CrowdFlowAnalyzer(config.WORLD_W, config.WORLD_H, config.HEATMAP_CELL_SIZE)
        self.yolo_model = YOLO(config.YOLO_MODEL_PATH)
        self.videoA_path = None
        self.videoB_path = None
        self.conf = 0.3
        self.frame_count = 0
        self.show_trajectories = True
        self.show_flow = True

    def start_processing(self, pathA, pathB, model_path, conf):
        self.videoA_path = pathA
        self.videoB_path = pathB
        self.conf = conf
        
        try:
            self.yolo_model = YOLO(model_path)
            self.capA = cv2.VideoCapture(self.videoA_path)
            self.capB = cv2.VideoCapture(self.videoB_path)
            
            if not self.capA.isOpened() or not self.capB.isOpened():
                raise ValueError("Could not open one or both video files")
            
            self._running = True
            self.frame_count = 0
            threading.Thread(target=self.run, daemon=True).start()
        except Exception as e:
            self.error_signal.emit(f"Error starting processing: {str(e)}")

    def stop_processing(self):
        self._running = False
        # Wait a bit for thread to finish
        QtCore.QThread.msleep(100)
        self._cleanup_resources()

    def _cleanup_resources(self):
        """Safely release video capture resources"""
        if self.capA is not None:
            try:
                self.capA.release()
            except:
                pass
            self.capA = None
        if self.capB is not None:
            try:
                self.capB.release()
            except:
                pass
            self.capB = None

    def pause(self):
        with self.pause_condition:
            self.paused = True

    def resume(self):
        with self.pause_condition:
            self.paused = False
            self.pause_condition.notify_all()

    def run(self):
        last_time = time.time()
        frames_processed = 0

        while self._running:
            t0 = time.time()

            # ----------------------
            # Pause handling
            # ----------------------
            with self.pause_condition:
                while self.paused:
                    self.pause_condition.wait()

            # ----------------------
            # Read frames
            # ----------------------
            retA, frameA = (False, None)
            retB, frameB = (False, None)
            if self.capA:
                retA, frameA = self.capA.read()
            if self.capB:
                retB, frameB = self.capB.read()

            if not retA and not retB:
                self.status_signal.emit("Reached end of both videos.")
                break

            # ----------------------------
            # Warp frames to top-down (Option 2)
            # ----------------------------
            if retA and frameA is not None:
                warpA = cv2.warpPerspective(
                    frameA,
                    self.S @ self.H_A,
                    (self.output_w, self.output_h)
                )
            else:
                warpA = np.zeros((self.output_h, self.output_w, 3), np.uint8)

            if retB and frameB is not None:
                warpB = cv2.warpPerspective(
                    frameB,
                    self.S @ self.H_B,
                    (self.output_w, self.output_h)
                )
            else:
                warpB = np.zeros((self.output_h, self.output_w, 3), np.uint8)

            # ----------------------------
            # Overlay both warped frames
            # ----------------------------
            topdown = cv2.addWeighted(warpA, 0.5, warpB, 0.5, 0)

            # ----------------------------
            # Draw grid overlay
            # ----------------------------
            GRID_W, GRID_H = config.GRID_W, config.GRID_H
            cell_w = config.WORLD_W / GRID_W
            cell_h = config.WORLD_H / GRID_H

            for i in range(GRID_W + 1):
                x = int(config.MARGIN + i * cell_w * config.SCALE)
                cv2.line(topdown, (x, config.MARGIN), (x, self.output_h - config.MARGIN), (100, 100, 100), 1)

            for j in range(GRID_H + 1):
                y = int(self.output_h - config.MARGIN - j * cell_h * config.SCALE)
                cv2.line(topdown, (config.MARGIN, y), (self.output_w - config.MARGIN, y), (100, 100, 100), 1)

            # Optional: axis labels
            try:
                vu.draw_axis_labels(
                    topdown, GRID_W, GRID_H, cell_w, cell_h,
                    config.WORLD_W, config.WORLD_H, config.SCALE, config.MARGIN
                )
            except Exception:
                pass

            # ----------------------------
            # Process camera A (YOLO + top-down points)
            # ----------------------------
            if retA and frameA is not None:
                try:
                    annotatedA = resA[0].plot() if self.model else frameA.copy()
                    if self.model:
                        resA = self.model(frameA, conf=self.confidence, classes=[0])
                except Exception as e:
                    annotatedA = frameA.copy()
                    self.status_signal.emit(f"YOLO A error: {e}")

                qimA = cv2_to_qimage(annotatedA)
                if qimA:
                    self.frameA_signal.emit(qimA)

                # Draw projected points on topdown
                try:
                    if self.model:
                        for box in resA[0].boxes.xyxy:
                            x1, y1, x2, y2 = box.tolist()
                            wx, wy = vu.project_to_world(((x1 + x2) / 2, y2), self.H_A)
                            if 0 <= wx <= config.WORLD_W and 0 <= wy <= config.WORLD_H:
                                px, py = vu.world_to_topdown(wx, wy, self.S)
                                cv2.circle(topdown, (px, py), 5, (0, 0, 255), -1)
                except Exception:
                    pass

            # ----------------------------
            # Process camera B (YOLO + top-down points)
            # ----------------------------
            if retB and frameB is not None:
                try:
                    annotatedB = resB[0].plot() if self.model else frameB.copy()
                    if self.model:
                        resB = self.model(frameB, conf=self.confidence, classes=[0])
                except Exception as e:
                    annotatedB = frameB.copy()
                    self.status_signal.emit(f"YOLO B error: {e}")

                qimB = cv2_to_qimage(annotatedB)
                if qimB:
                    self.frameB_signal.emit(qimB)

                # Draw projected points on topdown
                try:
                    if self.model:
                        for box in resB[0].boxes.xyxy:
                            x1, y1, x2, y2 = box.tolist()
                            wx, wy = vu.project_to_world(((x1 + x2) / 2, y2), self.H_B)
                            if 0 <= wx <= config.WORLD_W and 0 <= wy <= config.WORLD_H:
                                px, py = vu.world_to_topdown(wx, wy, self.S)
                                cv2.circle(topdown, (px, py), 5, (255, 0, 0), -1)
                except Exception:
                    pass

            # ----------------------------
            # Emit top-down frame
            # ----------------------------
            qim_td = cv2_to_qimage(topdown)
            if qim_td:
                self.topdown_signal.emit(qim_td)

            # ----------------------------
            # FPS calculation
            # ----------------------------
            frames_processed += 1
            t1 = time.time()
            elapsed = t1 - last_time
            if elapsed >= 1.0:
                self.fps_signal.emit(frames_processed / elapsed)
                last_time = t1
                frames_processed = 0

            # ----------------------------
            # Loop timing
            # ----------------------------
            loop_time = time.time() - t0
            sleep_time = max(0.001, 1.0 / 60.0 - loop_time)
            time.sleep(sleep_time)

        # ----------------------------
        # Cleanup
        # ----------------------------
        if self.capA:
            self.capA.release()
            self.capA = None
        if self.capB:
            self.capB.release()
            self.capB = None
        self.status_signal.emit("Worker stopped.")


# ==============================
# MainWindow
# ==============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crowd Analysis UI")
        self.resize(1200, 700)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Displays
        display_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(display_layout)
        self.labelA = QtWidgets.QLabel()
        self.labelA.setFixedSize(360, 240)
        self.labelA.setStyleSheet("background:black;")
        self.labelA.setScaledContents(False)
        self.labelA.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.labelB = QtWidgets.QLabel()
        self.labelB.setFixedSize(360, 240)
        self.labelB.setStyleSheet("background:black;")
        self.labelB.setScaledContents(False)
        self.labelB.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.labelTD = QtWidgets.QLabel()
        self.labelTD.setFixedSize(480, 360)
        self.labelTD.setStyleSheet("background:black;")
        self.labelTD.setScaledContents(False)
        self.labelTD.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        display_layout.addWidget(self.labelA)
        display_layout.addWidget(self.labelB)
        display_layout.addWidget(self.labelTD)

        # Controls row 1
        controls_layout1 = QtWidgets.QHBoxLayout()
        layout.addLayout(controls_layout1)
        
        self.btn_loadA = QtWidgets.QPushButton("Load Video A")
        self.btn_loadA.clicked.connect(self.on_loadA)
        self.btn_loadB = QtWidgets.QPushButton("Load Video B")
        self.btn_loadB.clicked.connect(self.on_loadB)
        self.btn_loadCalib = QtWidgets.QPushButton("Load Calibration Images")
        self.btn_loadCalib.clicked.connect(self.on_load_calib)
        self.btn_calibrate = QtWidgets.QPushButton("Calibrate Videos")
        self.btn_calibrate.clicked.connect(self.on_calibrate_videos)
        
        controls_layout1.addWidget(self.btn_loadA)
        controls_layout1.addWidget(self.btn_loadB)
        controls_layout1.addWidget(self.btn_loadCalib)
        controls_layout1.addWidget(self.btn_calibrate)

        # Controls row 2
        controls_layout2 = QtWidgets.QHBoxLayout()
        layout.addLayout(controls_layout2)
        
        controls_layout2.addWidget(QtWidgets.QLabel("Model:"))
        self.model_path_edit = QtWidgets.QLineEdit(config.YOLO_MODEL_PATH)
        controls_layout2.addWidget(self.model_path_edit)
        
        controls_layout2.addWidget(QtWidgets.QLabel("Confidence:"))
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(30)
        self.conf_slider.setMaximumWidth(150)
        self.conf_label = QtWidgets.QLabel("0.30")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v/100:.2f}"))
        controls_layout2.addWidget(self.conf_slider)
        controls_layout2.addWidget(self.conf_label)

        # Controls row 3
        controls_layout3 = QtWidgets.QHBoxLayout()
        layout.addLayout(controls_layout3)
        
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
        
        # Toggle buttons
        self.btn_toggle_trails = QtWidgets.QPushButton("Toggle Trails")
        self.btn_toggle_trails.clicked.connect(self.on_toggle_trails)
        self.btn_toggle_flow = QtWidgets.QPushButton("Toggle Flow")
        self.btn_toggle_flow.clicked.connect(self.on_toggle_flow)

        controls_layout3.addWidget(self.btn_start)
        controls_layout3.addWidget(self.btn_pause)
        controls_layout3.addWidget(self.btn_resume)
        controls_layout3.addWidget(self.btn_stop)
        controls_layout3.addWidget(self.btn_toggle_trails)
        controls_layout3.addWidget(self.btn_toggle_flow)
        controls_layout3.addStretch()

        self.status = QtWidgets.QLabel("Idle")
        layout.addWidget(self.status)

        # Worker
        self.worker = VideoWorker()
        self.worker.frame_updated.connect(self.on_frame_update)
        self.worker.topdown_signal.connect(self.update_topdown)
        self.worker.status_signal.connect(self.update_status)
        self.worker.error_signal.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)

    # ------------------------
    # File selectors
    # ------------------------
    def on_loadA(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video for Camera A", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.update_status(f"Selected A: {path}")
            self.worker.videoA_path = path

    def on_loadB(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video for Camera B", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.update_status(f"Selected B: {path}")
            self.worker.videoB_path = path

    def on_load_calib(self):
        a_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select calibration image A", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if a_path:
            config.CALIB_A = a_path
            self.update_status(f"Calibration A set: {a_path}")
        
        b_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select calibration image B", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if b_path:
            config.CALIB_B = b_path
            self.update_status(f"Calibration B set: {b_path}")

    # ------------------------
    # Calibration
    # ------------------------
    def on_calibrate_videos(self):
        if not self.worker.videoA_path or not self.worker.videoB_path:
            self.update_status("Load both videos first.")
            return
        
        try:
            capA = cv2.VideoCapture(self.worker.videoA_path)
            capB = cv2.VideoCapture(self.worker.videoB_path)
            retA, frameA = capA.read()
            retB, frameB = capB.read()
            capA.release()
            capB.release()
            
            if not retA or not retB:
                self.update_status("Failed to read frames.")
                return
            
            ptsA = self.get_points_from_image(frameA, "Select 4 points Camera A")
            if ptsA is None:
                self.update_status("Calibration cancelled.")
                return
            
            ptsB = self.get_points_from_image(frameB, "Select 4 points Camera B")
            if ptsB is None:
                self.update_status("Calibration cancelled.")
                return

            ptsA_np = np.array(ptsA, np.float32)
            ptsB_np = np.array(ptsB, np.float32)
            pts_world = np.array([
                [0, 0],
                [0, config.WORLD_H],
                [config.WORLD_W, config.WORLD_H],
                [config.WORLD_W, 0]
            ], np.float32)
            
            H_A, H_B = vu.compute_homographies(ptsA_np, ptsB_np, pts_world)
            self.worker.H_A = H_A
            self.worker.H_B = H_B
            
            warpedA = cv2.warpPerspective(frameA, H_A, 
                                         (self.worker.output_w, self.worker.output_h))
            warpedB = cv2.warpPerspective(frameB, H_B, 
                                         (self.worker.output_w, self.worker.output_h))
            preview_overlay = cv2.addWeighted(warpedA, 0.5, warpedB, 0.5, 0)
            
            cv2.imshow("Calibration preview", preview_overlay)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            self.update_status("Calibration applied.")
        except Exception as e:
            self.update_status(f"Calibration error: {str(e)}")

    def get_points_from_image(self, frame, title):
        dlg = CalibrationDialog(frame, title=title, parent=self)
        accepted = dlg.exec()
        if accepted == QtWidgets.QDialog.DialogCode.Accepted:
            return dlg.points
        return None

    # ------------------------
    # VideoWorker slots
    # ------------------------
    def on_start(self):
        if not self.worker.videoA_path or not self.worker.videoB_path:
            self.update_status("Please load both videos first.")
            return
        
        model_path = self.model_path_edit.text()
        conf = self.conf_slider.value() / 100.0
        self.worker.start_processing(self.worker.videoA_path, self.worker.videoB_path, 
                                     model_path, conf)
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.update_status("Processing started...")

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

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.update_status("Processing finished")

    def on_error(self, error_msg):
        QtWidgets.QMessageBox.critical(self, "Error", error_msg)
        self.update_status(f"Error: {error_msg}")
        self.on_finished()

    # ------------------------
    # Frame update slots
    # ------------------------
    def on_frame_update(self, cam, frame):
        qimg = cv2_to_qimage(frame)
        if qimg is None:
            return
        
        if cam == 'A':
            self.labelA.setPixmap(
                QPixmap.fromImage(qimg).scaled(
                    self.labelA.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
            )
        elif cam == 'B':
            self.labelB.setPixmap(
                QPixmap.fromImage(qimg).scaled(
                    self.labelB.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
            )

    def update_topdown(self, img):
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

    def update_status(self, text):
        self.status.setText(text)

    def closeEvent(self, event):
        """Handle window close event"""
        if self.worker._running:
            reply = QtWidgets.QMessageBox.question(
                self,
                'Confirm Exit',
                'Processing is still running. Are you sure you want to exit?',
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

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())