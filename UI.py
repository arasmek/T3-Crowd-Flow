# UI.py (corrected, consolidated)

import sys
import time
import threading
import os

import cv2
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

import config
import vision_utils as vu
from ultralytics import YOLO


# -----------------------------
# Helper: Convert OpenCV BGR → QImage
# -----------------------------
def cv2_to_qimage(bgr_img):
    if bgr_img is None:
        return None
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()


# -----------------------------
# Clickable QLabel
# -----------------------------
class ClickableLabel(QtWidgets.QLabel):
    mouse_clicked = QtCore.pyqtSignal(int, int)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            # ev.position() returns a QPointF; convert to int
            pos = ev.position()
            self.mouse_clicked.emit(int(pos.x()), int(pos.y()))
        super().mousePressEvent(ev)


# ==============================
# Calibration Dialog (single)
# ==============================
class CalibrationDialog(QtWidgets.QDialog):
    """
    Shows the provided OpenCV BGR frame and lets user click 4 points.
    Emits points as simple list of (x,y). The dialog returns accepted/rejected like exec().
    """

    def __init__(self, frame_bgr: np.ndarray, title: str = "Select 4 points", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.points = []

        # Convert BGR -> RGB QImage for display
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)

        layout = QtWidgets.QVBoxLayout(self)

        self.label = ClickableLabel()
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))
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

        # make dialog comfortable size
        self.setMinimumSize(min(w + 40, 1600), min(h + 100, 1000))

    def on_click(self, x: int, y: int):
        if len(self.points) >= 4:
            return
        # Ensure clicks correspond to image pixel coords; label may be scaled by layout if widget resized.
        # We assume label uses original pixmap size (no scaling) because we set minimum size above.
        self.points.append((int(x), int(y)))
        self.info.setText(f"{len(self.points)}/4 points selected")

        # Visual feedback: draw on pixmap copy and set
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
# VideoWorker (cleaned from earlier parts)
# ==============================
class VideoWorker(QtCore.QThread):
    frameA_signal = QtCore.pyqtSignal(QtGui.QImage)
    frameB_signal = QtCore.pyqtSignal(QtGui.QImage)
    topdown_signal = QtCore.pyqtSignal(QtGui.QImage)
    status_signal = QtCore.pyqtSignal(str)
    fps_signal = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.paused = False
        self.pause_condition = threading.Condition()

        self.model = None
        self.capA = None
        self.capB = None
        self.confidence = 0.3

        # default homographies: try compute from defaults, else load or identity
        try:
            self.H_A, self.H_B = vu.compute_homographies(
                np.array([[289, 577], [689, 156], [1102, 174], [1236, 680]], np.float32),
                np.array([[14, 691], [126, 233], [477, 207], [801, 544]], np.float32),
                np.array([
                    [0, 0], [0, config.WORLD_H], [config.WORLD_W, config.WORLD_H], [config.WORLD_W, 0]
                ], np.float32)
            )
        except Exception:
            try:
                self.H_A, self.H_B = vu.load_homographies()
            except Exception:
                self.H_A = np.eye(3, dtype=np.float32)
                self.H_B = np.eye(3, dtype=np.float32)

        self.S = np.array([
            [config.SCALE, 0, config.MARGIN],
            [0, -config.SCALE, config.WORLD_H * config.SCALE + config.MARGIN],
            [0, 0, 1]
        ], np.float32)

        self.output_w = int(config.WORLD_W * config.SCALE) + config.MARGIN * 2
        self.output_h = int(config.WORLD_H * config.SCALE) + config.MARGIN * 2

        self.bg_photo = cv2.imread(config.TOPDOWN_REF)
        if self.bg_photo is None:
            self.bg_photo = np.zeros((self.output_h, self.output_w, 3), np.uint8)
        self.bg_faint = vu.make_faint_background(self.bg_photo, alpha=0.18)

        # Paths - set from start_processing
        self.videoA_path = None
        self.videoB_path = None
        self.model_path = None

    def start_processing(self, videoA_path=None, videoB_path=None, model_path=None, confidence=0.3):
        """Initialize model and captures, then start thread loop."""
        self.confidence = confidence
        self.videoA_path = videoA_path or config.VIDEO_A
        self.videoB_path = videoB_path or config.VIDEO_B
        self.model_path = model_path or config.MODEL_PATH

        # Load model
        try:
            self.model = YOLO(self.model_path)
            self.status_signal.emit(f"Loaded model: {self.model_path}")
        except Exception as e:
            self.status_signal.emit(f"Model load error: {e}")
            self.model = None

        # Initialize captures
        self.capA = cv2.VideoCapture(self.videoA_path)
        self.capB = cv2.VideoCapture(self.videoB_path)

        if not self.capA.isOpened():
            self.status_signal.emit(f"Error opening video A: {self.videoA_path}")
        else:
            self.status_signal.emit(f"Opened A: {self.videoA_path}")
        if not self.capB.isOpened():
            self.status_signal.emit(f"Error opening video B: {self.videoB_path}")
        else:
            self.status_signal.emit(f"Opened B: {self.videoB_path}")

        # Skip initial seconds if configured
        try:
            fpsA = self.capA.get(cv2.CAP_PROP_FPS) or 10.0
            fpsB = self.capB.get(cv2.CAP_PROP_FPS) or 10.0
            fps = min(fpsA, fpsB)
            skip_frames = int(config.SKIP_SECONDS * fps) if fps > 0 else 0
            if skip_frames > 0:
                self.capA.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
                self.capB.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
                self.status_signal.emit(f"Skipped first {config.SKIP_SECONDS}s ({skip_frames} frames)")
        except Exception:
            pass

        self._running = True
        with self.pause_condition:
            self.paused = False
            self.pause_condition.notify_all()

        if not self.isRunning():
            self.start()

    def stop_processing(self):
        """Stop the worker loop; releases captures in run() cleanup."""
        self._running = False
        with self.pause_condition:
            self.paused = False
            self.pause_condition.notify_all()

    def pause(self):
        with self.pause_condition:
            self.paused = True
        self.status_signal.emit("Paused")

    def resume(self):
        with self.pause_condition:
            self.paused = False
            self.pause_condition.notify_all()
        self.status_signal.emit("Resumed")

    def run(self):
        last_time = time.time()
        frames_processed = 0

        while self._running:
            t0 = time.time()

            # Pause handling
            with self.pause_condition:
                while self.paused:
                    self.pause_condition.wait()

            retA, frameA = (False, None)
            retB, frameB = (False, None)
            if self.capA:
                retA, frameA = self.capA.read()
            if self.capB:
                retB, frameB = self.capB.read()

            if not retA and not retB:
                self.status_signal.emit("Reached end of both videos.")
                break

            # Warp frames to top-down
            warpA = cv2.warpPerspective(frameA, self.S @ self.H_A, (self.output_w, self.output_h)) if retA and frameA is not None else np.zeros((self.output_h, self.output_w, 3), np.uint8)
            warpB = cv2.warpPerspective(frameB, self.S @ self.H_B, (self.output_w, self.output_h)) if retB and frameB is not None else np.zeros((self.output_h, self.output_w, 3), np.uint8)

            # overlay
            topdown = cv2.addWeighted(warpA, 0.5, warpB, 0.5, 0)

            # grid overlay
            GRID_W, GRID_H = config.GRID_W, config.GRID_H
            cell_w = config.WORLD_W / GRID_W
            cell_h = config.WORLD_H / GRID_H
            for i in range(GRID_W + 1):
                x = int(config.MARGIN + i * cell_w * config.SCALE)
                cv2.line(topdown, (x, config.MARGIN), (x, self.output_h - config.MARGIN), (100, 100, 100), 1)
            for j in range(GRID_H + 1):
                y = int(self.output_h - config.MARGIN - j * cell_h * config.SCALE)
                cv2.line(topdown, (config.MARGIN, y), (self.output_w - config.MARGIN, y), (100, 100, 100), 1)
            try:
                vu.draw_axis_labels(topdown, GRID_W, GRID_H, cell_w, cell_h,
                                    config.WORLD_W, config.WORLD_H, config.SCALE, config.MARGIN)
            except Exception:
                pass

            # process camera A
            if retA and frameA is not None:
                try:
                    if self.model is not None:
                        resA = self.model(frameA, conf=self.confidence, classes=[0])
                        annotatedA = resA[0].plot()
                    else:
                        annotatedA = frameA.copy()
                except Exception as e:
                    annotatedA = frameA.copy()
                    self.status_signal.emit(f"YOLO A error: {e}")

                qimA = cv2_to_qimage(annotatedA)
                if qimA:
                    self.frameA_signal.emit(qimA)

                try:
                    if self.model is not None:
                        for box in resA[0].boxes.xyxy:
                            x1, y1, x2, y2 = box.tolist()
                            wx, wy = vu.project_to_world(((x1 + x2) / 2, y2), self.H_A)
                            if 0 <= wx <= config.WORLD_W and 0 <= wy <= config.WORLD_H:
                                px, py = vu.world_to_topdown(wx, wy, self.S)
                                cv2.circle(topdown, (px, py), 5, (0, 0, 255), -1)
                except Exception:
                    pass

            # process camera B
            if retB and frameB is not None:
                try:
                    if self.model is not None:
                        resB = self.model(frameB, conf=self.confidence, classes=[0])
                        annotatedB = resB[0].plot()
                    else:
                        annotatedB = frameB.copy()
                except Exception as e:
                    annotatedB = frameB.copy()
                    self.status_signal.emit(f"YOLO B error: {e}")

                qimB = cv2_to_qimage(annotatedB)
                if qimB:
                    self.frameB_signal.emit(qimB)

                try:
                    if self.model is not None:
                        for box in resB[0].boxes.xyxy:
                            x1, y1, x2, y2 = box.tolist()
                            wx, wy = vu.project_to_world(((x1 + x2) / 2, y2), self.H_B)
                            if 0 <= wx <= config.WORLD_W and 0 <= wy <= config.WORLD_H:
                                px, py = vu.world_to_topdown(wx, wy, self.S)
                                cv2.circle(topdown, (px, py), 5, (255, 0, 0), -1)
                except Exception:
                    pass

            # emit topdown
            qim_td = cv2_to_qimage(topdown)
            if qim_td:
                self.topdown_signal.emit(qim_td)

            # fps
            frames_processed += 1
            t1 = time.time()
            elapsed = t1 - last_time
            if elapsed >= 1.0:
                self.fps_signal.emit(frames_processed / elapsed)
                last_time = t1
                frames_processed = 0

            loop_time = time.time() - t0
            sleep_time = max(0.001, 1.0 / 60.0 - loop_time)
            time.sleep(sleep_time)

        # cleanup
        if self.capA:
            self.capA.release()
            self.capA = None
        if self.capB:
            self.capB.release()
            self.capB = None
        self.status_signal.emit("Worker stopped.")

class CalibrationPreviewDialog(QtWidgets.QDialog):
    def __init__(self, imgA, imgB, ptsA=None, ptsB=None, parent=None):
        """
        imgA, imgB: OpenCV BGR images from cameras
        ptsA, ptsB: Optional 4 calibration points for each image
        """
        super().__init__(parent)
        self.setWindowTitle("Calibration Preview")
        self.resize(1000, 700)

        self.imgA = imgA
        self.imgB = imgB
        self.ptsA = ptsA
        self.ptsB = ptsB
        self.alpha = 0.5  # overlay transparency

        # QLabel to show overlay
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Slider to adjust transparency
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(self.alpha * 100))
        self.slider.valueChanged.connect(self.on_slider_change)

        # Buttons
        btn_accept = QtWidgets.QPushButton("Accept Calibration")
        btn_retry = QtWidgets.QPushButton("Retry")
        btn_accept.clicked.connect(self.accept)
        btn_retry.clicked.connect(self.reject)

        # Layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addWidget(QtWidgets.QLabel("Overlay Transparency"))
        main_layout.addWidget(self.slider)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_accept)
        btn_layout.addWidget(btn_retry)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

        # Initial preview
        self.update_preview()

    def draw_points(self, image, points, color=(0, 255, 0)):
        """Draw numbered points on an image"""
        if points is None:
            return image
        img_copy = image.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(img_copy, (int(x), int(y)), 6, color, -1)
            cv2.putText(img_copy, str(i + 1), (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return img_copy

    def update_preview(self):
        # Ensure both images are same size
        h, w = self.imgA.shape[:2]
        imgB_resized = cv2.resize(self.imgB, (w, h), interpolation=cv2.INTER_LINEAR)

        # Draw calibration points if provided
        imgA_pts = self.draw_points(self.imgA, self.ptsA, color=(0, 0, 255))
        imgB_pts = self.draw_points(imgB_resized, self.ptsB, color=(255, 0, 0))

        # Overlay images with alpha
        overlay = cv2.addWeighted(imgA_pts, self.alpha, imgB_pts, 1 - self.alpha, 0)

        # Convert to QImage
        qimg = cv2_to_qimage(overlay)

        # Scale to fit label while keeping aspect ratio
        label_w = self.label.width()
        label_h = self.label.height()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            label_w, label_h,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.label.setPixmap(pix)

    def on_slider_change(self, value):
        self.alpha = value / 100.0
        self.update_preview()

    def resizeEvent(self, event):
        self.update_preview()
        super().resizeEvent(event)

# ==============================
# MainWindow (integrated)
# ==============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Project - PyQt UI")
        self.resize(1200, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # displays
        display_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(display_layout)

        self.labelA = QtWidgets.QLabel("Camera A")
        self.labelA.setFixedSize(360, 240)
        self.labelA.setStyleSheet("background: black;")
        display_layout.addWidget(self.labelA)

        self.labelB = QtWidgets.QLabel("Camera B")
        self.labelB.setFixedSize(360, 240)
        self.labelB.setStyleSheet("background: black;")
        display_layout.addWidget(self.labelB)

        self.labelTD = QtWidgets.QLabel("Top-down")
        self.labelTD.setFixedSize(480, 360)
        self.labelTD.setStyleSheet("background: black;")
        display_layout.addWidget(self.labelTD)

        # controls
        controls_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(controls_layout)

        self.btn_loadA = QtWidgets.QPushButton("Load Video A")
        self.btn_loadA.clicked.connect(self.on_loadA)
        controls_layout.addWidget(self.btn_loadA)

        self.btn_loadB = QtWidgets.QPushButton("Load Video B")
        self.btn_loadB.clicked.connect(self.on_loadB)
        controls_layout.addWidget(self.btn_loadB)

        self.btn_loadCalib = QtWidgets.QPushButton("Load Calibration Images")
        self.btn_loadCalib.clicked.connect(self.on_load_calib)
        controls_layout.addWidget(self.btn_loadCalib)

        # Calibrate Videos button
        self.btn_calibrate = QtWidgets.QPushButton("Calibrate Videos")
        self.btn_calibrate.clicked.connect(self.on_calibrate_videos)
        controls_layout.addWidget(self.btn_calibrate)

        self.model_path_edit = QtWidgets.QLineEdit(config.MODEL_PATH)
        self.model_path_edit.setPlaceholderText("Model path")
        controls_layout.addWidget(self.model_path_edit)

        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(int(0.3 * 100))
        controls_layout.addWidget(QtWidgets.QLabel("Confidence"))
        controls_layout.addWidget(self.conf_slider)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.clicked.connect(self.on_start)
        controls_layout.addWidget(self.btn_start)

        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_pause.setEnabled(False)
        controls_layout.addWidget(self.btn_pause)

        self.btn_resume = QtWidgets.QPushButton("Resume")
        self.btn_resume.clicked.connect(self.on_resume)
        self.btn_resume.setEnabled(False)
        controls_layout.addWidget(self.btn_resume)

        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        controls_layout.addWidget(self.btn_stop)

        self.status = QtWidgets.QLabel("Idle")
        layout.addWidget(self.status)

        # worker
        self.worker = VideoWorker()
        self.worker.frameA_signal.connect(self.update_frameA)
        self.worker.frameB_signal.connect(self.update_frameB)
        self.worker.topdown_signal.connect(self.update_topdown)
        self.worker.status_signal.connect(self.update_status)
        self.worker.fps_signal.connect(self.update_fps)

        # default paths
        self.videoA_path = config.VIDEO_A
        self.videoB_path = config.VIDEO_B

        self.calibA_pts = None
        self.calibB_pts = None

    # ---------------------
    # File selectors
    # ---------------------
    def on_loadA(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video for Camera A")
        if path:
            self.videoA_path = path
            self.update_status(f"Selected A: {path}")

    def on_loadB(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video for Camera B")
        if path:
            self.videoB_path = path
            self.update_status(f"Selected B: {path}")

    def on_load_calib(self):
        a_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select calibration image A")
        if a_path:
            config.CALIB_A = a_path
            self.update_status(f"Calibration A set: {a_path}")
        b_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select calibration image B")
        if b_path:
            config.CALIB_B = b_path
            self.update_status(f"Calibration B set: {b_path}")


    def get_points_from_image(self, img, window_title="Select points"):
        """
        Display the image in a temporary window and let the user click 4 points.
        Returns a list of 4 (x, y) tuples in the order clicked.
        """

        points = []

        clone = img.copy()
        cv2.namedWindow(window_title)
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 4:
                    points.append((x, y))
                    cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(clone, str(len(points)), (x+5, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.imshow(window_title, clone)

        cv2.setMouseCallback(window_title, click_event)
        cv2.imshow(window_title, clone)

        while True:
            key = cv2.waitKey(1) & 0xFF
            # Press ESC to cancel
            if key == 27:
                points = None
                break
            # Wait until 4 points are selected
            if points is not None and len(points) == 4:
                break

        cv2.destroyWindow(window_title)
        return points

    # ============================================
    # Calibration: grab first frames, allow clicks, save Option C files
    # ============================================
    def on_calibrate_videos(self):
        if not self.videoA_path or not self.videoB_path:
            self.update_status("Please load both videos first.")
            return

        # ---------------------------
        # Read first frames
        # ---------------------------
        capA = cv2.VideoCapture(self.videoA_path)
        retA, frameA = capA.read()
        capA.release()

        capB = cv2.VideoCapture(self.videoB_path)
        retB, frameB = capB.read()
        capB.release()

        if not retA or not retB:
            self.update_status("Failed to read first frames for calibration.")
            return

        # ---------------------------
        # Select points Camera A
        # ---------------------------
        ptsA = self.get_points_from_image(
            frameA, "Select 4 points for Camera A in CLOCKWISE order"
        )
        if ptsA is None:
            self.update_status("Calibration cancelled.")
            return

        # ---------------------------
        # Select points Camera B
        # ---------------------------
        ptsB = self.get_points_from_image(
            frameB, "Select 4 points for Camera B in SAME ORDER as Camera A"
        )
        if ptsB is None:
            self.update_status("Calibration cancelled.")
            return

        ptsA_np = np.array(ptsA, np.float32)
        ptsB_np = np.array(ptsB, np.float32)

        # ---------------------------
        # World-space reference square
        # ---------------------------
        pts_world = np.array([
            [0, 0],
            [0, config.WORLD_H],
            [config.WORLD_W, config.WORLD_H],
            [config.WORLD_W, 0]
        ], np.float32)

        # ---------------------------
        # Compute homographies
        # ---------------------------
        H_A, H_B = vu.compute_homographies(ptsA_np, ptsB_np, pts_world)

        # Store into worker
        self.worker.H_A = H_A
        self.worker.H_B = H_B

        # ---------------------------
        # Preview warped results
        # ---------------------------
        out_w = config.WORLD_W
        out_h = config.WORLD_H

        warpedA = cv2.warpPerspective(frameA, H_A, (self.worker.output_w, self.worker.output_h))
        warpedB = cv2.warpPerspective(frameB, H_B, (self.worker.output_w, self.worker.output_h))


        # Visual alignment check using overlay
        preview_overlay = cv2.addWeighted(warpedA, 0.5, warpedB, 0.5, 0)

        # ---------------------------
        # Show preview dialog
        # ---------------------------
        #preview_dialog = CalibrationPreviewDialog(warpedA, preview_overlay)
        #accepted = preview_dialog.exec()

        #if accepted == 0:
        #    self.update_status(
        #        "Calibration rejected. Please select points again."
        #    )
        #    return

        # ---------------------------
        # User Accepted → Save Files
        # ---------------------------
        imgA = frameA.copy()
        imgB = frameB.copy()
        for (x, y) in ptsA:
            cv2.circle(imgA, (x, y), 6, (0, 0, 255), -1)
        for (x, y) in ptsB:
            cv2.circle(imgB, (x, y), 6, (0, 255, 0), -1)

        cv2.imwrite("CALIB_A.png", imgA)
        cv2.imwrite("CALIB_B.png", imgB)

        with open("calibration_pts.txt", "w") as f:
            f.write("A_points:\n")
            for p in ptsA:
                f.write(f"{p[0]} {p[1]}\n")
            f.write("\nB_points:\n")
            for p in ptsB:
                f.write(f"{p[0]} {p[1]}\n")

        self.calibA_pts = ptsA
        self.calibB_pts = ptsB

        self.update_status("Calibration ACCEPTED and saved successfully!")

    # ------------------------------------
    # Start / Pause / Resume / Stop (use correct worker API)
    # ------------------------------------
    def on_start(self):
        model_path = self.model_path_edit.text().strip() or config.MODEL_PATH
        if not os.path.exists(model_path):
            self.update_status("Model path invalid.")
            return
        conf = max(0.01, min(0.99, self.conf_slider.value() / 100.0))

        # start processing (this will init model, opens caps, and start thread)
        self.worker.start_processing(self.videoA_path, self.videoB_path, model_path, confidence=conf)

        # enable/disable buttons
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.update_status("Starting...")

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
        # call stop_processing (not stop)
        self.worker.stop_processing()
        # reset buttons
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.update_status("Stopping...")

    # =============================
    # UI update slots
    # =============================
    def update_frameA(self, qimg: QtGui.QImage):
        if isinstance(qimg, QtGui.QImage):
            pix = QtGui.QPixmap.fromImage(qimg).scaled(self.labelA.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.labelA.setPixmap(pix)

    def update_frameB(self, qimg: QtGui.QImage):
        if isinstance(qimg, QtGui.QImage):
            pix = QtGui.QPixmap.fromImage(qimg).scaled(self.labelB.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.labelB.setPixmap(pix)

    def update_topdown(self, qimg: QtGui.QImage):
        if isinstance(qimg, QtGui.QImage):
            pix = QtGui.QPixmap.fromImage(qimg).scaled(self.labelTD.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.labelTD.setPixmap(pix)

    def update_status(self, text: str):
        self.status.setText(text)

    def update_fps(self, fps):
        self.status.setText(f"FPS: {fps:.1f}")

    def closeEvent(self, event):
        # ensure worker stops and thread finishes
        self.worker.stop_processing()
        self.worker.wait(2000)
        event.accept()


# =================================================
# Run App
# =================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()