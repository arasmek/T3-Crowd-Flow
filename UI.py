# UI.py
import sys
import time
import threading

import cv2
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

# Use your project modules
import config
import vision_utils as vu
from ultralytics import YOLO

# Helper: convert OpenCV BGR image to QImage
def cv2_to_qimage(bgr_img):
    if bgr_img is None:
        return None
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    # Format_RGB888 maps to 24-bit RGB
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()


class VideoWorker(QtCore.QThread):
    # Signals for frames and status
    frameA_signal = QtCore.pyqtSignal(QtGui.QImage)
    frameB_signal = QtCore.pyqtSignal(QtGui.QImage)
    topdown_signal = QtCore.pyqtSignal(QtGui.QImage)
    status_signal = QtCore.pyqtSignal(str)
    fps_signal = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        # pause control
        self.paused = False
        self.pause_condition = threading.Condition()

        # runtime objects
        self.model = None
        self.capA = None
        self.capB = None
        self.confidence = 0.3

        # Video / geometry defaults (will be recomputed if config changes)
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

        # Background/topdown ref
        self.bg_photo = cv2.imread(config.TOPDOWN_REF)
        if self.bg_photo is None:
            self.bg_photo = np.zeros((self.output_h, self.output_w, 3), np.uint8)
        self.bg_faint = vu.make_faint_background(self.bg_photo, alpha=0.18)

        # Paths used by the worker
        self.videoA_path = None
        self.videoB_path = None
        self.model_path = None

    def start_processing(self, videoA_path=None, videoB_path=None, model_path=None, confidence=0.3):
        """Initialize model and captures, then start thread loop."""
        self.confidence = confidence
        self.videoA_path = videoA_path or config.VIDEO_A
        self.videoB_path = videoB_path or config.VIDEO_B
        self.model_path = model_path or config.MODEL_PATH

        # Load model (this may take some time)
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
        # ensure paused is False when starting
        with self.pause_condition:
            self.paused = False
            self.pause_condition.notify_all()

        if not self.isRunning():
            self.start()

    def stop_processing(self):
        """Stop the worker loop; releases captures in run() cleanup."""
        self._running = False
        # unpause to let thread finish cleanly if paused
        with self.pause_condition:
            self.paused = False
            self.pause_condition.notify_all()

    def pause(self):
        """Pause processing (keeps video position)."""
        with self.pause_condition:
            self.paused = True
        self.status_signal.emit("Paused")

    def resume(self):
        """Resume processing from same frame."""
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

            retA, frameA = self.capA.read() if self.capA else (False, None)
            retB, frameB = self.capB.read() if self.capB else (False, None)

            if not retA and not retB:
                self.status_signal.emit("Reached end of both videos.")
                break

            # =====================
            # Warp frames to top-down
            # =====================
            warpA = cv2.warpPerspective(frameA, self.S @ self.H_A, (self.output_w, self.output_h)) if retA else np.zeros((self.output_h, self.output_w, 3), np.uint8)
            warpB = cv2.warpPerspective(frameB, self.S @ self.H_B, (self.output_w, self.output_h)) if retB else np.zeros((self.output_h, self.output_w, 3), np.uint8)

            # Overlay the two warped frames
            topdown = cv2.addWeighted(warpA, 0.5, warpB, 0.5, 0)

            # =====================
            # Draw grid overlay
            # =====================
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

            # =====================
            # Process Camera A
            # =====================
            if retA and frameA is not None:
                try:
                    resA = self.model(frameA, conf=self.confidence, classes=[0])
                    annotatedA = resA[0].plot()
                except Exception as e:
                    annotatedA = frameA.copy()
                    self.status_signal.emit(f"YOLO A error: {e}")

                # Emit Camera A frame
                qimA = cv2_to_qimage(annotatedA)
                if qimA:
                    self.frameA_signal.emit(QtGui.QPixmap.fromImage(qimA).toImage())

                # Draw detections on top-down
                try:
                    H_invA = np.linalg.inv(self.H_A)
                    for box in resA[0].boxes.xyxy:
                        x1, y1, x2, y2 = box.tolist()
                        wx, wy = vu.project_to_world(((x1 + x2) / 2, y2), self.H_A)
                        if 0 <= wx <= config.WORLD_W and 0 <= wy <= config.WORLD_H:
                            px, py = vu.world_to_topdown(wx, wy, self.S)
                            cv2.circle(topdown, (px, py), 5, (0, 0, 255), -1)
                            cv2.putText(topdown, f"({wx:.1f},{wy:.1f})", (px + 6, py - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
                except Exception:
                    pass

            # =====================
            # Process Camera B
            # =====================
            if retB and frameB is not None:
                try:
                    resB = self.model(frameB, conf=self.confidence, classes=[0])
                    annotatedB = resB[0].plot()
                except Exception as e:
                    annotatedB = frameB.copy()
                    self.status_signal.emit(f"YOLO B error: {e}")

                # Emit Camera B frame
                qimB = cv2_to_qimage(annotatedB)
                if qimB:
                    self.frameB_signal.emit(QtGui.QPixmap.fromImage(qimB).toImage())

                # Draw detections on top-down
                try:
                    H_invB = np.linalg.inv(self.H_B)
                    for box in resB[0].boxes.xyxy:
                        x1, y1, x2, y2 = box.tolist()
                        wx, wy = vu.project_to_world(((x1 + x2) / 2, y2), self.H_B)
                        if 0 <= wx <= config.WORLD_W and 0 <= wy <= config.WORLD_H:
                            px, py = vu.world_to_topdown(wx, wy, self.S)
                            cv2.circle(topdown, (px, py), 5, (255, 0, 0), -1)
                            cv2.putText(topdown, f"({wx:.1f},{wy:.1f})", (px + 6, py - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1, cv2.LINE_AA)
                except Exception:
                    pass

            # =====================
            # Emit top-down combined frame
            # =====================
            qim_td = cv2_to_qimage(topdown)
            if qim_td:
                self.topdown_signal.emit(QtGui.QPixmap.fromImage(qim_td).toImage())

            # =====================
            # FPS calculation
            # =====================
            frames_processed += 1
            t1 = time.time()
            elapsed = t1 - last_time
            if elapsed >= 1.0:
                self.fps_signal.emit(frames_processed / elapsed)
                last_time = t1
                frames_processed = 0

            # tiny sleep to avoid 100% CPU
            loop_time = time.time() - t0
            sleep_time = max(0.001, 1.0 / 60.0 - loop_time)
            time.sleep(sleep_time)

        # cleanup
        if self.capA:
            self.capA.release()
        if self.capB:
            self.capB.release()
        self.status_signal.emit("Worker stopped.")




class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Project - PyQt UI")
        self.resize(1200, 700)

        # central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Top split: video displays
        display_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(display_layout)

        # Camera A
        self.labelA = QtWidgets.QLabel("Camera A")
        self.labelA.setFixedSize(360, 240)
        self.labelA.setStyleSheet("background: black;")
        display_layout.addWidget(self.labelA)

        # Camera B
        self.labelB = QtWidgets.QLabel("Camera B")
        self.labelB.setFixedSize(360, 240)
        self.labelB.setStyleSheet("background: black;")
        display_layout.addWidget(self.labelB)

        # Topdown
        self.labelTD = QtWidgets.QLabel("Top-down")
        self.labelTD.setFixedSize(480, 360)
        self.labelTD.setStyleSheet("background: black;")
        display_layout.addWidget(self.labelTD)

        # Controls
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

        # Auto-load model path from config
        self.model_path_edit = QtWidgets.QLineEdit(config.MODEL_PATH)
        self.model_path_edit.setPlaceholderText("Model path")
        controls_layout.addWidget(self.model_path_edit)

        # Confidence slider
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(int(0.3 * 100))
        controls_layout.addWidget(QtWidgets.QLabel("Confidence"))
        controls_layout.addWidget(self.conf_slider)

        # Buttons
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

        # Status bar / log
        self.status = QtWidgets.QLabel("Idle")
        layout.addWidget(self.status)

        # Worker
        self.worker = VideoWorker()
        self.worker.frameA_signal.connect(self.update_frameA)
        self.worker.frameB_signal.connect(self.update_frameB)
        self.worker.topdown_signal.connect(self.update_topdown)
        self.worker.status_signal.connect(self.update_status)
        self.worker.fps_signal.connect(self.update_fps)

        # Store default paths from config
        self.videoA_path = config.VIDEO_A
        self.videoB_path = config.VIDEO_B

        # Optional: auto-start
        # Uncomment next line if you want the worker to start immediately, gal nereikia :p
        # self.on_start()


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
        # allow user to update calibration images referenced in config
        a_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select calibration image A")
        if a_path:
            config.CALIB_A = a_path
            self.update_status(f"Calibration A set: {a_path}")
        b_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select calibration image B")
        if b_path:
            config.CALIB_B = b_path
            self.update_status(f"Calibration B set: {b_path}")

    def on_start(self):
        model_path = self.model_path_edit.text().strip() or config.MODEL_PATH
        conf = max(0.01, min(0.99, self.conf_slider.value() / 100.0))
        self.update_status("Starting...")
        # enable/disable appropriate buttons
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(True)
        # start worker
        self.worker.start_processing(self.videoA_path, self.videoB_path, model_path, confidence=conf)

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
        # reset buttons
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.update_status("Stopping...")

    # slots that receive QImage
    def update_frameA(self, qimage):
        pix = QtGui.QPixmap.fromImage(qimage).scaled(self.labelA.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.labelA.setPixmap(pix)

    def update_frameB(self, qimage):
        pix = QtGui.QPixmap.fromImage(qimage).scaled(self.labelB.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.labelB.setPixmap(pix)

    def update_topdown(self, qimage):
        pix = QtGui.QPixmap.fromImage(qimage).scaled(self.labelTD.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.labelTD.setPixmap(pix)

    def update_status(self, text):
        self.status.setText(text)

    def update_fps(self, fps):
        # show FPS briefly in status
        self.status.setText(f"FPS: {fps:.1f}")

    def closeEvent(self, event):
        # ensure worker stops
        self.worker.stop_processing()
        # wait a short while for cleanup
        self.worker.wait(2000)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
