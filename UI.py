# UI.py - Fixed version with predetermined grid points
import sys
import threading
import time
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

# ==============================
# VideoWorker for threading
# ==============================
class VideoWorker(QtCore.QObject):
    topdown_signal = QtCore.pyqtSignal(np.ndarray)
    frame_updated = QtCore.pyqtSignal(str, np.ndarray)
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
        
        # PREDETERMINED CALIBRATION POINTS (from old version)
        camA_pts = np.array([
            [289, 577], [689, 156], [1102, 174], [1236, 680]
        ], np.float32)
        camB_pts = np.array([
            [14, 691], [126, 233], [477, 207], [801, 544]
        ], np.float32)
        world_pts = np.array([
            [0, 0], [0, config.WORLD_H], [config.WORLD_W, config.WORLD_H], [config.WORLD_W, 0]
        ], np.float32)
        
        # Compute homographies from predetermined points
        self.H_A, _ = cv2.findHomography(camA_pts, world_pts)
        self.H_B, _ = cv2.findHomography(camB_pts, world_pts)
        
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
        self.yolo_model = None
        self.videoA_path = None
        self.videoB_path = None
        self.conf = 0.3
        self.frame_count = 0
        self.show_trajectories = True
        self.show_flow = True
        self.show_heatmap = True

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
        
        GRID_W, GRID_H = config.GRID_W, config.GRID_H
        cell_w = config.WORLD_W / GRID_W
        cell_h = config.WORLD_H / GRID_H
        H_invA = np.linalg.inv(self.H_A)
        H_invB = np.linalg.inv(self.H_B)

        while self._running:
            t0 = time.time()

            # Pause handling
            with self.pause_condition:
                while self.paused:
                    self.pause_condition.wait()

            # Read frames
            retA, frameA = (False, None)
            retB, frameB = (False, None)
            if self.capA:
                retA, frameA = self.capA.read()
            if self.capB:
                retB, frameB = self.capB.read()

            if not retA and not retB:
                self.status_signal.emit("Reached end of both videos.")
                break

            self.frame_count += 1
            tracks_A, tracks_B = [], []
            
            # Camera A Detection & Tracking
            if retA and frameA is not None and self.yolo_model:
                try:
                    resA = self.yolo_model.track(frameA, conf=self.conf, 
                                               classes=[0], persist=False, verbose=False)
                    
                    if len(resA[0].boxes) > 0:
                        tracks_A = self.tracker.update_tracks(
                            resA[0].boxes, frameA, 'A', self.H_A, 
                            (config.WORLD_W, config.WORLD_H)
                        )
                    
                    # Annotate frame
                    annotatedA = frameA.copy()
                    for track in tracks_A:
                        ltrb = track.ltrb
                        x1, y1, x2, y2 = map(int, ltrb)
                        cv2.rectangle(annotatedA, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotatedA, f"ID:{track.local_id}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    annotatedA = vu.draw_world_grid_on_camera(
                        annotatedA, H_invA, GRID_W, GRID_H, cell_w, cell_h,
                        config.WORLD_W, config.WORLD_H, (100, 200, 100))
                    self.frame_updated.emit('A', annotatedA)
                except Exception as e:
                    self.status_signal.emit(f"Error processing camera A: {str(e)}")
            
            # Camera B Detection & Tracking
            if retB and frameB is not None and self.yolo_model:
                try:
                    resB = self.yolo_model.track(frameB, conf=self.conf,
                                               classes=[0], persist=False, verbose=False)
                    
                    if len(resB[0].boxes) > 0:
                        tracks_B = self.tracker.update_tracks(
                            resB[0].boxes, frameB, 'B', self.H_B,
                            (config.WORLD_W, config.WORLD_H)
                        )
                    
                    # Annotate frame
                    annotatedB = frameB.copy()
                    for track in tracks_B:
                        ltrb = track.ltrb
                        x1, y1, x2, y2 = map(int, ltrb)
                        cv2.rectangle(annotatedB, (x1, y1), (x2, y2), (255, 100, 0), 2)
                        cv2.putText(annotatedB, f"ID:{track.local_id}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                    
                    annotatedB = vu.draw_world_grid_on_camera(
                        annotatedB, H_invB, GRID_W, GRID_H, cell_w, cell_h,
                        config.WORLD_W, config.WORLD_H, (200, 150, 100))
                    self.frame_updated.emit('B', annotatedB)
                except Exception as e:
                    self.status_signal.emit(f"Error processing camera B: {str(e)}")
            
            # Merge tracks and update analytics
            all_tracks = self.tracker.merge_camera_tracks(tracks_A, tracks_B)
            self.crowd_analyzer.update(all_tracks)
            
            # Top-down visualization
            topdown = self.bg_faint.copy()
            
            # Draw heatmap overlay
            if self.show_heatmap:
                heatmap = self.crowd_analyzer.get_density_heatmap(smooth_sigma=2.5)
                heatmap_colored = np.zeros((self.crowd_analyzer.hmap_h, self.crowd_analyzer.hmap_w, 3), dtype=np.uint8)
                
                for hy in range(self.crowd_analyzer.hmap_h):
                    for hx in range(self.crowd_analyzer.hmap_w):
                        value = heatmap[hy, hx]
                        if value > 0:
                            if value < 0.33:
                                r = 0
                                g = int(255 * (value / 0.33))
                                b = int(255 * (1 - value / 0.33))
                            elif value < 0.66:
                                r = int(255 * ((value - 0.33) / 0.33))
                                g = 255
                                b = 0
                            else:
                                r = 255
                                g = int(255 * (1 - (value - 0.66) / 0.34))
                                b = 0
                            
                            heatmap_colored[hy, hx] = [b, g, r]
                
                heatmap_resized = cv2.resize(
                    heatmap_colored,
                    (int(config.WORLD_W * config.SCALE), int(config.WORLD_H * config.SCALE)),
                    interpolation=cv2.INTER_LINEAR
                )
                
                y_start = config.MARGIN
                y_end = config.MARGIN + int(config.WORLD_H * config.SCALE)
                x_start = config.MARGIN
                x_end = config.MARGIN + int(config.WORLD_W * config.SCALE)
                
                mask = (heatmap_resized.sum(axis=2) > 0).astype(np.uint8) * 255
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                roi = topdown[y_start:y_end, x_start:x_end]
                blended = cv2.addWeighted(roi, 0.6, heatmap_resized, 0.4, 0)
                topdown[y_start:y_end, x_start:x_end] = np.where(mask > 0, blended, roi)
            
            # Draw grid
            for i in range(GRID_W + 1):
                x = int(config.MARGIN + i * cell_w * config.SCALE)
                cv2.line(topdown, (x, config.MARGIN), (x, self.output_h - config.MARGIN), (80, 80, 80), 1)
            for j in range(GRID_H + 1):
                y = int(self.output_h - config.MARGIN - j * cell_h * config.SCALE)
                cv2.line(topdown, (config.MARGIN, y), (self.output_w - config.MARGIN, y), (80, 80, 80), 1)
            
            vu.draw_axis_labels(topdown, GRID_W, GRID_H, cell_w, cell_h,
                               config.WORLD_W, config.WORLD_H, config.SCALE, config.MARGIN)
            
            # Draw flow vectors
            if self.show_flow:
                flow_vectors = self.crowd_analyzer.get_flow_vectors()
                for vec in flow_vectors:
                    px, py = vu.world_to_topdown(vec['x'], vec['y'], self.S)
                    arrow_scale = min(vec['magnitude'] * 300, 50)
                    end_x = int(px + vec['vx'] * arrow_scale)
                    end_y = int(py - vec['vy'] * arrow_scale)
                    cv2.arrowedLine(topdown, (px, py), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)
            
            # Draw tracks
            for track in all_tracks:
                wx, wy = track.world_x, track.world_y
                px, py = vu.world_to_topdown(wx, wy, self.S)
                
                if hasattr(track, 'merged_from'):
                    color = (255, 0, 255)
                    thickness = 3
                elif track.camera_id == 'A':
                    color = (0, 255, 0)
                    thickness = 2
                else:
                    color = (255, 100, 0)
                    thickness = 2
                
                cv2.circle(topdown, (px, py), 8, color, thickness)
                cv2.circle(topdown, (px, py), 10, (255, 255, 255), 1)
                
                cv2.putText(topdown, str(track.global_id), (px+12, py-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                if self.show_trajectories:
                    trajectory = self.crowd_analyzer.get_trajectory(track.global_id)
                    if len(trajectory) > 1:
                        for i in range(len(trajectory) - 1):
                            pt1 = vu.world_to_topdown(trajectory[i][0], trajectory[i][1], self.S)
                            pt2 = vu.world_to_topdown(trajectory[i+1][0], trajectory[i+1][1], self.S)
                            alpha = i / len(trajectory)
                            fade_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
                            cv2.line(topdown, pt1, pt2, fade_color, 2)
            
            # Draw statistics
            stats = self.crowd_analyzer.get_statistics()
            info_text = [
                f"Frame: {self.frame_count}",
                f"People: {stats['current_count']}",
                f"Total: {stats['total_unique']}",
                f"Heatmap: {self.crowd_analyzer.heatmap_min_people}-{self.crowd_analyzer.heatmap_max_people}"
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(topdown, (8, y_offset + i*25 - 18), (tw + 12, y_offset + i*25 + 5), (0, 0, 0), -1)
                cv2.putText(topdown, text, (10, y_offset + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            
            self.topdown_signal.emit(topdown)
            
            # FPS calculation
            frames_processed += 1
            t1 = time.time()
            elapsed = t1 - last_time
            if elapsed >= 1.0:
                self.fps_signal.emit(frames_processed / elapsed)
                last_time = t1
                frames_processed = 0
            
            # Loop timing
            loop_time = time.time() - t0
            sleep_time = max(0.001, 1.0 / 30.0 - loop_time)
            time.sleep(sleep_time)

        self._cleanup_resources()
        self.finished.emit()

# ==============================
# MainWindow
# ==============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crowd Analysis UI - Heatmap System")
        self.resize(1400, 800)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Displays
        display_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(display_layout)
        
        self.labelA = QtWidgets.QLabel("Camera A")
        self.labelA.setFixedSize(360, 240)
        self.labelA.setStyleSheet("background:black; color:white;")
        self.labelA.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.labelB = QtWidgets.QLabel("Camera B")
        self.labelB.setFixedSize(360, 240)
        self.labelB.setStyleSheet("background:black; color:white;")
        self.labelB.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.labelTD = QtWidgets.QLabel("Top-Down Heatmap View")
        self.labelTD.setFixedSize(600, 450)
        self.labelTD.setStyleSheet("background:black; color:white;")
        self.labelTD.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        display_layout.addWidget(self.labelA)
        display_layout.addWidget(self.labelB)
        display_layout.addWidget(self.labelTD)

        # Controls
        controls_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(controls_layout)
        
        self.btn_loadA = QtWidgets.QPushButton("Load Video A")
        self.btn_loadA.clicked.connect(self.on_loadA)
        self.btn_loadB = QtWidgets.QPushButton("Load Video B")
        self.btn_loadB.clicked.connect(self.on_loadB)
        
        controls_layout.addWidget(self.btn_loadA)
        controls_layout.addWidget(self.btn_loadB)
        
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
        layout.addLayout(buttons_layout)
        
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

        self.status = QtWidgets.QLabel("Ready. Using predetermined calibration points.")
        layout.addWidget(self.status)

        # Worker
        self.worker = VideoWorker()
        self.worker.frame_updated.connect(self.on_frame_update)
        self.worker.topdown_signal.connect(self.update_topdown)
        self.worker.status_signal.connect(self.update_status)
        self.worker.error_signal.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)

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
        self.update_status("Processing started with heatmap...")

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
        self.update_status("Processing finished")

    def on_error(self, error_msg):
        QtWidgets.QMessageBox.critical(self, "Error", error_msg)
        self.update_status(f"Error: {error_msg}")
        self.on_finished()

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
        if self.worker._running:
            reply = QtWidgets.QMessageBox.question(
                self,
                'Confirm Exit',
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

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())