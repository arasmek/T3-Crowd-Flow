# main.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import config
import vision_utils as vu

# ==============================================================
# =============== CALIBRATION + GRID PREVIEW ===================
# ==============================================================

imgA = cv2.imread(config.CALIB_A)
imgB = cv2.imread(config.CALIB_B)
if imgA is None or imgB is None:
    raise SystemExit("Error: Could not load calibration images.")

camA_pts = np.array([
    [289, 577], [689, 156], [1102, 174], [1236, 680]
], np.float32)
camB_pts = np.array([
    [14, 691], [126, 233], [477, 207], [801, 544]
], np.float32)

world_pts = np.array([
    [0, 0], [0, config.WORLD_H], [config.WORLD_W, config.WORLD_H], [config.WORLD_W, 0]
], np.float32)

H_A, H_B = vu.compute_homographies(camA_pts, camB_pts, world_pts)
scale, margin = config.SCALE, config.MARGIN
output_w = int(config.WORLD_W * scale) + margin * 2
output_h = int(config.WORLD_H * scale) + margin * 2
S = np.array([
    [scale, 0, margin],
    [0, -scale, config.WORLD_H * scale + margin],
    [0, 0, 1]
], np.float32)

warpA = cv2.warpPerspective(imgA, S @ H_A, (output_w, output_h))
warpB = cv2.warpPerspective(imgB, S @ H_B, (output_w, output_h))
topdown_photo = cv2.addWeighted(warpA, 0.5, warpB, 0.5, 0)

GRID_W, GRID_H = config.GRID_W, config.GRID_H
cell_w = config.WORLD_W / GRID_W
cell_h = config.WORLD_H / GRID_H

# Draw grid for preview
preview = topdown_photo.copy()
for i in range(GRID_W + 1):
    x = int(margin + i * cell_w * scale)
    cv2.line(preview, (x, margin), (x, output_h - margin), (0, 255, 0), 1)
for j in range(GRID_H + 1):
    y = int(output_h - margin - j * cell_h * scale)
    cv2.line(preview, (margin, y), (output_w - margin, y), (0, 255, 0), 1)

cv2.imwrite(config.TOPDOWN_REF, topdown_photo)
print("Saved reference background:", config.TOPDOWN_REF)

plt.figure(figsize=(12, 6))
plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)); plt.title("Camera A")
plt.subplot(1,3,2); plt.imshow(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)); plt.title("Camera B")
plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)); plt.title("Top-down grid preview")
plt.tight_layout(); plt.show()

# ==============================================================
# =============== YOLO + VISUALIZATION =========================
# ==============================================================

model = YOLO(config.MODEL_PATH)
H_A, H_B = vu.load_homographies()
H_invA, H_invB = np.linalg.inv(H_A), np.linalg.inv(H_B)

capA = cv2.VideoCapture(config.VIDEO_A)
capB = cv2.VideoCapture(config.VIDEO_B)
if not capA.isOpened() or not capB.isOpened():
    raise SystemExit("Error opening video files.")

fpsA = capA.get(cv2.CAP_PROP_FPS) or 10.0
fpsB = capB.get(cv2.CAP_PROP_FPS) or 10.0
fps = min(fpsA, fpsB)
skip_frames = int(config.SKIP_SECONDS * fps)
capA.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
capB.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
print(f"Skipping to {config.SKIP_SECONDS}s ({skip_frames} frames)")

bg_photo = cv2.imread(config.TOPDOWN_REF)
if bg_photo is None:
    bg_photo = np.zeros((output_h, output_w, 3), np.uint8)
bg_faint = vu.make_faint_background(bg_photo, alpha=0.18)

print("Running YOLO... Press ESC to quit.")

while True:
    retA, frameA = capA.read()
    retB, frameB = capB.read()
    if not retA and not retB:
        break

    topdown = bg_faint.copy()

    # grid overlay
    for i in range(GRID_W + 1):
        x = int(margin + i * cell_w * scale)
        cv2.line(topdown, (x, margin), (x, output_h - margin), (100, 100, 100), 1)
    for j in range(GRID_H + 1):
        y = int(output_h - margin - j * cell_h * scale)
        cv2.line(topdown, (margin, y), (output_w - margin, y), (100, 100, 100), 1)
    vu.draw_axis_labels(topdown, GRID_W, GRID_H, cell_w, cell_h,
                    config.WORLD_W, config.WORLD_H, scale, margin)
    

    # === Camera A ===
    if retA:
        resA = model(frameA, conf=0.3, classes=[0])
        annotatedA = resA[0].plot()
        annotatedA = vu.draw_world_grid_on_camera(
            annotatedA, H_invA, GRID_W, GRID_H, cell_w, cell_h,
            config.WORLD_W, config.WORLD_H, (0, 0, 255))
        cv2.imshow("Camera A", annotatedA)

        for box in resA[0].boxes.xyxy:
            x1, y1, x2, y2 = box.tolist()
            wx, wy = vu.project_to_world(((x1+x2)/2, y2), H_A)
            if 0 <= wx <= config.WORLD_W and 0 <= wy <= config.WORLD_H:
                px, py = vu.world_to_topdown(wx, wy, S)
                cv2.circle(topdown, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(topdown, f"({wx:.2f},{wy:.2f})", (px+6, py-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1, cv2.LINE_AA)

    # === Camera B ===
    if retB:
        resB = model(frameB, conf=0.3, classes=[0])
        annotatedB = resB[0].plot()
        annotatedB = vu.draw_world_grid_on_camera(
            annotatedB, H_invB, GRID_W, GRID_H, cell_w, cell_h,
            config.WORLD_W, config.WORLD_H, (255, 0, 0))
        cv2.imshow("Camera B", annotatedB)

        for box in resB[0].boxes.xyxy:
            x1, y1, x2, y2 = box.tolist()
            wx, wy = vu.project_to_world(((x1+x2)/2, y2), H_B)
            if 0 <= wx <= config.WORLD_W and 0 <= wy <= config.WORLD_H:
                px, py = vu.world_to_topdown(wx, wy, S)
                cv2.circle(topdown, (px, py), 5, (255, 0, 0), -1)
                cv2.putText(topdown, f"({wx:.2f},{wy:.2f})", (px+6, py-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0), 1, cv2.LINE_AA)

    cv2.imshow("Top-down view", topdown)
    if cv2.waitKey(1) == 27:
        break

capA.release()
capB.release()
cv2.destroyAllWindows()
print("Done.")
