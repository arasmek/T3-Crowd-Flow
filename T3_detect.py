import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==============================================================
# =============== CALIBRATION + GRID PREVIEW ===================
# ==============================================================

# --- Load calibration images ---
imgA = cv2.imread("camA_calib.jpg")
imgB = cv2.imread("camB_calib.jpg")

if imgA is None or imgB is None:
    print("Error: Could not load calibration images.")
    raise SystemExit

# --- 4 grid corner points from each camera ---
camA_pts = np.array([
    [289, 577], [689, 156], [1102, 174], [1236, 680]
], np.float32)
camB_pts = np.array([
    [14, 691], [126, 233], [477, 207], [801, 544]
], np.float32)

# --- World plane (arbitrary units) ---
WORLD_W, WORLD_H = 4.0, 3.0
world_pts = np.array([
    [0, 0], [0, WORLD_H], [WORLD_W, WORLD_H], [WORLD_W, 0]
], np.float32)

# --- Compute homographies ---
H_A, _ = cv2.findHomography(camA_pts, world_pts)
H_B, _ = cv2.findHomography(camB_pts, world_pts)
if H_A is None or H_B is None:
    raise SystemExit("Homography computation failed.")
np.save("H_A.npy", H_A)
np.save("H_B.npy", H_B)

# --- Projection parameters ---
scale = 250
margin = 40
output_w = int(WORLD_W * scale) + margin * 2
output_h = int(WORLD_H * scale) + margin * 2
S = np.array([
    [scale, 0, margin],
    [0, -scale, WORLD_H * scale + margin],
    [0, 0, 1]
], np.float32)

# --- Warp calibration images for background ---
warpA = cv2.warpPerspective(imgA, S @ H_A, (output_w, output_h))
warpB = cv2.warpPerspective(imgB, S @ H_B, (output_w, output_h))
overlay_preview = cv2.addWeighted(warpA, 0.5, warpB, 0.5, 0)

# --- Draw grid lines on overlay ---
GRID_W, GRID_H = 20, 15
cell_w = WORLD_W / GRID_W
cell_h = WORLD_H / GRID_H
for i in range(GRID_W + 1):
    x = int(margin + i * cell_w * scale)
    cv2.line(overlay_preview, (x, margin),
             (x, output_h - margin), (0, 255, 0), 1)
for j in range(GRID_H + 1):
    y = int(output_h - margin - j * cell_h * scale)
    cv2.line(overlay_preview, (margin, y),
             (output_w - margin, y), (0, 255, 0), 1)

cv2.imwrite("topdown_reference.jpg", overlay_preview)
print("Saved reference top-down background as topdown_reference.jpg")

# --- Quick preview ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB))
plt.title("Camera A")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB))
plt.title("Camera B")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(overlay_preview, cv2.COLOR_BGR2RGB))
plt.title("Top-down grid preview")
plt.tight_layout()
plt.show()

# ==============================================================
# =============== YOLO + LIVE FEED VISUALIZATION ===============
# ==============================================================

model = YOLO("yolov8s.pt")
H_A = np.load("H_A.npy")
H_B = np.load("H_B.npy")

capA = cv2.VideoCapture("camA.mp4")
capB = cv2.VideoCapture("camB.mp4")

if not capA.isOpened() or not capB.isOpened():
    raise SystemExit("Error opening videos.")

# --- Skip to desired point ---
SKIP_SECONDS = 200
fpsA = capA.get(cv2.CAP_PROP_FPS) or 10
fpsB = capB.get(cv2.CAP_PROP_FPS) or 10
fps = min(fpsA, fpsB)
skip_frames = int(SKIP_SECONDS * fps)
capA.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
capB.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
print(f"Skipping to ~{SKIP_SECONDS:.1f}s ({skip_frames} frames)")

# --- Preload reference top-down background ---
background = cv2.imread("topdown_reference.jpg")
if background is None:
    background = np.zeros((output_h, output_w, 3), np.uint8)

# --- Grid overlay for cameras (project world grid back into each view) ---
def draw_world_grid_on_camera(frame, H_inv, color=(0, 255, 0)):
    """Project the world grid (in meters) onto the camera frame."""
    # Vertical lines (constant X, vary Y)
    for i in range(GRID_W + 1):
        pts_world = np.array([
            [i * cell_w, 0],
            [i * cell_w, WORLD_H]
        ], np.float32).reshape(-1, 1, 2)
        pts_img = cv2.perspectiveTransform(pts_world, H_inv)
        p1, p2 = tuple(map(int, pts_img[0, 0])), tuple(map(int, pts_img[1, 0]))
        cv2.line(frame, p1, p2, color, 1)

    # Horizontal lines (constant Y, vary X)
    for j in range(GRID_H + 1):
        pts_world = np.array([
            [0, j * cell_h],
            [WORLD_W, j * cell_h]
        ], np.float32).reshape(-1, 1, 2)
        pts_img = cv2.perspectiveTransform(pts_world, H_inv)
        p1, p2 = tuple(map(int, pts_img[0, 0])), tuple(map(int, pts_img[1, 0]))
        cv2.line(frame, p1, p2, color, 1)

    return frame



H_invA = np.linalg.inv(H_A)
H_invB = np.linalg.inv(H_B)

print("Running YOLO + grid overlays. Press ESC to quit.")
frame_id = 0

while True:
    retA, frameA = capA.read()
    retB, frameB = capB.read()
    if not retA and not retB:
        break

    topdown = background.copy()
    projected_points = []

    # --- Camera A ---
    if retA:
        resA = model(frameA, conf=0.3, classes=[0])
        annotatedA = resA[0].plot()
        annotatedA = draw_world_grid_on_camera(annotatedA, H_invA, (0, 255, 0))
        cv2.imshow("Camera A (with grid)", annotatedA)

        for box in resA[0].boxes.xyxy:
            x1, y1, x2, y2 = box.tolist()
            foot = np.array([[[ (x1 + x2)/2, y2 ]]], np.float32)
            world = cv2.perspectiveTransform(foot, H_A)[0, 0]
            wx, wy = world
            if 0 <= wx <= WORLD_W and 0 <= wy <= WORLD_H:
                P = S @ np.array([[wx, wy, 1]], np.float32).T
                px, py = int(P[0]/P[2]), int(P[1]/P[2])
                cv2.circle(topdown, (px, py), 5, (0, 0, 255), -1)
                projected_points.append((px, py))

    # --- Camera B ---
    if retB:
        resB = model(frameB, conf=0.3, classes=[0])
        annotatedB = resB[0].plot()
        annotatedB = draw_world_grid_on_camera(annotatedB, H_invB, (0, 255, 255))
        cv2.imshow("Camera B (with grid)", annotatedB)

        for box in resB[0].boxes.xyxy:
            x1, y1, x2, y2 = box.tolist()
            foot = np.array([[[ (x1 + x2)/2, y2 ]]], np.float32)
            world = cv2.perspectiveTransform(foot, H_B)[0, 0]
            wx, wy = world
            if 0 <= wx <= WORLD_W and 0 <= wy <= WORLD_H:
                P = S @ np.array([[wx, wy, 1]], np.float32).T
                px, py = int(P[0]/P[2]), int(P[1]/P[2])
                cv2.circle(topdown, (px, py), 5, (255, 0, 0), -1)
                projected_points.append((px, py))

    # --- Grid lines + coordinate markers on top-down ---
    for i in range(GRID_W + 1):
        x = int(margin + i * cell_w * scale)
        cv2.line(topdown, (x, margin),
                 (x, output_h - margin), (100, 100, 100), 1)
    for j in range(GRID_H + 1):
        y = int(output_h - margin - j * cell_h * scale)
        cv2.line(topdown, (margin, y),
                 (output_w - margin, y), (100, 100, 100), 1)

    cv2.imshow("Top-down with photo & dots", topdown)

    frame_id += 1
    print(f"Frame {frame_id} | Dots: {len(projected_points)}")
    if cv2.waitKey(1) == 27:
        break

capA.release()
capB.release()
cv2.destroyAllWindows()
print("Done.")
