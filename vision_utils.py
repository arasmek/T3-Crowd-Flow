# vision_utils.py
import cv2
import numpy as np
import config

# ========== Homography utils ==========
def compute_homographies(camA_pts, camB_pts, world_pts):
    H_A, _ = cv2.findHomography(camA_pts, world_pts)
    H_B, _ = cv2.findHomography(camB_pts, world_pts)
    np.save(config.H_A_PATH, H_A)
    np.save(config.H_B_PATH, H_B)
    return H_A, H_B

def load_homographies():
    H_A = np.load(config.H_A_PATH)
    H_B = np.load(config.H_B_PATH)
    return H_A, H_B

# ========== Projection ==========
def project_to_world(foot_point, H):
    """Convert a single (x, y) image pixel coordinate to world coordinate."""
    # foot_point must be (x, y)
    pts = np.array([[foot_point]], dtype=np.float32)  # shape (1, 1, 2)
    world = cv2.perspectiveTransform(pts, H)[0, 0]
    return float(world[0]), float(world[1])

def world_to_topdown(wx, wy, S):
    """Convert world coordinate -> pixel in top-down view."""
    P = S @ np.array([[wx, wy, 1]], np.float32).T
    return int(P[0] / P[2]), int(P[1] / P[2])

# ========== Grid visualization ==========
def draw_world_grid_on_camera(frame, H_inv, grid_w, grid_h, cell_w, cell_h, world_w, world_h, color=(0,255,0)):
    for i in range(grid_w + 1):
        pts = np.array([[i * cell_w, 0], [i * cell_w, world_h]], np.float32).reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(pts, H_inv)
        cv2.line(frame, tuple(img_pts[0,0].astype(int)), tuple(img_pts[1,0].astype(int)), color, 1)
    for j in range(grid_h + 1):
        pts = np.array([[0, j * cell_h], [world_w, j * cell_h]], np.float32).reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(pts, H_inv)
        cv2.line(frame, tuple(img_pts[0,0].astype(int)), tuple(img_pts[1,0].astype(int)), color, 1)
    return frame

def make_faint_background(photo, alpha=0.18):
    """Darken or fade background photo for top-down grid overlay."""
    return cv2.addWeighted(photo, alpha, np.zeros_like(photo), 1.0 - alpha, 0)

def draw_axis_labels(topdown, grid_w, grid_h, cell_w, cell_h, world_w, world_h, scale, margin):
    """Draws x/y coordinate labels on grid edges."""
    output_h, output_w = topdown.shape[:2]

    step_x = max(1, grid_w // 4)
    for i in range(0, grid_w + 1, step_x):
        x = int(margin + i * cell_w * scale)
        wx = i * cell_w
        cv2.putText(topdown, f"x={wx:.1f}", (x - 18, output_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    step_y = max(1, grid_h // 3)
    for j in range(0, grid_h + 1, step_y):
        y = int(output_h - margin - j * cell_h * scale)
        wy = j * cell_h
        cv2.putText(topdown, f"y={wy:.1f}", (6, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return topdown

# ========== Top-down visualization ==========
def draw_topdown(base_image, tracks, analyzer=None, draw_heatmap=True):
    """
    Draw tracks and optional crowd flow on top of a base image (warped camera overlay).
    base_image: np.ndarray (BGR)
    tracks: list of dicts with 'x', 'y' coordinates in top-down space
    analyzer: optional CrowdFlowAnalyzer instance for heatmaps/arrows
    draw_heatmap: whether to overlay heatmap/flow
    """
    img = base_image.copy()  # avoid modifying original

    # Draw each track
    for track in tracks:
        x, y = int(track['x']), int(track['y'])
        color = (0, 0, 255)  # red for tracked person
        cv2.circle(img, (x, y), 5, color, -1)

    # Optional: overlay crowd flow heatmap
    if analyzer and draw_heatmap:
        if hasattr(analyzer, "draw_flow"):
            img = analyzer.draw_flow(img)

    return img
