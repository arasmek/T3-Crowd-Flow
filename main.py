import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import config
import vision_utils as vu
from deepsort_tracker import MultiCameraTracker
from crowd_analytics import CrowdFlowAnalyzer

imgA = cv2.imread(config.CALIB_A)
imgB = cv2.imread(config.CALIB_B)
if imgA is None or imgB is None:
    raise SystemExit("Error: Could not load calibration images.")

# Camera reference points (PREDETERMINED - from old version)
camA_pts = np.array([
    [289, 577], [689, 156], [1102, 174], [1236, 680]
], np.float32)
camB_pts = np.array([
    [14, 691], [126, 233], [477, 207], [801, 544]
], np.float32)

# World coordinates
world_pts = np.array([
    [0, 0], [0, config.WORLD_H], [config.WORLD_W, config.WORLD_H], [config.WORLD_W, 0]
], np.float32)

# Compute homography
H_A, H_B = vu.compute_homographies(camA_pts, camB_pts, world_pts)
scale, margin = config.SCALE, config.MARGIN
output_w = int(config.WORLD_W * scale) + margin * 2
output_h = int(config.WORLD_H * scale) + margin * 2

# Transformation matrix for top-down view
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

# Preview grid
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

# ========== DEEPSORT TRACKING + CROWD ANALYTICS + HEATMAP ===============

print("Initializing YOLO and DeepSORT...")
print(f"Optimization: Resizing frames to {config.INFERENCE_WIDTH}x{config.INFERENCE_HEIGHT} for inference")
model = YOLO(config.YOLO_MODEL_PATH)
tracker = MultiCameraTracker()
analyzer = CrowdFlowAnalyzer(config.WORLD_W, config.WORLD_H, config.HEATMAP_CELL_SIZE)

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

print("Running DeepSORT tracking with heatmap...")
print("Controls: ESC=quit, 't'=toggle trajectories, 'f'=toggle flow vectors, 'h'=toggle heatmap")
print("          '+'/'-' adjust heatmap max people")
frame_count = 0
show_trajectories = config.SHOW_TRAJECTORIES
show_flow = config.SHOW_VELOCITY_VECTORS
show_heatmap = True

# Main tracking loop
while True:
    retA, frameA = capA.read()
    retB, frameB = capB.read()
    if not retA and not retB:
        break
    
    frame_count += 1
    tracks_A, tracks_B = [], []
    
    # Camera A Detection & Tracking
    if retA:
        # Resize frame for faster inference
        frameA_resized, scale_x_A, scale_y_A = tracker.resize_frame_for_inference(frameA)
        
        # Run YOLO on resized frame
        resA = model.track(frameA_resized, conf=config.DETECTION_CONFIDENCE, 
                          classes=[0], persist=False, verbose=False)
        
        if len(resA[0].boxes) > 0:
            # Scale detections back to original frame coordinates
            scaled_boxes = tracker.scale_detections(resA[0].boxes, scale_x_A, scale_y_A)
            
            # Create temporary detection objects with scaled coordinates
            class ScaledBox:
                def __init__(self, xyxy, conf):
                    self.xyxy = [np.array(xyxy)]
                    self.conf = [conf]
            
            scaled_detection_objects = [ScaledBox(box['xyxy'], box['conf']) for box in scaled_boxes]
            
            # Update tracker with original frame and scaled detections
            tracks_A = tracker.update_tracks(
                scaled_detection_objects, frameA, 'A', H_A, 
                (config.WORLD_W, config.WORLD_H)
            )
    
    # Camera B Detection & Tracking
    if retB:
        # Resize frame for faster inference
        frameB_resized, scale_x_B, scale_y_B = tracker.resize_frame_for_inference(frameB)
        
        # Run YOLO on resized frame
        resB = model.track(frameB_resized, conf=config.DETECTION_CONFIDENCE,
                          classes=[0], persist=False, verbose=False)
        
        if len(resB[0].boxes) > 0:
            # Scale detections back to original frame coordinates
            scaled_boxes = tracker.scale_detections(resB[0].boxes, scale_x_B, scale_y_B)
            
            # Create temporary detection objects with scaled coordinates
            class ScaledBox:
                def __init__(self, xyxy, conf):
                    self.xyxy = [np.array(xyxy)]
                    self.conf = [conf]
            
            scaled_detection_objects = [ScaledBox(box['xyxy'], box['conf']) for box in scaled_boxes]
            
            # Update tracker with original frame and scaled detections
            tracks_B = tracker.update_tracks(
                scaled_detection_objects, frameB, 'B', H_B,
                (config.WORLD_W, config.WORLD_H)
            )
    
    # Merge tracks and update analytics
    all_tracks = tracker.merge_camera_tracks(tracks_A, tracks_B)
    analyzer.update(all_tracks)
    
    # Create a lookup dict for global IDs
    global_id_map = {}
    for track in all_tracks:
        global_id_map[track.local_id] = track.global_id
        if hasattr(track, 'merged_from'):
            for local_id in track.merged_from:
                global_id_map[local_id] = track.global_id
    
    # Display camera views
    if retA:
        annotatedA = frameA.copy()
        for track in tracks_A:
            ltrb = track.ltrb
            x1, y1, x2, y2 = map(int, ltrb)
            
            global_id = global_id_map.get(track.local_id, "?")
            merged_track = next((t for t in all_tracks if t.local_id == track.local_id), None)
            is_merged = merged_track and hasattr(merged_track, 'merged_from')
            
            color = (255, 0, 255) if is_merged else (0, 255, 0)
            cv2.rectangle(annotatedA, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{global_id}"
            cv2.putText(annotatedA, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            foot_x = int((x1 + x2) / 2)
            foot_y = y2
            cv2.circle(annotatedA, (foot_x, foot_y), 5, (0, 255, 255), -1)
        
        annotatedA = vu.draw_world_grid_on_camera(
            annotatedA, H_invA, GRID_W, GRID_H, cell_w, cell_h,
            config.WORLD_W, config.WORLD_H, (100, 200, 100))
        cv2.imshow("Camera A", annotatedA)
    
    if retB:
        annotatedB = frameB.copy()
        for track in tracks_B:
            ltrb = track.ltrb
            x1, y1, x2, y2 = map(int, ltrb)
            
            global_id = global_id_map.get(track.local_id, "?")
            merged_track = next((t for t in all_tracks if t.local_id == track.local_id), None)
            is_merged = merged_track and hasattr(merged_track, 'merged_from')
            
            color = (255, 0, 255) if is_merged else (255, 100, 0)
            cv2.rectangle(annotatedB, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{global_id}"
            cv2.putText(annotatedB, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            foot_x = int((x1 + x2) / 2)
            foot_y = y2
            cv2.circle(annotatedB, (foot_x, foot_y), 5, (0, 255, 255), -1)
        
        annotatedB = vu.draw_world_grid_on_camera(
            annotatedB, H_invB, GRID_W, GRID_H, cell_w, cell_h,
            config.WORLD_W, config.WORLD_H, (200, 150, 100))
        cv2.imshow("Camera B", annotatedB)
    
    # Top-down visualization
    topdown = bg_faint.copy()
    
    # Draw heatmap overlay
    if show_heatmap:
        heatmap = analyzer.get_density_heatmap(smooth_sigma=2.5)
        
        # Create colored heatmap overlay
        heatmap_colored = np.zeros((analyzer.hmap_h, analyzer.hmap_w, 3), dtype=np.uint8)
        
        # Apply colormap (hot colors for high density)
        for hy in range(analyzer.hmap_h):
            for hx in range(analyzer.hmap_w):
                value = heatmap[hy, hx]
                if value > 0:
                    # Color gradient: blue -> green -> yellow -> red
                    if value < 0.33:
                        # Blue to green
                        r = 0
                        g = int(255 * (value / 0.33))
                        b = int(255 * (1 - value / 0.33))
                    elif value < 0.66:
                        # Green to yellow
                        r = int(255 * ((value - 0.33) / 0.33))
                        g = 255
                        b = 0
                    else:
                        # Yellow to red
                        r = 255
                        g = int(255 * (1 - (value - 0.66) / 0.34))
                        b = 0
                    
                    heatmap_colored[hy, hx] = [b, g, r]
        
        # Resize heatmap to match topdown view dimensions
        heatmap_resized = cv2.resize(
            heatmap_colored,
            (int(config.WORLD_W * scale), int(config.WORLD_H * scale)),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create overlay region
        y_start = margin
        y_end = margin + int(config.WORLD_H * scale)
        x_start = margin
        x_end = margin + int(config.WORLD_W * scale)
        
        # Blend heatmap with topdown
        mask = (heatmap_resized.sum(axis=2) > 0).astype(np.uint8) * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        roi = topdown[y_start:y_end, x_start:x_end]
        blended = cv2.addWeighted(roi, 0.6, heatmap_resized, 0.4, 0)
        topdown[y_start:y_end, x_start:x_end] = np.where(mask > 0, blended, roi)
    
    # Draw grid
    for i in range(GRID_W + 1):
        x = int(margin + i * cell_w * scale)
        cv2.line(topdown, (x, margin), (x, output_h - margin), (80, 80, 80), 1)
    for j in range(GRID_H + 1):
        y = int(output_h - margin - j * cell_h * scale)
        cv2.line(topdown, (margin, y), (output_w - margin, y), (80, 80, 80), 1)
    
    vu.draw_axis_labels(topdown, GRID_W, GRID_H, cell_w, cell_h,
                       config.WORLD_W, config.WORLD_H, scale, margin)
    
    # Draw flow vectors
    if show_flow:
        flow_vectors = analyzer.get_flow_vectors()
        for vec in flow_vectors:
            px, py = vu.world_to_topdown(vec['x'], vec['y'], S)
            
            arrow_scale = min(vec['magnitude'] * 300, 50)
            end_x = int(px + vec['vx'] * arrow_scale)
            end_y = int(py - vec['vy'] * arrow_scale)
            
            cv2.arrowedLine(topdown, (px, py), (end_x, end_y),
                          (0, 255, 255), 2, tipLength=0.3)
    
    # Draw current tracks
    for track in all_tracks:
        wx, wy = track.world_x, track.world_y
        px, py = vu.world_to_topdown(wx, wy, S)
        
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
        cv2.putText(topdown, str(track.global_id), (px+12, py-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        if show_trajectories:
            trajectory = analyzer.get_trajectory(track.global_id)
            if len(trajectory) > 1:
                for i in range(len(trajectory) - 1):
                    pt1 = vu.world_to_topdown(trajectory[i][0], trajectory[i][1], S)
                    pt2 = vu.world_to_topdown(trajectory[i+1][0], trajectory[i+1][1], S)
                    alpha = i / len(trajectory)
                    fade_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
                    cv2.line(topdown, pt1, pt2, fade_color, 2)
        
        pred_pos = analyzer.predict_position(track.global_id, config.PREDICTION_HORIZON)
        if pred_pos:
            pred_px, pred_py = vu.world_to_topdown(pred_pos[0], pred_pos[1], S)
            cv2.circle(topdown, (pred_px, pred_py), 6, (0, 200, 255), 1)
            cv2.line(topdown, (px, py), (pred_px, pred_py), (0, 200, 255), 1)
    
    # Draw statistics
    stats = analyzer.get_statistics()
    info_text = [
        f"Frame: {frame_count}",
        f"People in grid: {stats['current_count']}",
        f"Total tracked: {stats['total_unique']}",
        f"Heatmap range: {analyzer.heatmap_min_people}-{analyzer.heatmap_max_people}",
        "",
        "t=trails | f=flow | h=heatmap",
        "+/- adjust max people"
    ]
    
    y_offset = 30
    for i, text in enumerate(info_text):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(topdown, (8, y_offset + i*25 - 18), (tw + 12, y_offset + i*25 + 5),
                     (0, 0, 0), -1)
        cv2.putText(topdown, text, (10, y_offset + i*25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(topdown, text, (10, y_offset + i*25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Legend
    legend_y = output_h - 80
    cv2.rectangle(topdown, (8, legend_y - 5), (200, output_h - 8), (0, 0, 0), -1)
    cv2.putText(topdown, "Legend:", (15, legend_y + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(topdown, (20, legend_y + 30), 6, (0, 255, 0), 2)
    cv2.putText(topdown, "Camera A", (35, legend_y + 33),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(topdown, (120, legend_y + 30), 6, (255, 100, 0), 2)
    cv2.putText(topdown, "Camera B", (135, legend_y + 33),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(topdown, (20, legend_y + 50), 6, (255, 0, 255), 3)
    cv2.putText(topdown, "Both Cams", (35, legend_y + 53),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.imshow("Crowd Flow Analysis", topdown)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('t'):
        show_trajectories = not show_trajectories
        print(f"Trajectories: {'ON' if show_trajectories else 'OFF'}")
    elif key == ord('f'):
        show_flow = not show_flow
        print(f"Flow vectors: {'ON' if show_flow else 'OFF'}")
    elif key == ord('h'):
        show_heatmap = not show_heatmap
        print(f"Heatmap: {'ON' if show_heatmap else 'OFF'}")
    elif key == ord('+') or key == ord('='):
        analyzer.heatmap_max_people += 1
        print(f"Heatmap max: {analyzer.heatmap_max_people}")
    elif key == ord('-') or key == ord('_'):
        analyzer.heatmap_max_people = max(analyzer.heatmap_min_people + 1, analyzer.heatmap_max_people - 1)
        print(f"Heatmap max: {analyzer.heatmap_max_people}")

capA.release()
capB.release()
cv2.destroyAllWindows()