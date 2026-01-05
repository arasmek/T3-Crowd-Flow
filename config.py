# World and grid settings
WORLD_W, WORLD_H = 4.0, 3.0
GRID_W, GRID_H = 20, 15
SCALE = 250
MARGIN = 40

# File paths
CALIB_A = "data/camA_calib.jpg"
CALIB_B = "data/camB_calib.jpg"
VIDEO_A = "data/camA.mp4"
VIDEO_B = "data/camB.mp4"

# Model paths
YOLO_MODEL_PATH = "yolov9s.pt"
DEEPSORT_MODEL_PATH = "deep_sort/deep/checkpoint/ckpt.t7"

# Homography paths
H_A_PATH = "data/H_A.npy"
H_B_PATH = "data/H_B.npy"
TOPDOWN_REF = "data/topdown_reference.jpg"

# Processing settings
SKIP_SECONDS = 170
DETECTION_CONFIDENCE = 0.4

# Image optimization settings
INFERENCE_WIDTH = 640
INFERENCE_HEIGHT = 480
MAINTAIN_ASPECT_RATIO = True

# DeepSORT settings
MAX_COSINE_DISTANCE = 0.3
NN_BUDGET = 100
MAX_IOU_DISTANCE = 0.7
MAX_AGE = 70
N_INIT = 3

# Crowd flow analysis settings
TRAJECTORY_HISTORY = 30
MIN_TRAJECTORY_LENGTH = 5
HEATMAP_DECAY = 0.95
HEATMAP_CELL_SIZE = 0.2
PREDICTION_HORIZON = 30
VELOCITY_SMOOTHING_FRAMES = 5

# Visualization settings
SHOW_TRAJECTORIES = True
SHOW_VELOCITY_VECTORS = True