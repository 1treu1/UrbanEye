"""Configuration and global state management."""

import os
# Suppress TF noise and force TF to CPU — PyTorch/YOLO will own the GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
# Use legacy Keras 2 API via tf_keras (required by DeepFace with Keras 3)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Force TensorFlow to CPU BEFORE any framework loads CUDA
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

from typing import Dict, Any, Optional
from collections import deque
from ultralytics import YOLO

# ROI tracking configuration
ROI_MODE: str = "none"  # none | consolidated | visits
CSV_PATH: str = ""
FPS_ASSUMED: int = 14.90
WINDOW_FRAMES: int = 0
GLOBAL_FRAME_IDX: int = 0

# Location metadata
LATITUD: Optional[str] = None
LONGITUD: Optional[str] = None
LUGAR: Optional[str] = None

# YOLOv11 tracker
YOLO_MODEL: Optional[YOLO] = None
YOLO_MODEL_INITIALIZED: bool = False

# Track data storage
TRACKS: Dict[int, Dict[str, Any]] = {}  # track_id -> {bbox, attributes, trail, enter_time, etc.}
TRAILS: Dict[int, deque] = {}  # track_id -> deque of (x, y) positions for trail drawing
MAX_TRAIL_LENGTH: int = 50  # Maximum points in trail
LEAVE_TOL: int = 5  # visits mode tolerance
NEXT_TRACK_ID: int = 1  # Next available track ID

# Optimization: DeepFace throttling and caching
DEEPFACE_FRAME_SKIP: int = 15  # Run DeepFace every 15 frames (~2 times per second)
LAST_DEEPFACE_FRAME: int = -1  # Last frame where DeepFace was executed
DEEPFACE_CACHE: Dict[int, Dict[str, Any]] = {}  # Cache DeepFace results per track_id
DEEPFACE_CACHE_AGE: int = 30  # Frames before cache expires

# GPU setup flag
GPU_READY: bool = False

