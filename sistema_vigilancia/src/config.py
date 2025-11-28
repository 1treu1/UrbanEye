"""Configuration and global state management."""

from typing import Dict, Any, Optional
from collections import deque
from ultralytics import YOLO

# ROI tracking configuration
ROI_MODE: str = "none"  # none | consolidated | visits
CSV_PATH: str = ""
FPS_ASSUMED: int = 22
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
DEEPFACE_FRAME_SKIP: int = 1  # Run DeepFace every frame (set to 1 for full data collection)
LAST_DEEPFACE_FRAME: int = -1  # Last frame where DeepFace was executed
DEEPFACE_CACHE: Dict[int, Dict[str, Any]] = {}  # Cache DeepFace results per track_id
DEEPFACE_CACHE_AGE: int = 5  # Frames before cache expires

# GPU setup flag
GPU_READY: bool = False

