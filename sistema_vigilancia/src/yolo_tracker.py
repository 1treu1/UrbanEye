"""YOLOv11 person tracking module."""

from typing import Dict, Any, Optional, Tuple
import os
from ultralytics import YOLO

import sistema_vigilancia.src.config as config


def init_yolo_model(model_path: Optional[str] = None) -> None:
    """Initialize YOLOv11 model for person tracking.
    
    Args:
        model_path: Optional path to model file. If None, searches in current dir and workspace root.
    """
    if config.YOLO_MODEL_INITIALIZED and config.YOLO_MODEL is not None:
        return
    
    try:
        if model_path is None:
            model_path = "yolo11n.pt"
            # Check if model exists in current directory or workspace root
            if not os.path.exists(model_path):
                # Try workspace root
                workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                model_path_alt = os.path.join(workspace_root, "yolo11n.pt")
                if os.path.exists(model_path_alt):
                    model_path = model_path_alt
        
        config.YOLO_MODEL = YOLO(model_path)  # nano version for speed
        config.YOLO_MODEL_INITIALIZED = True
        print(f"YOLO11 model initialized successfully from {model_path}")
    except Exception as e:
        print(f"Warning: Could not initialize YOLO11 model: {e}")
        config.YOLO_MODEL = None
        config.YOLO_MODEL_INITIALIZED = False


def detect_and_track_persons(
    frame_bgr,
    roi_coords: Tuple[int, int, int, int],
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.45
) -> Dict[int, Dict[str, Any]]:
    """Detect and track persons using YOLOv11.
    
    Args:
        frame_bgr: Frame in BGR format
        roi_coords: ROI coordinates as (rx0, ry0, rx1, ry1)
        conf_threshold: Confidence threshold (0.0-1.0)
        iou_threshold: IoU threshold for NMS (0.0-1.0)
    
    Returns:
        Dictionary mapping track_id to track info: {bbox, conf, inside_roi, center}
    """
    yolo_tracks: Dict[int, Dict[str, Any]] = {}
    rx0, ry0, rx1, ry1 = roi_coords
    
    if config.YOLO_MODEL is None:
        print("ERROR: YOLO_MODEL is None in detect_and_track_persons!")
        return yolo_tracks
    
    try:
        # Run YOLOv11 tracking
        results = config.YOLO_MODEL.track(
            frame_bgr,
            persist=True,
            classes=[0],  # Only detect persons (class 0)
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and result.boxes.id is not None and len(result.boxes.id) > 0:
                boxes = result.boxes
                
                # Process each tracked person
                for i in range(len(boxes)):
                    track_id = int(boxes.id[i].item())
                    box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[i].item())
                    
                    x1, y1, x2, y2 = box
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    cx, cy = x + w // 2, y + h // 2
                    
                    # Check if center is inside ROI
                    inside_roi = (rx0 <= cx <= rx1) and (ry0 <= cy <= ry1)
                    
                    yolo_tracks[track_id] = {
                        "bbox": (x, y, w, h),
                        "bbox_xyxy": box.tolist(),
                        "conf": conf,
                        "inside_roi": inside_roi,
                        "center": (cx, cy)
                    }
    except Exception as e:
        print(f"YOLOv11 tracking error: {e}")
    
    return yolo_tracks

