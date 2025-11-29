"""DeepFace face analysis and attribute extraction module."""

from typing import Dict, Any, List, Tuple, Optional
import cv2
from deepface import DeepFace

import sistema_vigilancia.src.config as config
from .roi_manager import VideoState


def should_run_deepface(state: VideoState) -> bool:
    """Check if DeepFace should run on this frame based on throttling."""
    return (state.global_frame_idx - state.last_deepface_frame) >= config.DEEPFACE_FRAME_SKIP


def clean_old_cache(state: VideoState) -> None:
    """Remove old entries from DeepFace cache."""
    tracks_to_remove = []
    for cached_tid, cached_data in state.deepface_cache.items():
        if cached_data.get("frame", 0) < state.global_frame_idx - config.DEEPFACE_CACHE_AGE:
            tracks_to_remove.append(cached_tid)
    for tid in tracks_to_remove:
        del state.deepface_cache[tid]


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0.0 and 1.0
    """
    xi1 = max(bbox1[0], bbox2[0])
    yi1 = max(bbox1[1], bbox2[1])
    xi2 = min(bbox1[2], bbox2[2])
    yi2 = min(bbox1[3], bbox2[3])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def match_deepface_to_yolo(
    df_bbox: List[float],
    df_center: Tuple[int, int],
    yolo_tracks: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """Match a DeepFace detection to the best YOLO track.
    
    Args:
        df_bbox: DeepFace bbox as [x1, y1, x2, y2]
        df_center: DeepFace center as (cx, cy)
        yolo_tracks: Dictionary of YOLO tracks
    
    Returns:
        Best matching track_id or None
    """
    best_track_id = None
    best_score = -1.0
    df_cx, df_cy = df_center
    df_w = df_bbox[2] - df_bbox[0]
    df_h = df_bbox[3] - df_bbox[1]
    
    for track_id, yolo_track in yolo_tracks.items():
        if not yolo_track["inside_roi"]:
            continue
        
        yx, yy, yw, yh = yolo_track["bbox"]
        yolo_bbox_xyxy = [yx, yy, yx + yw, yy + yh]
        yolo_cx, yolo_cy = yolo_track["center"]
        
        iou = calculate_iou(df_bbox, yolo_bbox_xyxy)
        center_dist = ((df_cx - yolo_cx) ** 2 + (df_cy - yolo_cy) ** 2) ** 0.5
        max_allowed_dist = max(df_w, df_h) * 1.5
        
        distance_score = 1.0 - min(center_dist / max_allowed_dist, 1.0) if max_allowed_dist > 0 else 0.0
        score = iou * 2.0 + distance_score * 1.0
        
        if (iou > 0.05 or center_dist < max_allowed_dist) and score > best_score:
            best_score = score
            best_track_id = track_id
    
    return best_track_id


def extract_deepface_attributes(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract attributes from DeepFace result.
    
    Args:
        result: DeepFace analysis result
    
    Returns:
        Dictionary with age, gender, race, emotion
    """
    age_val = result.get("age")
    if age_val is None:
        age_val = result.get("age_estimate") or result.get("estimated_age")
    
    gender_val = result.get("dominant_gender") or result.get("gender") or ""
    race_val = result.get("dominant_race") or result.get("race") or ""
    emotion_val = result.get("dominant_emotion") or result.get("emotion") or ""
    
    return {
        "age": age_val,
        "gender": str(gender_val).strip() if gender_val else None,
        "race": str(race_val).strip() if race_val else None,
        "emotion": str(emotion_val).strip() if emotion_val else None,
    }


def parse_deepface_region(region: Dict[str, Any], roi_offset: Tuple[int, int] = (0, 0)) -> Optional[Tuple[int, int, int, int, Tuple[int, int]]]:
    """Parse bounding box from DeepFace region.
    
    Args:
        region: DeepFace region dictionary
        roi_offset: (rx0, ry0) offset if ROI was extracted from full frame
    
    Returns:
        (x, y, w, h, center) or None if invalid
    """
    df_x = int(region.get("x", region.get("left", 0)))
    df_y = int(region.get("y", region.get("top", 0)))
    df_w = int(region.get("w", region.get("width", 0) or 0))
    df_h = int(region.get("h", region.get("height", 0) or 0))
    
    if df_w == 0 and "right" in region and "left" in region:
        df_w = int(region["right"]) - int(region.get("left", 0))
    if df_h == 0 and "bottom" in region and "top" in region:
        df_h = int(region["bottom"]) - int(region.get("top", 0))
    
    if df_w <= 0 or df_h <= 0:
        return None
    
    rx0, ry0 = roi_offset
    df_x += rx0
    df_y += ry0
    
    df_cx, df_cy = df_x + df_w // 2, df_y + df_h // 2
    df_bbox_xyxy = [df_x, df_y, df_x + df_w, df_y + df_h]
    
    return (df_x, df_y, df_w, df_h, (df_cx, df_cy), df_bbox_xyxy)


def analyze_roi_with_deepface(
    roi_frame_rgb,
    yolo_tracks: Dict[int, Dict[str, Any]],
    roi_coords: Tuple[int, int, int, int],
    state: VideoState
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """Analyze ROI region with DeepFace and match to YOLO tracks.
    
    Args:
        roi_frame_rgb: ROI region in RGB format
        yolo_tracks: Dictionary of YOLO tracks
        roi_coords: (rx0, ry0, rx1, ry1) ROI coordinates
        state: VideoState instance
    
    Returns:
        Tuple of (deepface_results, deepface_full_roi_results)
        - deepface_results: track_id -> attributes for matched tracks
        - deepface_full_roi_results: unmatched DeepFace detections
    """
    deepface_results: Dict[int, Dict[str, Any]] = {}
    deepface_full_roi_results: List[Dict[str, Any]] = []
    rx0, ry0, _, _ = roi_coords
    
    state.last_deepface_frame = state.global_frame_idx
    clean_old_cache(state)
    
    try:
        df_roi_results = DeepFace.analyze(
            roi_frame_rgb,
            actions=["age", "gender", "race", "emotion"],
            enforce_detection=False,
            align=True,
            detector_backend="retinaface",
            silent=True
        )
        
        if isinstance(df_roi_results, dict):
            df_roi_results = [df_roi_results]
        
        for res in df_roi_results or []:
            region = res.get("region") or {}
            parsed = parse_deepface_region(region, (rx0, ry0))
            
            if parsed is None:
                continue
            
            df_x, df_y, df_w, df_h, (df_cx, df_cy), df_bbox_xyxy = parsed
            
            # Match to YOLO track
            best_track_id = match_deepface_to_yolo(df_bbox_xyxy, (df_cx, df_cy), yolo_tracks)
            
            # Extract attributes
            attrs = extract_deepface_attributes(res)
            df_attrs = {
                "bbox": (df_x, df_y, df_w, df_h),
                **attrs
            }
            
            if best_track_id is not None:
                # Merge with existing if present
                if best_track_id not in deepface_results:
                    deepface_results[best_track_id] = df_attrs
                else:
                    existing = deepface_results[best_track_id]
                    deepface_results[best_track_id] = {
                        "bbox": df_attrs.get("bbox", existing.get("bbox")),
                        "age": df_attrs.get("age") if df_attrs.get("age") is not None else existing.get("age"),
                        "gender": df_attrs.get("gender") if df_attrs.get("gender") else existing.get("gender"),
                        "race": df_attrs.get("race") if df_attrs.get("race") else existing.get("race"),
                        "emotion": df_attrs.get("emotion") if df_attrs.get("emotion") else existing.get("emotion"),
                    }
                
                # Update cache
                state.deepface_cache[best_track_id] = {
                    **deepface_results[best_track_id],
                    "frame": state.global_frame_idx
                }
            else:
                # No match: standalone detection
                deepface_full_roi_results.append(df_attrs)
    
    except Exception:
        # DeepFace failed, continue with cached results
        pass
    
    return deepface_results, deepface_full_roi_results


def populate_from_cache(
    yolo_tracks: Dict[int, Dict[str, Any]],
    deepface_results: Dict[int, Dict[str, Any]],
    state: VideoState
) -> None:
    """Populate deepface_results from cache for tracks in ROI."""
    for track_id, yolo_track in yolo_tracks.items():
        if yolo_track["inside_roi"] and track_id not in deepface_results:
            if track_id in state.deepface_cache:
                cached = state.deepface_cache[track_id]
                if cached.get("frame", 0) >= state.global_frame_idx - config.DEEPFACE_CACHE_AGE:
                    deepface_results[track_id] = {
                        "age": cached.get("age"),
                        "gender": cached.get("gender"),
                        "race": cached.get("race"),
                        "emotion": cached.get("emotion"),
                    }


def match_standalone_detections(
    yolo_tracks: Dict[int, Dict[str, Any]],
    deepface_results: Dict[int, Dict[str, Any]],
    deepface_full_roi_results: List[Dict[str, Any]],
    state: VideoState
) -> None:
    """Match standalone DeepFace detections to YOLO tracks that don't have results."""
    for track_id, yolo_track in yolo_tracks.items():
        if yolo_track["inside_roi"] and track_id not in deepface_results and deepface_full_roi_results:
            yx, yy, yw, yh = yolo_track["bbox"]
            yolo_cx, yolo_cy = yolo_track["center"]
            
            best_match = None
            best_dist = float('inf')
            
            for df_det in deepface_full_roi_results:
                df_x, df_y, df_w, df_h = df_det["bbox"]
                df_cx, df_cy = df_x + df_w // 2, df_y + df_h // 2
                center_dist = ((df_cx - yolo_cx) ** 2 + (df_cy - yolo_cy) ** 2) ** 0.5
                max_dist = (yw + yh) / 2 * 2.0
                
                if center_dist < max_dist and center_dist < best_dist:
                    best_dist = center_dist
                    best_match = df_det
            
            if best_match is not None:
                deepface_results[track_id] = {
                    "age": best_match.get("age"),
                    "gender": best_match.get("gender"),
                    "race": best_match.get("race"),
                    "emotion": best_match.get("emotion"),
                }
                state.deepface_cache[track_id] = {
                    **deepface_results[track_id],
                    "frame": state.global_frame_idx
                }

