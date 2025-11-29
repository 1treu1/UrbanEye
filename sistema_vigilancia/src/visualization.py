"""Visualization and drawing utilities."""

from typing import Dict, Any, List, Tuple, Deque
import cv2
import numpy as np

import sistema_vigilancia.src.config as config
from .roi_manager import VideoState


def draw_trails(annotated: np.ndarray, trails: Dict[int, Deque]) -> None:
    """Draw trails for tracked objects.
    
    Args:
        annotated: Frame to draw on
        trails: Dictionary of trails (deque of points)
    """
    for track_id, trail in trails.items():
        # Draw trail as connected lines
        if len(trail) > 1:
            points = list(trail)
            for i in range(1, len(points)):
                pt1 = points[i - 1]
                pt2 = points[i]
                # Use gradient color - newer points brighter
                alpha = i / len(points)
                color_intensity = int(255 * alpha)
                cv2.line(annotated, pt1, pt2, (color_intensity, color_intensity, 255), 2)


def draw_yolo_tracks(
    annotated: np.ndarray,
    yolo_tracks: Dict[int, Dict[str, Any]],
    deepface_results: Dict[int, Dict[str, Any]],
    state: VideoState
) -> int:
    """Draw YOLO tracks with DeepFace attributes.
    
    Args:
        annotated: Frame to draw on
        yolo_tracks: Dictionary of YOLO tracks
        deepface_results: Dictionary of DeepFace results per track_id
        state: VideoState instance
    
    Returns:
        Number of tracks drawn
    """
    num_tracks = 0
    
    for track_id, yolo_track in yolo_tracks.items():
        x, y, w, h = yolo_track["bbox"]
        
        # Draw track box in cyan
        color = (255, 255, 0)  # Cyan in BGR
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, f"ID: {track_id}", (x, y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        num_tracks += 1
        
        # If inside ROI and has DeepFace attributes, draw them
        if yolo_track["inside_roi"] and track_id in deepface_results:
            df_attrs = deepface_results[track_id]
            age = df_attrs.get("age")
            gender = df_attrs.get("gender")
            race = df_attrs.get("race")
            emotion = df_attrs.get("emotion")
            
            elapsed_txt = ""
            if track_id in state.tracks and "enter_time" in state.tracks[track_id]:
                enter_time_min = state.tracks[track_id]["enter_time"]
                current_time_min = state.global_frame_idx / max(1, state.fps_assumed) / 60.0
                elapsed_sec = (current_time_min - enter_time_min) * 60.0
                elapsed_txt = f" • {elapsed_sec:.1f}s"
            
            label = f"age: {age}  gender: {gender}  race: {race}  emotion: {emotion}{elapsed_txt}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(0, y - 35)
            cv2.rectangle(annotated, (x, max(0, y_text - th - 4)), (x + tw + 6, y_text + 2), (0, 0, 0), -1)
            cv2.putText(annotated, label, (x + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return num_tracks


def draw_deepface_detections(annotated: np.ndarray, deepface_full_roi_results: List[Dict[str, Any]]) -> int:
    """Draw standalone DeepFace detections.
    
    Args:
        annotated: Frame to draw on
        deepface_full_roi_results: List of unmatched DeepFace detections
    
    Returns:
        Number of DeepFace detections drawn
    """
    num_deepface = 0
    
    for df_det in deepface_full_roi_results:
        df_x, df_y, df_w, df_h = df_det["bbox"]
        # Draw DeepFace detection box in green
        cv2.rectangle(annotated, (df_x, df_y), (df_x + df_w, df_y + df_h), (0, 255, 0), 2)
        num_deepface += 1
        
        # Draw DeepFace attributes
        age = df_det.get("age")
        gender = df_det.get("gender")
        race = df_det.get("race")
        emotion = df_det.get("emotion")
        
        label = f"DF: age:{age} g:{gender} r:{race} e:{emotion}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        y_text = max(0, df_y - 10)
        cv2.rectangle(annotated, (df_x, max(0, y_text - th - 4)), (df_x + tw + 6, y_text + 2), (0, 0, 0), -1)
        cv2.putText(annotated, label, (df_x + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    
    return num_deepface


def draw_status(annotated: np.ndarray, num_tracks: int, num_deepface: int) -> None:
    """Draw status text on frame.
    
    Args:
        annotated: Frame to draw on
        num_tracks: Number of YOLO tracks
        num_deepface: Number of DeepFace detections
    """
    status_text = f"tracks: {num_tracks} | DeepFace: {num_deepface}"
    cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    if num_tracks == 0 and num_deepface == 0:
        cv2.putText(annotated, "No persons detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)


def draw_roi(annotated: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> None:
    """Draw ROI rectangle.
    
    Args:
        annotated: Frame to draw on
        roi_coords: (rx0, ry0, rx1, ry1) ROI coordinates
    """
    rx0, ry0, rx1, ry1 = roi_coords
    cv2.rectangle(annotated, (rx0, ry0), (rx1, ry1), (0, 0, 255), 2)

