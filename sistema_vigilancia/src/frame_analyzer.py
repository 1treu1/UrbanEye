"""Frame analysis orchestrator - main logic for processing video frames."""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from collections import deque

import src.config as config
from .yolo_tracker import init_yolo_model, detect_and_track_persons
from .deepface_analyzer import (
    should_run_deepface, analyze_roi_with_deepface,
    populate_from_cache, match_standalone_detections
)
from .gpu_setup import ensure_tf_gpu
from .visualization import (
    draw_trails, draw_yolo_tracks, draw_deepface_detections,
    draw_status, draw_roi
)
from .roi_manager import write_track_to_csv, flush_window, VideoState


def calculate_roi_coords(frame_shape: Tuple[int, int, int], roi_size: float) -> Tuple[int, int, int, int]:
    """Calculate ROI coordinates.
    
    Args:
        frame_shape: (height, width, channels)
        roi_size: ROI size as fraction of frame (0.0-1.0)
    
    Returns:
        (rx0, ry0, rx1, ry1) ROI coordinates
    """
    h_img, w_img = frame_shape[:2]
    rw = int(roi_size * w_img)
    rh = int(roi_size * h_img)
    rx0 = (w_img - rw) // 2
    ry0 = (h_img - rh) // 2
    rx1 = rx0 + rw
    ry1 = ry0 + rh
    return (rx0, ry0, rx1, ry1)


def update_tracks_and_trails(
    yolo_tracks: Dict[int, Dict[str, Any]],
    deepface_results: Dict[int, Dict[str, Any]],
    state: VideoState
) -> set:
    """Update global tracks and trails from YOLO tracks.
    
    Args:
        yolo_tracks: Dictionary of YOLO tracks
        deepface_results: Dictionary of DeepFace results
        state: VideoState instance
    
    Returns:
        Set of current track IDs
    """
    current_track_ids = set()
    
    for track_id, yolo_track in yolo_tracks.items():
        current_track_ids.add(track_id)
        
        # Update or create track
        if track_id not in state.tracks:
            state.tracks[track_id] = {
                "bbox": yolo_track["bbox"],
                "ages": [],
                "genders": [],
                "races": [],
                "emotions": [],
                "inside": False,
                "frames_in": 0,
            }
            state.trails[track_id] = deque(maxlen=config.MAX_TRAIL_LENGTH)
        
        track = state.tracks[track_id]
        track["bbox"] = yolo_track["bbox"]
        track["last_seen"] = state.global_frame_idx
        
        # Update trail with center position
        cx, cy = yolo_track["center"]
        state.trails[track_id].append((cx, cy))
        
        # Process ROI tracking
        inside_roi = yolo_track["inside_roi"]
        was_inside = track.get("inside", False)
        
        # Get DeepFace attributes if available
        df_attrs = deepface_results.get(track_id, {})
        
        if inside_roi:
            if not was_inside:
                # Just entered ROI
                track["enter_time"] = state.global_frame_idx / state.fps_assumed / 60.0
                track["frames_in"] = 0
                track["inside"] = True
                track["ages"] = []
                track["genders"] = []
                track["races"] = []
                track["emotions"] = []
            
            track["frames_in"] = track.get("frames_in", 0) + 1
            
            # Accumulate DeepFace attributes when available
            if df_attrs:
                # Age
                age_val = df_attrs.get("age")
                if age_val is not None:
                    try:
                        age_float = float(age_val)
                        if age_float > 0:
                            track["ages"].append(age_float)
                    except (ValueError, TypeError):
                        pass
                
                # Gender
                gender_val = df_attrs.get("gender")
                if gender_val and str(gender_val).strip():
                    track["genders"].append(str(gender_val).strip())
                
                # Race
                race_val = df_attrs.get("race")
                if race_val and str(race_val).strip():
                    track["races"].append(str(race_val).strip())
                
                # Emotion
                emotion_val = df_attrs.get("emotion")
                if emotion_val and str(emotion_val).strip():
                    track["emotions"].append(str(emotion_val).strip())
        else:
            if was_inside:
                # Just exited ROI
                track["exit_time"] = state.global_frame_idx / state.fps_assumed / 60.0
                track["inside"] = False
                write_track_to_csv(track_id, track, state)
                # Clear for potential re-entry
                if "enter_time" in track:
                    del track["enter_time"]
                if "exit_time" in track:
                    del track["exit_time"]
                track["ages"] = []
                track["genders"] = []
                track["races"] = []
                track["emotions"] = []
                track["frames_in"] = 0
    
    return current_track_ids


def cleanup_old_tracks(current_track_ids: set, state: VideoState) -> None:
    """Remove tracks that are no longer active.
    
    Args:
        current_track_ids: Set of currently active track IDs
        state: VideoState instance
    """
    tracks_to_remove = []
    
    for track_id in list(state.tracks.keys()):
        if track_id not in current_track_ids:
            # Track disappeared - if inside ROI, flush it
            if state.tracks[track_id].get("inside", False):
                track = state.tracks[track_id]
                track["exit_time"] = state.global_frame_idx / state.fps_assumed / 60.0
                track["inside"] = False
                write_track_to_csv(track_id, track, state)
                # Clear for potential re-entry
                if "enter_time" in track:
                    del track["enter_time"]
                if "exit_time" in track:
                    del track["exit_time"]
            tracks_to_remove.append(track_id)
    
    for track_id in tracks_to_remove:
        if track_id in state.tracks:
            del state.tracks[track_id]
        if track_id in state.trails:
            del state.trails[track_id]


def analyze_frame(
    frame_bgr: np.ndarray,
    state: VideoState,
    roi_size: float = 0.65,
    yolo_conf: float = 0.15,
    yolo_iou: float = 0.45,
    visualize: bool = True
) -> np.ndarray:
    """Analyze a single frame: detect, track, and analyze persons.
    
    Args:
        frame_bgr: Frame in BGR format
        state: VideoState instance
        roi_size: ROI size as fraction (0.0-1.0)
        yolo_conf: YOLO confidence threshold
        yolo_iou: YOLO IoU threshold
        visualize: Whether to draw visualizations on the frame
    
    Returns:
        Annotated frame with detections and labels
    """
    # Note: state.global_frame_idx is incremented in stream_generator, not here
    
    # Initialize models
    init_yolo_model()
    ensure_tf_gpu()
    
    # Verify YOLO model is ready
    if config.YOLO_MODEL is None:
        print("WARNING: YOLO_MODEL is None! Check model initialization.")
    
    annotated = frame_bgr.copy()
    
    # Calculate ROI
    roi_coords = calculate_roi_coords(annotated.shape, roi_size)
    rx0, ry0, rx1, ry1 = roi_coords
    
    # Step 1: YOLO tracking
    yolo_tracks = detect_and_track_persons(frame_bgr, roi_coords, yolo_conf, yolo_iou)
    
    # Debug: Log if no tracks found
    if len(yolo_tracks) == 0 and config.YOLO_MODEL is not None:
        # Don't spam, but useful for debugging
        pass  # Can add logging here if needed
    
    # Step 2: DeepFace analysis
    deepface_results: Dict[int, Dict[str, Any]] = {}
    deepface_full_roi_results: List[Dict[str, Any]] = []
    
    if should_run_deepface(state):
        # Lazy RGB conversion
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        roi_frame_rgb = frame_rgb[ry0:ry1, rx0:rx1] if (ry1 > ry0 and rx1 > rx0) else frame_rgb
        
        deepface_results, deepface_full_roi_results = analyze_roi_with_deepface(
            frame_rgb, yolo_tracks, roi_coords, state
        )
    else:
        # Use cache
        populate_from_cache(yolo_tracks, deepface_results, state)
    
    # Fallback matching
    match_standalone_detections(yolo_tracks, deepface_results, deepface_full_roi_results, state)
    
    # Step 3: Update tracks and process ROI
    current_track_ids = update_tracks_and_trails(yolo_tracks, deepface_results, state)
    cleanup_old_tracks(current_track_ids, state)
    
    # Step 4: Visualization
    if visualize:
        draw_trails(annotated, state.trails)
        # Draw YOLO tracks (includes DeepFace info)
        num_tracks = draw_yolo_tracks(annotated, yolo_tracks, deepface_results, state)
        # Draw standalone DeepFace detections
        num_deepface = draw_deepface_detections(annotated, deepface_full_roi_results)
        # Draw status info
        draw_status(annotated, num_tracks, num_deepface)
        # Draw ROI
        if state.roi_mode != "none":
            draw_roi(annotated, roi_coords)
    
    # Window flush
    if state.roi_mode != "none" and state.window_frames > 0 and state.global_frame_idx >= state.window_frames:
        flush_window(state)
    
    return annotated

