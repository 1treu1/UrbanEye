"""ROI tracking and CSV export management."""

import os
import csv
import threading
from datetime import date, datetime, timedelta
from typing import Dict, Any, Deque, Optional
from collections import deque

import pandas as pd
import src.config as config
from .domain.utils import gcs_utils

# Global lock for CSV writing
CSV_LOCK = threading.Lock()


class VideoState:
    """Encapsulates the state for a single video processing session."""
    
    def __init__(
        self,
        roi_mode: str,
        duration_min: int,
        fps: float,
        latitud: str = "",
        longitud: str = "",
        lugar: str = "",
        video_start_time: Optional[datetime] = None,
        video_name: str = "unknown_video"
    ):
        self.roi_mode = roi_mode
        self.fps_assumed = fps
        self.window_frames = max(1, duration_min * 60 * fps)
        self.global_frame_idx = 0
        
        self.latitud = latitud
        self.longitud = longitud
        self.lugar = lugar
        self.video_start_time = video_start_time
        self.video_name = video_name
        
        # Buffer for completed tracks
        self.tracks_buffer: List[Dict[str, Any]] = []
        
        # DeepFace state
        self.last_deepface_frame = -config.DEEPFACE_FRAME_SKIP
        self.deepface_cache: Dict[int, Dict[str, Any]] = {}
        
        # Tracker state
        self.next_track_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.trails: Dict[int, Deque] = {}


def write_track_to_csv(tid: int, track: Dict[str, Any], state: VideoState) -> None:
    """Buffer a completed track for Parquet export.
    
    Args:
        tid: Track ID
        track: Track data dictionary
        state: VideoState instance
    """
    if state.roi_mode == "none":
        return
    
    if "enter_time" not in track or "exit_time" not in track:
        return
    
    # Calculate values
    time_input = track["enter_time"]  # already in minutes
    time_out = track["exit_time"]  # already in minutes
    time_2 = time_out - time_input  # duration in minutes
    
    # Calculate average age
    if track.get("ages") and len(track["ages"]) > 0:
        avg_age = sum(track["ages"]) / len(track["ages"])
    else:
        avg_age = 0.0
    
    # For gender, race, emotion: join unique values with commas if multiple
    genders = track.get("genders", [])
    if genders:
        genders_unique = list(dict.fromkeys(genders))  # Preserve order, remove duplicates
        genders_str = ",".join(genders_unique) if len(genders_unique) > 1 else genders_unique[0] if genders_unique else ""
    else:
        genders_str = ""
    
    races = track.get("races", [])
    if races:
        races_unique = list(dict.fromkeys(races))
        races_str = ",".join(races_unique) if len(races_unique) > 1 else races_unique[0] if races_unique else ""
    else:
        races_str = ""
    
    emotions = track.get("emotions", [])
    if emotions:
        emotions_unique = list(dict.fromkeys(emotions))
        emotions_str = ",".join(emotions_unique) if len(emotions_unique) > 1 else emotions_unique[0] if emotions_unique else ""
    else:
        emotions_str = ""
    
    # Calculate date
    if state.video_start_time:
        # Date = Metadata Time + Elapsed Time (seconds) - 5 hours
        # time_input is in minutes, so convert to seconds
        elapsed_seconds = time_input * 60
        calculated_date = state.video_start_time + timedelta(seconds=elapsed_seconds) - timedelta(hours=5)
        date_str = calculated_date.strftime("%Y-%m-%d %H:%M:%S.%f")
    else:
        date_str = date.today().isoformat()
    
    # Add to buffer
    row_data = {
        "time_input": time_input,
        "time_out": time_out,
        "track_id": tid,
        "age": avg_age,
        "gender": genders_str,
        "race": races_str,
        "emotion": emotions_str,
        "time_2": time_2,
        "date": date_str,
        "latitud": state.latitud or "",
        "longitud": state.longitud or "",
        "lugar": state.lugar or ""
    }
    state.tracks_buffer.append(row_data)

def save_parquet_to_gcs(state: VideoState) -> None:
    """Save buffered tracks to Parquet and upload to GCS."""
    if not state.tracks_buffer:
        return

    try:
        df = pd.DataFrame(state.tracks_buffer)
        
        # Create temp file
        temp_file = f"temp_{state.video_name}.parquet"
        df.to_parquet(temp_file, index=False)
        
        # Upload to GCS
        destination = f"data/{state.video_name}.parquet"
        gcs_utils.upload_blob("bk-urbaneye-videos", temp_file, destination)
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        print(f"Saved {len(state.tracks_buffer)} tracks to {destination}")
        
    except Exception as e:
        print(f"Error saving Parquet to GCS: {e}")


def flush_window(state: VideoState) -> None:
    """Flush tracks that are still in ROI after window expires or video ends."""
    tracks_to_flush = []
    
    for tid, track in list(state.tracks.items()):
        # Check if track has entered ROI but hasn't exited yet
        if "enter_time" in track and track.get("inside", False):
            track["exit_time"] = state.global_frame_idx / state.fps_assumed / 60.0  # minutes
            track["inside"] = False
            tracks_to_flush.append((tid, track.copy()))
        # Also check inactive tracks that might have entered but weren't detected exiting
        elif "enter_time" in track and "exit_time" not in track:
            track["exit_time"] = state.global_frame_idx / state.fps_assumed / 60.0
            track["inside"] = False
            tracks_to_flush.append((tid, track.copy()))
    
    for tid, track in tracks_to_flush:
        write_track_to_csv(tid, track, state)
        
    # Save to GCS
    save_parquet_to_gcs(state)

