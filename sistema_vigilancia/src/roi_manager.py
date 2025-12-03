"""ROI tracking and CSV export management."""

import os
import csv
import threading
from datetime import date
from typing import Dict, Any, Deque
from collections import deque

import sistema_vigilancia.src.config as config

# Global lock for CSV writing
CSV_LOCK = threading.Lock()


class VideoState:
    """Encapsulates the state for a single video processing session."""
    
    def __init__(
        self,
        roi_mode: str,
        csv_path: str,
        duration_min: int,
        fps: float,
        latitud: str = "",
        longitud: str = "",
        lugar: str = ""
    ):
        self.roi_mode = roi_mode
        self.csv_path = csv_path
        self.fps_assumed = fps
        self.window_frames = max(1, duration_min * 60 * fps)
        self.global_frame_idx = 0
        
        self.latitud = latitud
        self.longitud = longitud
        self.lugar = lugar
        
        # DeepFace state
        self.last_deepface_frame = -config.DEEPFACE_FRAME_SKIP
        self.deepface_cache: Dict[int, Dict[str, Any]] = {}
        
        # Tracker state
        self.next_track_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.trails: Dict[int, Deque] = {}
        
        # Initialize CSV
        self._init_csv()

    def _init_csv(self):
        if self.roi_mode != "none" and self.csv_path:
            with CSV_LOCK:
                if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                    try:
                        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["time_input", "time_out", "track_id", "age", "gender", "race", "emotion", "time_2", "date", "latitud", "longitud", "lugar"])
                    except Exception as e:
                        print(f"Error initializing CSV: {e}")


def write_track_to_csv(tid: int, track: Dict[str, Any], state: VideoState) -> None:
    """Write a completed track to CSV.
    
    Args:
        tid: Track ID
        track: Track data dictionary
        state: VideoState instance
    """
    if not state.csv_path or state.roi_mode == "none":
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
    
    # Write to CSV
    try:
        with CSV_LOCK:
            file_exists = os.path.exists(state.csv_path) and os.path.getsize(state.csv_path) > 0
            
            with open(state.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["time_input", "time_out", "track_id", "age", "gender", "race", "emotion", "time_2", "date", "latitud", "longitud", "lugar"])
                
                current_date = date.today().isoformat()
                
                writer.writerow([
                    f"{time_input:.6f}",
                    f"{time_out:.6f}",
                    tid,
                    f"{avg_age:.1f}",
                    genders_str,
                    races_str,
                    emotions_str,
                    f"{time_2:.6f}",
                    current_date,
                    state.latitud or "",
                    state.longitud or "",
                    state.lugar or ""
                ])
                f.flush()
                os.fsync(f.fileno())
    except Exception as e:
        print(f"Error writing to CSV: {e}")


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

