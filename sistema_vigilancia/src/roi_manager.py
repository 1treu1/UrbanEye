"""ROI tracking and CSV export management."""

import os
import csv
from typing import Dict, Any

import sistema_vigilancia.src.config as config


def set_roi_config(roi_mode: str, csv_path: str, duration_min: int, fps: int) -> None:
    """Configure ROI tracking settings.
    
    Args:
        roi_mode: Tracking mode ("none", "consolidated", "visits")
        csv_path: Path to CSV file for exporting data
        duration_min: Window duration in minutes
        fps: Assumed frames per second
    """
    config.ROI_MODE = roi_mode
    config.CSV_PATH = csv_path
    config.FPS_ASSUMED = fps
    config.WINDOW_FRAMES = max(1, duration_min * 60 * fps)
    config.GLOBAL_FRAME_IDX = 0
    
    # Reset tracker
    config.NEXT_TRACK_ID = 1
    config.TRACKS.clear()
    
    # Write header if file doesn't exist
    if config.ROI_MODE != "none" and config.CSV_PATH:
        if not os.path.exists(config.CSV_PATH) or os.path.getsize(config.CSV_PATH) == 0:
            with open(config.CSV_PATH, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time_input", "time_out", "track_id", "age", "gender", "race", "emotion", "time_2"])


def write_track_to_csv(tid: int, track: Dict[str, Any]) -> None:
    """Write a completed track to CSV.
    
    Args:
        tid: Track ID
        track: Track data dictionary
    """
    if not config.CSV_PATH or config.ROI_MODE == "none":
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
    file_exists = os.path.exists(config.CSV_PATH) and os.path.getsize(config.CSV_PATH) > 0
    
    with open(config.CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["time_input", "time_out", "track_id", "age", "gender", "race", "emotion", "time_2"])
        
        writer.writerow([
            f"{time_input:.6f}",
            f"{time_out:.6f}",
            tid,
            f"{avg_age:.1f}",
            genders_str,
            races_str,
            emotions_str,
            f"{time_2:.6f}"
        ])


def flush_window() -> None:
    """Flush tracks that are still in ROI after window expires or video ends."""
    tracks_to_flush = []
    
    for tid, track in list(config.TRACKS.items()):
        # Check if track has entered ROI but hasn't exited yet
        if "enter_time" in track and track.get("inside", False):
            track["exit_time"] = config.GLOBAL_FRAME_IDX / config.FPS_ASSUMED / 60.0  # minutes
            track["inside"] = False
            tracks_to_flush.append((tid, track.copy()))
        # Also check inactive tracks that might have entered but weren't detected exiting
        elif "enter_time" in track and "exit_time" not in track:
            track["exit_time"] = config.GLOBAL_FRAME_IDX / config.FPS_ASSUMED / 60.0
            track["inside"] = False
            tracks_to_flush.append((tid, track.copy()))
    
    for tid, track in tracks_to_flush:
        write_track_to_csv(tid, track)

