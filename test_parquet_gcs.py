import os
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch
from sistema_vigilancia.src.roi_manager import VideoState, write_track_to_csv, save_parquet_to_gcs

def test_parquet_gcs():
    # Mock gcs_utils
    with patch("sistema_vigilancia.src.roi_manager.gcs_utils") as mock_gcs:
        # Setup
        video_name = "test_video_segment"
        start_time = datetime(2025, 11, 3, 17, 22, 55, 980014)
        
        # Initialize state
        state = VideoState("consolidated", 30, 30.0, video_start_time=start_time, video_name=video_name)
        
        # Mock track
        track = {
            "enter_time": 3.0 / 60.0,
            "exit_time": 6.0 / 60.0,
            "ages": [25],
            "genders": ["Man"],
            "races": ["Latino"],
            "emotions": ["neutral"]
        }
        
        # Write track (should buffer)
        write_track_to_csv(1, track, state)
        
        print(f"Buffered tracks: {len(state.tracks_buffer)}")
        
        # Trigger save
        save_parquet_to_gcs(state)
        
        # Verify upload_blob called
        expected_dest = f"data/{video_name}.parquet"
        mock_gcs.upload_blob.assert_called_once()
        args = mock_gcs.upload_blob.call_args[0]
        bucket = args[0]
        source_file = args[1]
        dest_blob = args[2]
        
        print(f"Upload called with: Bucket={bucket}, Source={source_file}, Dest={dest_blob}")
        
        if dest_blob == expected_dest:
            print("SUCCESS: Destination blob path is correct.")
        else:
            print(f"FAILURE: Expected {expected_dest}, got {dest_blob}")
            
        # Verify temp file was created (it's deleted in the function, so we check if it WAS created by checking if pandas to_parquet was called implicitly via the fact that upload happened with a file)
        # Actually, since we mocked upload_blob, the file might still exist if we didn't mock os.remove? 
        # Wait, the code deletes it. But we can check if the source file passed to upload_blob actually existed at some point?
        # Simpler: just trust the mock call for now.
        
        # Verify DataFrame content logic by inspecting buffer
        row = state.tracks_buffer[0]
        expected_date = "2025-11-03 12:22:58.980014"
        if row["date"] == expected_date:
             print("SUCCESS: Date calculation preserved.")
        else:
             print(f"FAILURE: Date calculation mismatch. Got {row['date']}")

if __name__ == "__main__":
    test_parquet_gcs()
