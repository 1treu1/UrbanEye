import os
import csv
from datetime import datetime, timedelta
from sistema_vigilancia.src.roi_manager import VideoState, write_track_to_csv

def test_date_calculation():
    # Setup
    csv_path = "test_output.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    # Mock start time: 2025-11-03 17:22:55.980014
    start_time = datetime(2025, 11, 3, 17, 22, 55, 980014)
    
    # Initialize state
    state = VideoState("consolidated", csv_path, 30, 30.0, video_start_time=start_time)
    
    # Mock track
    # Enter at 3 seconds (0.05 minutes)
    # Exit at 6 seconds (0.1 minutes)
    track = {
        "enter_time": 3.0 / 60.0,
        "exit_time": 6.0 / 60.0,
        "ages": [25],
        "genders": ["Man"],
        "races": ["Latino"],
        "emotions": ["neutral"]
    }
    
    # Write to CSV
    write_track_to_csv(1, track, state)
    
    # Verify
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        
        # Expected calculation:
        # Start: 2025-11-03 17:22:55.980014
        # + 3s : 2025-11-03 17:22:58.980014
        # - 5h : 2025-11-03 12:22:58.980014
        
        expected_date = "2025-11-03 12:22:58.980014"
        actual_date = row["date"]
        
        print(f"Expected: {expected_date}")
        print(f"Actual:   {actual_date}")
        
        if expected_date == actual_date:
            print("SUCCESS: Date calculation is correct.")
        else:
            print("FAILURE: Date calculation is incorrect.")

if __name__ == "__main__":
    test_date_calculation()
