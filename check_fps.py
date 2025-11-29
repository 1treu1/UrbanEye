import cv2
import os
from sistema_vigilancia.src.domain.utils import gcs_utils

def check_fps():
    bucket_name = "bk-urbaneye-videos"
    blob_name = "videos/viva_envigado_2025-11-25_camara_01_segment_20251125_114644.mp4"
    temp_file = "temp_fps_check.mp4"

    print(f"Downloading {blob_name} from {bucket_name}...")
    try:
        gcs_utils.download_blob(bucket_name, blob_name, temp_file)
        
        cap = cv2.VideoCapture(temp_file)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"\n--- Video Info ---")
        print(f"FPS: {fps}")
        print(f"Total Frames: {frame_count}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"------------------\n")
        
        cap.release()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print("Temp file removed.")

if __name__ == "__main__":
    check_fps()
