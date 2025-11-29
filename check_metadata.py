import os
import subprocess
import json
from datetime import datetime
from sistema_vigilancia.src.domain.utils import gcs_utils

def check_metadata():
    bucket_name = "bk-urbaneye-videos"
    blob_name = "videos/viva_envigado_2025-11-25_camara_01_segment_20251125_114644.mp4"
    temp_file = "temp_metadata_check.mp4"

    print(f"--- Checking GCS Object Metadata ---")
    try:
        client = gcs_utils.get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.get_blob(blob_name)
        
        if not blob:
            print(f"Error: Blob {blob_name} not found.")
            return

        print(f"Name: {blob.name}")
        print(f"Time Created: {blob.time_created}")
        print(f"Time Updated: {blob.updated}")
        print(f"Custom Metadata: {blob.metadata}")
        
        print(f"\n--- Downloading for Internal Metadata Check ---")
        blob.download_to_filename(temp_file)
        
        # Use ffprobe to get metadata
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            temp_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            format_tags = data.get("format", {}).get("tags", {})
            
            print("\n--- Video File Metadata (ffprobe) ---")
            print("Format Tags:")
            for k, v in format_tags.items():
                print(f"  {k}: {v}")
                
            # Check streams for creation_time
            for i, stream in enumerate(data.get("streams", [])):
                stream_tags = stream.get("tags", {})
                if stream_tags:
                    print(f"\nStream {i} Tags:")
                    for k, v in stream_tags.items():
                        print(f"  {k}: {v}")
        else:
            print("Error running ffprobe.")
            print(result.stderr)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print("\nTemp file removed.")

if __name__ == "__main__":
    check_metadata()
