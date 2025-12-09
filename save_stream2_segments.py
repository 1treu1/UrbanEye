import subprocess
import argparse
import os
import time
import datetime
import signal
import glob
import threading
from google.cloud import storage
import wifi_geolocation

def upload_to_gcs_thread(bucket_name, source_file_name, destination_blob_name, credentials_path, metadata=None):
    """Uploads a file to the bucket in a separate thread."""
    def _upload():
        max_retries = 5
        base_delay = 10 # Seconds

        for attempt in range(1, max_retries + 1):
            try:
                print(f"[Upload] Attempt {attempt}/{max_retries}: {source_file_name} -> {destination_blob_name}")
                # Set a timeout for the client operations (connection and read)
                storage_client = storage.Client.from_service_account_json(credentials_path)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(destination_blob_name)
                
                if metadata:
                    blob.metadata = metadata
                
                # Explicitly setting timeout=300 (5 minutes) for the upload call.
                blob.upload_from_filename(source_file_name, timeout=300)
                
                print(f"[Upload] Success: {source_file_name}")
                
                # Delete local file after successful upload
                if os.path.exists(source_file_name):
                    os.remove(source_file_name)
                    print(f"[Upload] Deleted local file: {source_file_name}")
                
                # Break the loop if successful
                return 

            except Exception as e:
                print(f"[Upload] Error on attempt {attempt}: {e}")
                if attempt < max_retries:
                    # Exponential backoff: 10s, 20s, 40s, 80s...
                    sleep_time = base_delay * (2 ** (attempt - 1))
                    print(f"[Upload] Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"[Upload] FINAL FAILURE. Deleting local file to save space: {source_file_name}")
                    if os.path.exists(source_file_name):
                        os.remove(source_file_name)

    thread = threading.Thread(target=_upload)
    thread.daemon = True # Daemon thread so it doesn't block script exit
    thread.start()

def stop_ffmpeg_gracefully(process):
    """Stops FFMPEG gracefully to ensure MP4 trailer is written."""
    print("Stopping FFMPEG gracefully...")
    
    # Method 1: Send 'q' to stdin (Universal and preferred)
    try:
        if process.stdin:
            process.stdin.write(b'q')
            process.stdin.flush()
            try:
                process.wait(timeout=3)
                return True
            except subprocess.TimeoutExpired:
                pass
    except Exception as e:
        print(f"Stdin 'q' failed: {e}")

    # Method 2: OS-Specific Signals
    # Linux/Mac: SIGINT (equivalent to Ctrl+C)
    # Windows: CTRL_BREAK_EVENT (requires creationflags=CREATE_NEW_PROCESS_GROUP)
    try:
        if os.name == 'nt':
            print("Sending CTRL_BREAK_EVENT (Windows)...")
            os.kill(process.pid, signal.CTRL_BREAK_EVENT)
        else:
            print("Sending SIGINT (Linux/Mac)...")
            process.send_signal(signal.SIGINT)
            
        try:
            process.wait(timeout=15)
            return True
        except subprocess.TimeoutExpired:
            pass
    except Exception as e:
        print(f"Signal failed: {e}")

    # Method 3: Force Kill (Last Resort)
    print("Graceful exit failed. Forcing kill (file may be corrupt).")
    process.kill()
    process.wait()
    return False

def save_stream_segments_async(stream_url, segment_duration_minutes, output_dir, bucket_name, place_name, camera_name, lat=None, lon=None):
    """
    Records a stream, stops gracefully, and uploads asynchronously.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Find credentials file
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    cred_files = glob.glob(os.path.join(config_dir, "*.json"))
    if not cred_files:
        print("Error: No JSON credentials found in config/ directory.")
        return
    credentials_path = cred_files[0]
    print(f"Using credentials: {credentials_path}")

    segment_duration_seconds = segment_duration_minutes * 60
    
    print(f"Starting recording from {stream_url}")
    print(f"Segment duration: {segment_duration_minutes} minutes")
    print(f"Output directory: {output_dir}")
    print(f"Target Bucket: {bucket_name}")
    print("Press Ctrl+C to stop.")

    while True:
        process = None
        output_filename = ""
        try:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            date_folder = now.strftime("%Y-%m-%d")
            filename_only = f"segment_{timestamp}.mp4"
            output_filename = os.path.join(output_dir, filename_only)
            
            # Command to record
            # -rtsp_transport tcp: Force TCP for RTSP (more reliable)
            # -timeout: Socket timeout for RTSP in microseconds (5000000 = 5s)
            # -use_wallclock_as_timestamps 1: Fix non-monotonic DTS by using local time
            # -fflags +genpts: Generate new PTS
            # -c:v copy: Copy video stream (fast, no re-encoding)
            # -an: Disable audio (save bandwidth/storage)
            command = [
                "ffmpeg",
                "-y",
                "-rtsp_transport", "tcp",
                "-timeout", "5000000",
                "-use_wallclock_as_timestamps", "1",
                "-fflags", "+genpts",
                "-i", stream_url,
                "-c:v", "copy",
                "-an",
                output_filename
            ]
            
            print(f"Recording segment: {output_filename}")
            
            # Start FFMPEG
            # On Windows, we need CREATE_NEW_PROCESS_GROUP to send CTRL_BREAK_EVENT
            creationflags = 0
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            
            process = subprocess.Popen(command, stdin=subprocess.PIPE, creationflags=creationflags)
            
            # Wait for the specified duration
            try:
                process.wait(timeout=segment_duration_seconds)
            except subprocess.TimeoutExpired:
                # Time is up! Stop gracefully
                stop_ffmpeg_gracefully(process)
            
            print(f"Finished recording: {output_filename}")
            
            # Upload to GCS (Async)
            if os.path.exists(output_filename):
                # Check if file size is > 0 (basic corruption check)
                if os.path.getsize(output_filename) > 0:
                    gcs_path = f"{place_name}/{date_folder}/{camera_name}/{filename_only}"
                    
                    metadata = {
                        "created_at": now.isoformat(),
                        "segment_seconds": str(segment_duration_seconds)
                    }
                    if lat:
                        metadata["latitude"] = str(lat)
                    if lon:
                        metadata["longitude"] = str(lon)
                        
                    upload_to_gcs_thread(bucket_name, output_filename, gcs_path, credentials_path, metadata)
                else:
                    print(f"Warning: File {output_filename} is empty. Skipping upload.")
            
        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
            if process:
                stop_ffmpeg_gracefully(process)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            if process:
                process.kill()
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record stream and upload to GCS (Async).")
    parser.add_argument("--url", type=str, required=True, help="Stream URL")
    parser.add_argument("--minutes", type=int, default=1, help="Duration of each segment in minutes")
    parser.add_argument("--out", type=str, default="recordings", help="Output directory")
    parser.add_argument("--bucket", type=str, required=True, help="GCS Bucket Name")
    parser.add_argument("--lugar", type=str, required=True, help="Place name (e.g., 'oficina')")
    parser.add_argument("--camara", type=str, required=True, help="Camera name (e.g., 'camara_1')")
    
    parser.add_argument("--lat", type=str, help="Latitude")
    parser.add_argument("--lon", type=str, help="Longitude")
    
    args = parser.parse_args()

    # Auto-detect location if not provided
    if not args.lat or not args.lon:
        print("[Auto-Location] Latitude/Longitude not provided. Attempting to detect via Wi-Fi...")
        try:
            detected_lat, detected_lon, _ = wifi_geolocation.get_device_location()
            if detected_lat and detected_lon:
                args.lat = str(detected_lat)
                args.lon = str(detected_lon)
                print(f"[Auto-Location] Success: {args.lat}, {args.lon}")
            else:
                print("[Auto-Location] Failed to detect location.")
        except Exception as e:
            print(f"[Auto-Location] Error during detection: {e}")
    
    save_stream_segments_async(args.url, args.minutes, args.out, args.bucket, args.lugar, args.camara, args.lat, args.lon)
