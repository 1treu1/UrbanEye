import os
import json
import base64
import logging
import tempfile
import cv2
from flask import Flask, request
from google.cloud import storage
from google.oauth2 import service_account

# Import internal modules
from . import config
from .frame_analyzer import analyze_frame
from .roi_manager import set_roi_config, flush_window

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global configuration
# Allow overriding via env var, default to local config for dev
SERVICE_ACCOUNT_FILE = os.environ.get('SERVICE_ACCOUNT_FILE', 
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'service.json')
)
OUTPUT_BUCKET_SUFFIX = "/data" # As requested: bk-urbaneye-videos/data

def get_storage_client():
    """Get GCS client using service account if available, else ADC."""
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        try:
            credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
            return storage.Client(credentials=credentials, project=credentials.project_id)
        except Exception as e:
            logger.error(f"Failed to load service account credentials: {e}")
            # Fallback to ADC
            return storage.Client()
    else:
        return storage.Client()

def process_video(video_path, output_csv_path):
    """Process video file and generate CSV report."""
    # Configure ROI manager (headless mode)
    # We use "consolidated" mode by default as per gradio_stream.py
    set_roi_config(roi_mode="consolidated", csv_path=output_csv_path, duration_min=30, fps=22)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return False

    try:
        frame_idx = 0
        analyze_every_n = 10 # Same as gradio_stream.py default
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            # Update global frame index
            config.GLOBAL_FRAME_IDX += 1
            
            # Analyze every N frames
            if frame_idx % analyze_every_n == 0:
                # We don't need the annotated frame for headless, but we need to run analysis
                # to update tracks and stats
                _ = analyze_frame(frame, roi_size=0.65) # Default ROI size
            
            frame_idx += 1
            
        # Flush any remaining tracks
        if config.ROI_MODE != "none":
            flush_window()
            
        return True
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return False
    finally:
        cap.release()

@app.route('/', methods=['POST'])
def handle_event():
    """Handle Eventarc event."""
    envelope = request.get_json()
    if not envelope:
        return 'No event received', 400

    logger.info(f"Received event: {json.dumps(envelope)}")

    # Extract bucket and file name using logic from auditoria.py
    bucket_name = 'Desconocido'
    file_name = 'Desconocido'
    
    # Logic adapted from auditoria.py
    if 'protoPayload' in envelope and 'resourceName' in envelope['protoPayload']:
        resource_name = envelope['protoPayload']['resourceName']
        if 'objects/' in resource_name:
            try:
                file_name = resource_name.split('objects/')[-1]
            except Exception:
                pass
        if 'resource' in envelope and 'labels' in envelope['resource']:
             bucket_name = envelope['resource']['labels'].get('bucket_name', 'Desconocido')

    # Fallbacks
    if bucket_name == 'Desconocido':
        # Try to find in data/message
        pass # Add more robust extraction if needed based on auditoria.py

    # If we still don't have it, try the payload directly if it's a direct GCS event
    if bucket_name == 'Desconocido':
         bucket_name = envelope.get('bucket') or envelope.get('bucketId') or 'Desconocido'
    if file_name == 'Desconocido':
         file_name = envelope.get('name') or envelope.get('objectId') or 'Desconocido'

    if bucket_name == 'Desconocido' or file_name == 'Desconocido':
        logger.error("Could not extract bucket or filename from event")
        return 'Bad Request: Missing bucket or file info', 400

    logger.info(f"Processing file: {file_name} from bucket: {bucket_name}")

    # Download video
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        logger.info(f"Downloading to {temp_video.name}...")
        blob.download_to_filename(temp_video.name)
        temp_video_path = temp_video.name

    # Prepare CSV output
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_csv:
        temp_csv_path = temp_csv.name

    try:
        # Process
        logger.info("Starting video processing...")
        success = process_video(temp_video_path, temp_csv_path)
        
        if success and os.path.exists(temp_csv_path) and os.path.getsize(temp_csv_path) > 0:
            # Upload CSV
            # Target: bk-urbaneye-videos/data/FILENAME.csv
            # We assume the user wants it in the same bucket but under 'data/' prefix
            # The user said: "el csv se debe guardar en bk-urbaneye-videos/data"
            # If the input bucket is bk-urbaneye-videos, we use that.
            
            target_bucket_name = "bk-urbaneye-videos" # Hardcoded based on request, or use bucket_name
            target_blob_name = f"data/{os.path.basename(file_name)}.csv"
            
            logger.info(f"Uploading results to gs://{target_bucket_name}/{target_blob_name}")
            
            target_bucket = storage_client.bucket(target_bucket_name)
            target_blob = target_bucket.blob(target_blob_name)
            target_blob.upload_from_filename(temp_csv_path)
            
            logger.info("Upload complete.")
            return 'OK', 200
        else:
            logger.warning("Processing finished but no CSV generated or error occurred.")
            return 'Processing failed or no data', 500

    except Exception as e:
        logger.error(f"Error in processing loop: {e}")
        return f"Internal Error: {e}", 500
    finally:
        # Cleanup
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
