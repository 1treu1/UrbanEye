#!/bin/bash

# start_urbaneye.sh
# Wrapper script to launch UrbanEye on startup
# Usage: ./start_urbaneye.sh

# Navigate to script directory
cd "$(dirname "$0")"

# Activate Virtual Environment
source .venv/bin/activate

# --- CONFIGURATION ---
# PLEASE EDIT THESE VALUES
STREAM_URL="rtsp://admin:password@192.168.1.100:554/stream"
BUCKET_NAME="my-gcs-bucket-name"
LUGAR="home"
CAMARA="cam_front"
# ---------------------

echo "Starting UrbanEye..."
echo "URL: $STREAM_URL"
echo "Bucket: $BUCKET_NAME"

# Run the Python script
# Add --lat and --lon arguments if you want to hardcode location instead of auto-detect
python save_stream2_segments.py \
    --url "$STREAM_URL" \
    --bucket "$BUCKET_NAME" \
    --lugar "$LUGAR" \
    --camara "$CAMARA" \
    --minutes 1 \
    --out "recordings"

