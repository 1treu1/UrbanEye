#!/bin/bash

# start_urbaneye.sh
# Wrapper script to launch UrbanEye on startup
# Usage: ./start_urbaneye.sh

# Navigate to script directory
cd "$(dirname "$0")"

# Activate Virtual Environment
source .venv/bin/activate

# --- CONFIGURATION ---
STREAM_URL="rtsp://1treu1:treutapo@192.168.0.102:554/stream2"
BUCKET_NAME="bk-urbaneye-videos"
LUGAR="videos/Bogota"
CAMARA="camara_01"
OUTPUT_DIR="sistema_vigilancia/videos"
DURATION_MINUTES=30
# ---------------------

echo "Starting UrbanEye..."
echo "URL: $STREAM_URL"
echo "Bucket: $BUCKET_NAME"
echo "Camara: $CAMARA"

# Run the Python script
python -u save_stream2_segments.py \
    --url "$STREAM_URL" \
    --bucket "$BUCKET_NAME" \
    --lugar "$LUGAR" \
    --camara "$CAMARA" \
    --minutes "$DURATION_MINUTES" \
    --out "$OUTPUT_DIR"
