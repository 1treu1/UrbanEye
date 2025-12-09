# UrbanEye System Walkthrough

This document outlines the usage and architecture of the UrbanEye surveillance system.

## Quick Start

To start the system (recording + upload), run the startup wrapper:

```bash
./start_urbaneye.sh
```

## Overview

The system captures an RTSP video stream, segments it into video files (default 30 minutes), and uploads them to Google Cloud Storage (GCS). It also attempts to tag each segment with geolocation data derived from Wi-Fi scanning.

### Core Components

1.  **`start_urbaneye.sh`**: The entry point.
    -   Activates the Python virtual environment.
    -   Sets configuration variables (URL, Bucket, Location, Camera Name).
    -   Launches the main Python script.

2.  **`save_stream2_segments.py`**: The main logic.
    -   Connects to the RTSP stream using `ffmpeg`.
    -   Records segments (`segment_TIMESTAMP.mp4`).
    -   Auto-detects location via `wifi_geolocation.py` if not provided.
    -   Uploads segments to GCS bucket `bk-urbaneye-videos` in a background thread.
    -   Handles graceful shutdowns to prevent video corruption.

3.  **`wifi_geolocation.py`**: Helper module to determine device coordinates based on available Wi-Fi networks.

## Configuration

You can configure the system by editing variables in `start_urbaneye.sh`:

```bash
STREAM_URL="rtsp://user:pass@ip:port/stream"
BUCKET_NAME="bk-urbaneye-videos"
LUGAR="videos/Bogota"   # Folder path in bucket
CAMARA="camara_01"      # Subfolder for specific camera
DURATION_MINUTES=30     # Length of each video segment
```

## Streaming to Web (Optional)

If you need to expose the RTSP stream to the web (e.g., for testing), refer to `STREAM_RTSP_README.md`. It documents how to use `stream_rtsp_to_ngrok.py` to push the stream via HLS and ngrok.

## Troubleshooting

-   **Process Dies**: Check `ffmpeg` logs in the console. Ensure the RTSP URL is reachable.
-   **No Upload**: Verify `config/` contains a valid Google Cloud Service Account JSON key.
-   **Location Failed**: Ensure Wi-Fi is enabled on the device for `wifi_geolocation` to work.
