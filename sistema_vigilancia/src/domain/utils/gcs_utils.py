from __future__ import annotations

import os
from typing import List
from google.cloud import storage

from google.oauth2 import service_account

def get_client() -> storage.Client:
    """Get Google Cloud Storage client using service account credentials."""
    # Path relative to the workspace root or absolute path
    creds_path = os.path.join(os.getcwd(), "config", "service.json")
    if os.path.exists(creds_path):
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        return storage.Client(credentials=credentials)
    else:
        # Fallback to default credentials if file not found
        print(f"Warning: {creds_path} not found, using default credentials.")
        return storage.Client()

def list_folders(bucket_name: str, prefix: str = "") -> List[str]:
    """List 'folders' (common prefixes) in a bucket."""
    client = get_client()
    bucket = client.bucket(bucket_name)
    
    # Use delimiter to emulate directory listing
    blobs = bucket.list_blobs(prefix=prefix, delimiter="/")
    
    # Force iteration to populate prefixes
    _ = list(blobs)
    
    return list(blobs.prefixes)

def list_all_folders_recursive(bucket_name: str, prefix: str = "") -> List[str]:
    """List all unique folders containing files recursively."""
    client = get_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    folders = set()
    for blob in blobs:
        # Get directory name
        dirname = os.path.dirname(blob.name)
        if dirname:
            # Normalize to ensure trailing slash for consistency
            folder_path = dirname.replace("\\", "/") + "/"
            folders.add(folder_path)
            
    return sorted(list(folders))

def list_videos_in_folder(bucket_name: str, folder_name: str) -> List[str]:
    """List video files in a specific folder."""
    client = get_client()
    bucket = client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=folder_name)
    
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    videos = []
    for blob in blobs:
        if any(blob.name.lower().endswith(ext) for ext in video_extensions):
            videos.append(blob.name)
            
    return videos

def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: str) -> None:
    """Download a blob to a local file."""
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
