import os
import glob
import sys
from google.cloud import storage

def load_credentials():
    """Finds and returns the path to the JSON credentials file."""
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    cred_files = glob.glob(os.path.join(config_dir, "*.json"))
    if not cred_files:
        print("Error: No JSON credentials found in config/ directory.")
        sys.exit(1)
    return cred_files[0]

def validate_metadata(full_path):
    """
    Validates metadata for a GCS object.
    path format expected: bucket_name/path/to/object
    """
    # Parse the full path into bucket and object name
    parts = full_path.split('/', 1)
    if len(parts) != 2:
        print(f"Error: Invalid path format '{full_path}'. Expected 'bucket_name/path/to/object'")
        return

    bucket_name = parts[0]
    blob_name = parts[1]

    print(f"Checking Bucket: {bucket_name}")
    print(f"Checking Blob: {blob_name}")

    try:
        credentials_path = load_credentials()
        storage_client = storage.Client.from_service_account_json(credentials_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.get_blob(blob_name)

        if not blob:
            print("Error: Blob not found.")
            return

        print("\n--- Metadata Found ---")
        if blob.metadata:
            for key, value in blob.metadata.items():
                print(f"{key}: {value}")
            
            lat = blob.metadata.get('latitude')
            lon = blob.metadata.get('longitude')
            
            print("\n--- Validation Result ---")
            if lat and lon:
                print(f"SUCCESS: Coordinates found. Lat: {lat}, Lon: {lon}")
            else:
                print("FAILURE: Coordinates NOT found in metadata.")
                if not lat: print("- Missing 'latitude'")
                if not lon: print("- Missing 'longitude'")
        else:
            print("No metadata found for this blob.")
            print("\n--- Validation Result ---")
            print("FAILURE: No metadata exists.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default for testing based on user request if no arg provided
        default_path = "bk-urbaneye-videos/videos/Pereira/2025-12-08/camara_2/segment_20251208_134256.mp4"
        print(f"No path provided, using default: {default_path}")
        validate_metadata(default_path)
    else:
        validate_metadata(sys.argv[1])
