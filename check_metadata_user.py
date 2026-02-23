from google.cloud import storage
import os

def check_metadata():
    bucket_name = "bk-urbaneye-videos"
    blob_name = "videos/Popayan/2025-12-03/camara_01t/segment_20251203_172255.mp4"

    # Path to credentials
    cred_path = os.path.join(os.getcwd(), "config", "service.json")

    print(f"Using credentials: {cred_path}")
    print(f"Checking blob: {blob_name} in bucket: {bucket_name}")

    try:
        storage_client = storage.Client.from_service_account_json(cred_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.get_blob(blob_name)

        if blob:
            print("\n--- Metadata Found ---")
            if blob.metadata:
                for key, value in blob.metadata.items():
                    print(f"{key}: {value}")
            else:
                print("No custom metadata found on this blob.")

            print("\n--- Standard Properties ---")
            print(f"Created: {blob.time_created}")
            print(f"Updated: {blob.updated}")
            print(f"Size: {blob.size} bytes")
        else:
            print(f"Error: Blob {blob_name} not found in bucket {bucket_name}.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_metadata()
