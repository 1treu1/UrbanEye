from sistema_vigilancia.src.domain.utils import gcs_utils
import sys

def test_gcs():
    print("Testing GCS connection...")
    try:
        folders = gcs_utils.list_folders("bk-urbaneye-videos")
        print(f"Folders found: {folders}")
        
        if folders:
            first_folder = folders[0]
            print(f"Listing videos in {first_folder}...")
            videos = gcs_utils.list_videos_in_folder("bk-urbaneye-videos", first_folder)
            print(f"Videos found: {videos}")
        else:
            print("No folders found (or empty bucket).")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_gcs()
