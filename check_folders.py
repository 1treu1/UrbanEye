from sistema_vigilancia.src.domain.utils import gcs_utils

def check_folders():
    folders = ["videos/", "viva_envigado/"]
    for folder in folders:
        print(f"--- Checking {folder} ---")
        try:
            videos = gcs_utils.list_videos_in_folder("bk-urbaneye-videos", folder)
            print(f"Found {len(videos)} videos:")
            for v in videos:
                print(f"  - {v}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_folders()
