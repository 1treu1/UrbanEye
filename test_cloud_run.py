import unittest
from unittest.mock import MagicMock, patch
import json
import os
import sys

# Mock cv2 and other dependencies before importing cloud_run_main
sys.modules['cv2'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['deepface'] = MagicMock()
sys.modules['gradio'] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sistema_vigilancia.src import cloud_run_main

class TestCloudRun(unittest.TestCase):
    def setUp(self):
        self.app = cloud_run_main.app.test_client()
        self.app.testing = True

    @patch('sistema_vigilancia.src.cloud_run_main.get_storage_client')
    @patch('sistema_vigilancia.src.cloud_run_main.process_video')
    @patch('os.remove')
    def test_handle_event(self, mock_remove, mock_process_video, mock_get_storage_client):
        # Mock GCS
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        mock_get_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Mock process_video success
        mock_process_video.return_value = True
        
        # Mock file existence for CSV upload check
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            with patch('os.path.getsize') as mock_getsize:
                mock_getsize.return_value = 100 # Fake size
                
                # Payload based on user example
                payload = {
                    "protoPayload": {
                        "resourceName": "projects/_/buckets/bk-urbaneye-videos/objects/videos/test_video.mp4"
                    },
                    "resource": {
                        "labels": {
                            "bucket_name": "bk-urbaneye-videos"
                        }
                    }
                }
                
                response = self.app.post('/', json=payload)
                
                self.assertEqual(response.status_code, 200)
                
                # Verify GCS interactions
                mock_client.bucket.assert_any_call('bk-urbaneye-videos')
                mock_blob.download_to_filename.assert_called()
                
                # Verify processing called
                mock_process_video.assert_called()
                
                # Verify upload
                # Should upload to data/test_video.mp4.csv
                # Note: The code logic splits by objects/ so filename is videos/test_video.mp4
                # Then basename is test_video.mp4
                # So target is data/test_video.mp4.csv
                
                # We need to check if upload_from_filename was called on the target blob
                # The target blob is created from target_bucket.blob()
                # We can check the calls to bucket and blob
                
                # Check that we accessed the bucket again for upload
                # (It might be the same mock object depending on how we set it up, 
                # but here we return the same mock_bucket for any bucket() call? 
                # No, mock_client.bucket is a method, we can check calls)
                
                # Verify upload call
                # We can't easily check the exact blob name without more complex mocking, 
                # but we can check if upload_from_filename was called
                self.assertTrue(mock_blob.upload_from_filename.called or mock_bucket.blob().upload_from_filename.called)

if __name__ == '__main__':
    unittest.main()
