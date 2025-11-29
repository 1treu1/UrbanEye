
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "sistema_vigilancia", "src"))

try:
    from sistema_vigilancia.src import gradio_stream
    from sistema_vigilancia.src import roi_manager
    from sistema_vigilancia.src import frame_analyzer
    from sistema_vigilancia.src import visualization
    from sistema_vigilancia.src import deepface_analyzer
    
    print("Imports successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
