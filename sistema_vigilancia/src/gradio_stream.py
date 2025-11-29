"""Gradio interface for video stream analysis - Refactored modular version."""

from __future__ import annotations

import os
import time
from typing import Generator, Optional, Tuple
import cv2
import gradio as gr
import numpy as np

from .frame_analyzer import analyze_frame
from .roi_manager import set_roi_config, flush_window, write_track_to_csv
import sistema_vigilancia.src.config as config
from .domain.utils import gcs_utils



def get_video_fps(video_path: str) -> float:
    """Detect FPS from video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 22.0  # Default fallback
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 22.0
    except Exception:
        return 22.0

def stream_generator(
    video_path: str,
    analyze_every_n: int = 10,
    max_width: int = 640,
    roi_size: float = 0.65,
    yield_every_n: int = 3
) -> Generator[np.ndarray, None, None]:
    """Generate annotated video frames for Gradio streaming.
    
    Args:
        video_path: Path to video file
        analyze_every_n: Analyze every N frames (reuse last annotation otherwise)
        max_width: Maximum frame width for processing
        roi_size: ROI size as fraction (0.0-1.0)
    
    Yields:
        Annotated frames in RGB format
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    
    try:
        # Small buffer to reduce latency
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        
        frame_idx = 0
        last_annotated: Optional[np.ndarray] = None
        last_yield_time = 0.0
        
        while True:
            ok, frame = cap.read()
            if not ok:
                # Video ended - flush any remaining tracks in ROI
                if config.ROI_MODE != "none":
                    flush_window()
                break
            
            # Resize for speed, preserve aspect ratio
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame
            
            # Increment global frame counter for each frame (for accurate time tracking)
            config.GLOBAL_FRAME_IDX += 1
            
            # Analyze every N frames; reuse last annotation otherwise for smoothness
            if frame_idx % analyze_every_n == 0 or last_annotated is None:
                annotated_small = analyze_frame(frame_small, roi_size=roi_size)
                # If we downscaled, upscale annotations back to original size for display stability
                if annotated_small.shape[1] != w:
                    annotated = cv2.resize(annotated_small, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    annotated = annotated_small
                last_annotated = annotated
            else:
                annotated = last_annotated
            
            frame_idx += 1
            
            # Heartbeat log
            if frame_idx % 30 == 0:
                print(f"Processing frame {frame_idx}...")
            
            # Yield first frame and then every N frames
            if frame_idx == 1 or frame_idx % yield_every_n == 0:
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                yield annotated_rgb
    
    finally:
        # Final flush of any remaining tracks in ROI when stream ends
        if config.ROI_MODE != "none":
            flush_window()
            # Additional check: flush any tracks with enter_time but no exit_time
            for tid, track in list(config.TRACKS.items()):
                if "enter_time" in track and "exit_time" not in track:
                    track["exit_time"] = config.GLOBAL_FRAME_IDX / config.FPS_ASSUMED / 60.0
                    track["inside"] = False
                    write_track_to_csv(tid, track)
        cap.release()


def build_demo(
    default_video: str,
    roi_mode: str = "consolidated",
    csv_path: str = "",
    duration_min: int = 30,
    fps: int = 22
) -> gr.Blocks:
    """Build Gradio demo interface.
    
    Args:
        default_video: Default video path
        roi_mode: ROI tracking mode
        csv_path: Path to CSV output file
        duration_min: Window duration in minutes
        fps: Assumed frames per second
    
    Returns:
        Gradio Blocks interface
    """
    # Set ROI configuration - auto-configure CSV path if not provided
    if not csv_path:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "roi_stats.csv")
    set_roi_config(roi_mode=roi_mode, csv_path=csv_path, duration_min=duration_min, fps=fps)
    
    
    # Get GCS folders (if credentials available)
    gcs_folders = []
    try:
        gcs_folders = gcs_utils.list_all_folders_recursive("bk-urbaneye-videos")
    except Exception as e:
        print(f"Warning: Could not list GCS folders: {e}")

    with gr.Blocks() as demo:
        gr.Markdown("## Sistema de Vigilancia - DeepFace Stream (video)")
        
        with gr.Row():
            source_mode = gr.Dropdown(
                choices=["Local File", "GCS Folder"],
                value="Local File",
                label="Modo de Fuente",
                interactive=True
            )
        
        with gr.Group(visible=True) as local_group:
            path_in = gr.Textbox(value=default_video, label="Ruta del video local", interactive=True)
            
        with gr.Group(visible=False) as gcs_group:
            folder_dd = gr.Dropdown(choices=gcs_folders, label="Carpeta GCS", interactive=True)
            refresh_btn = gr.Button("Refrescar Carpetas")
            progress_output = gr.Textbox(label="Progreso del Lote", interactive=False)

        with gr.Row():
            lat_input = gr.Textbox(value=config.LATITUD, label="Latitud", interactive=True)
            lon_input = gr.Textbox(value=config.LONGITUD, label="Longitud", interactive=True)
            lugar_input = gr.Textbox(value=config.LUGAR, label="Lugar", interactive=True)

        roi_size_slider = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.65,
            step=0.05,
            label="Tamaño del rectángulo ROI (porcentaje)",
            info="0.7 = 70% del tamaño de la imagen"
        )
        output = gr.Image(label="Stream", streaming=True, type="numpy")
        start_btn = gr.Button("Iniciar")
        
        def toggle_inputs(mode):
            return {
                local_group: gr.Group(visible=(mode == "Local File")),
                gcs_group: gr.Group(visible=(mode == "GCS Folder"))
            }
            
        source_mode.change(fn=toggle_inputs, inputs=source_mode, outputs=[local_group, gcs_group])
        
        def refresh_gcs():
            try:
                folders = gcs_utils.list_all_folders_recursive("bk-urbaneye-videos")
                return gr.Dropdown(choices=folders)
            except Exception as e:
                return gr.Dropdown(choices=[])
                
        refresh_btn.click(fn=refresh_gcs, inputs=[], outputs=folder_dd)
        
        def start(mode: str, vpath: str, folder: str, roi_size: float, lat: str, lon: str, place: str) -> Generator[Tuple[np.ndarray, str], None, None]:
            
            if mode == "Local File":
                # Detect FPS for local video
                real_fps = get_video_fps(vpath)
                print(f"Detected FPS for {vpath}: {real_fps}")
                
                # Reset ROI config
                set_roi_config(
                    roi_mode=roi_mode, 
                    csv_path=csv_path, 
                    duration_min=duration_min, 
                    fps=real_fps,
                    latitud=lat,
                    longitud=lon,
                    lugar=place
                )
                
                # Generator wrapper to yield (image, progress)
                gen = stream_generator(vpath, analyze_every_n=10, max_width=640, roi_size=roi_size, yield_every_n=3)
                for frame in gen:
                    yield frame, "Procesando video local..."
            else:
                # GCS Batch Mode
                if not folder:
                    yield np.zeros((100, 100, 3), dtype=np.uint8), "Error: Seleccione una carpeta"
                    return
                try:
                    videos = gcs_utils.list_videos_in_folder("bk-urbaneye-videos", folder)
                    total_videos = len(videos)
                    
                    for i, video_blob in enumerate(videos):
                        video_name = os.path.basename(video_blob)
                        progress_msg = f"Procesando video {i+1} de {total_videos}: {video_name}"
                        
                        # Yield loading state
                        yield np.zeros((100, 100, 3), dtype=np.uint8), f"Descargando {video_name}..."
                        
                        # Download to temp file
                        temp_path = os.path.join(os.getcwd(), "temp_video.mp4")
                        gcs_utils.download_blob("bk-urbaneye-videos", video_blob, temp_path)
                        
                        try:
                            # Detect FPS for downloaded video
                            real_fps = get_video_fps(temp_path)
                            print(f"Detected FPS for {video_blob}: {real_fps}")
                            
                            # Reset/Update ROI config for this video
                            set_roi_config(
                                roi_mode=roi_mode, 
                                csv_path=csv_path, 
                                duration_min=duration_min, 
                                fps=real_fps,
                                latitud=lat,
                                longitud=lon,
                                lugar=place
                            )
                            
                            gen = stream_generator(temp_path, analyze_every_n=10, max_width=640, roi_size=roi_size, yield_every_n=3)
                            for frame in gen:
                                yield frame, progress_msg
                                
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                except Exception as e:
                    print(f"Error processing GCS folder: {e}")
                    yield np.zeros((100, 100, 3), dtype=np.uint8), f"Error: {e}"
        
        start_btn.click(
            fn=start, 
            inputs=[source_mode, path_in, folder_dd, roi_size_slider, lat_input, lon_input, lugar_input], 
            outputs=[output, progress_output]
        )
    
    # Warm-up: run a quick analyze on first frame to load models
    try:
        cap = cv2.VideoCapture(default_video)
        ok, frame = cap.read()
        if ok:
            _ = analyze_frame(frame)
        cap.release()
    except Exception:
        pass
    
    return demo


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(base_dir, "videos", "face-demographics-walking-and-pause.mp4")
    port = int(os.getenv("PORT", "7860"))
    demo = build_demo(default_path)
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=True)
