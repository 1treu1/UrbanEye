"""Gradio interface for video stream analysis - Refactored modular version."""

from __future__ import annotations

import os
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Generator, Optional, Tuple, List
import cv2
import gradio as gr
import numpy as np

from .frame_analyzer import analyze_frame
from .roi_manager import flush_window, write_track_to_csv, VideoState
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


def process_video_headless(
    video_blob: str,
    roi_mode: str,
    csv_path: str,
    duration_min: int,
    roi_size: float,
    lat: str,
    lon: str,
    place: str
) -> str:
    """Process a video in background without UI updates."""
    temp_path = ""
    try:
        # Create unique temp file
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        
        print(f"Background processing: Downloading {video_blob}...")
        gcs_utils.download_blob("bk-urbaneye-videos", video_blob, temp_path)
        
        real_fps = get_video_fps(temp_path)
        print(f"Background processing: {video_blob} (FPS: {real_fps})")
        
        state = VideoState(roi_mode, csv_path, duration_min, real_fps, lat, lon, place)
        
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return f"Error opening {video_blob}"
            
        frame_idx = 0
        analyze_every_n = 10
        max_width = 640
        
        while True:
            ok, frame = cap.read()
            if not ok:
                if state.roi_mode != "none":
                    flush_window(state)
                break
            
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame
            
            state.global_frame_idx += 1
            
            if frame_idx % analyze_every_n == 0:
                analyze_frame(frame_small, state, roi_size=roi_size)
            
            frame_idx += 1
            
            # Heartbeat for logs
            if frame_idx % 500 == 0:
                print(f"Background {video_blob}: Frame {frame_idx}")
                
        cap.release()
        return f"Completed {video_blob}"
        
    except Exception as e:
        print(f"Error in background processing {video_blob}: {e}")
        return f"Error {video_blob}: {e}"
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def stream_generator(
    video_path: str,
    roi_mode: str,
    csv_path: str,
    duration_min: int,
    lat: str,
    lon: str,
    place: str,
    analyze_every_n: int = 10,
    max_width: int = 640,
    roi_size: float = 0.65,
    yield_every_n: int = 3,
    visualize: bool = True
) -> Generator[np.ndarray, None, None]:
    """Generate annotated video frames for Gradio streaming."""
    
    # Initialize state for this video
    real_fps = get_video_fps(video_path)
    state = VideoState(roi_mode, csv_path, duration_min, real_fps, lat, lon, place)
    print(f"Streaming video: {video_path} (FPS: {real_fps})")
    
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
        
        while True:
            ok, frame = cap.read()
            if not ok:
                if state.roi_mode != "none":
                    flush_window(state)
                break
            
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame
            
            state.global_frame_idx += 1
            
            # Analyze every N frames; reuse last annotation otherwise for smoothness
            if frame_idx % analyze_every_n == 0 or last_annotated is None:
                annotated_small = analyze_frame(frame_small, state, roi_size=roi_size, visualize=visualize)
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
                print(f"Streaming frame {frame_idx}...")
            
            # Yield first frame and then every N frames
            if frame_idx == 1 or frame_idx % yield_every_n == 0:
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                yield annotated_rgb
    
    finally:
        cap.release()


def build_demo(
    default_video: str,
    roi_mode: str = "consolidated",
    csv_path: str = "",
    duration_min: int = 30,
    fps: int = 22
) -> gr.Blocks:
    """Build Gradio demo interface."""
    # Set ROI configuration - auto-configure CSV path if not provided
    if not csv_path:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "roi_stats.csv")
    
    # Get GCS folders (if credentials available)
    gcs_folders = []
    try:
        gcs_folders = gcs_utils.list_all_folders_recursive("bk-urbaneye-videos")
    except Exception as e:
        print(f"Warning: Could not list GCS folders: {e}")

    with gr.Blocks() as demo:
        gr.Markdown("## Sistema de Vigilancia - DeepFace Stream (Procesamiento Paralelo)")
        
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

        with gr.Row():
            show_visuals = gr.Checkbox(value=True, label="Mostrar Visualizaciones (Bounding Boxes/ROI)", interactive=True)
            num_workers = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Workers Paralelos (GCS)", interactive=True)

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
        
        def start(mode: str, vpath: str, folder: str, roi_size: float, lat: str, lon: str, place: str, visualize: bool, max_workers: int) -> Generator[Tuple[np.ndarray, str], None, None]:
            
            if mode == "Local File":
                # Generator wrapper to yield (image, progress)
                gen = stream_generator(
                    vpath, roi_mode, csv_path, duration_min, lat, lon, place,
                    analyze_every_n=10, max_width=640, roi_size=roi_size, yield_every_n=3,
                    visualize=visualize
                )
                for frame in gen:
                    yield frame, "Procesando video local..."
            else:
                # GCS Batch Mode with Parallel Processing
                if not folder:
                    yield np.zeros((100, 100, 3), dtype=np.uint8), "Error: Seleccione una carpeta"
                    return
                try:
                    videos = gcs_utils.list_videos_in_folder("bk-urbaneye-videos", folder)
                    total_videos = len(videos)
                    
                    if total_videos == 0:
                        yield np.zeros((100, 100, 3), dtype=np.uint8), "No se encontraron videos en la carpeta"
                        return

                    # We will process N videos in parallel: 1 in main thread (streaming), N-1 in background
                    # Ensure at least 1 worker
                    max_workers = int(max(1, max_workers))
                    
                    executor = ThreadPoolExecutor(max_workers=max_workers)
                    futures = []
                    
                    # Queue of videos to process
                    video_queue = list(videos)
                    completed_count = 0
                    
                    while video_queue or futures:
                        # Fill background workers
                        # We keep 1 slot "virtual" for the main thread streaming, so we submit max_workers - 1 to background
                        # But wait, ThreadPoolExecutor manages threads. If we submit max_workers tasks, they will all run.
                        # The main thread is separate. So if we want TOTAL parallelism of max_workers + 1 (main), we can submit max_workers.
                        # If user sets "Workers" to 4, they probably expect 4 total.
                        # Let's say max_workers is the BACKGROUND workers count.
                        # Or better: max_workers is total concurrent videos.
                        # If max_workers = 1, then only main thread runs (sequential).
                        # If max_workers = 2, 1 main + 1 background.
                        
                        bg_workers = max(0, max_workers - 1)
                        
                        while len(futures) < bg_workers and len(video_queue) > 1:
                            # Pop from end or start? Let's pop from start, but reserve index 0 for UI if not running
                            # Actually, let's just pop the next available video for background
                            # Strategy: Always keep 1 video for the UI thread to pick up
                            
                            # If we have at least 2 videos, send one to background
                            bg_video = video_queue.pop(0)
                            print(f"Submitting background task: {bg_video}")
                            f = executor.submit(
                                process_video_headless, 
                                bg_video, roi_mode, csv_path, duration_min, roi_size, lat, lon, place
                            )
                            futures.append(f)
                            
                            # If we only have 1 left, break so UI can take it
                            if len(video_queue) == 0:
                                break
                                
                        # Clean up finished futures
                        done_futures = [f for f in futures if f.done()]
                        for f in done_futures:
                            futures.remove(f)
                            completed_count += 1
                            try:
                                res = f.result()
                                print(f"Background task result: {res}")
                            except Exception as e:
                                print(f"Background task failed: {e}")
                        
                        # If we have a video for the UI, process it
                        if video_queue:
                            ui_video_blob = video_queue.pop(0)
                            video_name = os.path.basename(ui_video_blob)
                            
                            yield np.zeros((100, 100, 3), dtype=np.uint8), f"Descargando {video_name} (Lote: {completed_count}/{total_videos})..."
                            
                            # Download to temp file
                            fd, temp_path = tempfile.mkstemp(suffix=".mp4")
                            os.close(fd)
                            gcs_utils.download_blob("bk-urbaneye-videos", ui_video_blob, temp_path)
                            
                            try:
                                gen = stream_generator(
                                    temp_path, roi_mode, csv_path, duration_min, lat, lon, place,
                                    analyze_every_n=10, max_width=640, roi_size=roi_size, yield_every_n=3,
                                    visualize=visualize
                                )
                                for frame in gen:
                                    # Check background futures occasionally?
                                    # We can't easily check futures inside this loop without blocking or complex logic
                                    # Just yield frames
                                    yield frame, f"Procesando {video_name} (Lote: {completed_count}/{total_videos} completados + {len(futures)} en segundo plano)"
                                
                                completed_count += 1
                            finally:
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                        else:
                            # No videos for UI, but background tasks might be running
                            if futures:
                                yield np.zeros((100, 100, 3), dtype=np.uint8), f"Esperando tareas de fondo ({len(futures)} activas)..."
                                time.sleep(1)
                            else:
                                break
                                
                    executor.shutdown(wait=True)
                    yield np.zeros((100, 100, 3), dtype=np.uint8), f"Procesamiento completado: {total_videos} videos."

                except Exception as e:
                    print(f"Error processing GCS folder: {e}")
                    yield np.zeros((100, 100, 3), dtype=np.uint8), f"Error: {e}"
        
        start_btn.click(
            fn=start, 
            inputs=[source_mode, path_in, folder_dd, roi_size_slider, lat_input, lon_input, lugar_input, show_visuals, num_workers], 
            outputs=[output, progress_output]
        )
    
    # Warm-up: run a quick analyze on first frame to load models
    print("Inicializando modelos de IA (DeepFace + YOLO)... Por favor espere.")
    try:
        # Create dummy state for warm-up
        dummy_state = VideoState("none", "", 30, 22.0)
        cap = cv2.VideoCapture(default_video)
        ok, frame = cap.read()
        if ok:
            _ = analyze_frame(frame, dummy_state)
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
