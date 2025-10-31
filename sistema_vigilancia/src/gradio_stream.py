"""Gradio interface for video stream analysis - Refactored modular version."""

from __future__ import annotations

import os
from typing import Generator, Optional
import cv2
import gradio as gr
import numpy as np

from .frame_analyzer import analyze_frame
from .roi_manager import set_roi_config, flush_window, write_track_to_csv
import sistema_vigilancia.src.config as config


def stream_generator(
    video_path: str,
    analyze_every_n: int = 10,
    max_width: int = 640,
    roi_size: float = 0.65
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
    
    with gr.Blocks() as demo:
        gr.Markdown("## Sistema de Vigilancia - DeepFace Stream (video)")
        path_in = gr.Textbox(value=default_video, label="Ruta del video", interactive=True)
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
        
        def start(vpath: str, roi_size: float) -> Generator[np.ndarray, None, None]:
            # Reset ROI config for new video
            set_roi_config(roi_mode=roi_mode, csv_path=csv_path, duration_min=duration_min, fps=fps)
            # Stream with throttled DeepFace analysis for smoother playback
            yield from stream_generator(vpath, analyze_every_n=10, max_width=640, roi_size=roi_size)
        
        start_btn.click(fn=start, inputs=[path_in, roi_size_slider], outputs=output)
    
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
