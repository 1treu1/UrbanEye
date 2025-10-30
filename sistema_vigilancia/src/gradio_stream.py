from __future__ import annotations

import os
from typing import Generator, List, Tuple, Optional

import cv2
import gradio as gr
import numpy as np
from deepface import DeepFace


_GPU_READY = False

def _ensure_tf_gpu() -> None:
    global _GPU_READY
    if _GPU_READY:
        return
    # Prefer graceful memory growth to avoid pre-allocating full VRAM
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    try:
        import tensorflow as tf  # type: ignore
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)  # type: ignore
            except Exception:
                pass
        # Optional XLA; keep conservative to avoid surprises
        try:
            tf.config.optimizer.set_jit(False)  # type: ignore
        except Exception:
            pass
    except Exception:
        # If TF not available, proceed silently (DeepFace may still use PyTorch)
        pass
    _GPU_READY = True


def analyze_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Run DeepFace on the whole frame and draw boxes + attributes.

    Detects faces and annotates dominant age, gender, race, emotion.
    """
    # GPU/TF setup (no-op on subsequent calls)
    _ensure_tf_gpu()
    # DeepFace expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Single-call analyze on full frame; DeepFace returns regions and attributes
    results = []
    detector_used = ""
    # Single fast backend for speed
    try:
        res = DeepFace.analyze(
            frame_rgb,
            actions=["age", "gender", "race", "emotion"],
            enforce_detection=False,
            align=True,
            detector_backend="retinaface",
        )
        if isinstance(res, dict):
            res = [res]
        results = res or []
        detector_used = "retinaface"
    except Exception:
        results = []

    if isinstance(results, dict):
        results = [results]

    annotated = frame_bgr.copy()
    num_faces = 0
    for res in results or []:
        region = res.get("region") or {}
        x = int(region.get("x", region.get("left", 0)))
        y = int(region.get("y", region.get("top", 0)))
        w = int(region.get("w", region.get("width", 0) or 0))
        h = int(region.get("h", region.get("height", 0) or 0))
        if w == 0 and "right" in region and "left" in region:
            w = int(region["right"]) - int(region.get("left", 0))
        if h == 0 and "bottom" in region and "top" in region:
            h = int(region["bottom"]) - int(region.get("top", 0))

        if w > 0 and h > 0:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            num_faces += 1

        age = res.get("age")
        gender = res.get("dominant_gender") or res.get("gender")
        race = res.get("dominant_race") or res.get("race")
        emotion = res.get("dominant_emotion") or res.get("emotion")

        label = f"age: {age}  gender: {gender}  race: {race}  emotion: {emotion}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y - 10) if h > 0 else 10
        cv2.rectangle(annotated, (x, max(0, y_text - th - 4)), (x + tw + 6, y_text + 2), (0, 0, 0), -1)
        cv2.putText(annotated, label, (x + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    status_text = f"faces: {num_faces}"
    try:
        status_text += f"  backend: {detector_used}"
    except Exception:
        pass
    cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    if num_faces == 0:
        cv2.putText(annotated, "No faces detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw centered red rectangle that covers 65% of the image
    h_img, w_img = annotated.shape[:2]
    rw = int(0.65 * w_img)
    rh = int(0.65 * h_img)
    x0 = (w_img - rw) // 2
    y0 = (h_img - rh) // 2
    x1 = x0 + rw
    y1 = y0 + rh
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2)
    return annotated


def stream_generator(video_path: str, analyze_every_n: int = 10, max_width: int = 640) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    try:
        # small buffer to reduce latency
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        frame_idx = 0
        last_annotated: Optional[np.ndarray] = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Resize for speed, preserve aspect ratio
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame

            # Analyze every N frames; reuse last annotation otherwise for smoothness
            if frame_idx % analyze_every_n == 0 or last_annotated is None:
                annotated_small = analyze_frame(frame_small)
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
        cap.release()


def build_demo(default_video: str) -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("## Sistema de Vigilancia - DeepFace Stream (video)")
        path_in = gr.Textbox(value=default_video, label="Ruta del video", interactive=True)
        output = gr.Image(label="Stream", streaming=True, type="numpy")
        start_btn = gr.Button("Iniciar")

        def start(vpath: str) -> Generator[np.ndarray, None, None]:
            # Stream with throttled DeepFace analysis for smoother playback
            yield from stream_generator(vpath, analyze_every_n=10, max_width=640)

        start_btn.click(fn=start, inputs=path_in, outputs=output)

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


