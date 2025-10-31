from __future__ import annotations

import os
from typing import Generator, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from .gradio_stream import analyze_frame

# Ensure TF GPU opts are set for this process as well (idempotent in analyze_frame)
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def mjpeg_generator(video_path: str, analyze_every_n: int = 1, max_width: int = 720, loop: bool = False, roi_size: float = 0.65) -> Generator[bytes, None, None]:
    cap = cv2.VideoCapture(0 if video_path == "0" else video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {video_path}")
    try:
        frame_idx = 0
        last_annotated: Optional[np.ndarray] = None
        while True:
            ok, frame = cap.read()
            if not ok:
                # Si es archivo y loop está activo, reiniciar al inicio
                if loop and video_path not in ("0", 0):
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    except Exception:
                        break
                break

            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame

            if frame_idx % analyze_every_n == 0 or last_annotated is None:
                annotated_small = analyze_frame(frame_small, roi_size=roi_size)
                if annotated_small.shape[1] != w:
                    annotated = cv2.resize(annotated_small, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    annotated = annotated_small
                last_annotated = annotated
            else:
                annotated = last_annotated

            frame_idx += 1

            ok, enc = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            jpg = enc.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
    finally:
        cap.release()


def build_app(default_video: str) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (
            """
            <html>
              <head>
                <title>DeepFace Stream</title>
                <style>body{font-family:sans-serif;margin:16px} input,button{margin:4px}</style>
              </head>
              <body>
                <h2>DeepFace - age / gender / race / emotion</h2>
                <div>
                  <label>Fuente (ruta o 0 para webcam):</label>
                  <input id="source" type="text" style="width:360px" placeholder="/ruta/video.mp4" />
                  <label>Loop</label>
                  <input id="loop" type="checkbox" />
                  <button onclick="startStream()">Iniciar</button>
                  <button onclick="restartStream()">Reiniciar</button>
                </div>
                <div style="margin-top:8px">
                  <img id="stream" src="video_feed" style="max-width: 100%; border:1px solid #ccc"/>
                </div>
                <script>
                  function qs(){
                    const s = document.getElementById('source').value;
                    const loop = document.getElementById('loop').checked ? '1' : '0';
                    const t = Date.now();
                    const p = new URLSearchParams();
                    if(s) p.set('source', s);
                    p.set('loop', loop);
                    p.set('t', t);
                    return p.toString();
                  }
                  function startStream(){
                    document.getElementById('stream').src = 'video_feed?' + qs();
                  }
                  function restartStream(){
                    const img = document.getElementById('stream');
                    const url = new URL(img.src, window.location.href);
                    url.searchParams.set('t', Date.now());
                    img.src = url.toString();
                  }
                </script>
              </body>
            </html>
            """
        )

    @app.get("/video_feed")
    def video_feed(source: str | None = None, loop: int = 0, an: int = 1, mw: int = 720, roi_size: float = 0.65):
        vsrc = source if source is not None and source != "" else default_video
        return StreamingResponse(
            mjpeg_generator(vsrc, analyze_every_n=an, max_width=mw, loop=bool(loop), roi_size=roi_size),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


