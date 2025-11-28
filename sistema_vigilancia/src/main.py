from __future__ import annotations

import argparse
import os
from typing import Optional

import cv2
from .gradio_stream import build_demo, analyze_frame
from .http_stream import build_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sistema de Vigilancia - Hexagonal")
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Fuente de video: ruta, 0 para webcam, o URL RTSP. Por defecto usa el video local",
    )
    parser.add_argument(
        "--ui",
        type=str,
        choices=["gradio", "cv2", "http"],
        default="gradio",
        help="Interfaz de visualización: gradio (web), cv2 (ventana) o http (MJPEG)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "7860")),
        help="Puerto HTTP para Gradio (solo si --ui=gradio)",
    )
    parser.add_argument(
        "--roi-size",
        type=float,
        default=0.65,
        help="Tamaño del rectángulo ROI como porcentaje (0.0 a 1.0, default 0.65 para 65%)",
    )
    parser.add_argument(
        "--latitud",
        type=str,
        default="",
        help="Latitud de la ubicación de la cámara",
    )
    parser.add_argument(
        "--longitud",
        type=str,
        default="",
        help="Longitud de la ubicación de la cámara",
    )
    parser.add_argument(
        "--lugar",
        type=str,
        default="",
        help="Nombre del lugar/ubicación",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Set global config from args
    from . import config
    config.LATITUD = args.latitud
    config.LONGITUD = args.longitud
    config.LUGAR = args.lugar

    # Default to bundled demo video if not provided and UI is gradio
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_video = os.path.join(base_dir, "videos", "face-demographics-walking-and-pause.mp4")

    if args.ui == "gradio":
        vsource = args.source if args.source else default_video
        demo = build_demo(vsource)
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=True)
        return

    if args.ui == "http":
        # Simple servidor HTTP con MJPEG streaming para entornos headless
        import uvicorn
        vsource = args.source if args.source else default_video
        app = build_app(vsource)
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return

    # cv2 window pipeline using only DeepFace
    if args.source == "0":
        source: Optional[int | str] = 0
    else:
        source = args.source if args.source else default_video

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {source}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated = analyze_frame(frame, roi_size=args.roi_size)
            cv2.imshow("DeepFace - age/gender/race/emotion", annotated)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




