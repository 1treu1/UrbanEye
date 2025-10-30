from __future__ import annotations

import argparse
import os
import cv2


def extract_frames(input_path: str, output_dir: str, max_frames: int = 1000) -> int:
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(0 if input_path == "0" else input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {input_path}")

    count = 0
    try:
        while count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            fname = os.path.join(output_dir, f"frame_{count+1:04d}.jpg")
            cv2.imwrite(fname, frame)
            count += 1
    finally:
        cap.release()
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae los primeros N frames de un video")
    parser.add_argument("--input", type=str, required=False, default="", help="Ruta del video o 0 para webcam")
    parser.add_argument("--outdir", type=str, required=False, default="", help="Directorio de salida para frames")
    parser.add_argument("--max", type=int, required=False, default=1000, help="Cantidad máxima de frames a extraer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_video = os.path.join(base_dir, "videos", "face-demographics-walking-and-pause.mp4")
    input_path = args.input or default_video
    output_dir = args.outdir or os.path.join(base_dir, "frames")

    num = extract_frames(input_path=input_path, output_dir=output_dir, max_frames=args.max)
    print(f"Frames guardados: {num} en {output_dir}")


if __name__ == "__main__":
    main()


