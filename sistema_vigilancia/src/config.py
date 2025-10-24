from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


def build_default_polygon(frame_width: int = 1280, frame_height: int = 720) -> List[Tuple[int, int]]:
    w, h = frame_width, frame_height
    margin_w, margin_h = int(w * 0.25), int(h * 0.25)
    return [
        (margin_w, margin_h),
        (w - margin_w, margin_h),
        (w - margin_w, h - margin_h),
        (margin_w, h - margin_h),
    ]


@dataclass
class AppConfig:
    source: Optional[int | str] = 0
    visualize: bool = True
    detector_model: str = "yolov8n.pt"  # Modelo más ligero para CPU
    midas_model: str = "MiDaS_small"  # Modelo más ligero para CPU
    deepface_every_n_frames: int = 15
    polygon_points: List[Tuple[int, int]] = field(default_factory=build_default_polygon)
    csv_output_path: str = str(Path(__file__).resolve().parent.parent / "data" / "logs" / "events.csv")
    use_gpu: bool = True




