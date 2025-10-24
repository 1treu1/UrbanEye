from __future__ import annotations

import argparse
from typing import Optional

from .config import AppConfig, build_default_polygon
from .application.video_service import VideoProcessingService
from .application.event_logger import EventLogger

# Ports
from .ports.detection_port import DetectionPort
from .ports.tracking_port import TrackingPort
from .ports.depth_port import DepthEstimationPort

# Adapters (sin DeepFace)
from .adapters.yolo_adapter import YOLOv8Detector
from .adapters.deepsort_adapter import DeepSortTracker
from .adapters.midas_adapter import MiDaSDepthEstimator


class DummyAgeGenderPort:
    """Puerto dummy para reemplazar DeepFace"""
    def estimate(self, frame_bgr, person_bbox):
        from .ports.age_gender_port import AgeGender
        return AgeGender(age=None, gender=None)


def build_services(config: AppConfig) -> VideoProcessingService:
    detection: DetectionPort = YOLOv8Detector(
        model_name=config.detector_model, 
        use_gpu=config.use_gpu
    )
    tracking: TrackingPort = DeepSortTracker(max_age=30, n_init=2)
    age_gender = DummyAgeGenderPort()  # Sin DeepFace
    depth: DepthEstimationPort = MiDaSDepthEstimator(model_name=config.midas_model)

    logger = EventLogger(csv_path=config.csv_output_path)
    service = VideoProcessingService(
        detection=detection,
        tracking=tracking,
        age_gender=age_gender,
        depth=depth,
        event_logger=logger,
        polygon_points=config.polygon_points,
        visualize=config.visualize,
    )
    return service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sistema de Vigilancia - Sin DeepFace")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Fuente de video: ruta, 0 para webcam, o URL RTSP",
    )
    parser.add_argument(
        "--polygon",
        type=str,
        default="",
        help="Puntos del polígono 'x1,y1;x2,y2;...'; por defecto un rectángulo central",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Usar GPU para aceleración (por defecto: True)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Forzar uso de CPU (desactiva GPU)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "0":
        source: Optional[int | str] = 0
    else:
        source = args.source

    polygon = (
        [tuple(map(int, p.split(","))) for p in args.polygon.split(";")]
        if args.polygon
        else build_default_polygon()
    )

    # Determinar si usar GPU
    use_gpu = args.gpu and not args.no_gpu

    config = AppConfig(source=source, polygon_points=polygon, use_gpu=use_gpu)
    service = build_services(config)
    service.run(source=config.source)


if __name__ == "__main__":
    main()

