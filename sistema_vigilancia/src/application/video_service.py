from __future__ import annotations

from typing import List, Tuple, Optional

import cv2

from ..domain.services import SurveillanceOrchestrator
from ..application.event_logger import EventLogger
from ..ports.detection_port import DetectionPort
from ..ports.tracking_port import TrackingPort
from ..ports.age_gender_port import AgeGenderPort
from ..ports.depth_port import DepthEstimationPort


class VideoProcessingService:
    def __init__(
        self,
        detection: DetectionPort,
        tracking: Optional[TrackingPort],
        age_gender: AgeGenderPort,
        depth: Optional[DepthEstimationPort],
        event_logger: EventLogger,
        polygon_points: List[Tuple[int, int]],
        visualize: bool = True,
    ) -> None:
        self.orchestrator = SurveillanceOrchestrator(
            detection=detection,
            tracking=tracking,
            age_gender=age_gender,
            depth=depth,
            polygon_points=polygon_points,
        )
        self.logger = event_logger
        self.visualize = visualize

    def run(self, source: Optional[int | str] = 0) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente: {source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                vis, events = self.orchestrator.process_frame(frame)
                self.logger.append(events)
                if self.visualize:
                    cv2.imshow("Vigilancia", vis)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()




