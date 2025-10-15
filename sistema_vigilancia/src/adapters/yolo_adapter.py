from __future__ import annotations

from typing import List

import numpy as np
from ultralytics import YOLO

from ..ports.detection_port import Detection, DetectionPort


class YOLOv8Detector(DetectionPort):
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25) -> None:
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame_bgr) -> List[Detection]:
        res = self.model.predict(frame_bgr[..., ::-1], conf=self.conf, verbose=False)
        detections: List[Detection] = []
        for r in res:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0].item())
                cls_id = int(b.cls[0].item())
                cls_name = r.names.get(cls_id, str(cls_id))
                detections.append(Detection((x1, y1, x2, y2), conf, cls_id, cls_name))
        return detections



