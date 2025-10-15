from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..ports.detection_port import DetectionPort
from ..ports.tracking_port import TrackingPort
from ..ports.age_gender_port import AgeGenderPort
from ..ports.depth_port import DepthEstimationPort
from .models import PersonState
from .utils.area_utils import is_point_in_polygon, bbox_center
from .utils.draw_utils import draw_annotations, draw_polygon


class SurveillanceOrchestrator:
    """Orquesta detección, tracking, edad/género, profundidad y eventos."""

    def __init__(
        self,
        detection: DetectionPort,
        tracking: TrackingPort,
        age_gender: AgeGenderPort,
        depth: DepthEstimationPort,
        polygon_points: List[Tuple[int, int]],
    ) -> None:
        self.detection = detection
        self.tracking = tracking
        self.age_gender = age_gender
        self.depth = depth
        self.polygon_points = polygon_points
        self.track_states: Dict[int, PersonState] = {}

    def process_frame(self, frame_bgr):
        detections = self.detection.detect(frame_bgr)
        det_for_tracker = [(*d.bbox, float(d.confidence)) for d in detections if d.class_name == "person"]
        tracks = self.tracking.update(frame_bgr, det_for_tracker)

        depth_map = self.depth.estimate_depth(frame_bgr)

        events: List[PersonState] = []
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            track_id = t.track_id
            state = self.track_states.get(track_id) or PersonState(track_id=track_id, bbox=t.bbox)
            state.bbox = t.bbox

            # Edad/género (implementación decide la frecuencia)
            ag = self.age_gender.estimate(frame_bgr, t.bbox)
            state.age = ag.age
            state.gender = ag.gender

            # Heurística simple: si la caja superior contiene un bulto, marcar carries_object (placeholder, detector real podría reemplazar)
            state.carries_object = self._heuristic_carries_object(frame_bgr, t.bbox)

            # Profundidad mediana en la caja
            crop_depth = depth_map[y1:y2, x1:x2]
            if crop_depth.size > 0:
                state.depth_median = float(np.median(crop_depth))

            # Inside area
            cx, cy = bbox_center(t.bbox)
            inside = is_point_in_polygon((cx, cy), self.polygon_points)
            state.inside_area = inside

            self.track_states[track_id] = state
            events.append(state)

        # Dibujo
        vis = frame_bgr.copy()
        vis = draw_polygon(vis, self.polygon_points)
        vis = draw_annotations(vis, events)
        return vis, events

    @staticmethod
    def _heuristic_carries_object(frame_bgr, bbox: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return False
        top = frame_bgr[y1 : y1 + max(1, h // 3), x1:x2]
        if top.size == 0:
            return False
        # varianza de bordes como proxy (muy simple):
        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(edges.var()) > 500.0



