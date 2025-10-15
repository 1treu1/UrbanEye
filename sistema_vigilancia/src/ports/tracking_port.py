from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float


class TrackingPort(Protocol):
    """Puerto de seguimiento multi-objeto (asignación de ID)."""

    def update(self, frame_bgr, detections: List[Tuple[int, int, int, int, float]]) -> List[Track]:
        """Actualiza el estado del seguidor y retorna pistas activas.

        - detections: lista de (x1,y1,x2,y2,conf)
        """
        ...



