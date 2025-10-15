from __future__ import annotations

from typing import Protocol


class DepthEstimationPort(Protocol):
    """Puerto para estimación de profundidad monocular por pixel."""

    def estimate_depth(self, frame_bgr):
        """Devuelve un mapa de profundidad (float32) normalizado 0..1 del frame."""
        ...



