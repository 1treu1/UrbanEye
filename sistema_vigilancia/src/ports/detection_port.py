from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


class DetectionPort(Protocol):
    """Puerto de detección de objetos.

    Cualquier detector (YOLO, etc.) debe implementar esta interfaz para ser
    intercambiable sin cambiar la lógica de la aplicación.
    """

    def detect(self, frame_bgr) -> List[Detection]:
        """Retorna lista de detecciones para un frame BGR.

        - frame_bgr: imagen en formato OpenCV BGR.
        - Devuelve cajas como (x1,y1,x2,y2) en pixeles, confianza y clase.
        """
        ...





