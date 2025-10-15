from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class AgeGender:
    age: Optional[int]
    gender: Optional[str]


class AgeGenderPort(Protocol):
    """Puerto para estimación de edad y género en un recorte facial."""

    def estimate(self, frame_bgr, person_bbox) -> AgeGender:
        """Devuelve edad y género estimados para una persona.

        La implementación puede internamente detectar rostro en el recorte.
        """
        ...



