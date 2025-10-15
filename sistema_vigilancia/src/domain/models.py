from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PersonState:
    track_id: int
    bbox: Tuple[int, int, int, int]
    age: Optional[int] = None
    gender: Optional[str] = None
    carries_object: bool = False
    depth_median: Optional[float] = None  # 0..1 normalizado o metros si calibrado
    inside_area: bool = False



