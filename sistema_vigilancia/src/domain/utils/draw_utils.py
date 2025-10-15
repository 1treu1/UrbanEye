from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..models import PersonState


def draw_polygon(img, points: List[Tuple[int, int]]):
    if len(points) >= 3:
        cv2.polylines(img, [np.array(points, dtype="int32")], True, (0, 255, 255), 2)
    return img


def draw_annotations(img, people: List[PersonState]):
    for p in people:
        x1, y1, x2, y2 = p.bbox
        color = (0, 255, 0) if p.inside_area else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        gender_str = p.gender or "?"
        age_str = str(p.age) if p.age is not None else "?"
        depth_str = f"{p.depth_median:.2f}" if p.depth_median is not None else "0.00"
        label = f"ID:{p.track_id} {gender_str} {age_str} obj:{int(p.carries_object)} d:{depth_str}"
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


