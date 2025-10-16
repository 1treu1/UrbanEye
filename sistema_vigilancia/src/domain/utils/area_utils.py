from __future__ import annotations

from typing import List, Tuple


def is_point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        if ((y1 > y) != (y0 > y)) and (
            x < (x0 - x1) * (y - y1) / (y0 - y1 + 1e-6) + x1
        ):
            inside = not inside
        x0, y0 = x1, y1
    return inside


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2





