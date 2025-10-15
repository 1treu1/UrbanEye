from __future__ import annotations

import cv2
import numpy as np


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    d = depth_map.astype("float32")
    d = cv2.normalize(d, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return d



