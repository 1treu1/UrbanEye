from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from ..ports.depth_port import DepthEstimationPort
from ..domain.utils.depth_utils import normalize_depth


class MiDaSDepthEstimator(DepthEstimationPort):
    def __init__(self, model_name: str = "DPT_Large") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", model_name)
        self.model.to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def estimate_depth(self, frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy().astype("float32")
        return normalize_depth(depth)



