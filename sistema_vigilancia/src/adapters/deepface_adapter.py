from __future__ import annotations

from typing import Optional

import cv2
from deepface import DeepFace

from ..ports.age_gender_port import AgeGender, AgeGenderPort


class DeepFaceEstimator(AgeGenderPort):
    def __init__(self, analyze_every_n_frames: int = 15) -> None:
        self.counter = 0
        self.every = max(1, analyze_every_n_frames)
        self.cache: dict[int, AgeGender] = {}

    def estimate(self, frame_bgr, person_bbox) -> AgeGender:
        self.counter += 1
        if self.counter % self.every != 0:
            # devuelve último estimado si existe
            return AgeGender(age=None, gender=None)

        x1, y1, x2, y2 = person_bbox
        crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return AgeGender(age=None, gender=None)
        try:
            # DeepFace espera RGB
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            analysis = DeepFace.analyze(rgb, actions=['age', 'gender'], enforce_detection=False, prog_bar=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            age = int(analysis.get('age')) if analysis.get('age') is not None else None
            gender = analysis.get('dominant_gender')
            return AgeGender(age=age, gender=gender)
        except Exception:
            return AgeGender(age=None, gender=None)



