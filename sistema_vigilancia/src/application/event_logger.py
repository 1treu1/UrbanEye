from __future__ import annotations

from pathlib import Path
from typing import Iterable
from datetime import datetime

import pandas as pd

from ..domain.models import PersonState


class EventLogger:
    """Registra eventos de personas en CSV."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            df = pd.DataFrame(
                columns=[
                    "timestamp",
                    "track_id",
                    "age",
                    "gender",
                    "carries_object",
                    "depth",
                    "inside_area",
                ]
            )
            df.to_csv(self.csv_path, index=False)

    def append(self, people: Iterable[PersonState]) -> None:
        rows = []
        ts = datetime.utcnow().isoformat()
        for p in people:
            rows.append(
                {
                    "timestamp": ts,
                    "track_id": p.track_id,
                    "age": p.age,
                    "gender": p.gender,
                    "carries_object": int(p.carries_object),
                    "depth": p.depth_median if p.depth_median is not None else None,
                    "inside_area": int(p.inside_area),
                }
            )
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.csv_path, mode="a", header=False, index=False)





