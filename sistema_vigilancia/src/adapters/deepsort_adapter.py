from __future__ import annotations

from typing import List, Tuple

from deep_sort_realtime.deepsort_tracker import DeepSort

from ..ports.tracking_port import Track, TrackingPort


class DeepSortTracker(TrackingPort):
    def __init__(self, max_age: int = 30, n_init: int = 2) -> None:
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, frame_bgr, detections: List[Tuple[int, int, int, int, float]]) -> List[Track]:
        # expected by deep-sort-realtime: [ [[x1,y1,x2,y2], conf, cls], ... ]
        dets = [[[x1, y1, x2, y2], conf, 0] for (x1, y1, x2, y2, conf) in detections]
        tracks = self.tracker.update_tracks(dets, frame=frame_bgr)
        result: List[Track] = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            result.append(Track(track_id=int(t.track_id), bbox=(x1, y1, x2, y2), confidence=1.0))
        return result


