from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from yolox.tracker.byte_tracker import BYTETracker, STrack
from supervision.tools.detections import Detections

from utils.detections2boxes import detections2boxes
from utils.match_detections_with_tracks import match_detections_with_tracks


@dataclass
class TrackerConfig:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


class ByteTrackAdapter:
    def __init__(self, cfg: TrackerConfig) -> None:
        self._tracker = BYTETracker(cfg)

    def update(self, detections: Detections, frame_shape: tuple) -> List[STrack]:
        return self._tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame_shape,
            img_size=frame_shape,
        )

    @staticmethod
    def attach_tracker_ids(detections: Detections, tracks: List[STrack]) -> Detections:
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filter out unmatched
        mask = np.array([tid is not None for tid in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        return detections
