import numpy as np
from typing import List
from yolox.tracker.byte_tracker import STrack

__all__ = ['tracks2boxes']

# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)