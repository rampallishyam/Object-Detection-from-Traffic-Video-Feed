
import numpy as np
from supervision.tools.detections import Detections

__all__ = ["detections2boxes"]

def detections2boxes(detections: Detections) -> np.ndarray:
    """Convert Detections to a numpy array for tracking (xyxy + confidence)."""
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))

