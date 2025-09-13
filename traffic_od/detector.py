from __future__ import annotations

from typing import Any, Dict
import numpy as np
from ultralytics import YOLO
from supervision.tools.detections import Detections


class YoloV8Detector:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)
        # fuse for speed
        try:
            self.model.fuse()
        except Exception:
            pass
        self.class_names: Dict[int, str] = getattr(self.model.model, "names", {})

    def predict(self, frame: np.ndarray) -> Detections:
        results = self.model(frame)
        det = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )
        return det
