from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import argparse
import os


@dataclass(frozen=True)
class TrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


@dataclass
class AppConfig:
    model_yolo: str
    source_video: str
    interval_time: int = 2
    class_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    home: str = field(default_factory=lambda: os.getcwd())
    tracker: TrackerArgs = field(default_factory=TrackerArgs)

    @staticmethod
    def from_cli(default_model: Optional[str] = None, default_video: Optional[str] = None, default_interval: int = 2,
                 default_classes: Optional[List[int]] = None) -> "AppConfig":
        parser = argparse.ArgumentParser(description="Run Model on Source Video")
        parser.add_argument("--model_yolo", default=default_model, help="Path for the YOLO model")
        parser.add_argument("--source_video", default=default_video, help="Path of the input video")
        parser.add_argument("--interval_time", default=default_interval, type=int, help="Time interval between logs in seconds")
        parser.add_argument("--class_ids", default=default_classes or [0, 1, 2, 3, 4], nargs="+", type=int, help="Class IDs of interest")
        args = parser.parse_args()

        return AppConfig(
            model_yolo=args.model_yolo,
            source_video=args.source_video,
            interval_time=args.interval_time,
            class_ids=list(args.class_ids),
            home=os.getcwd(),
        )
