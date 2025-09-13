from __future__ import annotations

from typing import Dict, List
import os
import cv2
import numpy as np
from loguru import logger

from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo

from .config import AppConfig
from .detector import YoloV8Detector
from .tracker import ByteTrackAdapter, TrackerConfig
from .results import ResultPaths, init_vehicle_count_csvs, append_vehicle_counts, write_track_metrics, write_od_matrix

from utils.get_inputs import get_inputs
from utils.get_tracks import get_conv_mat, get_distance


class VideoPipeline:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.detector = YoloV8Detector(cfg.model_yolo)
        self.tracker = ByteTrackAdapter(TrackerConfig(**vars(cfg.tracker)))

    def run(self) -> None:
        logger.info("Starting pipeline")
        lines, Pnts, real_dim = get_inputs(self.cfg.source_video)

        video_info = VideoInfo.from_video_path(self.cfg.source_video)
        out_paths = ResultPaths(self.cfg.home, self.cfg.source_video)

        line_counters: List[LineCounter] = []
        for line in lines:
            if len(line) == 2:
                line_counters.append(LineCounter(
                    start=Point(line[0][0], line[0][1]),
                    end=Point(line[1][0], line[1][1]),
                    all_vehicle_class=self.cfg.class_ids,
                ))

        box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
        line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

        with out_paths.open_video_sink(video_info) as sink:
            video = cv2.VideoCapture(self.cfg.source_video)
            if not video.isOpened():
                raise RuntimeError(f"Could not open video at {self.cfg.source_video}")

            conv_mat = get_conv_mat(Pnts, real_dim)
            vehicle_turns: Dict[str, list] = {}
            init_vehicle_count_csvs(out_paths, self.detector.class_names, self.cfg.class_ids, len(line_counters))

            for i in range((video_info.total_frames // (video_info.fps * self.cfg.interval_time)) + 1):
                if i == (video_info.total_frames // (video_info.fps * self.cfg.interval_time)):
                    iter_frames = video_info.total_frames % (video_info.fps * self.cfg.interval_time)
                else:
                    iter_frames = video_info.fps * self.cfg.interval_time

                for j, line_counter in enumerate(line_counters):
                    append_vehicle_counts(out_paths, j, i * self.cfg.interval_time, line_counter.get_data())

                pos_real = {}
                success, frame = video.read()
                frame_count = 0

                while success and frame_count < iter_frames:
                    for l in range(4):
                        cv2.circle(frame, (Pnts[l][0], Pnts[l][1]), 5, (0, 255, 255), -1)

                    detections = self.detector.predict(frame)
                    mask = np.array([cid in self.cfg.class_ids for cid in detections.class_id], dtype=bool)
                    detections.filter(mask=mask, inplace=True)

                    tracks = self.tracker.update(detections, frame.shape)
                    detections = self.tracker.attach_tracker_ids(detections, tracks)

                    labels = [f"#{tracker_id}" for _, _, _, tracker_id in detections]

                    get_distance(conv_mat, detections, pos_real, frame_count)

                    for k, line_counter in enumerate(line_counters):
                        line_counter.update(detections=detections, corridor_id=k+1, turning_prop=vehicle_turns)

                    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                    for line_counter in line_counters:
                        line_annotator.annotate(frame=frame, line_counter=line_counter)
                    sink.write_frame(frame)

                    success, frame = video.read()
                    frame_count += 1

                # summarize tracks for this interval
                for track_id, positions in pos_real.items():
                    if len(positions) < 2:
                        continue
                    distance = np.sqrt((positions[-1][0] - positions[0][0]) ** 2 + (positions[-1][1] - positions[0][1]) ** 2)
                    time_interval = (positions[-1][2] - positions[0][2]) / video_info.fps
                    speed = 3.6 * abs(distance) / time_interval if time_interval > 0 else 0.0
                    write_track_metrics(out_paths, track_id, [[
                        np.round(i * self.cfg.interval_time + positions[0][2] / video_info.fps),
                        np.round(i * self.cfg.interval_time + positions[-1][2] / video_info.fps),
                        positions[0][0],
                        positions[0][1],
                        distance,
                        speed,
                    ]])

            video.release()

        # finalize OD matrix
        od_matrix: Dict[tuple, int] = {}
        for id, turns in vehicle_turns.items():
            if len(turns) >= 2:
                key = tuple(turns)
                od_matrix[key] = od_matrix.get(key, 0) + 1
        write_od_matrix(out_paths, od_matrix, len(line_counters))

        logger.info("Pipeline finished")
