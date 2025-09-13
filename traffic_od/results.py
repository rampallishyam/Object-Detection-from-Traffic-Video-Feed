from __future__ import annotations

import os
import csv
from typing import Dict, List, Tuple

from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink


class ResultPaths:
    def __init__(self, base_dir: str, video_path: str) -> None:
        video_name = os.path.basename(video_path)
        self.root = os.path.join(base_dir, "test", "results", f"{os.path.splitext(video_name)[0]}_results")
        if not os.path.exists(self.root):
            os.makedirs(os.path.join(self.root, "run1", "corridor_counts"), exist_ok=True)
            os.makedirs(os.path.join(self.root, "run1", "tracks"), exist_ok=True)
            self.run_path = os.path.join(self.root, "run1")
        else:
            runs_dir = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
            tot_runs = len(runs_dir)
            self.run_path = os.path.join(self.root, f"run{tot_runs+1}")
            os.makedirs(os.path.join(self.run_path, "corridor_counts"), exist_ok=True)
            os.makedirs(os.path.join(self.run_path, "tracks"), exist_ok=True)
        self.tracks_path = os.path.join(self.run_path, "tracks")
        self.corridor_counts = os.path.join(self.run_path, "corridor_counts")
        self.video_out = os.path.join(self.run_path, "result_video.mp4")

    def open_video_sink(self, video_info: VideoInfo) -> VideoSink:
        return VideoSink(self.video_out, video_info)


def init_vehicle_count_csvs(path: ResultPaths, class_names_dict: Dict[int, str], class_ids: List[int], lines_count: int) -> None:
    header = ["Time Interval (seconds)"] + [class_names_dict[id] for id in class_ids] + ["Total (cumulative)"]
    for i in range(lines_count):
        for flw_typ in ("inflow", "outflow"):
            with open(os.path.join(path.corridor_counts, f"vehicle_counts_corridor{i+1}_{flw_typ}.csv"), "a", newline="") as f:
                csv.writer(f).writerow(header)


def append_vehicle_counts(path: ResultPaths, line_idx: int, time_s: int, counts: Dict, flows=("inflow", "outflow")) -> None:
    for o, flw_typ in enumerate(flows):
        with open(os.path.join(path.corridor_counts, f"vehicle_counts_corridor{line_idx+1}_{flw_typ}.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time_s] + [cnt[o] for cnt in counts.values()])


def write_track_metrics(path: ResultPaths, track_id: int, rows: List[List]) -> None:
    csv_path = os.path.join(path.tracks_path, f"{track_id}.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Time Start (seconds)",
                "Time End (seconds)",
                "Real_X (meters)",
                "Real_Y (meters)",
                "Distance Travelled (meters)",
                "Speed (km/hr)",
            ])
        writer.writerows(rows)


def write_od_matrix(path: ResultPaths, od_matrix, lines_count: int) -> None:
    with open(os.path.join(path.run_path, "OD_matrix.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["v-- O , D -->"] + [f"corridor_{i}" for i in range(1, 1 + lines_count)])
        for i in range(1, 1 + lines_count):
            temp = [0] * lines_count
            for od, count in od_matrix.items():
                if od[0] == i:
                    temp[od[-1] - 1] = count
            writer.writerow([f"corridor_{i}"] + temp)
