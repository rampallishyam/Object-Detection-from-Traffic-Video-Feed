
import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink

from utils.detections2boxes import detections2boxes
from utils.get_inputs import get_inputs
from utils.get_tracks import get_conv_mat, get_distance
from utils.match_detections_with_tracks import match_detections_with_tracks



@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def main():
    """Main function to run vehicle detection, tracking, and analytics on a video file."""
    parser = argparse.ArgumentParser(description="Run Model on Source Video")
    parser.add_argument("--model_yolo", default=MODEL, help="Path for the YOLO model")
    parser.add_argument("--source_video", default=SOURCE_VIDEO_PATH, help="Path of the input video")
    parser.add_argument("--interval_time", default=interval_time, type=int, help="Time interval between logs in seconds")
    parser.add_argument("--class_ids", default=CLASS_ID, nargs="+", type=int, help="Class IDs of interest")
    args = parser.parse_args()

    lines, Pnts, real_dim = get_inputs(args.source_video)

    # Create results directory structure
    video_name = os.path.basename(args.source_video)
    results_path = os.path.join(HOME, "test", "results", f"{os.path.splitext(video_name)[0]}_results")
    if not os.path.exists(results_path):
        os.makedirs(os.path.join(results_path, "run1", "corridor_counts"))
        os.makedirs(os.path.join(results_path, "run1", "tracks"))
        run_path = os.path.join(results_path, "run1")
        track_path = os.path.join(run_path, "tracks")
    else:
        runs_dir = [d for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))]
        tot_runs = len(runs_dir)
        run_path = os.path.join(results_path, f"run{tot_runs+1}")
        os.makedirs(os.path.join(run_path, "corridor_counts"))
        os.makedirs(os.path.join(run_path, "tracks"))
        track_path = os.path.join(run_path, "tracks")

    # Load YOLO model
    model = YOLO(args.model_yolo)
    model.fuse()
    class_names_dict = model.model.names

    # Create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())

    # Get video info
    video_info = VideoInfo.from_video_path(args.source_video)

    # Create line counters
    line_counters = []
    for line in lines:
        if len(line) == 2:
            line_counter = LineCounter(
                start=Point(line[0][0], line[0][1]),
                end=Point(line[1][0], line[1][1]),
                all_vehicle_class=args.class_ids,
            )
            line_counters.append(line_counter)

    # Annotators
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
    line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

    # Open target video file
    with VideoSink(os.path.join(run_path, "result_video.mp4"), video_info) as sink:
        video = cv2.VideoCapture(args.source_video)
        if not video.isOpened():
            raise RuntimeError(f"Could not open video at {args.source_video}")

        # Conversion matrix for pixel to real-world coordinates
        conv_mat = get_conv_mat(Pnts, real_dim)
        vehicle_turns: Dict[str, list] = {}

        # Create vehicle count CSVs
        vehicle_count_header = ["Time Interval (seconds)"] + [class_names_dict[id] for id in args.class_ids] + ["Total (cumulative)"]
        flows_type = ["inflow", "outflow"]
        for i, _ in enumerate(line_counters):
            for flw_typ in flows_type:
                with open(os.path.join(run_path, f"corridor_counts/vehicle_counts_corridor{i+1}_{flw_typ}.csv"), "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(vehicle_count_header)

        # Loop over video frames in intervals
        for i in range((video_info.total_frames // (video_info.fps * args.interval_time)) + 1):
            if i == (video_info.total_frames // (video_info.fps * args.interval_time)):
                iter_frames = video_info.total_frames % (video_info.fps * args.interval_time)
            else:
                iter_frames = video_info.fps * args.interval_time

            # Append vehicle count to CSVs
            for j, line_counter in enumerate(line_counters):
                total_counts_this = line_counter.get_data()
                for o, flw_typ in enumerate(flows_type):
                    with open(os.path.join(run_path, f"corridor_counts/vehicle_counts_corridor{j+1}_{flw_typ}.csv"), "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([i * args.interval_time] + [cnt[o] for cnt in total_counts_this.values()])

            pos_real = {}  # Initialize positions dict
            success, frame = video.read()
            frame_count = 0

            while success and frame_count < iter_frames:
                for l in range(4):
                    cv2.circle(frame, (Pnts[l][0], Pnts[l][1]), 5, (0, 255, 255), -1)

                # Model prediction and conversion to Detections
                results = model(frame)
                detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                )

                # Filter unwanted classes
                mask = np.array([class_id in args.class_ids for class_id in detections.class_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)

                # Tracking
                tracks = byte_tracker.update(
                    output_results=detections2boxes(detections=detections),
                    img_info=frame.shape,
                    img_size=frame.shape,
                )

                tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
                detections.tracker_id = np.array(tracker_id)

                # Filter detections without trackers
                mask = np.array([tid is not None for tid in detections.tracker_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)

                # Format custom labels
                labels = [f"#{tracker_id}" for _, _, _, tracker_id in detections]

                # Track real-world positions
                get_distance(conv_mat, detections, pos_real, frame_count)

                # Update line counters
                for k, line_counter in enumerate(line_counters):
                    line_counter.update(detections=detections, corridor_id=k+1, turning_prop=vehicle_turns)

                # Annotate and write frame
                frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                for line_counter in line_counters:
                    line_annotator.annotate(frame=frame, line_counter=line_counter)
                sink.write_frame(frame)

                success, frame = video.read()
                frame_count += 1

            # Log trajectories and speeds
            for track_id, positions in pos_real.items():
                if len(positions) < 2:
                    continue
                distance = np.sqrt((positions[-1][0] - positions[0][0]) ** 2 + (positions[-1][1] - positions[0][1]) ** 2)
                time_interval = (positions[-1][2] - positions[0][2]) / video_info.fps
                speed = 3.6 * abs(distance) / time_interval if time_interval > 0 else 0.0
                csv_path = os.path.join(track_path, f"{track_id}.csv")
                write_header = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="") as file:
                    writer = csv.writer(file)
                    if write_header:
                        writer.writerow([
                            "Time Start (seconds)",
                            "Time End (seconds)",
                            "Real_X (meters)",
                            "Real_Y (meters)",
                            "Distance Travelled (meters)",
                            "Speed (km/hr)",
                        ])
                    writer.writerow([
                        np.round(i * args.interval_time + positions[0][2] / video_info.fps),
                        np.round(i * args.interval_time + positions[-1][2] / video_info.fps),
                        positions[0][0],
                        positions[0][1],
                        distance,
                        speed,
                    ])

        video.release()

    # OD matrix calculation
    od_matrix: Dict[tuple, int] = {}
    for id, turns in vehicle_turns.items():
        if len(turns) >= 2:
            key = tuple(turns)
            od_matrix[key] = od_matrix.get(key, 0) + 1

    with open(os.path.join(run_path, "OD_matrix.csv"), "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["v-- O , D -->"] + [f"corridor_{i}" for i in range(1, 1 + len(line_counters))])
        for i in range(1, 1 + len(line_counters)):
            temp = np.zeros(len(line_counters))
            for od, count in od_matrix.items():
                if od[0] == i:
                    temp[od[-1] - 1] = count
            writer.writerow([f"corridor_{i}"] + list(temp))


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.version.cuda)

    HOME = os.getcwd()
    MODEL = os.path.join(HOME, "models", "all_weights", "idd_dataset.pt")
    CLASS_ID = [0, 1, 2, 3, 4]  # class_ids of Indian_Traffic_model - bike, auto, car, bus and truck
    SOURCE_VIDEO_PATH = os.path.join(HOME, "test", "test_videos", "video1.mp4")
    interval_time = 2
    main()


