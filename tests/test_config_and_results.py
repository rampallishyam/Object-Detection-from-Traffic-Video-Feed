import os
import sys
from pathlib import Path

import builtins

from traffic_od.config import AppConfig
from traffic_od.results import ResultPaths


def test_app_config_from_cli_no_args(monkeypatch, tmp_path):
    monkeypatch.setenv("PWD", str(tmp_path))
    # Simulate argv with script name only; defaults will be None but we pass defaults via from_cli
    monkeypatch.setattr(sys, "argv", ["prog"])
    model = str(tmp_path / "models" / "all_weights" / "idd_dataset.pt")
    video = str(tmp_path / "video.mp4")
    cfg = AppConfig.from_cli(default_model=model, default_video=video, default_interval=3, default_classes=[1, 2])
    assert cfg.model_yolo == model
    assert cfg.source_video == video
    assert cfg.interval_time == 3
    assert cfg.class_ids == [1, 2]


def test_result_paths_creation(tmp_path):
    base = str(tmp_path)
    video_path = str(tmp_path / "test" / "test_videos" / "video1.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    Path(video_path).touch()

    rp = ResultPaths(base, video_path)
    assert os.path.isdir(rp.run_path)
    assert os.path.isdir(rp.tracks_path)
    assert os.path.isdir(rp.corridor_counts)
    assert rp.video_out.endswith("result_video.mp4")
