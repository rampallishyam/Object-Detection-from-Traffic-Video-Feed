
import argparse
import os
import torch

from traffic_od.config import AppConfig
from traffic_od.pipeline import VideoPipeline
from traffic_od.logging_utils import setup_logging


def main():
    HOME = os.getcwd()
    MODEL = os.path.join(HOME, "models", "all_weights", "idd_dataset.pt")
    SOURCE_VIDEO_PATH = os.path.join(HOME, "test", "test_videos", "video1.mp4")
    CLASS_ID = [0, 1, 2, 3, 4]
    DEFAULT_INTERVAL = 2

    cfg = AppConfig.from_cli(
        default_model=MODEL,
        default_video=SOURCE_VIDEO_PATH,
        default_interval=DEFAULT_INTERVAL,
        default_classes=CLASS_ID,
    )

    setup_logging(os.path.join(HOME, "logs"))
    pipeline = VideoPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.version.cuda)
    main()


