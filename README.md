# Traffic Detection and Travel Time Estimation

This project implements computer vision algorithms to detect, track, and count vehicles using YOLOv8 and ByteTrack. The Indian Driving Dataset (IDD) is utilized for training and evaluation. The goal of this project is to provide accurate vehicle monitoring and insights for traffic management.

> Note: This repository is a sandbox implementation of a larger production system. The full production setup is subject to IP restrictions and cannot be shared publicly.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [OD Estimation](#od_estimation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to leverage state-of-the-art computer vision techniques to address vehicle detection, tracking, and counting challenges in real-world traffic scenarios. YOLOv8 is employed for precise vehicle detection, while ByteTrack is utilized for robust vehicle tracking. The IDD Dataset provides a diverse and comprehensive training and validation dataset to ensure model performance across various Indian road conditions.

## Features

- Vehicle detection using YOLOv8
- Robust vehicle tracking with ByteTrack
- Accurate vehicle counting and turning proportion analysis
- Integration with the Indian Driving Dataset (IDD)
- Real-time and efficient algorithms for traffic monitoring

git clone https://github.com/rampallishyam/vehicle-detection-travel-time-estimation.git

## Installation

1. Clone this repository:

```bash
git clone https://github.com/rampallishyam/vehicle-detection-travel-time-estimation.git
cd vehicle-detection-travel-time-estimation
```

2. Install dependencies using [Poetry](https://python-poetry.org/):

```bash
# Install Poetry (if not already)
curl -sSL https://install.python-poetry.org | python3 -

# Install deps and activate shell
poetry install
poetry shell
```

> **Note:** If you need to use custom files from `modified_files/`, manually copy them to the appropriate location in your environment as needed.

## Usage

The app is now structured as a modular, production-ready pipeline under `traffic_od/`.

- CLI (backward-compatible defaults):

```bash
python main.py \
	--model_yolo models/all_weights/idd_dataset.pt \
	--source_video test/test_videos/video1.mp4 \
	--interval_time 2 \
	--class_ids 0 1 2 3 4
```

During the first frame, annotate corridors and conversion points via the interactive windows. Outputs will be written under `test/results/<video>_results/runN/`.

> [Demo Video: How to Add Corridors](https://tinyurl.com/55uwsxvt)

### Architecture overview

- `traffic_od/config.py` – CLI parsing and AppConfig dataclass
- `traffic_od/logging_utils.py` – Loguru-based logging
- `traffic_od/detector.py` – YOLOv8 wrapper
- `traffic_od/tracker.py` – BYTETracker adapter using existing utils
- `traffic_od/results.py` – Output directory management and CSV/video writing
- `traffic_od/pipeline.py` – Orchestration of detection, tracking, counting, and outputs

Training scripts and dataset tools remain under `models/` and root utilities.

## Project structure

```
.
├── Makefile                        # Poetry-based helpers (install, run, test, lint)
├── main.py                         # CLI entrypoint invoking the pipeline
├── models/                         # Training configs, weights, and runs
│   └── training/
│       └── runs/                   # YOLOv8 training outputs (artifacts)
├── traffic_od/                     # Production pipeline package
│   ├── config.py                   # AppConfig and CLI parser
│   ├── detector.py                 # YOLOv8 detector wrapper
│   ├── tracker.py                  # BYTETracker adapter (uses utils)
│   ├── results.py                  # Output dirs, CSVs and video sink
│   ├── logging_utils.py            # Loguru logging setup
│   └── pipeline.py                 # Orchestrates detection/tracking/OD
├── utils/                          # Supporting utilities
│   ├── detections2boxes.py         # Convert detections to tracker input
│   ├── get_inputs.py               # Interactive corridor and 4-point input
│   ├── get_tracks.py               # Pixel-to-real conversion and distances
│   ├── match_detections_with_tracks.py
│   └── tracks2boxes.py
├── test/                           # Example test videos and generated results
│   ├── test_videos/
│   └── results/
├── tests/                          # Unit tests
│   └── test_config_and_results.py
├── pyproject.toml                  # Poetry configuration and dependencies
└── README.md
```

Notes:
- `traffic_od/` contains the production-ready, modular pipeline.
- `models/` and `test/` include artifacts and sample data; they’re not core runtime code.
- Use the Makefile or Poetry to run, test, and lint consistently.

### Run tests

```bash
poetry run pytest -q
```

## OD Estimation

1. Run `python OD_Matrix_Estimation/main.py --help` for usage instructions.
2. Example: `python main.py --num_nodes 5 --distance_matrix sample_data/distance_matrix.csv --entry_counts sample_data/entry_counts.csv --exit_counts sample_data/exit_counts.csv`
3. Press Enter to run the OD Estimation Model.

## Contributing

We welcome contributions to enhance the project's capabilities, performance, and documentation. Please submit issues for bug reports, feature requests, or questions. Pull requests are also encouraged.

## License

This project is licensed under the [MIT License](LICENSE).
