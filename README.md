# Traffic Detection and Travel Time Estimation

This project implements computer vision algorithms to detect, track, and count vehicles using YOLOv8 and ByteTrack. The Indian Driving Dataset (IDD) is utilized for training and evaluation. The goal of this project is to provide accurate vehicle monitoring and insights for traffic management.

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
pip install -r requirements.txt

## Installation

1. Clone this repository:

```bash
git clone https://github.com/rampallishyam/vehicle-detection-travel-time-estimation.git
cd vehicle-detection-travel-time-estimation
```

2. Install dependencies using [Poetry](https://python-poetry.org/):

```bash
# If you don't have Poetry installed:
pip install poetry

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

> **Note:** If you need to use custom files from `modified_files/`, manually copy them to the appropriate location in your environment as needed.

## Usage

1. Download the IDD Dataset from https://idd.insaan.iiit.ac.in/dataset/download/ and preprocess it using `create_idd_dataset.py` for training and validation.
2. Train the YOLOv8 model on the IDD Dataset using `models/training/yolov8_train.py`.
3. Run `main.py` using the trained weights (or use the pre-trained weights in `models/all_weights/idd_dataset.pt`).
4. [Demo Video: How to Add Corridors](https://tinyurl.com/55uwsxvt)

## OD Estimation

1. Run `python OD_Matrix_Estimation/main.py --help` for usage instructions.
2. Example: `python main.py --num_nodes 5 --distance_matrix sample_data/distance_matrix.csv --entry_counts sample_data/entry_counts.csv --exit_counts sample_data/exit_counts.csv`
3. Press Enter to run the OD Estimation Model.

## Contributing

We welcome contributions to enhance the project's capabilities, performance, and documentation. Please submit issues for bug reports, feature requests, or questions. Pull requests are also encouraged.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Project improvements:**
- Dependency management is now handled by Poetry (`pyproject.toml`).
- Updated installation instructions for modern Python workflows.
- Cleaned up and clarified usage steps.
