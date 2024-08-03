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

## Installation

1. Clone this repository:

git clone https://github.com/AnupN29/Bosch_Internship_Project.git

cd Bosch_Internship_Project

2. Set up your Python environment:

conda create -n your_env_name python=3.8

conda activate your_env_name

pip install -r requirements.txt

replace files in site-packages/supervision with  the respective files in Multi_Vehicle_Tracking/modified_files

## Usage

1. Download IDD Dataset from https://idd.insaan.iiit.ac.in/dataset/download/ and preprocess using create_idd_dataset.py the  for training and validation.
2. Train the YOLOv8 model on the IDD Dataset using yolov8_train.py script.
3. Run the main.py using the trained weights (or you can use already trained weights in models/all_weights/idd_dataset.pt)
4. Demo Video To add Corridors : https://tinyurl.com/55uwsxvt

## OD Estimation

1. Run python OD_Matrix_Estimation/main.py --help
2. Sample Code : python main.py --num_nodes 5 --distance_matrix sample_data/distance_matrix.csv --entry_counts sample_data/entry_counts.csv --exit_counts sample_data/exit_counts.csv
3. Hit enter to run the OD Estimation Model

## Contributing

We welcome contributions to enhance the project's capabilities, performance, and documentation. Please submit issues for bug reports, feature requests, or questions. Pull requests are also encouraged.

## License

This project is licensed under the [MIT License](LICENSE).
