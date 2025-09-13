import os
import torch
import multiprocessing
from ultralytics import YOLO

def main():
    """Train a YOLOv8 model on the specified dataset."""
    model = YOLO("yolov8m.pt")
    model.train(data="/data.yaml", epochs=100, imgsz=640, device=0)
    print("DONE TRAINING")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} {device_name}")
    multiprocessing.freeze_support()
    main()


