import os
from ultralytics import YOLO
import multiprocessing
import torch

def main():
    # Load a model 
    model = YOLO('yolov8m.pt')

    # Train the model
    model.train(data='D:/Anup/MICP/dataset/data.yaml', epochs=100, imgsz=640, device=0)
    print("DONE TRAINING")


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    deviceN = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(deviceN)
    print("Device : " + str(deviceN) + " " + str(device_name))                                                     
    multiprocessing.freeze_support()

    main()


