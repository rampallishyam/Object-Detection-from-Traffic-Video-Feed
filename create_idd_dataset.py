import argparse
import os
import random
from utils.image_transform import img_transform
from utils.xml2yolo import xml2yolo

BASE_PATH = "/"
IMGS_PATH = os.path.join(BASE_PATH, "_/_old/idd-detection/IDD_Detection/JPEGImages")
LABELS_PATH = os.path.join(BASE_PATH, "_/_old/idd-detection/IDD_Detection/Annotations")
TRAIN_PATH = os.path.join(BASE_PATH, "dataset/train")
VAL_PATH = os.path.join(BASE_PATH, "dataset/val")

def create_dataset():
    """Create a YOLO-formatted dataset from the IDD dataset, splitting into train/val sets."""
    parser = argparse.ArgumentParser(description="Create YOLO dataset from IDD")
    parser.add_argument("--imgs_path", default=IMGS_PATH, help="Path to images")
    parser.add_argument("--labels_path", default=LABELS_PATH, help="Path to labels")
    parser.add_argument("--split", default=0.9, type=float, help="Train/val split ratio")
    args = parser.parse_args()

    primary_folders = [name for name in os.listdir(args.imgs_path) if os.path.isdir(os.path.join(args.imgs_path, name))]

    for primary_folder in primary_folders:
        primary_folder_path = os.path.join(args.imgs_path, primary_folder)
        secondary_folders = [name for name in os.listdir(primary_folder_path) if os.path.isdir(os.path.join(primary_folder_path, name))]

        for secondary_folder in secondary_folders:
            secondary_folder_path = os.path.join(primary_folder_path, secondary_folder)
            image_files = [name for name in os.listdir(secondary_folder_path) if name.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))]
            train_index = int(args.split * len(image_files))
            random.shuffle(image_files)

            # Training dataset
            for file_name in image_files[:train_index]:
                file_path = os.path.join(secondary_folder_path, file_name)
                if (os.path.exists(os.path.join(TRAIN_PATH, "images")) and file_name in os.listdir(os.path.join(TRAIN_PATH, "images"))) or \
                   (os.path.exists(os.path.join(VAL_PATH, "images")) and file_name in os.listdir(os.path.join(VAL_PATH, "images"))):
                    continue
                file_base, file_ext = os.path.splitext(file_name)
                transformed_image = img_transform(file_path, mode="train")
                xml_file = os.path.join(args.labels_path, primary_folder, secondary_folder, file_base + ".xml")
                output_txt = os.path.join(TRAIN_PATH, "labels", file_base + ".txt")
                try:
                    xml2yolo(xml_file, output_txt)
                    transformed_image.save(os.path.join(TRAIN_PATH, "images", file_base + file_ext))
                except Exception as e:
                    exp_txt = os.path.join(BASE_PATH, "err", file_base + ".txt")
                    with open(exp_txt, "w") as txt_file:
                        txt_file.write(f"ERROR: {output_txt}\n{e}")

            # Validation dataset
            for file_name in image_files[train_index:]:
                file_path = os.path.join(secondary_folder_path, file_name)
                if (os.path.exists(os.path.join(TRAIN_PATH, "images")) and file_name in os.listdir(os.path.join(TRAIN_PATH, "images"))) or \
                   (os.path.exists(os.path.join(VAL_PATH, "images")) and file_name in os.listdir(os.path.join(VAL_PATH, "images"))):
                    continue
                file_base, file_ext = os.path.splitext(file_name)
                transformed_image = img_transform(file_path, mode="val")
                xml_file = os.path.join(args.labels_path, primary_folder, secondary_folder, file_base + ".xml")
                output_txt = os.path.join(VAL_PATH, "labels", file_base + ".txt")
                try:
                    xml2yolo(xml_file, output_txt)
                    transformed_image.save(os.path.join(VAL_PATH, "images", file_base + file_ext))
                except Exception as e:
                    exp_txt = os.path.join(BASE_PATH, "err", file_base + ".txt")
                    with open(exp_txt, "w") as txt_file:
                        txt_file.write(f"ERROR: {output_txt}\n{e}")


if __name__ == "__main__":
    create_dataset()
    print("DONE")


                     
