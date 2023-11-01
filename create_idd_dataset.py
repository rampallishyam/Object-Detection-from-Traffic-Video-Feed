import argparse
import os
import random

from utils.image_transform import *
from utils.xml2yolo import *

BASE_PATH = "D:/Anup/MICP"
# Paths to Orginal IDD Dataset
imgs_path = os.path.join(BASE_PATH, "_/_old/idd-detection/IDD_Detection/JPEGImages")
labels_path = os.path.join(BASE_PATH, "_/_old/idd-detection/IDD_Detection/Annotations") 

# Paths to New IDD Dataset
train_path = os.path.join(BASE_PATH, "dataset/train")
val_path = os.path.join(BASE_PATH, "dataset/val")

def create_dataset():
    parser = argparse.ArgumentParser(description='create_dataset')
    parser.add_argument('--imgs_path', default=imgs_path, help='img_path_old')
    parser.add_argument('--labels_path', default=labels_path, help='labels_path_old')
    parser.add_argument('--split', default=0.9, help='split')
    args = parser.parse_args()

    primary_folders = [name for name in os.listdir(args.imgs_path) if os.path.isdir(os.path.join(args.imgs_path, name))]

    for primary_folder in primary_folders:
        primary_folder_path = os.path.join(args.imgs_path, primary_folder)
        secondary_folders = [name for name in os.listdir(primary_folder_path) if os.path.isdir(os.path.join(primary_folder_path, name))]

        for secondary_folder in secondary_folders:
            secondary_folder_path = os.path.join(primary_folder_path, secondary_folder)

            image_files = [name for name in os.listdir(secondary_folder_path) if name.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            train_index = int(args.split*len(image_files))
            random.shuffle(image_files)

            # training dataset 
            for file_name in image_files[:train_index]:
                file_path = os.path.join(secondary_folder_path, file_name)
                if os.path.exists(os.path.join(train_path, "images")) and file_name in os.listdir(os.path.join(train_path, "images")):
                     pass
                elif os.path.exists(os.path.join(val_path, "images")) and file_name in os.listdir(os.path.join(val_path, "images")):
                    pass
                else:
                    #img transform
                    file_name, file_ext = os.path.splitext(file_name)
                    transformed_image = img_transform(file_path,mode='train')
                    

                    #labels
                    xml_file = os.path.join(args.labels_path,primary_folder,secondary_folder, file_name+'.xml')
                    output_txt = os.path.join(train_path,"labels",file_name+".txt")
                    try:
                        xml2yolo(xml_file, output_txt)
                        transformed_image.save(os.path.join(train_path,"images", file_name + file_ext))
                    except:
                        exp_txt = os.path.join(BASE_PATH,"err",file_name+".txt")
                        with open(exp_txt, 'w') as txt_file:
                             txt_file.write('ERROR: '+ output_txt)


            # validation dataset 
            for file_name in image_files[train_index:]:
                file_path = os.path.join(secondary_folder_path, file_name)
                if os.path.exists(os.path.join(train_path, "images")) and file_name in os.listdir(os.path.join(train_path, "images")):
                     pass
                elif os.path.exists(os.path.join(val_path, "images")) and file_name in os.listdir(os.path.join(val_path, "images")):
                    pass
                else:
                    #img transform
                    file_name, file_ext = os.path.splitext(file_name)
                    transformed_image = img_transform(file_path,mode='val')
                    

                    #labels
                    xml_file = os.path.join(args.labels_path,primary_folder,secondary_folder, file_name+'.xml')
                    output_txt = os.path.join(val_path,"labels",file_name+".txt")
                    try:
                        xml2yolo(xml_file, output_txt)
                        transformed_image.save(os.path.join(val_path,"images", file_name + file_ext))

                    except:
                        exp_txt = os.path.join(BASE_PATH,"err",file_name+".txt")
                        with open(exp_txt, 'w') as txt_file:
                             txt_file.write('ERROR: '+ output_txt)

                

if __name__ == '__main__':
    create_dataset()
    print("DONE")


                     
