import json
import os
import shutil
import argparse
import sys
from sklearn.model_selection import train_test_split
from os import path

class_list = ["raspberry pi"]

def labelme_json_to_yolov7_seg(labelme_dataset_dir):

    json_files = [pos_json for pos_json in os.listdir(labelme_dataset_dir) if pos_json.endswith('.json')]
    for i in range (len(json_files)):

        with open(labelme_dataset_dir+"/"+json_files[i]) as f:
            data = json.load(f)
        width = data["imageWidth"]
        height = data["imageHeight"]
        shapes = data["shapes"]
        text_file_name = labelme_dataset_dir+"/"+json_files[i]
        text_file_name = text_file_name.replace("json","txt")
        text_file = open(text_file_name, 'w')
        for i in range (len(shapes)):
            class_name = shapes[i]["label"]
            class_id = class_list.index(str(class_name))
            points = data["shapes"][i]["points"]
            normalize_point_list = []
            normalize_point_list.append(class_id)
            for i in range (len(points)):
                normalize_x = points[i][0]/width
                normalize_y = points[i][1]/height
                normalize_point_list.append(normalize_x)
                normalize_point_list.append(normalize_y)
            for i in range (len(normalize_point_list)):
                text_file.write(str(normalize_point_list[i])+" ")
            text_file.write("\n")

def train_val_split(labelme_dataset_dir,ouput_dataset_dir,image_name,train_val_ratio=0.2):
    # input dataset dir setting
    json_files = [pos_json for pos_json in os.listdir(labelme_dataset_dir) if pos_json.endswith('.json')]

    # output dataset dir setting
    train_folder = os.path.join(ouput_dataset_dir, 'train/')
    val_folder = os.path.join(ouput_dataset_dir, 'val/')
    train_image_folder = os.path.join(train_folder, 'images/')
    train_label_folder = os.path.join(train_folder, 'labels/')
    train_json_folder = os.path.join(train_folder, 'json/')
    val_image_folder = os.path.join(val_folder, 'images/')
    val_label_folder = os.path.join(val_folder, 'labels/')
    val_json_folder = os.path.join(val_folder, 'json/')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    if not os.path.exists(train_image_folder):
        os.makedirs(train_image_folder)
    if not os.path.exists(train_label_folder):
        os.makedirs(train_label_folder)
    if not os.path.exists(train_json_folder):
        os.makedirs(train_json_folder)
    if not os.path.exists(val_image_folder):
        os.makedirs(val_image_folder)
    if not os.path.exists(val_label_folder):
        os.makedirs(val_label_folder)
    if not os.path.exists(val_json_folder):
        os.makedirs(val_json_folder)

    # make copy of json,txt,jpg to output dataset directory
    train_idxs, val_idxs = train_test_split(range(len(json_files)), test_size=train_val_ratio)
    for i in range (len(train_idxs)):
        if path.exists(labelme_dataset_dir+"/"+image_name+str(train_idxs[i])+".jpg"):
            shutil.copy2(labelme_dataset_dir+"/"+image_name+str(train_idxs[i])+".jpg", train_image_folder) # move train txt
        if path.exists(labelme_dataset_dir+"/"+image_name+str(train_idxs[i])+".txt"):
            shutil.copy2(labelme_dataset_dir+"/"+image_name+str(train_idxs[i])+".txt", train_label_folder) # move train image
        if path.exists(labelme_dataset_dir+"/"+image_name+str(train_idxs[i])+".json"):
            shutil.copy2(labelme_dataset_dir+"/"+image_name+str(train_idxs[i])+".json", train_json_folder) # move train json
    for i in range (len(val_idxs)):
        if path.exists(labelme_dataset_dir+"/"+image_name+str(val_idxs[i])+".jpg"):
            shutil.copy2(labelme_dataset_dir+"/"+image_name+str(val_idxs[i])+".jpg", val_image_folder) # move val txt
        if path.exists(labelme_dataset_dir+"/"+image_name+str(val_idxs[i])+".txt"):
            shutil.copy2(labelme_dataset_dir+"/"+image_name+str(val_idxs[i])+".txt", val_label_folder) # move val image
        if path.exists(labelme_dataset_dir+"/"+image_name+str(val_idxs[i])+".json"):
            shutil.copy2(labelme_dataset_dir+"/"+image_name+str(val_idxs[i])+".json", val_json_folder) # move val json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labelme_dataset_dir',type=str, default=None,
                        help='Please input the path of the labelme files (jpg and json)')
    parser.add_argument('--train_val_ratio',type=float, default=None,
                        help='Please input the validation dataset size, for example 0.1')
    parser.add_argument('--ouput_dataset_dir',type=str, default=None,
                        help='Please input desired processed data directory.')
    parser.add_argument('--image_name',type=str, default=None,
                        help='Please input image name without ids.')
    args = parser.parse_args(sys.argv[1:])
    labelme_json_to_yolov7_seg(args.labelme_dataset_dir)
    train_val_split(args.labelme_dataset_dir,args.ouput_dataset_dir,args.image_name,train_val_ratio=args.train_val_ratio)

# python3 labelme2yolov7seg.py --labelme_dataset_dir ./data --train_val_ratio 0.2 --ouput_dataset_dir ./yolov7_seg_grass_dataset --image_name frame
