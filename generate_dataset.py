""" This script takes care of dataset generation from custom data.
"""
from sys import argv
import get_motion_estimate as gme
import os
from PIL import Image
import numpy as np
import hashlib
from datetime import datetime

def main(train_video_path, test_video_path, output_dir):
    """Take training video path and test video path and output the dataset to output_dir

    Arguments:
        train_video_path {[type]} -- [description]
        test_video_path {[type]} -- [description]
        output_dir {[type]} -- [description]
    """
    # Read training and testing video data
    train_videodata = gme.load_video_data(train_video_path)
    test_videodata = gme.load_video_data(test_video_path)
    train_motion_xy = gme.get_motion_xy(train_videodata)
    test_motion_xy = gme.get_motion_xy(test_videodata)

    # Create sub directory for training and testing data
    directory_names = ["train", "train_mv", "eval", "eval_mv"]
    for dir_name in directory_names:
        dir_path = output_dir+"/"+dir_name
        try:
            os.mkdir(dir_path)
            print("Directory" , dir_path ,  "Created ") 
        except FileExistsError:
            print("Directory" , dir_path ,  "already exists")
    
    ### Video Data

    train_name = datetime.now().strftime("%Y%m%d&H%M%S")+"_train"
    train_dir_path = output_dir+"/train/"
    # Populate training video directories
    for f in range(train_videodata.shape[0]):
        seq = str(f).zfill(4)
        im = Image.fromarray(train_videodata[f,:,:,:])
        im.save(train_dir_path+train_name+"_"+seq)

    test_name = datetime.now().strftime("%Y%m%d&H%M%S")+"_test"
    test_dir_path = output_dir+"/test/"
    # Populate testing video directories
    for f in range(test_videodata.shape[0]):
        seq = str(f).zfill(4)
        im = Image.fromarray(test_videodata[f,:,:,:])
        im.save(test_dir_path+test_name+"_"+seq)
        



if __name__ == "__main__":
    if len(argv) != 4:
        print("Arguments: train_video_path test_video_path output_dir")
        exit(1)
    main(argv[1], argv[2], argv[3])