import os
import shutil 
from PIL import Image
import numpy as np 

from sklearn.model_selection import train_test_split
import torch 
from torchvision import datasets, models, transforms


origin = 'mandatory1_data'
target = 'mandatory1_data_smaller'

#method for making a smaller data set
def smaller_data(origin, target, make_folder=False):
    folders = os.listdir(f"{origin}")
    if make_folder:
        os.mkdir(target)
    for folder_name in folders:
        if make_folder:
            os.mkdir(f"{target}/{folder_name}")
        all_files = os.listdir(f"{origin}/{folder_name}")
        numb = 200
        for i,files in enumerate(all_files):
            shutil.copy(f"{origin}/{folder_name}/{files}", f"{target}/{folder_name}/{files}")
            if i == numb:
                break

try: 
    smaller_data(origin, target, make_folder=True)
except:
    print("mandatory1_data_smaller already exists")

#Method for splitting data into train, test and validation data
def split_data(data):
    ratio = 1000/17034 
    length = len(data)

    data_, data_test = train_test_split(data, test_size = int(length*3*ratio))
    data_train, data_val = train_test_split(data_, test_size = int(length*2*ratio))
    #print(len(data_train), len(data_test), len(data_val))
    return data_train, data_test, data_val

#Method for making a train, test and validation data set
#If the folder has been made before, make_folder should be False, else make it True
def train_test_val(path, target, make_folders = False):
    labels = os.listdir(path)

    train_data = {}
    test_data = {}
    val_data = {}

    if make_folders:
        for folder in ["train", "test", "val"]:
                new_folder = os.path.join(target, folder)
                os.mkdir(new_folder)
    

    for label in labels:
        if make_folders:
            for folder in ["train", "test", "val"]:
                new_folder = os.path.join(target, folder+"/"+label)
                #new_folder = os.listdir(f"{target}/{folder}/{label}")
                os.mkdir(new_folder)

        all_files = os.listdir(f"{path}/{label}")
        train_data[label], test_data[label], val_data[label] = split_data(all_files)
    
    for label in train_data:
        for file in train_data[label]:
            shutil.copy(f"{path}/{label}/{file}", f"{target}/train/{label}/{file}")

    for label in test_data:
        for file in test_data[label]:
            shutil.copy(f"{path}/{label}/{file}", f"{target}/test/{label}/{file}")
    
    for label in val_data:
        for file in val_data[label]:
            shutil.copy(f"{path}/{label}/{file}", f"{target}/val/{label}/{file}")
    


try:
    train_test_val("mandatory1_data_smaller", "data_split_smaller", True)
except:
    print("data_split_smaller already exists, delete folder if you want to test again or run with make_folder=False")
#train_test_val("mandatory1_data_smaller", "data_split_smaller")


try:
    train_test_val("mandatory1_data", "data_split_larger", True)
except:
    print("data_split_larger already exists, delete folder if you want to test again or run with make_folder=False")

#Method for verifying disjointness
def verify_disjoint(path):
    train = f"{path}/train"
    test = f"{path}/test"
    val = f"{path}/val"

    for classes in os.listdir(train):
        for train_item in os.listdir(f"{train}/{classes}"):
            for test_item in os.listdir(f"{test}/{classes}"):
                if train_item == test_item:
                    print("arent disjoint")
                    return 0
            for val_item in os.listdir(f"{val}/{classes}"):
                if train_item == val_item:
                    print("arent disjoint")
                    return 0
        for test_item in os.listdir(f"{test}/{classes}"):
            for val_item in os.listdir(f"{val}/{classes}"):
                if val_item == test_item:
                    print("arent disjoint")
                    return 0
    print(f"data set {path} is disjoint")

verify_disjoint("data_split_smaller")
verify_disjoint("data_split_larger")
