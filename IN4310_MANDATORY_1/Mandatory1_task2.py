import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
#import time
import os
import copy

import scipy.linalg as la
from PIL import Image


#Method for fetching the features with hooks and storing the values in the activation dictionary
activation = {}
def getActivation(name):
# the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

feature_maps = ["relu", "maxpool", "layer1", "layer4", "avgpool"]

#Takes in a dataloader and calculates and returns the feature map values.
def Calc_feature_maps(data):
    #Attaching the hooks
    h1 = model.relu.register_forward_hook(getActivation('relu'))
    h2 = model.maxpool.register_forward_hook(getActivation('maxpool'))
    h3 = model.layer1.register_forward_hook(getActivation('layer1'))
    h4 = model.layer4.register_forward_hook(getActivation('layer4'))
    h5 = model.avgpool.register_forward_hook(getActivation('avgpool'))

    relu_list = []
    maxpool_list = []
    layer1_list = []
    layer4_list = []
    avgpool_list = []

    #For loop for fetching the feature maps and storing them
    for i in range(13): #13*16 = 208, so we use 208 samples
        inputs, classes = next(iter(data))
        
        out = model(inputs)
        relu_list.append(activation["relu"])
        maxpool_list.append(activation["maxpool"])
        layer1_list.append(activation["layer1"])
        layer4_list.append(activation["layer4"])
        avgpool_list.append(activation["avgpool"])
        

    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()
    h5.remove()


    feature_maps_sum = {}
    total_list = {"relu": relu_list, "maxpool":maxpool_list, "layer1": layer1_list, "layer4": layer4_list, "avgpool":avgpool_list}

    #summing over the feature maps
    for feature in feature_maps:
        sum = 0
        for images in total_list[feature]:
            less = np.count_nonzero(images <= 0) 
            bigger = np.count_nonzero(images > 0)      
            sum += less/(less+bigger) #percentage of non-positive values
        sum /= 13
        feature_maps_sum[feature] = sum  
    print(feature_maps_sum)  #For task 2
    return feature_maps_sum, total_list


#Task 3
#Method for finding eigenvalues using a list of feature maps 
def find_eigenvalues(list):

    batches = list[0].shape[0]*len(list)
    channels = list[0].shape[1]
    mean_arr = np.zeros((batches, channels))


    idx = -1
    for batch in list: #13
        for image in batch: #16 elements
            idx += 1
            for i,channel in enumerate(image):
                mean = np.mean(np.array(channel))
                mean_arr[idx][i] = mean 



    covar_mat = np.zeros((channels, channels))
    expected_mean = 0
    for i in range(batches):
        expected_mean += mean_arr[i,:]

    expected_mean /= batches
    expected_mean = expected_mean.reshape(-1,1)
    expected_mean_T = np.transpose(expected_mean)



    for i in range(batches):
        f = mean_arr[i,:]
        f = f.reshape(-1,1)
        covar_mat += f*np.transpose(f) - expected_mean*expected_mean_T
        

    covar_mat /= batches


    eigen_values, _ = np.linalg.eig(covar_mat)
    eigen_values = np.sort(eigen_values)
    #I choose to plot the first 64 since 3 of the feature maps have only 64 channels
    return eigen_values

#Method for calculating and plotting the eigenvalues for each of our chosen features.
def plot_eigenvalues(savename, data, title):
    feature_maps_sum, total_list = Calc_feature_maps(data)

    x_arr = np.linspace(0,63,64)
    ls = ['-', '--', '-.', '-', "--"]
    for i,map in enumerate(feature_maps):
        list = total_list[map]
        eigen_values = find_eigenvalues(list)
        plt.plot(x_arr, eigen_values[-64:], label=map, linestyle=ls[i])

    plt.legend()
    plt.ylabel("eigenvalue")
    plt.title(f"eigenvalues sorted for all feature maps for {title}")
    plt.yscale("log")
    plt.savefig(f"Figures/{savename}")
    plt.show()

if __name__ == "__main__":
    path = "data_split_larger"

    config = {
            'batch_size': 16,
            'use_cuda': True if torch.cuda.is_available() else False,      #True=use Nvidia GPU | False use CPU
            'log_interval': 20,     #How often to display (batch) loss during training
            'epochs': 20,           #Number of epochs
            'learningRate': 0.001
            }

    device = "cpu" #using cpu since the calculations are pretty small.

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]),
        'cifar': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(path, x), data_transforms[x])
                    for x in ['train', 'test', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config['batch_size'],
                                                shuffle=True, num_workers=0)
                for x in ['train', 'test', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'val']}
    class_names = image_datasets['train'].classes

    model = torch.load("Models/model_large_full.pt") #Loading the model
    model = model.to(device)

    #finding eigenvalues and plotting them for our own dataset
    plot_eigenvalues("Eigenvalues_own_data_train",dataloaders['train'], "own data train")
    plot_eigenvalues("Eigenvalues_own_data_val",dataloaders['val'], "own data val")


    #now we want to to the same for cifar
    cifar_path = "" #Asumes cifar folder is extracted to cifar-10-batches-py and lies in same folder as this python file
    cifar_data = datasets.CIFAR10(root=cifar_path, train=False, download=False, transform=data_transforms['cifar'])
    cifar_dataloader = torch.utils.data.DataLoader(cifar_data, batch_size=16, shuffle=True)

    plot_eigenvalues("Eigenvalues_cifar_data",cifar_dataloader, "Cifar Data")

    #Same for imagnet data set   
    imagenet_path = "ILSVRC2012_img_val/"

    image_datasets = []
    for filename in os.listdir(imagenet_path):
        image_datasets.append(filename)


    class data_gen(torch.utils.data.Dataset):
        def __init__(self, files, path):
            
            self.files = files
            self.path = path
            
        def __getitem__(self, I):
            file_path = os.path.join(self.path + self.files[I])
            image = Image.open(file_path).convert('RGB')
            image = np.resize(image.getdata(), (3,224,224))
            image_tensor = torch.tensor(image, device="cpu", dtype=torch.float)
            y_label = 0

            return (image_tensor, y_label)
        
        def __len__(self):
            return len(self.files)


    imagenet_data = data_gen(image_datasets, imagenet_path)
    imagenet_dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=16, shuffle=True)

    plot_eigenvalues("Eigenvalues_imagenet_data",imagenet_dataloader, "Imagenet Data")




