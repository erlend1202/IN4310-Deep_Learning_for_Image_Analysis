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

#path = "data_split_smaller"
path = "data_split_larger"

config = {
          'batch_size': 16,
          'use_cuda': True if torch.cuda.is_available() else False,      #True=use Nvidia GPU | False use CPU
          'log_interval': 20,     #How often to display (batch) loss during training
          'epochs': 20,           #Number of epochs
          'learningRate': 0.001
         }


device = torch.device("cuda" if config['use_cuda'] else "cpu")


#####
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((150,150)),
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


#####
model = models.resnet18(pretrained=True)    


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr = config['learningRate'])


# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


#####
def loss_fn(prediction, labels):
    """Returns softmax cross entropy loss."""
    loss = nn.functional.cross_entropy(input=prediction, target=labels)
    return loss

#####
from sklearn import metrics
#Method for training and running through the model, taken from week 3 exercises
def run_epoch(model, epoch, data_loader, optimizer, is_training, config):
    """
    Args:
        model        (obj): The neural network model
        epoch        (int): The current epoch
        data_loader  (obj): A pytorch data loader "torch.utils.data.DataLoader"
        optimizer    (obj): A pytorch optimizer "torch.optim"
        is_training (bool): Whether to use train (update) the model/weights or not. 
        config      (dict): Configuration parameters

    Intermediate:
        totalLoss: (float): The accumulated loss from all batches. 
                            Hint: Should be a numpy scalar and not a pytorch scalar

    Returns:
        loss_avg         (float): The average loss of the dataset
        accuracy         (float): The average accuracy of the dataset
        confusion_matrix (float): A 6x6 matrix
    """
    
    if is_training==True: 
        model.train()
    else:
        model.eval()

    total_loss       = 0 
    correct          = 0 
    confusion_matrix = np.zeros(shape=(6,6))
    labels_list      = [0,1,2,3,4,5]

    for batch_idx, data_batch in enumerate(data_loader):
        if config['use_cuda'] == True:
            images = data_batch[0].to('cuda') # send data to GPU
            labels = data_batch[1].to('cuda') # send data to GPU
        else:
            images = data_batch[0]
            labels = data_batch[1]

        if not is_training:
            with torch.no_grad():
                prediction = model(images)
                loss        = loss_fn(prediction, labels)
                total_loss += loss.item()  
                
        elif is_training:
            prediction = model(images)
            loss        = loss_fn(prediction, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            

        # Update the number of correct classifications and the confusion matrix
        predicted_label  = prediction.max(1, keepdim=True)[1][:,0]
        correct          += predicted_label.eq(labels).cpu().sum().numpy()
        confusion_matrix += metrics.confusion_matrix(labels.cpu().numpy(), predicted_label.cpu().numpy(), labels=labels_list)

        # Print statistics
        #batchSize = len(labels)
        if batch_idx % config['log_interval'] == 0:
            print(f'Epoch={epoch} | {(batch_idx+1)/len(data_loader)*100:.2f}% | loss = {loss:.5f}', flush=True)

    loss_avg         = total_loss / len(data_loader)
    accuracy         = correct / len(data_loader.dataset)
    confusion_matrix = confusion_matrix / len(data_loader.dataset)

    return loss_avg, accuracy, confusion_matrix


# train the model and validating it
train_loss = np.zeros(shape=config['epochs'])
train_acc  = np.zeros(shape=config['epochs'])
val_loss   = np.zeros(shape=config['epochs'])
val_acc    = np.zeros(shape=config['epochs'])
train_confusion_matrix = np.zeros(shape=(6,6,config['epochs']))
val_confusion_matrix   = np.zeros(shape=(6,6,config['epochs']))

for epoch in range(config['epochs']):
    train_loss[epoch], train_acc[epoch], train_confusion_matrix[:,:,epoch] = \
                               run_epoch(model, epoch, dataloaders["train"], optimizer, is_training=True, config=config)

    val_loss[epoch], val_acc[epoch], val_confusion_matrix[:,:,epoch]     = \
                               run_epoch(model, epoch, dataloaders["val"], optimizer, is_training=False, config=config)
    



# Plot the loss and the accuracy in training and val
#plt.figure()
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
ax = plt.subplot(2, 1, 1)
# plt.subplots_adjust(hspace=2)
ax.plot(train_loss, 'b', label='train loss')
ax.plot(val_loss, 'r', label='val loss')
ax.grid()
plt.ylabel('Loss', fontsize=18)
plt.xlabel('Epochs', fontsize=18)
ax.legend(loc='upper right', fontsize=16)

ax = plt.subplot(2, 1, 2)
plt.subplots_adjust(hspace=0.4)
ax.plot(train_acc, 'b', label='train accuracy')
ax.plot(val_acc, 'r', label='val accuracy')
ax.grid()
plt.ylabel('Accuracy', fontsize=18)
plt.xlabel('Epochs', fontsize=18)
val_acc_max = np.max(val_acc)
val_acc_max_ind = np.argmax(val_acc)
plt.axvline(x=val_acc_max_ind, color='g', linestyle='--', label='Highest val accuracy')
plt.title('Highest val accuracy = %0.1f %%' % (val_acc_max*100), fontsize=16)
ax.legend(loc='lower right', fontsize=16)
plt.ion()

plt.savefig("accuracy_val.png")

ind = np.argmax(train_acc)
class_accuracy = train_confusion_matrix[:,:,ind]
for ii in range(len(class_names)):
    acc = train_confusion_matrix[ii,ii,ind] / np.sum(train_confusion_matrix[ii,:,ind])
    print(f'Accuracy of {str(class_names[ii]).ljust(15)}: {acc*100:.01f}%', flush=True)


#Average precision scores
ind = np.argmax(val_acc)
class_accuracy = val_confusion_matrix[:,:,ind]
for ii in range(len(class_names)):
    acc = val_confusion_matrix[ii,ii,ind] / np.sum(val_confusion_matrix[ii,:,ind])
    print(f'Accuracy of {str(class_names[ii]).ljust(15)}: {acc*100:.01f}%', flush=True)


torch.save(model.state_dict(), "model_large.pt")
torch.save(model, "model_large_full.pt")


# test the model
test_loss   = np.zeros(shape=config['epochs'])
test_acc    = np.zeros(shape=config['epochs'])
test_confusion_matrix   = np.zeros(shape=(6,6,config['epochs']))

for epoch in range(config['epochs']):
    test_loss[epoch], test_acc[epoch], test_confusion_matrix[:,:,epoch]     = \
                               run_epoch(model, epoch, dataloaders["test"], optimizer, is_training=False, config=config)
    

#mean accuracy
ind = np.argmax(test_acc)
class_accuracy = test_confusion_matrix[:,:,ind]
total_acc = 0
for ii in range(len(class_names)):
    acc = test_confusion_matrix[ii,ii,ind] / np.sum(test_confusion_matrix[ii,:,ind])
    print(f'Accuracy of {str(class_names[ii]).ljust(15)}: {acc*100:.01f}%', flush=True)
    total_acc += acc 

print(f'Accuracy of {str("all combined").ljust(15)}: {total_acc*100/6:.01f}%', flush=True)


temp_loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'test', 'val']}



"""
Modified the run_epoch method so it stores and returns arrays that we want.
"""
def get_predictions(model, data_loader, is_training=False, config=config):
    """
    Args:
        model        (obj): The neural network model
        data_loader  (obj): A pytorch data loader "torch.utils.data.DataLoader"
        is_training (bool): Whether to use train (update) the model/weights or not. 
        config      (dict): Configuration parameters


    Returns: pred_arr, label_arr, idx_arr, pred2_arr, im_arr
        pred_arr          (np.array): Contains the prediction value for each image.
        pred2_arr         (np.array): Contains the prediction class for each image.
        label_arr         (np.array): Contains the correct class for each image.
        idx_arr           (np.array): Contains the corresponding index for each image.
        im_arr            (np.array): Contains each image.
    """
    
    if is_training==True: 
        model.train()
    else:
        model.eval()

    pred_arr = np.zeros(len(data_loader))
    label_arr = np.zeros(len(data_loader)) 
    idx_arr = np.zeros(len(data_loader))
    pred2_arr = np.zeros(len(data_loader))
    im_arr = []

    for batch_idx, data_batch in enumerate(data_loader):
        if config['use_cuda'] == True:
            images = data_batch[0].to('cuda') # send data to GPU
            labels = data_batch[1].to('cuda') # send data to GPU
        else:
            images = data_batch[0]
            labels = data_batch[1]

        with torch.no_grad():
            prediction = model(images)

            im_arr.append(images)
            pred_arr[batch_idx] = torch.max(prediction)
            pred2_arr[batch_idx] = torch.argmax(prediction)
            label_arr[batch_idx] = labels
            idx_arr[batch_idx] = batch_idx

    return pred_arr, label_arr, idx_arr, pred2_arr, im_arr


"""
My attempt at finding bot and top 10 for each class. For some reason it did not work properly.
pred_arr_classes = [[] for _ in range(len(class_names))]
idx_arr_classes = [[] for _ in range(len(class_names))]


for i, label in enumerate(label_arr):
    label = int(label)
    pred_arr_classes[label].append(pred_arr[i])
    idx_arr_classes[label].append(idx_arr[i])


for i, label in enumerate(class_names):
    pred_arr_classes[i], idx_arr_classes[i] = zip(*sorted(zip(pred_arr_classes[i], idx_arr_classes[i])))

"""


pred_arr, label_arr, idx_arr, pred2_arr, im_arr = get_predictions(model, temp_loader["test"])

pred_arr, idx_arr = zip(*sorted(zip(pred_arr, idx_arr))) #Sorting idx_arr to match the predictions when sorted.
top_10 = idx_arr[len(idx_arr)-10: len(idx_arr)] #fetching top 10
bottom_10 = idx_arr[0:10] #fetching bottom 10


#Calculating top 10 labeled images
rows = 2
columns = 5
fig = plt.figure(figsize=(20, 7))

for i,idx in enumerate(top_10):
    idx = int(idx)
    im = im_arr[idx]
    im = im.cpu()
    im = torch.squeeze(im,0)
    im = im.numpy().transpose((1,2,0))
    im = np.clip(im, 0, 1)
    fig.add_subplot(rows, columns, i+1)
    plt.title(f"{class_names[int(label_arr[idx])]}, {class_names[int(pred2_arr[idx])]}")
    plt.axis('off')
    plt.imshow(im)

plt.savefig("top_10")


#bottom 10 labeled images
rows = 2
columns = 5
fig = plt.figure(figsize=(20, 7))

for i,idx in enumerate(bottom_10):
    idx = int(idx)
    im = im_arr[idx]
    im = im.cpu()
    im = torch.squeeze(im,0)
    im = im.numpy().transpose((1,2,0))
    im = np.clip(im, 0, 1)
    fig.add_subplot(rows, columns, i+1)
    plt.title(f"{class_names[int(label_arr[idx])]}, {class_names[int(pred2_arr[idx])]}")
    plt.axis('off')
    plt.imshow(im)

plt.savefig("bottom_10")

#Average precision scores
from sklearn.metrics import average_precision_score 
for i, label in enumerate(class_names):
    temp_pred = [1 if j == i else 0 for j in pred2_arr]
    temp_label = [1 if j == i else 0 for j in label_arr]

    AP = average_precision_score(temp_pred, temp_label) 
    print(f'Average precision of {str(label).ljust(15)}: {AP*100:.01f}%', flush=True)

