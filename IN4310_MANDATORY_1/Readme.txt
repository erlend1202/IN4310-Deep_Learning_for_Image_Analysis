To compile, first run makeDATA.py to split the data into train, test and val. 
For this, it is assumed that you have mandatory1_data in same folder as the python file.

The code for task 1 is in mandatory1_task1python.py. The paths are at the top, if you want to test for a faster and smaller
dataset, simply comment out the path for data_set_larger and uncomment for data_set_smaller, else it will use the entire dataset. 
Here, you need to have compiled makeDATA.py first. 

The code for task 2 and 3 can be found in mandatory1_task2.py. Here, the paths to the cifar-10 data set assumes that 
you have cifar-10-batches-py in same folder as the python file. The paths to the imagenet dataset assumes that you have the 
data located in ILSVRC2012_img_val, where the folder should also be located in same folder as the python file.

The model i trained can be found in the foler Models as model_large_full.pt, the other is the state dict stored.
My outprints from task1 can be found in out1.txt