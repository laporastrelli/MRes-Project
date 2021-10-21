############ IMPORTS ############
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time 

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from models import CNN1, CNN2, CNN3
from utils import train_utils, test_utils

# select device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device name: ", device)

############ PREPARING INPUTS ############
## !IMPORTANT!: select model tag to be one of the following: CNN1, CNN2, CNN3
partioned_dataset = input("Enter 0 for partioned dataset, any other number for standard dataset:")
model_tag = input("Enter model tag: ")

# create the name of the model based on input and date
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
run_name = model_tag + '_' + dt_string

if int(partioned_dataset) == 0:
    partion_dataset = True
    run_name = 'partioned_' + run_name

# create a folder in memeory where all model Tensorbaord session are stored
os.mkdir("./runs/" + run_name)
# initialize summary writer for Tensorboard
writer = SummaryWriter('./runs/' + run_name)

############ LOADING DATA ############
# data parameters and path
data_path = "/data2/users/lr4617/data/"
download = False
batch_size = 64

# prevents from downloading it if dataset already downloaded
if len(os.listdir(data_path)) == 0:
    download = True

# performs transforms on data
transform_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))])

# get data
cifar10_train = datasets.CIFAR10(data_path, train=True, download=download, transform=transform_data)
cifar10_test = datasets.CIFAR10(data_path, train=False, download=download, transform=transform_data)

# create loaders
## it is important NOT to shuffle the test dataset since the adversarial variation 
## delta are going to be saved in memory in the same order as the test samples are. 
train_loader = DataLoader(cifar10_train, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size = batch_size, shuffle=False)

############ TRAINING/TESTING ITERATION ############

models_path = './models/'
num_Of_models = len(os.listdir(models_path))
partition_size = int(len(cifar10_train)/num_Of_models)

for i in range(num_Of_models):
    # load partioned dataset
    indices = range(i*partition_size, (i+1)*partition_size)
    train_sampler = torch.utils.data.SubsetRandomSampler(indices)
    train_loader = DataLoader(cifar10_train, batch_size = batch_size, sampler=indices)

    # select model based on iteration
    if i == 0:
        net = CNN1.CNN1(device)
    
    elif i == 1:
        net = CNN2.CNN2(device)
        model_tag='CNN2'
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        run_name = model_tag + '_' + dt_string
        if partion_dataset == True:
            run_name = 'partioned_' + run_name

    elif i == 2:
        net = CNN3.CNN3(device)
        model_tag='CNN3'
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        run_name = model_tag + '_' + dt_string
        if partion_dataset == True:
            run_name = 'partioned_' + run_name


    # train model
    net = train_utils.train(train_loader, net.cuda(), device)

    # save trained model 
    PATH = './../../../../../data2/users/lr4617/partioned_models/' + run_name + '.pth'
    dir_path = './../../../../../data2/users/lr4617/partioned_models/'
    if os.path.isdir(dir_path) == False:
        os.mkdir(dir_path)

    saved_models = os.listdir(dir_path)
    save = True
    for saved_model in saved_models:
        # if saved_model.find(run_name.split('_')[1])!= -1:
        if saved_model.find(model_tag)!= -1:
            print("found")
            save = False
            break

    if save:
        torch.save(net, PATH)

    # test model
    test_utils.test(PATH, test_loader, device)

    # adversarial attack and eval
    deltas_path = './deltas_partioned/'
    test_utils.adversarial_test(PATH, model_tag, test_loader, deltas_path, device)

    # cross model testing
    test_utils.cross_model_testing(test_loader, dir_path, deltas_path, model_tag, device)

