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
import torchvision.models as models
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils_ import train_utils, test_utils, get_model, utils_flags, load_data
from absl import app
from absl import flags

def train(model_name, where_bn):

    FLAGS = flags.FLAGS

    # choose device
    device = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")

    # load dataset
    train_loader, test_loader = load_data.get_data()

    # set input paramters
    FLAGS.model_name = model_name
    FLAGS.where_bn = where_bn

    # create run name
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
 
    # based on input bn locations detect whther we are using it or not
    if sum(FLAGS.where_bn) == 0:
        batch_norm = False
        bn = 'no_bn'
    elif sum(FLAGS.where_bn) >1:
        bn = 'bn'
        batch_norm = True
    else:
        bn = str(where_bn.index(1))
        batch_norm = True

    # create run name to contain info about where the BN layer is 
    run_name = FLAGS.model_name + '_' + bn + '_' + dt_string
    if FLAGS.model_name == 'ResNet50':
        if FLAGS.version == 2:
            run_name = FLAGS.model_name + 'v' + str(FLAGS.version) + '_' + bn + '_' + dt_string
    writer = SummaryWriter('./runs/' + run_name)
    PATH_to_model = FLAGS.root_path + '/models/' + FLAGS.model_name + '/' + run_name + '.pth'
    
    net = get_model.get_model(FLAGS.model_name, FLAGS.where_bn, run_name=run_name, train_mode=True)

    # train model
    return_s = train_utils.train(train_loader, 
                                test_loader, 
                                net.to(device), 
                                FLAGS.device, 
                                FLAGS.model_name, 
                                batch_norm, 
                                writer, 
                                run_name)
        
    if FLAGS.save_to_wandb:
        net, run = return_s
    else:
        net = return_s

    # save model
    if not os.path.isdir(FLAGS.root_path + '/models/' + FLAGS.model_name):
        os.mkdir(FLAGS.root_path + '/models/' + FLAGS.model_name)

    torch.save(net.state_dict(), PATH_to_model)

    if FLAGS.save_to_wandb:
        return run_name, run
    else:
        return run_name