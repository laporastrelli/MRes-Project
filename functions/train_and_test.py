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
from models import ResNet, ResNet_bn
from utils import train_utils, test_utils, get_model, utils_flags, load_data
from absl import app
from absl import flags

colums = ['Model', 'Dataset', 'Batch-Normalization', 
          'Training Mode', 'Test Accuracy', 'Attack',
          'Epsilon-Budget', 'Adversarial Test Accuracy', 'Transfer adv. Test Accuracy']


def train_and_test(model_name_in, where_bn, attack=None):

    FLAGS = flags.FLAGS

    # list to store the results to append to df
    to_df = []

    # choose device
    device = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")

    # load dataset
    train_loader, test_loader = load_data.get_data()

    # set input paramters
    FLAGS.model_name = model_name_in
    FLAGS.where_bn = where_bn
    
    # create run name
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")    
    if 1 not in FLAGS.where_bn:
        bn = ''
        batch_norm = False
    else:
        bn = 'bn'
        batch_norm = True

    run_name = FLAGS.model_name + '_' + dt_string

    # create run name (Tensorboard)
    writer = SummaryWriter('./runs/' + run_name)

    # set paths
    PATH_to_deltas = FLAGS.root_path + '/deltas/'
    PATH_to_model = FLAGS.root_path + '/models/' + FLAGS.model_name + bn + '/' + run_name + '.pth'

    # train model 
    if FLAGS.train: 
        to_df.append(dt_string)  
        net = get_model.get_model(FLAGS.model_name, FLAGS.where_bn)
    
        # train model
        net = train_utils.train(train_loader, 
                                test_loader, 
                                net.cuda(), 
                                FLAGS.device, 
                                FLAGS.model_name, 
                                batch_norm, 
                                writer, 
                                optimum=FLAGS.optimum)

        # save model
        if not os.path.isdir(FLAGS.root_path + '/models/' + FLAGS.model_name + bn):
            os.mkdir(FLAGS.root_path + '/models/' + FLAGS.model_name + bn)

        torch.save(net.state_dict(), PATH_to_model)

    # test model (if requested)
    if FLAGS.test:
        # re initialize model to apply saved weights
        net = get_model.get_model(FLAGS.model_name, FLAGS.batch_norm)
           
        # test model
        test_acc = test_utils.test(net, PATH_to_model, test_loader, device)

        # write results to list
        to_df.append(test_acc)

    # adversarial test (if requested)
    if FLAGS.adversarial_test:
        # test model (adversarial)
        adv_test_acc = test_utils.adversarial_test(net, 
                                                  PATH_to_model, 
                                                  FLAGS.model_name, 
                                                  run_name, 
                                                  test_loader, 
                                                  PATH_to_deltas, 
                                                  device, 
                                                  attack=FLAGS.attack, 
                                                  epsilon=FLAGS.epsilon)
        
        # write results to list
        to_df.append(adv_test_acc)

    return to_df
