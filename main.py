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
from utils import utils_flags, load_data, test_utils
from absl import app
from absl import flags
from .train_and_test import train_and_test
from .adversarial_test import adversarial_test

from utils.get_model import get_model

colums = ['Model', 'Dataset', 'Batch-Normalization', 
          'Training Mode', 'Test Accuracy', 'Attack',
          'Epsilon-Budget', 'Adversarial Test Accuracy', 'Transfer adv. Test Accuracy']


def main(argv):
    
    del argv

    FLAGS = flags.FLAGS

    # open pandas results log file
    df = pd.read_pickle('./logs/results.pkl')

    # train and test the model and get results
    index, test_acc, bn_test_accuracy = train_and_test()

    # carry out adversarial training as well as cross-BN adversarial transferability test
    adv_acc, bn_adv_acc, bn_on_std, std_on_bn = adversarial_test(index)

    # create dictonaries to be inserted into Pandas Dataframe
    if FLAGS.where_bn == [0,0,0,0]:
        bn_string = 'No'
    else:
        bn_string = 'Yes - ' + str(FLAGS.where_bn)
    
    # NO BN
    df_dict = {
    colums[0] : FLAGS.model_name,
    colums[1] : FLAGS.dataset,
    colums[2] : bn_string, 
    colums[3] : FLAGS.training_mode, 
    colums[4] : test_acc,
    colums[5] : FLAGS.attack, 
    colums[6] : FLAGS.epsilon, 
    colums[7] : adv_acc, 
    colums[8] : std_on_bn
    }

    df_dict = {
    colums[0] : FLAGS.model_name,
    colums[1] : FLAGS.dataset,
    colums[2] : 'Yes',
    colums[3] : FLAGS.training_mode, 
    colums[4] : test_acc,
    colums[5] : FLAGS.attack, 
    colums[6] : FLAGS.epsilon, 
    colums[7] : adv_acc, 
    colums[8] : std_on_bn
    }
    

if __name__ == '__main__':
    app.run(main)