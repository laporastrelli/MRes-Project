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

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import utils_flags, load_data, test_utils
from absl import app
from absl import flags

from utils.get_model import get_model

'''
    Given a trained model we want to:
    - generate adversarial examples
    - test each model on adversarial examples
    - carry out cross-mode testing
'''

def adversarial_test(index):
    
    FLAGS = flags.FLAGS

    # list to store the results to append to df
    to_df = []

    # set dt_string (in this script indicated as "run_name") as input index
    dt_string = index

    # choose device
    device = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")

    # load dataset
    _, test_loader = load_data.get_data()
            
    # generate adversarial exmaples and test on them
    modes = ['', 'bn']

    # setting filename for saving deltas in the proper folder
    filename = FLAGS.model_name + '_' + dt_string

    # setting path for saving logs
    path_to_logs = './logs/adv/' + FLAGS.model_name + '/'

    for mode in modes:
        # retieve model
        model_name = FLAGS.model_name + mode
        print('MODEL NAME: ', model_name)
        
        PATH_to_model = FLAGS.root_path + '/models/' + model_name + '/' + model_name + '_' + dt_string + '.pth'
        PATH_to_deltas = FLAGS.root_path + '/deltas/'

        if mode == '': batch_norm=False
        else: batch_norm=True

        net = get_model(FLAGS.model_name, batch_norm)

        adv_test_acc = test_utils.adversarial_test(net, 
                                                PATH_to_model, 
                                                model_name, 
                                                filename, 
                                                test_loader, 
                                                PATH_to_deltas, 
                                                device, 
                                                attack=FLAGS.attack, 
                                                epsilon=FLAGS.epsilon)
        # save results    
        if isinstance(adv_test_acc, list):
            np.save(path_to_logs + mode + '_' + dt_string + '.npy', np.array(adv_test_acc))

        elif isinstance(adv_test_acc, float) or (isinstance(adv_test_acc, list) and len(adv_test_acc)==1):
            f = open(path_to_logs + filename + ".txt", "a")
            f.write(FLAGS.attack)
            f.write("Adversarial Test Accuracy " + mode + " : " + str(adv_test_acc) + "\n")

        # write results to list
        to_df.append(adv_test_acc)
    
    ########################################################################################################################

    # cross-mode test (BN on Standard)
    adv_test_acc = test_utils.cross_model_testing(FLAGS.model_name + '_' + dt_string, 
                                                'BN_on_STD', 
                                                FLAGS.root_path, 
                                                test_loader, 
                                                device, 
                                                attack=FLAGS.attack, 
                                                epsilon=FLAGS.epsilon)

    if isinstance(adv_test_acc, float):
        # write results to file
        f = open(path_to_logs + filename + ".txt", "a")
        f.write("Adversarial Test Accuracy (BN on Standard): " + str(adv_test_acc) + "\n")
        f.close()  
    else:
        np.save(path_to_logs + 'BN_on_STD_' + dt_string + '.npy', np.array(adv_test_acc))

    # write results to list
    to_df.append(adv_test_acc)

    # cross-mode test (Standard on BN)
    adv_test_acc = test_utils.cross_model_testing(FLAGS.model_name + '_' + dt_string, 
                                                'STD_ON_BN', 
                                                FLAGS.root_path, 
                                                test_loader, 
                                                device, 
                                                attack=FLAGS.attack, 
                                                epsilon=FLAGS.epsilon)
    if isinstance(adv_test_acc, float):
        # write results to file 
        f = open(path_to_logs + filename + ".txt", "a")
        f.write("Adversarial Test Accuracy (Standard on BN): " + str(adv_test_acc) + "\n")
        f.close() 
    else:
        np.save(path_to_logs + 'STD_on BN' + dt_string + '.npy', np.array(adv_test_acc))

    # write results to list
    to_df.append(adv_test_acc)