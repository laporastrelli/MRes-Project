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


def main(argv):
    
    del argv

    FLAGS = flags.FLAGS

    # open pandas results log file
    df = pd.read_pickle('./logs/results.pkl')

    # choose device
    device = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")

    # load dataset
    train_loader, test_loader = load_data.get_data()
    
    # create run name
    model_name = FLAGS.model_name
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    run_name = model_name + '_' + dt_string

    # create logs filename
    path_to_logs = 'logs/adv/' + model_name + '/'
    if not os.path.isdir(path_to_logs):
        os.mkdir(path_to_logs)
    logs_filename = path_to_logs + run_name  + ".txt"
    f = open(logs_filename, "a")

    # create run name (Tensorboard)
    writer = SummaryWriter('./runs/' + run_name)

    # set paths
    PATH_to_deltas = FLAGS.root_path + '/deltas/'

    ############ STANDARD ############
    FLAGS.batch_norm = False 
    f.write("Batch Normalization: False" + "\n")
    PATH_to_model = FLAGS.root_path + '/models/' + model_name + '/' + run_name + '.pth'

    # train model (if requested)
    if FLAGS.train:   
        net = get_model.get_model(model_name, FLAGS.batch_norm)
    
        # train model
        net = train_utils.train(train_loader, test_loader, net.to(device), FLAGS.device, model_name, FLAGS.n_epochs, FLAGS.batch_norm, writer)

        # save model
        if not os.path.isdir(FLAGS.root_path + '/models/' + model_name):
            os.mkdir(FLAGS.root_path + '/models/' + model_name)

        torch.save(net.state_dict(), PATH_to_model)

    # test model (if requested)
    if FLAGS.test:
        # re initialize model to apply saved weights
        net = get_model.get_model(model_name, FLAGS.batch_norm)
           
        # test model
        test_acc = test_utils.test(net, PATH_to_model, test_loader, device)

        # write results
        f.write("Test Accuracy: " + str(test_acc) + "\n")  

    # adversarial test (if requested)
    if FLAGS.adversarial_test:
        # test model (adversarial)
        adv_test_acc = test_utils.adversarial_test(net, PATH_to_model, model_name, run_name, test_loader, PATH_to_deltas, device)

        # save results    
        if isinstance(adv_test_acc, list):
            np.save(np.array(adv_test_acc), path_to_logs + dt_string + '.npy')
        else:
            f.write("Adversarial Test Accuracy: " + str(adv_test_acc) + "\n")
    
    f.close()

    ############ BATCH-NORMALIZATION ############
    FLAGS.batch_norm = True
    model_name = FLAGS.model_name + 'bn'
    f = open(logs_filename, "a")
    f.write("Batch Normalization: True" + "\n")

    # create run name
    run_name = model_name + '_' + dt_string
    writer = SummaryWriter('./runs/' + run_name)

    PATH_to_model = FLAGS.root_path + '/models/' + model_name + '/' + run_name + '.pth'

    if FLAGS.train:
        # retrieve model
        net = get_model.get_model(FLAGS.model_name, FLAGS.batch_norm)

        # train model
        net = train_utils.train(train_loader, test_loader, net.to(device), FLAGS.device, FLAGS.model_name, FLAGS.n_epochs, FLAGS.batch_norm, writer)

        # save model
        if not os.path.isdir(FLAGS.root_path + '/models/' + model_name):
            os.mkdir(FLAGS.root_path + '/models/' + model_name)
        
        torch.save(net.state_dict(), PATH_to_model)

    if FLAGS.test:
        # re initialize model to apply saved weights
        net = get_model.get_model(FLAGS.model_name, FLAGS.batch_norm)
            
        # test model
        test_acc = test_utils.test(net, PATH_to_model, test_loader, device)

        # save results
        f.write("Test Accuracy: " + str(test_acc) +"\n")

    if FLAGS.adversarial_test:
        # adversarial test
        adv_test_acc = test_utils.adversarial_test(net, PATH_to_model, model_name, run_name, test_loader, PATH_to_deltas, device)

        # save results    
        if isinstance(adv_test_acc, list):
            np.save(np.array(adv_test_acc), path_to_logs + dt_string + '.npy')
        else:
            f.write("Adversarial Test Accuracy: " + str(adv_test_acc) + "\n")
    
    f.close()

if __name__ == '__main__':
    app.run(main)