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

from zmq import proxy
from utils_ import test_utils, utils_flags, load_data
from utils_.get_model import get_model
from absl import app
from absl import flags
from functions.get_entropy import get_layer_entropy


def test(run_name, standard=True, adversarial=False, get_features=False):

    if get_features:
        standard = False

    if adversarial:
        standard = False    

    FLAGS = flags.FLAGS

    outputs = []
    
    # choose device
    device = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")

    # load dataset
    _, test_loader = load_data.get_data()
    
    # retrive model details from run_name
    model_name = run_name.split('_')[0]
    if run_name.split('_')[1] == 'no':
        if model_name.find('ResNet')!= -1:
            where_bn = [0,0,0,0]
        else:
            where_bn = [0,0,0,0,0]
    elif run_name.split('_')[1] == 'bn':
        if model_name.find('ResNet')!= -1:
            where_bn = [1,1,1,1]
        else:
            where_bn = [1,1,1,1,1]
    else:
        where_bn_idx = int(run_name.split('_')[1])
        if model_name.find('ResNet')!= -1:
            where_bn = [i*0 for i in range(4)]
        elif model_name.find('VGG')!= -1:
            where_bn = [i*0 for i in range(5)]
        where_bn[where_bn_idx] = 1

    # re initialize model to apply saved weights
    net = get_model(model_name, where_bn)

    # set model path
    PATH_to_model = FLAGS.root_path + '/models/' + model_name + '/' + run_name + '.pth'
    
    # test model (standard)
    if standard:
        test_acc = test_utils.test(net, 
                                   PATH_to_model, 
                                   test_loader, 
                                   device, 
                                   inject_noise=FLAGS.test_noisy, 
                                   noise_variance=FLAGS.noise_variance)
        outputs.append(test_acc)
    
    # get features of specific layers in the model
    if get_features:
        layer_outputs = test_utils.get_layer_output(net, 
                                                    PATH_to_model, 
                                                    test_loader, 
                                                    device, 
                                                    get_adversarial=adversarial,
                                                    attack=FLAGS.attack, 
                                                    epsilon=FLAGS.epsilon, 
                                                    num_iter=FLAGS.PGD_iterations)

        layer_entropy = get_layer_entropy(layer_outputs)
    
        if adversarial:
            np.save('./entropy_analysis/adversarial_entropies/' + run_name + '_layer_entropy.npy', layer_entropy)
        else:
            np.save('./entropy_analysis/entropies/' + run_name + '_layer_entropy.npy', layer_entropy)

        # reset adversarial to default to prevent from entering next if statement
        adversarial = False

    # test model (adversarial)
    if adversarial:
        print('Adversarial attack used: ', FLAGS.attack)
        print('Epsilon Budget: ', FLAGS.epsilon)

        PATH_to_deltas = FLAGS.root_path + '/deltas/'
        adv_test_acc = test_utils.adversarial_test(net, 
                                                   PATH_to_model, 
                                                   model_name, 
                                                   run_name, 
                                                   test_loader, 
                                                   PATH_to_deltas, 
                                                   device, 
                                                   attack=FLAGS.attack, 
                                                   epsilon=FLAGS.epsilon, 
                                                   num_iter=FLAGS.PGD_iterations,
                                                   use_pop_stats=FLAGS.use_pop_stats,
                                                   inject_noise=FLAGS.test_noisy, 
                                                   noise_variance=FLAGS.noise_variance, 
                                                   no_eval_clean=FLAGS.no_eval_clean)
        
        outputs.append(adv_test_acc)
        
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs