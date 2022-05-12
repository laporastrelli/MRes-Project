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
from utils_.miscellaneous import get_model_path, get_model_specs
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
    device = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")
    _, test_loader = load_data.get_data()
    model_name, where_bn = get_model_specs(run_name)
    net = get_model(model_name, where_bn)
    PATH_to_model = get_model_path(FLAGS.root_path, model_name, run_name)
    
    # test model (standard)
    if standard:
        test_acc = test_utils.test(net, 
                                   PATH_to_model, 
                                   test_loader, 
                                   device, 
                                   run_name=run_name,
                                   eval_mode=FLAGS.use_pop_stats,
                                   inject_noise=FLAGS.test_noisy, 
                                   noise_variance=FLAGS.noise_variance, 
                                   random_resizing=FLAGS.random_resizing,
                                   noise_capacity_constraint=FLAGS.noise_capacity_constraint,
                                   capacity=FLAGS.capacity,
                                   get_logits=FLAGS.get_logits)
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
                                                   capacity=FLAGS.capacity,
                                                   noise_capacity_constraint=FLAGS.noise_capacity_constraint,
                                                   capacity_calculation=FLAGS.capacity_calculation,
                                                   use_pop_stats=FLAGS.use_pop_stats,
                                                   inject_noise=FLAGS.test_noisy, 
                                                   noise_variance=FLAGS.noise_variance, 
                                                   no_eval_clean=FLAGS.no_eval_clean,
                                                   random_resizing=FLAGS.random_resizing)
        
        outputs.append(adv_test_acc)
        
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs