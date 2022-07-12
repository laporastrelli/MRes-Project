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


def test(run_name,
         standard=False, 
         adversarial=False, 
         get_features=False, 
         get_saliency_map=False, 
         capacity_calculation=False, 
         channel_transfer=False, 
         frequency_analysis=False, 
         IB_noise_calculation=False, 
         test_frequency=False,
         parametric_frequency=False):

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
    
    if test_frequency:
        test_acc = test_utils.test_frequency(net, 
                                            PATH_to_model, 
                                            test_loader, 
                                            device, 
                                            run_name=run_name,
                                            eval_mode=FLAGS.use_pop_stats, 
                                            which_frequency=FLAGS.which_frequency,
                                            frequency_radius=int(FLAGS.frequency_radius))
        outputs.append(test_acc)
    
    # get saliency map of largest logit with respect to a given layer activation
    elif get_saliency_map:
        test_utils.saliency_map(net, 
                                PATH_to_model, 
                                test_loader, 
                                device, 
                                run_name=run_name,
                                eval_mode=FLAGS.use_pop_stats)
    
    # get features of specific layers in the model
    elif get_features:
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
    elif adversarial:
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
                                                   random_resizing=FLAGS.random_resizing, 
                                                   scaled_noise=FLAGS.scaled_noise, 
                                                   scaled_noise_norm=FLAGS.scaled_noise_norm, 
                                                   scaled_noise_total = FLAGS.scaled_noise_total,
                                                   get_similarity=FLAGS.get_similarity,
                                                   relative_accuracy=FLAGS.relative_accuracy,
                                                   get_max_indexes=FLAGS.get_max_indexes, 
                                                   channel_transfer=FLAGS.channel_transfer, 
                                                   n_channels=FLAGS.n_channels_transfer,
                                                   transfer_mode=FLAGS.transfer_mode)
        
        outputs.append(adv_test_acc)
    
    # calculate capacity mode
    elif capacity_calculation:
        _ = test_utils.calculate_capacity(net, 
                                         PATH_to_model,
                                         run_name, 
                                         test_loader, 
                                         device, 
                                         attack=FLAGS.attack, 
                                         epsilon=FLAGS.epsilon,
                                         num_iter=FLAGS.PGD_iterations,
                                         use_pop_stats=FLAGS.use_pop_stats, 
                                         capacity_regularization=FLAGS.capacity_regularization,
                                         regularization_mode=FLAGS.regularization_mode,
                                         beta=FLAGS.beta)

    # channel transfer mode
    elif channel_transfer:
        _ = test_utils.channel_transfer(net, 
                                        PATH_to_model,
                                        run_name, 
                                        test_loader, 
                                        device,  
                                        epsilon_list=FLAGS.epsilon_in,
                                        num_iter=FLAGS.PGD_iterations,
                                        attack=FLAGS.attack,
                                        channel_transfer=FLAGS.channel_transfer,
                                        transfer_mode=FLAGS.transfer_mode,
                                        layer_to_test=FLAGS.layer_to_test,
                                        use_pop_stats=FLAGS.use_pop_stats)
    
    # frequency analysis
    elif frequency_analysis:
        test_utils.get_frequency_images(net,
                                        PATH_to_model,
                                        test_loader,
                                        device,
                                        run_name,
                                        eval_mode=FLAGS.use_pop_stats, 
                                        layer_to_test=FLAGS.layer_to_test, 
                                        frequency_radius=int(FLAGS.frequency_radius))

    elif IB_noise_calculation:
        test_utils.IB_noise_calculation(net, 
                                        PATH_to_model, 
                                        test_loader,
                                        device,
                                        run_name, 
                                        eval_mode=FLAGS.use_pop_stats, 
                                        layer_to_test=FLAGS.layer_to_test,
                                        capacity_regularization=FLAGS.capacity_regularization)
    
    elif parametric_frequency:
        test_utils.get_parametric_frequency(net, 
                                            PATH_to_model, 
                                            test_loader,
                                            device,
                                            run_name, 
                                            eval_mode=FLAGS.use_pop_stats, 
                                            layer_to_test=FLAGS.layer_to_test,
                                            get_parametric_frequency_MSE_only=FLAGS.parametric_frequency_MSE,
                                            get_parametric_frequency_MSE_CE=FLAGS.parametric_frequency_MSE_CE,
                                            capacity_regularization=FLAGS.capacity_regularization)

    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs