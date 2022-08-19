############ IMPORTS ############
from importlib.resources import path
from itertools import count
from statistics import mean
from tkinter import Y
from cv2 import ellipse2Poly
import torch
from torch._C import Size
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
from utils_ import load_data, utils_flags, test_utils
from absl import app
from absl import flags
# from advertorch.attacks import FABAttack, LinfFABAttack
from autoattack import AutoAttack
from utils_.get_model import get_model
from utils_.adversarial_attack import pgd_linf, pgd_linf_loss_analysis, pgd_linf_grad_analysis
from utils_.miscellaneous import get_bn_config, get_minmax, get_model_name, get_model_path, set_eval_mode
from torch import linalg as LA
from models.noisy_VGG import noisy_VGG
from models.proxy_VGG import proxy_VGG


def get_FAB_acc(run_name, attack, verbose=True):

    # for each iterable batch we create a smaller batch of length equal to the number 
    # of correctly classified samples in that batch and calculate the L-infinity norm
    # between the original sample and the adversarial sample. The idea is that if FAB
    # is not able to find an adversarial examples with smaller perturbation than PGD does,
    # then this might imply that the selected samples are more globally robust, not just 
    # based on gradient measures.

    FLAGS = flags.FLAGS

    print('ATTACK: ', FLAGS.attack)
    print('EPSILON: ', FLAGS.epsilon)
    
    device = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")
    _, test_loader = load_data.get_data()
    where_bn = get_bn_config(run_name)
    model_name = get_model_name(run_name)
    net = get_model(model_name, where_bn)
    PATH_to_model = get_model_path(FLAGS.root_path, model_name, run_name)
    net.load_state_dict(torch.load(PATH_to_model, map_location='cuda:0'))
    net.to(device)
    set_eval_mode(net, FLAGS.use_pop_stats)

    if FLAGS.attack == 'Square':
        print('Using Proxy Model ...')
        net = proxy_VGG(net, 
                        eval_mode=FLAGS.use_pop_stats,
                        device=device,
                        run_name=run_name)

    elif FLAGS.test_noisy and FLAGS.noise_before_PGD:
        print('Adversarial Test With Noise ...')
        net = noisy_VGG(net, 
                        eval_mode=FLAGS.use_pop_stats,
                        noise_variance=FLAGS.noise_variance, 
                        device=device,
                        capacity_=FLAGS.capacity,
                        noise_capacity_constraint=FLAGS.noise_capacity_constraint,
                        run_name=run_name)

    if verbose:
        if not net.training:
            print('MODE: EVAL')
        elif net.training:
            print('MODE: not EVAL')

    # initialize helper vars
    correct = 0
    counter = 0
    correct_clean = 0
    correct_samples = np.zeros((len(test_loader), FLAGS.batch_size))
    min_tensor, max_tensor = get_minmax(test_loader, device)
    all_samples = True  
    loss_analysis = False # TODO: make it FLAGS
    grad_analysis = False # TODO: make it FLAGS

    # First run PGD and identify for each batch the correctly classified samples
    if not all_samples:
        print('Attacking -- PGD')
        for i, data in enumerate(test_loader, 0):
            X, y = data
            X, y = X.to(device), y.to(device)

            # get timestamp
            dir_perturbations = './sample_delta/'
            
            # set directories and decide wheter to run PGD
            if not net.training:
                if len(os.listdir(dir_perturbations + 'eval' + '/')) < len(test_loader):
                    run = True
                else: run = False
            elif net.training:
                if len(os.listdir(dir_perturbations + 'no_eval' + '/')) < len(test_loader):
                    run = True
                else: run = False
            
            # compute adversarial examples
            if run:
                delta = pgd_linf(net, 
                                 X, 
                                 y, 
                                 FLAGS.epsilon, 
                                 max_tensor, 
                                 min_tensor, 
                                 alpha=FLAGS.epsilon/10, 
                                 num_iter=FLAGS.PGD_iterations)

            # save computed deltas
            if not net.training:
                if len(os.listdir(dir_perturbations + 'eval' + '/')) < len(test_loader):
                    torch.save(delta, dir_perturbations + 'eval' + '/' + 'delta_' + str(i) + '.pt')
            else:
                if len(os.listdir(dir_perturbations + 'no_eval' + '/')) < len(test_loader):
                    torch.save(delta, dir_perturbations + 'no_eval' + '/' + 'delta_' + str(i) + '.pt')

            # predict and identify correctly classified samples
            with torch.no_grad():
                if not net.training:
                    delta = torch.load(dir_perturbations + 'eval' + '/' + 'delta_' + str(i) + '.pt')
                else:
                    delta = torch.load(dir_perturbations + 'no_eval' + '/' + 'delta_' + str(i) + '.pt')

                outputs = net(X+delta[0])
            _, predicted = torch.max(outputs.data, 1)

            # one-hot encoded vector for correctly classified samples in the batch 
            correct_samples[i, 0:X.size()[0]] = (predicted == y).cpu().numpy()
            correct_samples.astype(int)
            counter += np.sum(correct_samples[i, :])
                                                                                                                    
    # Run boundary attack on correctly classified samples
    for j, datapoints in enumerate(test_loader, 0):
        X, y = datapoints
        X, y = X.to(device), y.to(device)
        if not all_samples:
            # populate smaller batches
            X_positive = torch.zeros((int(np.sum(correct_samples[j, :])), X.size()[1], X.size()[2], X.size()[3]))
            Y_positive = torch.zeros((int(np.sum(correct_samples[j, :]))))
            cnt = 0
            for idx, x in enumerate(X, 0):
                if correct_samples[j, idx] == 1:
                    X_positive[cnt] = x
                    Y_positive[cnt] = y[idx]
                    cnt+=1
            X_positive = X_positive.to(device)
            Y_positive = Y_positive.to(device)
        else:
            X_positive, Y_positive = X.to(device), y.to(device)
        
        ####################################################################
        if j == 5 and FLAGS.attack == 'Square':
            break
        ####################################################################
        
        # initialize attack
        if attack == 'FAB':
            version = 'custom'
            adversary = AutoAttack(net, 
                                   norm='Linf', 
                                   eps=FLAGS.epsilon, 
                                   version=version,
                                   device=FLAGS.device,
                                   min_tensor=min_tensor, 
                                   max_tensor=max_tensor)
            if version == 'custom':
                adversary.attacks_to_run = ['fab']

        elif attack.find('APGD') != -1:
            version = 'custom'
            adversary = AutoAttack(net, 
                                   norm='Linf', 
                                   eps=FLAGS.epsilon, 
                                   version=version,
                                   device=FLAGS.device,
                                   min_tensor=min_tensor, 
                                   max_tensor=max_tensor)

            if version == 'custom':
                if attack == 'APGD_DLR':
                    adversary.attacks_to_run = ['apgd-dlr']
                elif attack == 'APGD_CE':
                    adversary.attacks_to_run = ['apgd-ce']
        
        elif attack == 'Square':
            version = 'custom'
            adversary = AutoAttack(net, 
                                   norm='Linf', 
                                   eps=FLAGS.epsilon, 
                                   version=version,
                                   device=FLAGS.device,
                                   verbose=True,
                                   min_tensor=min_tensor, 
                                   max_tensor=max_tensor)
            if version == 'custom':
                adversary.attacks_to_run = ['square']

        elif attack == 'PGD':
            if loss_analysis:
                if j < 5:
                    deltas, step_loss = pgd_linf_loss_analysis(net, 
                                                               X, 
                                                               y, 
                                                               FLAGS.epsilon, 
                                                               max_tensor, 
                                                               min_tensor, 
                                                               alpha=FLAGS.epsilon/10, 
                                                               num_iter=FLAGS.PGD_iterations)
                    if not os.path.isdir('./results/VGG/loss_landscape/' + run_name):
                        os.mkdir('./results/VGG/loss_landscape/' + run_name)
                    np.save('./results/VGG/loss_landscape/' + run_name + '/step_loss_' + str(j) + '.npy', step_loss)
                    fig = plt.figure()
                    for k, loss_per_step in enumerate(step_loss):
                        x_axis = np.arange(k*step_loss.shape[1], k*step_loss.shape[1] + step_loss.shape[1], 1)
                        plt.plot(x_axis, loss_per_step)
                        plt.xlabel('iteration')
                        plt.ylabel('loss')
                    plt.savefig('./results/VGG/loss_landscape/' + run_name + '/step_loss_' + str(j) + '.jpg')
                else:
                    deltas = [torch.zeros_like(X)]
                advimg = X+deltas[0]
           
            elif grad_analysis:
                if j < 5:
                    deltas, step_norm = pgd_linf_grad_analysis(net, 
                                                               X, 
                                                               y, 
                                                               FLAGS.epsilon, 
                                                               max_tensor, 
                                                               min_tensor, 
                                                               alpha=FLAGS.epsilon/10, 
                                                               num_iter=FLAGS.PGD_iterations)
                    if not os.path.isdir('./results/VGG/gradient_predictiveness/' + run_name):
                        os.mkdir('./results/VGG/gradient_predictiveness/' + run_name)
                    np.save('./results/VGG/gradient_predictiveness/' + run_name + '/step_norm_' + str(j) + '.npy', step_norm)
                    fig = plt.figure()
                    for k, norm_per_step in enumerate(step_norm):
                        x_axis = np.arange(k*step_norm.shape[1], k*step_norm.shape[1] + step_norm.shape[1], 1)
                        plt.plot(x_axis, norm_per_step)
                        plt.xlabel('iteration')
                        plt.ylabel('gradient norm')
                    plt.savefig('./results/VGG/gradient_predictiveness/' + run_name + '/step_norm_' + str(j) + '.jpg')
                else:
                    deltas = [torch.zeros_like(X)]
                advimg = X+deltas[0]
        
        # employ attack
        if attack != 'PGD':
            advimg = adversary.run_standard_evaluation(X_positive, Y_positive.type(torch.LongTensor).to(device), bs=FLAGS.batch_size)
            if FLAGS.attack == 'Square':
                layers = [0,1,2,5,8,10,12,15]
                layers_keys = ['BN_' + str(t) for t in layers]
                legend_labels = ['Training Variance', 'Test Variance']

                fig_var, axs_var = plt.subplots(nrows=2, ncols=4, sharey=False, figsize=(13,7))
                axs_var = axs_var.ravel()
                for h, layer_key in enumerate(layers_keys):
                    test_var = net.test_variance[layer_key]
                    running_var = net.get_running_variance()[layer_key]
                    dict_to_plot = {'Training Variance': running_var, 'Test Variance': test_var}
                    axs_var[h].sns.displot(data=dict_to_plot, kind="kde", legend=False)

                plt.suptitle('Comparison of Training and Test (Adversarial) Channel Variance')
                fig_var.text(0.45, 0.03, 'Channel Variance', va='center', fontsize=14)
                fig_var.text(0.02, 0.5, 'Density', va='center', rotation='vertical', fontsize=14)
                legend = fig_var.legend(legend_labels, ncol=len(legend_labels),loc="upper center", bbox_to_anchor=[0.5, 0.92])
                frame = legend.get_frame()
                frame.set_color('white')
                frame.set_edgecolor('red')
                fig_var.tight_layout(pad=2, rect=[0.03, 0.05, 1, 0.95])
                root_to_save = './results/'
                path_out = root_to_save + run_name.split('_')[0] + '/' 
                path_out += 'no_eval' + '/' 
                path_out += str(FLAGS.attack) + '/'
                path_out += 'statistics_comaprison' + '/'
                if not os.path.isdir(path_out): os.mkdir(path_out)
                path_out += run_name +'/'
                if not os.path.isdir(path_out): os.mkdir(path_out)
                path_out += 'batch_' + str(j) + '/'
                if not os.path.isdir(path_out): os.mkdir(path_out)
                path_out += 'epsilon_' + str(FLAGS.epsilon).replace('.', '') + '/'
                if not os.path.isdir(path_out): os.mkdir(path_out)
                fig_var.savefig(path_out + 'variance_comaprison.jpg')

                fig_mean, axs_mean = plt.subplots(nrows=2, ncols=4, sharey=False, figsize=(13,7))
                axs_mean = axs_mean.ravel()
                legend_labels = ['Training Mean', 'Test Mean']
                for h, layer_key in enumerate(layers_keys):
                    test_mean = net.test_mean[layer_key]
                    running_mean = net.get_running_mean()[layer_key]
                    dict_to_plot = {'Training Mean': test_mean, 'Test Mean': running_mean}
                    axs_var[h].sns.displot(data=dict_to_plot, kind="kde", legend=False)

                plt.suptitle('Comparison of Training and Test (Adversarial) Channel Mean')
                fig_mean.text(0.45, 0.03, 'Channel Mean', va='center', fontsize=14)
                fig_mean.text(0.02, 0.5, 'Density', va='center', rotation='vertical', fontsize=14)
                legend = fig_mean.legend(legend_labels, ncol=len(legend_labels),loc="upper center", bbox_to_anchor=[0.5, 0.92])
                frame = legend.get_frame()
                frame.set_color('white')
                frame.set_edgecolor('red')
                fig_mean.tight_layout(pad=2, rect=[0.03, 0.05, 1, 0.95])
                fig_mean.savefig(path_out + 'mean_comaprison.jpg')

        with torch.no_grad():
            if FLAGS.test_noisy and not FLAGS.noise_before_PGD:
                print('Adversarial Test With Noise ...')
                net = noisy_VGG(net, 
                                eval_mode=FLAGS.use_pop_stats,
                                noise_variance=FLAGS.noise_variance, 
                                device=device,
                                capacity_=FLAGS.capacity,
                                noise_capacity_constraint=FLAGS.noise_capacity_constraint,
                                run_name=run_name)
            outputs = net(advimg)
            outputs_clean = net(X_positive)
        
        _, predicted_clean = torch.max(outputs_clean.data, 1)
        _, predicted = torch.max(outputs.data, 1)
        
        correct_clean += (predicted_clean == Y_positive).sum().item()

        correct += (torch.logical_and(predicted == Y_positive, predicted_clean == Y_positive)).sum().item()
        counter += y.size(0)

        # calcluate norm if necessary
        if attack == 'FAB':
            dist = []
            misclassified_dist = []
            dist_ = []
            misclassified_dist_ = []

            pred_vector = (predicted == Y_positive)

            for h, _ in enumerate(pred_vector):

                np_X = X_positive[h, :, :, :].cpu().numpy()
                # plt.imshow(np.transpose(np_X, (1, 2, 0)))
                # plt.savefig('original_sample.png')

                adv_sample = advimg[h,:, :, :]
                np_adv_sample = adv_sample.cpu().numpy()
                # plt.imshow(np.transpose(np_adv_sample, (1, 2, 0)))
                # plt.savefig('adversarial_example.png')

                # calculated l-infinity distance between original and adversarial image
                if pred_vector[h]:
                    dist.append(torch.abs((X_positive[h, :, :, :] - advimg[h, :, :, :]).view(x.size(0), -1)).max(dim=1)[0])
                    diff = (X_positive[h, :, :, :] - advimg[h, :, :, :]).view(x.size(0), -1)
                    flat_diff = torch.transpose(diff, 0, 1)
                    flat_diff = torch.reshape(flat_diff, (-1,))
                    dist_.append(LA.norm(flat_diff, float('inf')))
                else:
                    misclassified_dist.append(torch.abs((X_positive[h, :, :, :] - advimg[h, :, :, :]).view(x.size(0), -1)).max(dim=1)[0])
                    diff = (X_positive[h, :, :, :] - advimg[h, :, :, :]).view(x.size(0), -1)
                    flat_diff = torch.transpose(diff, 0, 1)
                    flat_diff = torch.reshape(flat_diff, (-1,))
                    misclassified_dist_.append(LA.norm(flat_diff, float('inf')))
            
            if len(dist) > 0:
                mean_dist = sum(dist)/len(dist)
                mean_dist_ = sum(dist_)/len(dist_)

            mean_misclassified_dist = sum(misclassified_dist)/len(misclassified_dist)
            mean_misclassified_dist_ = sum(misclassified_dist_)/len(misclassified_dist_)

            # print("CORRECTLY CLASSIFIED - Mean L-Infinity distance: ", mean_dist, mean_dist_)
            print("IN-CORRECTLY CLASSIFIED - Mean L-Infinity distance: ", mean_misclassified_dist, mean_misclassified_dist_)

    print('ACCURACY: ', correct/counter)

    return correct/counter
