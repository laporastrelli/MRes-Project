from audioop import avg
from pdb import runcall
from sqlite3 import Timestamp
from tkinter.tix import Tree
from bleach import clean
from cv2 import mean
import torch.nn as nn
import torch
import os
import csv
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time

from torchvision import datasets, transforms
from torchvision.utils import save_image
from zmq import EVENT_CLOSE_FAILED
from utils_.adversarial_attack import fgsm, pgd_linf, pgd_linf_capacity, pgd_linf_total_capacity
from utils_ import get_model
from utils_.miscellaneous import get_minmax, get_path2delta, get_bn_int_from_name, CKA, cosine_similarity, get_model_name
from utils_.log_utils import get_csv_path, get_csv_keys

from advertorch.attacks import LinfPGDAttack
from autoattack import AutoAttack

from models.proxy_VGG import proxy_VGG
from models.proxy_VGG2 import proxy_VGG2
from models.proxy_VGG3 import proxy_VGG3
from models.noisy_VGG import noisy_VGG
from models.test_VGG import test_VGG
from models.proxy_ResNet import proxy_ResNet
from models.noisy_ResNet import noisy_ResNet


# TODO: ##################################################################################
# - Remember to change max and min of clamp in PGD (adversarial_attack.py) when using SVHN
# - BEWARE OF net.eval ---> CRUCIAL
# - Remember to save delta tensors if needed (it is now disabled)
# TODO: ##################################################################################

def model_setup(net, 
                model_path, 
                device,
                use_pop_stats, 
                run_name, 
                noise_variance, 
                regularization_mode='', 
                prune_mode='', 
                prune_percentage=1.0, 
                layer_to_test=0):

    net.load_state_dict(torch.load(model_path))
    net.to(device)

    ################## MODEL SELECTION ##################
    if run_name.find('VGG') != -1: 
        net = proxy_VGG(net, 
                        eval_mode=use_pop_stats,
                        device=device,
                        run_name=run_name,
                        noise_variance=noise_variance, 
                        regularization_mode=regularization_mode, 
                        prune_mode=prune_mode, 
                        prune_percentage=prune_percentage, 
                        layer_to_test=layer_to_test)
    elif run_name.find('ResNet') != -1: 
        net = proxy_ResNet(net, 
                            eval_mode=use_pop_stats,
                            device=device,
                            run_name=run_name,
                            noise_variance=noise_variance, 
                            regularization_mode=regularization_mode, 
                            prune_mode=prune_mode, 
                            prune_percentage=prune_percentage, 
                            layer_to_test=layer_to_test)
    #####################################################

    ################## EVAL MODE ##################
    if use_pop_stats: net.eval()
    # setting dropout to eval mode (in VGG)
    if run_name.find('VGG')!= -1: net.classifier.eval()
    ###############################################

    return net

def test(net, 
        model_path, 
        test_loader, 
        device, 
        run_name,
        eval_mode=True,
        inject_noise=False,
        noise_variance=0, 
        random_resizing=False, 
        noise_capacity_constraint=False,
        scaled_noise=False, 
        scaled_noise_norm=False, 
        scaled_noise_total=False,
        scaled_lambda=False,
        noise_first_layer=False,
        noise_not_first_layer=False,
        prune_mode='',
        prune_percentage=1.0,
        layer_to_test=0,
        attenuate_HF=False,
        capacity=0,
        get_logits=False):

    if prune_mode == '':
        # net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        net.to(device)

    if inject_noise:
        if run_name.find('VGG') != -1:
            noisy_net = noisy_VGG(net, 
                                  eval_mode=eval_mode,
                                  noise_variance=noise_variance, 
                                  device=device,
                                  capacity_=capacity,
                                  noise_capacity_constraint=noise_capacity_constraint, 
                                  run_name=run_name, 
                                  scaled_noise=scaled_noise, 
                                  scaled_noise_norm=scaled_noise_norm, 
                                  scaled_noise_total=scaled_noise_total,
                                  scaled_lambda=scaled_lambda,
                                  noise_first_layer=noise_first_layer,
                                  noise_not_first_layer=noise_not_first_layer)
                                  
        elif run_name.find('ResNet') != -1:
            net = noisy_ResNet(net,
                               eval_mode=eval_mode,
                               device=device,
                               run_name=run_name,
                               noise_variance=noise_variance)

        elif run_name.find('ResNet')!=-1:
            noisy_net = noisy_ResNet(net,
                                     eval_mode=eval_mode,
                                     device=device,
                                     run_name=run_name,
                                     noise_variance=noise_variance)

    if attenuate_HF:
        print('Testing Model w/ HF attenuation ...')
        if run_name.find('VGG')!= -1:
            net = proxy_VGG3(net, 
                             eval_mode=eval_mode,
                             device=device,
                             run_name=run_name,
                             noise_variance=0, 
                             attenuate_HF=attenuate_HF,
                             layer_to_test=int(layer_to_test))
    
    if len(prune_mode) > 0:
        net = model_setup(net, 
                          model_path, 
                          device,
                          eval_mode, 
                          run_name, 
                          noise_variance, 
                          prune_mode=prune_mode, 
                          prune_percentage=prune_percentage, 
                          layer_to_test=layer_to_test)

    if eval_mode:
        net.eval()

    correct = 0
    total = 0
    temp = 0

    for i, data in enumerate(test_loader, 0):  
        X, y = data
        X, y = X.to(device), y.to(device)

        if random_resizing:
            print('RANDOM RESIZING')
            size = random.randint(X.shape[2]-2, X.shape[2]-1)
            crop = transforms.RandomResizedCrop(X.shape[2]-1)
            X_crop = crop.forward(X)

            if size == X.shape[2]-2:
                choices = [[1, 1, 1, 1],
                            [2, 0, 2, 0], 
                            [2, 0, 0, 2],
                            [0, 2, 2, 0], 
                            [0, 2, 0, 2]] 
                rand_idx = random.randint(0,4)
                to_pad = choices[rand_idx]

            elif size == X.shape[2]-1:
                to_pad = [0, 0, 0, 0] # l - t - r - b
                while sum(to_pad) < 2:
                    rand_idx_lr = random.choice([0, 1])
                    rand_idx_tb = random.choice([2, 3])
                    rand_pad_side_lr = random.randint(0,1)
                    rand_pad_side_tb = random.randint(0,1)

                    if to_pad[0]+to_pad[1] == 0:
                        to_pad[rand_idx_lr] = rand_pad_side_lr
                    if to_pad[2]+to_pad[3] == 0:
                        to_pad[rand_idx_tb] = rand_pad_side_tb

            pad = torch.nn.ZeroPad2d(tuple(to_pad))
            X = pad(X_crop)
        if inject_noise:
            if run_name.find('VGG')!= -1:
                noisy_net = noisy_VGG(net, 
                                    eval_mode=eval_mode,
                                    noise_variance=noise_variance, 
                                    device=device,
                                    capacity_=capacity,
                                    noise_capacity_constraint=noise_capacity_constraint, 
                                    run_name=run_name)
            elif run_name.find('ResNet')!= -1:
                noisy_net = noisy_ResNet(net,
                                         eval_mode=eval_mode,
                                         device=device,
                                         run_name=run_name,
                                         noise_variance=noise_variance)
            outputs = noisy_net(X)
            temp = outputs
        elif get_logits:
            if i == 10:
                acc = 0
                break

            # utils
            logits_diff_correct = []
            logits_diff_incorrect = []

            # get clean outputs
            outputs = net(X) 
            _, predicted = torch.max(outputs.data, 1)

            print(net.parameters())

            for ii in range(X.size(0)):
                scaled_logits = torch.nn.functional.softmax(outputs[ii].data)
                sorted_logits, _ = torch.sort(scaled_logits, descending=True)
                if predicted[ii] == y[ii] and sorted_logits[0] == float(1):
                    sample = torch.unsqueeze(X[ii], 0)
                    label = torch.unsqueeze(y[ii], 0)
                    loss_before = nn.CrossEntropyLoss()(net(sample)/10 , label)
                    print('LOSS BEFORE: ', loss_before)

                    delta = torch.zeros_like(sample, requires_grad=True)
                    loss_after = nn.CrossEntropyLoss()(net(sample + delta)/10 , label)
                    loss_after.backward()

                    if torch.equal(loss_after, torch.zeros_like(loss_after)):
                        print(torch.max(delta.grad.detach()))
                    print('LOSS AFTER: ', loss_after)

            # run adversarial attack
            epsilon = 0.0392
            min_tensor, max_tensor = get_minmax(test_loader=test_loader, device=device)
            delta = pgd_linf(net, 
                             X, 
                             y, 
                             epsilon, 
                             max_tensor, 
                             min_tensor, 
                             alpha=epsilon/10, 
                             num_iter=40)
            adv_outputs = net(X+delta[0])
            _, adv_predicted = torch.max(adv_outputs.data, 1)
            
            # partition correct and incorrect PGD logits
            for ii in range(X.size(0)):
                scaled_logits = torch.nn.functional.softmax(outputs[ii].data)
                sorted_logits, _ = torch.sort(scaled_logits, descending=True)
                if predicted[ii] == y[ii] and adv_predicted[ii] == y[ii]:
                    logits_diff_correct.append((sorted_logits[0]).cpu().numpy())
                elif predicted[ii] == y[ii] and adv_predicted[ii] != y[ii]:
                    logits_diff_incorrect.append((sorted_logits[0]).cpu().numpy())
            
            if not os.path.isdir('./results/VGG19/VGG_no_eval/logits_analysis/' + run_name):
                os.mkdir('./results/VGG19/VGG_no_eval/logits_analysis/' + run_name)

            np.save('./results/VGG19/VGG_no_eval/logits_analysis/' + run_name + '/logits_diff_correct_' + str(i) + '.npy', np.asarray(logits_diff_correct))
            np.save('./results/VGG19/VGG_no_eval/logits_analysis/' + run_name + '/logits_diff_incorrect_' + str(i) + '.npy', np.asarray(logits_diff_incorrect))
        if not inject_noise:
            outputs = net(X)
            #print('time elapsed for single forward pass: ', timestamp2 - timestamp1)

        _, predicted = torch.max(outputs.data, 1)

        total += y.size(0)
        correct += (predicted == y).sum().item()

    print('Test Accuracy: %d %%' % (100 * correct / total))
       
    acc = correct / total
    return acc

def test_frequency(net, 
                   model_path, 
                   test_loader, 
                   device, 
                   run_name,
                   eval_mode=True, 
                   which_frequency='high', 
                   frequency_radius=4):

    net.load_state_dict(torch.load(model_path))
    net.to(device)

    if eval_mode: net.eval()

    correct = 0
    total = 0

    for i, data in enumerate(test_loader, 0):  
        X, y = data
        X, y = X.to(device), y.to(device)

        # get high and low frequency images for the given batch
        low_f_img, high_f_img = generateDataWithDifferentFrequencies_3Channel(X.cpu().numpy(), frequency_radius)

        # convert images to appropriate format to feed into model
        low_f_img = torch.tensor(low_f_img, dtype=torch.float32, device=device)
        high_f_img = torch.tensor(high_f_img, dtype=torch.float32, device=device)

        if which_frequency == 'high':
            batch = high_f_img
        elif which_frequency == 'low':
            batch = low_f_img

        outputs = net(batch)
        _, predicted = torch.max(outputs.data, 1)

        total += y.size(0)
        correct += (predicted == y).sum().item()

    acc = correct / total
    print('Accuracy: ', acc)
    return acc

def get_layer_output(net, 
        model_path, 
        test_loader,
        device,
        get_adversarial, 
        attack, 
        epsilon, 
        num_iter):

    net.load_state_dict(torch.load(model_path))
    net.to(device)
    # net.eval()

    # TO BE DELETED
    ######################################################### 
    print('TRAINING MODE: ', net.training)
    #########################################################

    proxy_net = proxy_VGG(net)
    test_layer_outputs = []
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0): 
            if i>0:
                break 
            X, y = data
            X, y = X.to(device), y.to(device)

            if get_adversarial:
                if attack == 'PGD':
                    delta = pgd_linf(net, X, y, epsilon, alpha=1e-2, num_iter=num_iter)
                    outputs = proxy_net(X+delta[0])
            else:
                outputs = proxy_net(X)
                
            test_layer_outputs.append(outputs)

    return test_layer_outputs

def calculate_capacity(net, 
                       model_path, 
                       run_name, 
                       test_loader,
                       device, 
                       attack,
                       epsilon, 
                       num_iter,
                       capacity_mode='variance_based',
                       use_pop_stats=True,
                       batch=1,
                       noise_variance=0,
                       capacity_regularization=False,
                       beta=0,
                       regularization_mode='euclidean',
                       save_analysis=True, 
                       get_BN_names=False, 
                       net_analysis=False, 
                       distribution_analysis=False, 
                       index_order_analysis=True):
    
    print('Calculating Capacity')

    # get model
    net = model_setup(net, 
                      model_path, 
                      device,
                      use_pop_stats, 
                      run_name, 
                      noise_variance)

    # getting min and max pixel values to be used in PGD for clamping
    min_tensor, max_tensor = get_minmax(test_loader=test_loader, device=device)
    
    # select which layer you want to calculate the capacity
    lambdas = net.get_bn_parameters()
    layer_key = ['BN_' + str(i) for i in range(len(list(lambdas.keys())))]

    # create root directory for saving content
    if save_analysis:
        path_out = './results/'
        path_out += get_model_name(run_name) + '/'
        if use_pop_stats:
            eval_mode_str = 'eval'
        else:
            eval_mode_str = 'no_eval'
        if capacity_regularization:
            path_out += eval_mode_str + '/' + attack +  '/capacity_regularization/capacity/' + capacity_mode + '/'
        else:
            path_out += eval_mode_str + '/' + attack +  '/capacity/' + capacity_mode + '/'

    # lambda-based capacity calculation
    if capacity_mode == 'lambda_based':
        lambdas = net.get_bn_parameters()
        if get_BN_names:
            net.get_BN_names()
            BN_names = net.BN_names
            np.save('./results/ResNet50/eval/PGD/capacity/bn_names_dict.npy', np.array(BN_names))
        if save_analysis:
            path_out += run_name + '/'
            if not os.path.isdir(path_out): 
                os.mkdir(path_out)
            if capacity_regularization:
                path_out += regularization_mode + '/'
                if not os.path.isdir(path_out): 
                    os.mkdir(path_out)
                path_out += str(beta).replace('.', '') + '/'
                if not os.path.isdir(path_out): 
                    os.mkdir(path_out)
            path_out += 'lambdas.npy'
            if not os.path.isfile(path_out):
                np.save(path_out, lambdas)

    # variance-based capacity calculation
    elif capacity_mode == 'variance_based':
        
        if net_analysis:
            net_change = np.zeros((batch, len(layer_key)))
            for i, data in enumerate(test_loader, 0):
                if i < batch:
                    print('Batch: ', i)
                    X, y = data
                    X, y = X.to(device), y.to(device)

                    net.set_verbose(verbose=True)
                    _, capacities, _ = pgd_linf_capacity(net, 
                                                        X, 
                                                        y, 
                                                        epsilon, 
                                                        max_tensor, 
                                                        min_tensor, 
                                                        alpha=epsilon/10, 
                                                        num_iter=num_iter, 
                                                        layer_key=layer_key)

                    net.set_verbose(verbose=False) 
                    for chn, key_ in enumerate(layer_key):
                        net_change[i, chn] = np.mean(capacities[key_][-1].numpy() - capacities[key_][0].numpy())
            
            mean_net_change = np.mean(net_change, axis=0)
            var_net_change = np.var(net_change, axis=0)

            path_out += 'all_layers/clean_test/global_analysis' + '/' 
            if not os.path.isdir(path_out): os.mkdir(path_out)
            path_out +=  run_name + '/'
            if not os.path.isdir(path_out): os.mkdir(path_out)

            np.save(path_out + '/mean_net_change_' + str(epsilon).replace('.', '') + '.npy', mean_net_change)
            np.save(path_out + '/var_net_change' + str(epsilon).replace('.', '') + '.npy', var_net_change)
        
        elif distribution_analysis:
            for i, data in enumerate(test_loader, 0):

                if i < batch:
                    print('Batch: ', i)
                    X, y = data
                    X, y = X.to(device), y.to(device)

                    net.set_verbose(verbose=True)
                    _, capacities, _ = pgd_linf_capacity(net, 
                                                        X, 
                                                        y, 
                                                        epsilon, 
                                                        max_tensor, 
                                                        min_tensor, 
                                                        alpha=epsilon/10, 
                                                        num_iter=num_iter, 
                                                        layer_key=layer_key)
                    net.set_verbose(verbose=False) 

                    path_to_save = path_out + 'all_layers/clean_test/distribution_analysis' + '/' 
                    if not os.path.isdir(path_to_save): os.mkdir(path_to_save)
                    path_to_save +=  run_name + '/'
                    if not os.path.isdir(path_to_save): os.mkdir(path_to_save)
                    path_to_save +=  'batch_' + str(i) + '/' 
                    if not os.path.isdir(path_to_save): os.mkdir(path_to_save)

                    if run_name.find('VGG') != -1:
                        layers_to_save = [0,1,2,5,8,10,12,15]
                    else:
                        layers_to_save = [0,1,2,10,20,30,39,49]
                    keys_to_save = ['BN_' + str(i) for i in layers_to_save]
                    for chn, key_ in enumerate(layer_key):
                        if key_ in keys_to_save:
                            temp = capacities[key_][-1].numpy() - capacities[key_][0].numpy()

                            path_to_save_file =  path_to_save + key_ + '/' 
                            if not os.path.isdir(path_to_save_file): os.mkdir(path_to_save_file)

                            np.save(path_to_save_file + 'diff_' + str(epsilon).replace('.', '') + '.npy', temp)
        
        elif index_order_analysis:
            use_lambda = True
            if run_name.find('VGG') != -1:
                layers_to_save = [0,1,2,5,8,10,12,15]
            else:
                layers_to_save = [0,1,2,10,20,30,39,49]
            keys_to_save = ['BN_' + str(i) for i in layers_to_save]

            capacity_diff = dict.fromkeys(keys_to_save, [])

            path_out += 'all_layers/clean_test/index_order_analysis' + '/' 
            if not os.path.isdir(path_out): os.mkdir(path_out)
            
            if use_lambda:
                path_out += 'use_lambda' + '/'
                if not os.path.isdir(path_out): os.mkdir(path_out)

            path_out +=  run_name + '/'
            if not os.path.isdir(path_out): os.mkdir(path_out)

            for i, data in enumerate(test_loader, 0):
                if i < batch:
                    print('Batch: ', i)
                    X, y = data
                    X, y = X.to(device), y.to(device)

                    net.set_verbose(verbose=True)
                    _, capacities, _ = pgd_linf_capacity(net, 
                                                        X, 
                                                        y, 
                                                        epsilon, 
                                                        max_tensor, 
                                                        min_tensor, 
                                                        alpha=epsilon/10, 
                                                        num_iter=num_iter, 
                                                        layer_key=layer_key)
                    net.set_verbose(verbose=False) 

                    for chn, key_ in enumerate(layer_key):
                        sorted_idxs = torch.argsort(capacities[key_][0])
                        if use_lambda:
                            sorted_idxs = torch.argsort(net.get_bn_parameters()[key_])
                        final_capacity = capacities[key_][-1]
                        temp = final_capacity[sorted_idxs].numpy()

                        if batch > 1:
                            if key_ in keys_to_save:
                                if i == 0:
                                    capacity_diff[key_] = temp
                                else:
                                    to_add = []
                                    if i == 1: 
                                        exists = [capacity_diff[key_]]
                                    else:
                                        exists = capacity_diff[key_]
                                    for i in range(len(exists) + 1):
                                        if i < len(exists):
                                            to_add.append(exists[i])
                                        else:
                                            to_add.append(temp)
                                    capacity_diff[key_] = to_add
                        else: 
                            if key_ in keys_to_save:
                                temp = final_capacity[sorted_idxs].numpy()
                                capacity_diff[key_] = temp
                
                for layer in list(capacity_diff.keys()):
                    path_to_save = path_out + layer + '/'
                    if not os.path.isdir(path_to_save): os.mkdir(path_to_save)

                    if batch > 1:
                        to_save_mean = np.mean(np.array(capacity_diff[layer]), axis=0)
                        to_save_var = np.var(np.array(capacity_diff[layer]), axis=0)
                    else:
                        to_save_mean = np.array(capacity_diff[layer])

                    np.save(path_to_save + 'diff_mean_eps' + str(epsilon).replace(',', '') + '.npy', to_save_mean)
                    if batch > 1:
                        np.save(path_to_save + 'diff_var_eps' + str(epsilon).replace(',', '') + '.npy', to_save_var)

        else:
            for i, data in enumerate(test_loader, 0):
                if i == batch:
                    X, y = data
                    X, y = X.to(device), y.to(device)
                    break
            net.set_verbose(verbose=True)
            _, capacities, _ = pgd_linf_capacity(net, 
                                                X, 
                                                y, 
                                                epsilon, 
                                                max_tensor, 
                                                min_tensor, 
                                                alpha=epsilon/10, 
                                                num_iter=num_iter, 
                                                layer_key=layer_key)
            net.set_verbose(verbose=False) 
            if save_analysis:
                if len(layer_key)==1:
                    if layer_key[0] == 'BN_0':
                        path_out += 'first_layer/'
                    elif layer_key[0] == 'BN_1':
                        path_out += 'second_layer/'
                else: path_out += 'all_layers/'
                if not os.path.isdir(path_out): os.mkdir(path_out)

                if net.get_noisy_mode():
                    if noise_variance > 0.05:
                        path_out += 'noisy_test/'
                    else:
                        path_out += 'small_noisy_test/'
                else: path_out += 'clean_test/'
                if not os.path.isdir(path_out): os.mkdir(path_out)

                path_out += 'BATCH_' + str(batch) + '/'
                if not os.path.isdir(path_out): os.mkdir(path_out)
                
                for folder_name in layer_key:
                    if not os.path.isdir(path_out + folder_name): os.mkdir(path_out + folder_name)
                    sub_folder_name = str(epsilon).replace('.', '')
                    if not os.path.isdir(path_out + folder_name + '/' + sub_folder_name):
                        os.mkdir(path_out + folder_name + '/' + sub_folder_name)
                    t = 0
                    fig = plt.figure()
                    if len(layer_key)==1:
                        key = layer_key[0]
                    else:
                        key = folder_name
                    for temp in capacities[key]:
                        x_axis = [t]
                        for x_, y_ in zip(x_axis, [temp]):
                            plt.scatter([x_] * len(y_), y_)
                            plt.xlabel('PGD steps')
                            plt.ylabel('Capacity Estimate')
                            if net.get_noisy_mode():
                                noisy_str = 'noisy - ' + r'$\sigma^2=$' + str(noise_variance)
                            else:
                                noisy_str = 'deterministic'
                            if get_bn_int_from_name(run_name) == 100:
                                title_str = get_model_name(run_name) + ' ' + 'FULL-BN' + ' ' + noisy_str
                            else:
                                title_str = get_model_name(run_name) + ' ' + 'BLOCK' + '-' + \
                                            str(get_bn_int_from_name(run_name)-1) + \
                                            ' ' + noisy_str
                            plt.title(title_str)
                            fig.savefig(path_out + folder_name + '/' \
                                + sub_folder_name + '/' + run_name + '_capacity.png') 
                        t+=1
                    plt.close(fig)

def channel_transfer(net, 
                     model_path, 
                     run_name, 
                     test_loader,
                     device, 
                     epsilon_list, 
                     num_iter,
                     attack='PGD',
                     channel_transfer='largest',
                     transfer_mode='frequency_based',
                     layer_to_test=0,
                     use_pop_stats=True, 
                     noise_variance=0):
    
    # get model
    net = model_setup(net, 
                      model_path, 
                      device,
                      use_pop_stats, 
                      run_name, 
                      noise_variance)
    
    # get layer key to test
    layer_key = ['BN_' + str(layer_to_test)]

    print('Layer being transferred: ', layer_key)

    # get number of channels in layer
    # print(net.get_bn_parameters())
    # channel_size = net.get_bn_parameters()[layer_key[0]].size(0)
    channel_size = 64

    # get number of channels to transfer
    if channel_transfer in ['largest', 'smallest']: transfers = np.arange(9)
    elif channel_transfer == 'individual': transfers = np.arange(0, channel_size, 9)
    print('Total number of channels to transfer: ', len(transfers))

    # create directory
    if model_path.find('bitbucket')!= -1:
        dir_path = './gpucluster/CIFAR10/' + get_model_name(run_name) + '/eval/PGD/channel_transfer/' \
                + run_name.split('_')[0]  + '_' +  run_name.split('_')[1] + '/'
    else:
        dir_path = './results/' + get_model_name(run_name) + '/eval/PGD/channel_transfer/' \
                + run_name.split('_')[0]  + '_' +  run_name.split('_')[1] + '/'
      
    if not os.path.isdir(dir_path): os.mkdir(dir_path)
    dir_path += layer_key[0] + '/'
    if not os.path.isdir(dir_path): os.mkdir(dir_path)
    dir_path += transfer_mode + '/'
    if not os.path.isdir(dir_path): os.mkdir(dir_path)

    # get min and max tensor
    min_tensor, max_tensor = get_minmax(test_loader=test_loader, device=device)
    
    for n_channels in transfers:
        print('channel IDX: ', n_channels)
        for eps in epsilon_list:
            epsilon = float(eps)

            correct_clean = 0
            correct_s = 0
            total = 0

            for i, data in enumerate(test_loader, 0):
                # get data
                X, y = data
                X, y = X.to(device), y.to(device)

                # create npy file
                fname = dir_path + channel_transfer + '.npy'
                if os.path.isfile(fname): csv_dict = np.load(fname, allow_pickle='TRUE').item()
                else: csv_dict = {}

                # check if the channel transfer has already been completed
                curr_key = str(run_name + '_' + str(epsilon))
                if curr_key in csv_dict.keys():
                    if (len(csv_dict[curr_key]) == 9 and channel_transfer in ['smallest', 'largest']) or \
                        (len(csv_dict[curr_key]) == 8 and channel_transfer=='individual'): 
                        print('------ KEY ALREADY FULL ------')
                        break
                
                # get adversarial activations
                net.set_verbose(verbose=True)
                _, capacities, adv_activations = pgd_linf_capacity(net, 
                                                                   X, 
                                                                   y, 
                                                                   epsilon, 
                                                                   max_tensor, 
                                                                   min_tensor, 
                                                                   alpha=epsilon/10, 
                                                                   num_iter=num_iter, 
                                                                   layer_key=layer_key)         
                net.set_verbose(verbose=False)

                # order channels based on transfer mode
                if channel_transfer == 'largest' or channel_transfer == 'individual': descending = True
                elif channel_transfer == 'smallest': descending = False
                                            
                if transfer_mode == 'capacity_based':
                    tmp_capacity_idx = torch.argsort(torch.Tensor(np.array(capacities[layer_key[0]][-1]) \
                                       - np.array(capacities[layer_key[0]][0])), descending=descending)

                elif transfer_mode == 'lambda_based':
                    lambdas = net.get_bn_parameters()[layer_key[0]]
                    tmp_capacity_idx = torch.argsort(lambdas, descending=descending)
                
                elif transfer_mode == 'frequency_based':
                    print('------------------------------------------------------')
                    print(os.getcwd())
                    if model_path.find('bitbucket')!= -1:
                        tmp_capacity_idx = np.load('./gpucluster/CIFAR10/VGG19/eval/PGD/Gaussian_Parametric_frequency/MSE_CE/'+ run_name + '/layer_0/frequency_ordered_channels.npy')
                    else:
                        tmp_capacity_idx = np.load('./results/VGG19/eval/PGD/Gaussian_Parametric_frequency/MSE_CE/'+ run_name + '/layer_0/frequency_ordered_channels.npy')

                # select channels (i.e. channels-corresponding channels) to transfer
                if channel_transfer in ['smallest', 'largest']:
                    if int(n_channels) != 0:
                        capacity_ch = tmp_capacity_idx[0:int(n_channels)*int((channel_size/8))].cpu().detach().numpy()
                    else:
                        capacity_ch = tmp_capacity_idx[0].cpu().detach().numpy()
                elif channel_transfer == 'individual':
                    if transfer_mode == 'frequency_based':
                        capacity_ch = tmp_capacity_idx[int(n_channels)]
                    else:
                        capacity_ch = tmp_capacity_idx[int(n_channels)].cpu().detach().numpy()

                capacity_activations = adv_activations[layer_key[0]][-1][:, capacity_ch, :, :]
                # print(capacity_ch, int(layer_key[0][-1]), capacity_activations.shape)
                transfer_activation = [capacity_ch, int(layer_key[0].split('_')[-1]), capacity_activations]

                # subsitute clean activation with adversarial activations 
                with torch.no_grad():
                    outputs = net(X, transfer_activation)
                    outputs_clean = net(X) 

                _, predicted_clean = torch.max(outputs_clean.data, 1)
                _, predicted = torch.max(outputs.data, 1)

                correct_clean += (predicted_clean == y).sum().item()

                #print('clean ------------------------------: ', (predicted_clean == y).sum().item())
                #print('adversarial ------------------------------: ', (predicted == y).sum().item())

                total += y.size(0)
                correct_s += (torch.logical_and(predicted == y, predicted_clean == y)).sum().item()

                # save result 
                if (channel_transfer in ['smallest', 'largest'] and i == 20) or (channel_transfer=='individual' and i == 7):
                    if not os.path.isfile(fname):
                        csv_dict = {curr_key: [correct_s/correct_clean]}
                        np.save(fname, csv_dict)
                    else:
                        csv_dict = np.load(fname, allow_pickle='TRUE').item()
                        if curr_key in csv_dict.keys():
                            temp = csv_dict[curr_key]
                            to_add = []
                            for iiii in range(len(temp)+1):
                                if iiii<len(temp):
                                    to_add.append(csv_dict[curr_key][iiii])
                                elif iiii == len(temp):
                                    to_add.append(correct_s/correct_clean)
                            csv_dict[curr_key] = to_add
                        else:
                            csv_dict.update({curr_key:[correct_s/correct_clean]})

                        np.save(fname, csv_dict)

                    # exit the loop upon saving the file 
                    break

def adversarial_test(net, 
                     model_path, 
                     model_tag, 
                     run_name, 
                     test_loader, 
                     PATH_to_deltas_, 
                     device, 
                     attack,
                     epsilon, 
                     num_iter,
                     noise_capacity_constraint,
                     capacity_calculation=False,
                     capacity=0,
                     use_pop_stats=False,
                     inject_noise=False,
                     noise_variance=0,
                     no_eval_clean=False,
                     random_resizing=False,
                     scaled_noise=False,
                     relative_accuracy=True,
                     scaled_noise_norm=False,
                     scaled_noise_total=False,
                     scaled_lambda=False,
                     noise_first_layer=False,
                     noise_not_first_layer=False,
                     get_similarity=False, 
                     eval=True, 
                     custom=True,
                     save=False, 
                     save_analysis=False,
                     get_max_indexes=False, 
                     channel_transfer='', 
                     n_channels=0, 
                     transfer_mode='capacity_based'):

    #net.load_state_dict(torch.load(model_path))
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.to(device)

    ################## MODE SELECTION ##################
    if inject_noise:
        print('Adversarial Test With Noise ...')
        if run_name.find('VGG')!=-1:
            net = noisy_VGG(net, 
                            eval_mode=use_pop_stats,
                            noise_variance=noise_variance, 
                            device=device,
                            capacity_=capacity,
                            noise_capacity_constraint=noise_capacity_constraint,
                            run_name=run_name, 
                            scaled_noise=scaled_noise, 
                            scaled_noise_norm=scaled_noise_norm, 
                            scaled_noise_total=scaled_noise_total,
                            scaled_lambda=scaled_lambda, 
                            noise_first_layer=noise_first_layer,
                            noise_not_first_layer=noise_not_first_layer)

        if run_name.find('ResNet')!= -1:
            net = noisy_ResNet(net,
                            eval_mode=use_pop_stats,
                            device=device,
                            run_name=run_name,
                            noise_variance=noise_variance)

    if capacity_calculation or len(get_similarity)>0 or len(channel_transfer)>0:
        if run_name.find('VGG') != -1: 
            net = proxy_VGG(net, 
                            eval_mode=use_pop_stats,
                            device=device,
                            run_name=run_name,
                            noise_variance=noise_variance)
        elif run_name.find('ResNet') != -1: 
            net = proxy_ResNet(net, 
                               eval_mode=use_pop_stats,
                               device=device,
                               run_name=run_name,
                               noise_variance=noise_variance)
    ####################################################

    ################## EVAL MODE ##################
    if use_pop_stats:
        net.eval()
    # setting dropout to eval mode (in VGG)
    if run_name.find('VGG')!= -1:
        net.classifier.eval()
    ###############################################

    print('EVAL MODE: ', not net.training)


    ################## VERBOSE ##################
    print('eval MODE:                       ', use_pop_stats)
    print('inject noise MODE:               ', inject_noise)
    print('test on clean batch stats MODE:  ', no_eval_clean)
    print('---------------------------------')
    if run_name.find('VGG')!= -1:
        print('features training MODE:          ', net.features.training)
        print('average pooling training MODE:   ', net.avgpool.training)
        print('classifier training MODE:        ', net.classifier.training)
    #############################################

    # getting min and max pixel values to be used in PGD for clamping
    min_tensor, max_tensor = get_minmax(test_loader=test_loader, device=device)

    # counter for correctly classified inputs
    if isinstance(epsilon, list):
        correct_s = np.zeros((1, len(epsilon)))
    else:
        correct_s = 0
    
    # counter for number of samples
    total = 0
    correct_clean = 0
    transfer_activation = []

    ################## ADVERSARIAL EVALUATION ##################
    if eval:
        for i, data in enumerate(test_loader, 0):
            X, y = data
            X, y = X.to(device), y.to(device)
            
            if random_resizing:
                size = random.randint(X.shape[2]-2, X.shape[2]-1)
                size = X.shape[2]-2
                crop = transforms.RandomResizedCrop(X.shape[2]-2)
                X_crop = crop.forward(X)

                if size == X.shape[2]-2:
                    choices = [[1, 1, 1, 1],
                               [2, 0, 2, 0], 
                               [2, 0, 0, 2],
                               [0, 2, 2, 0], 
                               [0, 2, 0, 2]] 
                    rand_idx = random.randint(0,4)
                    to_pad = choices[rand_idx]

                elif size == X.shape[2]-1:
                    to_pad = [0, 0, 0, 0] # l - t - r - b
                    while sum(to_pad) < 2:
                        rand_idx_lr = random.choice([0, 1])
                        rand_idx_tb = random.choice([2, 3])
                        rand_pad_side_lr = random.randint(0,1)
                        rand_pad_side_tb = random.randint(0,1)

                        if to_pad[0]+to_pad[1] == 0:
                            to_pad[rand_idx_lr] = rand_pad_side_lr
                        if to_pad[2]+to_pad[3] == 0:
                            to_pad[rand_idx_tb] = rand_pad_side_tb

                pad = torch.nn.ZeroPad2d(tuple(to_pad))
                X = pad(X_crop)

            if no_eval_clean:
                net = test_VGG(net, X)

            if attack == 'FGSM':
                delta = fgsm(net, X, y, epsilon)
                    
            elif attack == 'PGD':
                # ------ Custom PGD function
                if custom:
                    # get capacity mode
                    if i==0 and (capacity_calculation or len(get_similarity) > 0 or get_max_indexes):
                        ####################################################################################################################
                        # get similarity only for BN-0 configuration
                        if len(get_similarity) > 0 and get_bn_int_from_name(run_name)!=1:
                            correct_s = 100
                            total = 100
                            correct_clean = 100
                            break
                        # get adversarial perturbations, capacity, layer adversarial activations.  
                        net.set_verbose(verbose=True)
                        if run_name.find('bn')!= -1:
                            layer_key = ['BN_' + str(i) for i in range(16)]
                        else:
                            layer_key = ['BN_0', 'BN_1']
                        
                        if capacity_calculation:
                            lambdas = net.get_bn_parameters()
                            path_out_ = './results/VGG19/eval/PGD/capacity/lambda_based/'
                            path_out_ += run_name + '/'
                            if not os.path.isdir(path_out_): 
                                os.mkdir(path_out_)
                            path_out_ += 'lambdas.npy'
                            if not os.path.isfile(path_out_):
                                np.save(path_out_, lambdas)

                            correct_s = 100
                            total = 100
                            correct_clean = 100
                            break
                        
                        delta, capacities, adv_activations = pgd_linf_capacity(net, 
                                                                               X, 
                                                                               y, 
                                                                               epsilon, 
                                                                               max_tensor, 
                                                                               min_tensor, 
                                                                               alpha=epsilon/10, 
                                                                               num_iter=num_iter, 
                                                                               layer_key=layer_key)       
                        # get layer clean activations for similarity computation                                                                       
                        if len(get_similarity) > 0 and get_bn_int_from_name(run_name)==1:
                            _ = net(X)
                            clean_activations = net.get_activations()[layer_key[0]]   
                        # prevent model from returning capacity, layer adversarial activations when not necessary
                        net.set_verbose(verbose=False) 
                        ####################################################################################################################

                        # save analysis files
                        if save_analysis and (len(get_similarity) > 0 or capacity_calculation):
                            model_name = run_name.split('_')[0] 
                            if use_pop_stats:
                                eval_mode_str = 'eval'
                            else:
                                eval_mode_str = 'no_eval'
                            
                            # save channel idxs difference
                            if get_max_indexes and get_bn_int_from_name(run_name)==1:
                                tmp_capacity_idx, _ = torch.sort(torch.argsort(torch.Tensor(np.array(capacities[layer_key[0]][-1]) \
                                                - np.array(capacities[layer_key[0]][0])), descending=False)[0:10])
                                tmp_CKA_idx, _ = torch.sort(torch.argsort(torch.Tensor(np.array(CKA(clean_activations.to(device),\
                                            adv_activations[layer_key[0]][-1].to(device), device))), descending=True)[0:10])
                                ch_diff = torch.abs(tmp_capacity_idx - tmp_CKA_idx)
                                norm_ch_diff = (ch_diff > torch.zeros_like(ch_diff)).cpu().detach().numpy().tolist()
                                
                                path_out = './results/' + model_name + '/' + eval_mode_str + '/'\
                                            + attack + '/CKA_vs_capacity/' + run_name + '/'
                                if not os.path.isdir(path_out):
                                    os.mkdir(path_out)
            
                                np.save(path_out + 'idx_diff_' + str(epsilon).replace('.', ''), norm_ch_diff)

                            # create paths ########################################################################################
                            if capacity_calculation:
                                path_out = './results/' + model_name + '/' + eval_mode_str + '/'\
                                            + attack + '/capacity/'
                            elif len(get_similarity) > 0:
                                path_out = './results/' + model_name + '/' + eval_mode_str + '/'\
                                            + attack + '/' + get_similarity + '_similarity' + '/'                        

                            if len(layer_key)==1:
                                if layer_key[0] == 'BN_0':
                                    path_out += 'first_layer/'
                                elif layer_key[0] == 'BN_1':
                                    path_out += 'second_layer/'
                            else:
                                path_out += 'all_layers/'
                            
                            if net.get_noisy_mode():
                                if noise_variance > 0.05:
                                    path_out += 'noisy_test/'
                                else:
                                    path_out += 'small_noisy_test/'
                            else:
                                path_out += 'clean_test/'
                            
                            if not os.path.isdir(path_out):
                                os.mkdir(path_out)

                            if not os.path.isdir(path_out + 'BATCH_' + str(i)):
                                os.mkdir(path_out + 'BATCH_' + str(i))
                            path_out += 'BATCH_' + str(i) + '/'

                            if len(layer_key)==1:
                                folder_names = ['BN_' + str(get_bn_int_from_name(run_name)-1)]
                            else:
                                folder_names = layer_key
                            ########################################################################################################

                            # save selected modality
                            if not get_max_indexes:
                                for folder_name in folder_names:
                                    if not os.path.isdir(path_out + folder_name):
                                        os.mkdir(path_out + folder_name)
                                    sub_folder_name = str(epsilon).replace('.', '')
                                    if not os.path.isdir(path_out + folder_name + '/' + sub_folder_name):
                                        os.mkdir(path_out + folder_name + '/' + sub_folder_name)                                        

                                    if capacity_calculation:
                                        t = 0
                                        fig = plt.figure()
                                        if len(layer_key)==1:
                                            key = layer_key[0]
                                        else:
                                            key = folder_name
                                        for temp in capacities[key]:
                                            x_axis = [t]
                                            for x_, y_ in zip(x_axis, [temp]):
                                                plt.scatter([x_] * len(y_), y_)
                                                plt.xlabel('PGD steps')
                                                plt.ylabel('Capacity Estimate')
                                                if net.get_noisy_mode():
                                                    noisy_str = 'noisy - ' + r'$\sigma^2=$' + str(noise_variance)
                                                else:
                                                    noisy_str = 'deterministic'
                                                if get_bn_int_from_name(run_name) == 100:
                                                    title_str = model_name + ' ' + 'FULL-BN' + ' ' + noisy_str
                                                else:
                                                    title_str = model_name + ' ' + 'BLOCK' + '-' + \
                                                                str(get_bn_int_from_name(run_name)-1) + \
                                                                ' ' + noisy_str
                                                plt.title(title_str)
                                                fig.savefig(path_out + folder_name + '/' \
                                                    + sub_folder_name + '/' + run_name + '_capacity.png')
                                                
                                            t+=1
                                        plt.xticks(np.arange(0, t))
                                        print('SAVED!')

                                    if len(get_similarity) > 0 and get_bn_int_from_name(run_name)==1:
                                        fig = plt.figure()
                                        step = 0
                                        for adv in adv_activations[folder_name]:
                                            if get_similarity == 'CKA':
                                                temp = CKA(clean_activations.to(device), adv.to(device), device)
                                            elif get_similarity == 'cosine':
                                                temp = cosine_similarity(clean_activations.to(device), adv.to(device), device)
                                            x_axis = [step]                                     
                                            for x_, y_ in zip(x_axis, [temp]):
                                                plt.scatter([x_] * len(y_), y_)
                                                plt.xlabel('PGD steps')
                                                plt.ylabel('Channel Similarity ' + '(' + get_similarity + ')')
                                                fig.savefig(path_out + folder_name + '/' \
                                                    + sub_folder_name + '/' + run_name + '_' + get_similarity + '.png')
                                            step+=1
                    
                    else:     
                        # if capacity is recorded then break
                        if capacity_calculation or len(get_similarity) > 0:
                            correct_s = 100
                            total = 100
                            correct_clean = 100
                            break
                    
                        # channel transfer mode
                        elif len(channel_transfer) > 0 and get_bn_int_from_name(run_name) == 100:

                            if run_name.find('bn')!= -1:
                                # layer_key = ['BN_' + str(i) for i in range(16)]
                                layer_key = ['BN_2']
                            else:
                                layer_key = ['BN_0']

                            dir_path = './results/VGG19/eval/PGD/channel_transfer/' + run_name.split('_')[0] \
                                       + '_' +  run_name.split('_')[1] + '/' 
                            if not os.path.isdir(dir_path):
                                os.mkdir(dir_path)
                            dir_path += layer_key[0] + '/'
                            if not os.path.isdir(dir_path):
                                os.mkdir(dir_path)
                            dir_path += transfer_mode + '/'
                            if not os.path.isdir(dir_path):
                                os.mkdir(dir_path)

                            fname = dir_path + channel_transfer + '.npy'
                            curr_key = str(run_name + '_' + str(epsilon))
                            if os.path.isfile(fname):
                                csv_dict = np.load(fname, allow_pickle='TRUE').item()
                            else:
                                csv_dict = {}

                            print(net.get_bn_parameters()[layer_key[0]].size(0))
                            channel_size = net.get_bn_parameters()[layer_key[0]].size(0)

                            if curr_key in csv_dict.keys():
                                if (len(csv_dict[curr_key]) == 9 and channel_transfer in ['smallest', 'largest']) or (len(csv_dict[curr_key]) == channel_size and channel_transfer=='individual'): 
                                    # here 9 is the total number of times we transfer channels 
                                    # given a total number of 64 channels in the first layer
                                    # i.e. we first tranfer a single channel (the first of either
                                    # ascending or descending order) then we transfer 8 channel cumulatively. 
                                    correct_s = 100 
                                    total = 100
                                    correct_clean = 100
                                    print('------ KEY ALREADY FULL ------')
                                    break
                            
                            print('FEATURE TRANSFER MODE')
                            net.set_verbose(verbose=True)
                            _, capacities, adv_activations = pgd_linf_capacity(net, 
                                                                                X, 
                                                                                y, 
                                                                                epsilon, 
                                                                                max_tensor, 
                                                                                min_tensor, 
                                                                                alpha=epsilon/10, 
                                                                                num_iter=num_iter, 
                                                                                layer_key=layer_key)         
                            net.set_verbose(verbose=False)
                            if channel_transfer == 'largest' or channel_transfer == 'individual':
                                descending = True
                            elif channel_transfer == 'smallest':
                                descending = False
                                                        
                            if transfer_mode == 'capacity_based':
                                tmp_capacity_idx = torch.argsort(torch.Tensor(np.array(capacities[layer_key[0]][-1]) \
                                                   - np.array(capacities[layer_key[0]][0])), descending=descending)

                            elif transfer_mode == 'lambda_based':
                                lambdas = net.get_bn_parameters()[layer_key[0]]
                                tmp_capacity_idx = torch.argsort(lambdas, descending=descending)

                            tot_num_channels = tmp_capacity_idx.size(0)

                            if channel_transfer in ['smallest', 'largest']:
                                if int(n_channels) != 0:
                                    print('MULTIPLE FEATURES TRANSFER')
                                    capacity_ch = tmp_capacity_idx[0:int(n_channels)*int((tot_num_channels/8))].cpu().detach().numpy()
                                else:
                                    print('SINGLE FEATURE TRANSFER')
                                    capacity_ch = tmp_capacity_idx[0].cpu().detach().numpy()

                            elif channel_transfer == 'individual':
                                print('INDIVIDUAL FEATURE TRANSFER')
                                capacity_ch = tmp_capacity_idx[int(n_channels) - 1].cpu().detach().numpy()
                            
                            capacity_activations = adv_activations[layer_key[0]][-1][:, capacity_ch, :, :]
                            print(capacity_ch, int(layer_key[0][-1]), capacity_activations.size())
                            transfer_activation = [capacity_ch, int(layer_key[0].split('_')[-1]), capacity_activations]
                            delta = [torch.zeros_like(X).detach()]
                        
                        # total capacity scaling
                        elif scaled_noise_total:
                            if run_name.find('bn')!= -1:
                                layer_key = ['BN_' + str(i) for i in range(16)]
                            else:
                                layer_key = ['BN_0', 'BN_1']
                            
                            delta = pgd_linf_total_capacity(net, 
                                                            X, 
                                                            y, 
                                                            epsilon, 
                                                            max_tensor, 
                                                            min_tensor, 
                                                            alpha=epsilon/10, 
                                                            num_iter=num_iter, 
                                                            layer_key=layer_key)

                        # compute PGD 
                        else:        
                            print('Attacking with PGD (custom) ...')                   
                            delta = pgd_linf(net, 
                                             X, 
                                             y, 
                                             epsilon, 
                                             max_tensor, 
                                             min_tensor,
                                             alpha=epsilon/10, 
                                             num_iter=num_iter, 
                                             noise_injection=inject_noise)                                                
            
                    adv_inputs = X + delta[0]
                
                # ------ AdverTorch PGD function
                else:
                    adversary = LinfPGDAttack(
                                    net, loss_fn=nn.CrossEntropyLoss(), eps=epsilon,
                                    nb_iter=40, eps_iter=epsilon/10, rand_init=False, clip_min=min_tensor, clip_max=max_tensor,
                                    targeted=False)
                    adv_inputs = adversary.perturb(X, y)
                    delta = [adv_inputs-X]
            
            if len(delta) == 1:
                if save:     
                    path = get_path2delta(PATH_to_deltas_, model_tag, run_name, attack, epsilon)
                    name_out = 'adversarial_delta_' + str(i) + '.pth'

                    '''path = get_path2delta(PATH_to_deltas_, model_tag, run_name, attack)
                    eps_ = 'eps_' + str(epsilon).replace('.', '')
                    if not os.path.isdir(path + '/' + eps_ + '/'):
                        os.mkdir(path + '/' + eps_ + '/')
                    torch.save(delta[0], path + '/' + eps_ + "/adversarial_delta_" + str(i) + ".pth")''' 
                
                if use_pop_stats:
                    net.eval()

                with torch.no_grad():
                    if len(channel_transfer) > 0:
                        outputs = net(X, transfer_activation)
                    else:
                        outputs = net(adv_inputs)
                    
                    if scaled_noise_total:
                        outputs_clean = net(X, noise_in=net.get_noise())
                    else:
                        outputs_clean = net(X)

                _, predicted_clean = torch.max(outputs_clean.data, 1)
                _, predicted = torch.max(outputs.data, 1)
                
                correct_clean += (predicted_clean == y).sum().item()

                print('clean ------------------------------: ', (predicted_clean == y).sum().item())
                print('adversarial ------------------------------: ', (predicted == y).sum().item())

                total += y.size(0)
                correct_s += (torch.logical_and(predicted == y, predicted_clean == y)).sum().item()

                if len(channel_transfer) > 0:
                    if (channel_transfer in ['smallest', 'largest'] and i == 20) or (channel_transfer=='individual' and i == 6):
                        fname = dir_path + channel_transfer + '.npy'
                        curr_key = str(run_name + '_' + str(epsilon))

                        if not os.path.isfile(fname):
                            csv_dict = {curr_key: [correct_s/correct_clean]}
                            np.save(fname, csv_dict)
                        else:
                            csv_dict = np.load(fname, allow_pickle='TRUE').item()
                            print(csv_dict.keys())
                            
                            if curr_key in csv_dict.keys():
                                temp = csv_dict[curr_key]
                                to_add = []
                                for iiii in range(len(temp)+1):
                                    if iiii<len(temp):
                                        to_add.append(csv_dict[curr_key][iiii])
                                    elif iiii == len(temp):
                                        to_add.append(correct_s/correct_clean)
                                csv_dict[curr_key] = to_add
                            else:
                                csv_dict.update({curr_key:[correct_s/correct_clean]})

                            print(csv_dict)
                            np.save(fname, csv_dict)
                        break

            else:
                for k, idv_delta in enumerate(delta):
                    num = epsilon[k]
                    eps_ = 'eps_' + str(num).replace('.', '')
                    if not os.path.isdir(path + '/' + eps_ + '/'):
                        os.mkdir(path + '/' + eps_ + '/')

                    torch.save(idv_delta, path + '/' + eps_ + '/' 
                                + "/adversarial_delta_" + str(i) + '.pth') 

                    with torch.no_grad():
                        net.eval()
                        outputs = net(X+idv_delta)

                    _, predicted = torch.max(outputs.data, 1)
                    correct_s[0, k] += (predicted == y).sum().item()
    
    # adversarial evaluation if delta is given
    else:
        if isinstance(epsilon, list):
            correct_s = np.zeros((1, len(epsilon)))
        else:
            correct_s = 0
        total = 0
        PATH_to_deltas = PATH_to_deltas_ + model_tag
        
        for i, data in enumerate(test_loader, 0):
            X, y = data
            X, y = X.to(device), y.to(device)

            # load deltas 
            for h, eps in enumerate(epsilon):
                
                eps_ = 'eps_' + str(eps).replace('.', '')
                path = PATH_to_deltas + '/' + run_name + '/' + attack + '/' + eps_ + '/'
                deltas = os.listdir(path)
                delta = torch.load(path + "/adversarial_delta_" + str(i) + ".pth")
                
                with torch.no_grad():
                    outputs = net(X+delta)

                _, predicted = torch.max(outputs.data, 1)
                
                if isinstance(epsilon, list):
                    correct_s[0, h] += (predicted == y).sum().item()
                else:
                    correct_s += (predicted == y).sum().item()

            total += y.size(0)

    if relative_accuracy:
        acc = correct_s / correct_clean
    else:
        acc = correct_s / total
    
    print('Adversarial Test Accuracy: ', acc)

    if isinstance(epsilon, list):
        return acc.tolist()
    else:
        return acc

def saliency_map(model, 
                 model_path, 
                 test_loader, 
                 device, 
                 run_name,
                 eval_mode=True, 
                 adversarial=False, 
                 epsilon=0.0392, 
                 mode='max_loss'):
    
    print('Creating saliency maps')
        
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    if run_name.find('VGG')!= -1:
        net = proxy_VGG2(model, 
                        eval_mode=eval_mode,
                        device=device,
                        run_name=run_name,
                        noise_variance=0)
    elif run_name.find('ResNet')!= -1:
        net = proxy_ResNet(model, 
                           eval_mode=eval_mode,
                           device=device,
                           run_name=run_name,
                           noise_variance=0)

    if eval_mode: net.eval()

    X, y = next(iter(test_loader))
    X, y = X.to(device), y.to(device) 

    if adversarial:
        min_tensor, max_tensor = get_minmax(test_loader=test_loader, device=device)
        delta = pgd_linf(net, 
                        X, 
                        y, 
                        epsilon, 
                        max_tensor, 
                        min_tensor,
                        alpha=epsilon/10, 
                        num_iter=40)
        X = X + delta[0]   

    if mode == 'max_score':
        score = net(X)
        pred_score, predicted = torch.max(score, 1)
        max_score = torch.max(pred_score)
    elif mode == 'max_loss':
        score = net(X)
        _, predicted = torch.max(score, 1)
        loss = nn.CrossEntropyLoss()(net(X), y)

    for j, _ in enumerate(X): 
        print('IMAGE: ', j)
        if j > 9: break
        if mode == 'max_score':
            print('Mode: max-score')
            if (not adversarial and predicted[j] == y[j]) or (adversarial and predicted[j] != y[j]):
                score = net(X[j].unsqueeze(0))
                pred_score = torch.max(score)
                norm_score = pred_score/max_score
                norm_score.backward(retain_graph=True)
                if run_name.find('VGG')!=-1:
                    saliency_map = torch.abs(net.bn1.grad.detach())
        elif mode == 'max_loss':
            print('Mode: max-loss')
            if (not adversarial and predicted[j] == y[j]) or (adversarial and predicted[j] != y[j]):
                print('IN')
                loss.backward(retain_graph=True)
                if run_name.find('VGG')!=-1:
                    saliency_map = torch.abs(net.bn1.grad.detach())

        root_path = './results/' + get_model_name(run_name) + '/'

        if eval_mode: root_path += 'eval/'
        else: root_path += 'no_eval/' 
        
        root_path += 'PGD' + '/' + 'saliency_maps/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

        if adversarial: root_path += 'adversarial/'
        else: root_path += 'clean/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

        root_path += mode +'/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

        if adversarial: root_path += str(epsilon).replace('.', '') + '/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

        root_path += run_name + '/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

        root_path += 'img_' + str(j) + '/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

        if run_name.find('VGG')!= -1:
            ch_max = torch.argsort(net.bn.weight, descending=True)[0]
            ch_min = torch.argsort(net.bn.weight, descending=False)[0]
        elif run_name.find('ResNet')!= -1:
            ch_max = torch.argsort(net.bn1.weight, descending=True)[0]
            ch_min = torch.argsort(net.bn1.weight, descending=False)[0]
        chs = [ch_max, ch_min]
        
        inv_normalize = transforms.Normalize(
            mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
            std=[1/0.2023, 1/0.1994, 1/0.2010])

        input_img = inv_normalize(X[j])

        map_min = torch.min(saliency_map)    
        map_max = torch.max(saliency_map)

        for jj in range(saliency_map.size(1)):
            if jj in chs:
                if mode == 'max_score':
                    temp = saliency_map[0, jj, :, :]
                elif mode == 'max_loss':
                    temp = saliency_map[j, jj, :, :]
                temp_norm = (temp - map_min)/(map_max - map_min)
                plt.figure()
                if adversarial: plt.title('Advesarial Sample Saliency Map')
                else: plt.title('Clean Sample Saliency Map')
                plt.subplot(1, 3, 1)
                plt.imshow(np.transpose(input_img.cpu().detach().numpy(), (1, 2, 0)))
                plt.xticks([])
                plt.yticks([])
                plt.title('Original Sample')
                plt.subplot(1, 3, 2)
                if mode == 'max_score':
                    plt.imshow(net.bn1[0, jj, :, :].cpu().detach().numpy())
                elif mode == 'max_loss':
                    plt.imshow(net.bn1[j, jj, :, :].cpu().detach().numpy())
                plt.xticks([])
                plt.yticks([])
                plt.title('Channel Activation')
                plt.subplot(1, 3, 3)
                plt.imshow(temp_norm.cpu().numpy())
                plt.xticks([])
                plt.yticks([])
                mode_str = mode.replace('_', '-')
                plt.title('Saliency Map')
                if jj == ch_max:
                    plt.savefig(root_path + 'ch_max' + '.jpg')
                elif jj == ch_min:
                    plt.savefig(root_path + 'ch_min' + '.jpg')
                plt.close()

        chns = [0, 7, 15, 25, 35, 45, 55, 63]
        temp_count = 0
        ordered_channels = torch.argsort(net.bn.weight, descending=False)
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
        axs = axs.ravel()

        if mode == 'max_loss':
            for bb, ch_ in enumerate(ordered_channels):
                if bb in chns:
                    axs[temp_count].imshow(net.bn1[j, ch_, :, :].cpu().detach().numpy())
                    axs[temp_count].set_xticks([])
                    axs[temp_count].set_yticks([])
                    axs[temp_count].set_title('Ordered Channel #' + str(bb))
                    temp_count+= 1
            fig.savefig(root_path + 'channels_' + '.jpg')
        
        # zero gradients for next steps
        saliency_map.zero_()

def cross_model_testing(file_name, 
                        mode, 
                        root_path,
                        test_loader, 
                        device, 
                        attack, 
                        epsilon):

    model_name = file_name.split('_', 1)[0]
    og_model_name = model_name
    dt_string = file_name.split('_', 1)[1]

    PATH_to_deltas = root_path + '/deltas/'

    if mode=='BN_on_STD':
        # retrieve model object and get the trained model weights path
        batch_norm = False
        run_name = og_model_name + '_' + dt_string
        net = get_model.get_model(og_model_name, batch_norm)   
        PATH_to_model = root_path + '/models/' + og_model_name + '/' + run_name + '.pth'

        # model name of the deltas to retrieve
        model_name = og_model_name + 'bn'
        run_name = og_model_name + '_' + dt_string

        # adversarial test
        adv_test_acc = adversarial_test(net, 
                                        PATH_to_model, 
                                        model_name, 
                                        run_name, 
                                        test_loader, 
                                        PATH_to_deltas, 
                                        device, 
                                        attack, 
                                        epsilon, 
                                        eval=False)

    else:
        # retrieve model object and get the trained model weights path
        batch_norm = True
        run_name = og_model_name + 'bn' + '_' + dt_string
        net = get_model.get_model(og_model_name, batch_norm) 
        PATH_to_model = root_path + '/models/' + og_model_name + 'bn' + '/' + run_name + '.pth'

        # model name of the deltas to retrieve
        model_name = og_model_name
        run_name = model_name + '_' + dt_string

        # adversarial test
        adv_test_acc = adversarial_test(net, 
                                        PATH_to_model, 
                                        model_name, 
                                        run_name, 
                                        test_loader, 
                                        PATH_to_deltas, 
                                        device, 
                                        attack, 
                                        epsilon, 
                                        eval=False)

    return adv_test_acc

def cross_model_testing_(test_loader, models_path, deltas_path, model_tag, device):

    saved_models = os.listdir(models_path)
    deltas = os.listdir(deltas_path)

    # load desired model
    for model_path in saved_models:
        model = torch.load(models_path + model_path)
        model_tag = model_path.split('_')[1]

        # for the same model - test for each of the adversarial produced by all models (including model itself)
        for delta_path in deltas:
            spec_deltas = os.listdir(deltas_path + delta_path)
            delta_model_name = delta_path

            # load iterator of test image on which adversarial variation are then added
            test_iter = iter(test_loader)
            
            # load the model-specific delta and evaluate the selectd model
            correct = 0
            total = 0
            for i, model_delta in enumerate(spec_deltas):
                batch_delta = torch.load(deltas_path + delta_path + '/' + model_delta)
                X, y = test_iter.next()
                X, y = X.to(device), y.to(device)
                model.to(device)
                
                outputs = model(X+batch_delta)

                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            print('ACCURACY ' + model_tag + ' trained on ' + delta_model_name + ' ADVERSARIAL test images: %d %%' % (100 * correct / total))

def fft(img):
    return np.fft.fft2(img)

def fftshift(img):
    return np.fft.fftshift(fft(img))

def ifft(img):
    return np.fft.ifft2(img)

def ifftshift(img):
    return ifft(np.fft.ifftshift(img))

def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def distance_bigger(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis > r:
        return 1.0
    else:
        return 0

def get_cut_off_value(fft_img, percentage):
    temp = fft_img.flatten()
    ordered_temp = temp[::-1].sort()
    how_many = int(float(temp.size)*percentage)
    cut_off = ordered_temp[how_many]
    return cut_off, how_many

def mask_coefficient(fft_img, percentage):
    rows, cols = fft_img.shape
    mask = np.zeros((rows, cols))
    cut_off, how_many = get_cut_off_value(fft_img, percentage)
    count = 0
    for i in range(rows):
        for j in range(cols):
            if fft_img[i,j] >= cut_off: 
                mask[i, j] = 1.0
            else:
                mask[i, j] = 0.0

            count += 1
    return mask

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[2], Images.shape[3]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([3, Images.shape[2], Images.shape[3]])
        for j in range(3):
            fd = fftshift(Images[i, j, :, :])
            fd = fd * mask
            img_low = ifftshift(fd)
            tmp[j,:,:] = np.real(img_low)
        Images_freq_low.append(tmp)
        tmp = np.zeros([3, Images.shape[2], Images.shape[3]])
        for j in range(3):
            fd = fftshift(Images[i, j, :, :])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[j,:,:] = np.real(img_high)
        Images_freq_high.append(tmp)

    return np.array(Images_freq_low), np.array(Images_freq_high)

def get_frequency_components(Images, r):
    '''
    frequency component for activations
    '''
    frequency_component_low = []
    frequency_component_high = []
    mask = mask_radial(np.zeros([Images.shape[2], Images.shape[3]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[0], Images.shape[2], Images.shape[3]])
        for j in range(Images.shape[1]):
            fd = fftshift(Images[i, j, :, :])
            fd = fd * mask
            tmp[j,:,:] = np.real(fd)
        frequency_component_low.append(tmp)
        tmp = np.zeros([Images.shape[0], Images.shape[2], Images.shape[3]])
        for j in range(Images.shape[1]):
            fd = fftshift(Images[i, j, :, :])
            fd = fd * (1 - mask)
            tmp[j,:,:] = np.real(fd)
        frequency_component_high.append(tmp)
    
    return frequency_component_low, frequency_component_high

def get_flattend_frequency_components(frequency_image, r):
    rows, cols = frequency_image.shape
    flattened_frequency_seq = []
    for i in range(rows):
        for j in range(cols):
            if distance_bigger(i, j, imageSize=rows, r=r) > 0:
                flattened_frequency_seq.append(frequency_image[i, j])

    return flattened_frequency_seq

def mse_error(input1, input2):
    return((input1 - input2)**2).mean()

def get_frequency_images(model, 
                         model_path, 
                         test_loader, 
                         device, 
                         run_name,
                         eval_mode=True,
                         layer_to_test=0, 
                         frequency_radius=[i for i in range(2,16)], 
                         visualization=False, 
                         mse_comparison=True, 
                         use_conv=False):
    '''
    input(s):
        - model
        - samples
        - layer to test 
        - frequency radius

    return(s):
        - comparison of low-high frequency input
        - qualitative average distance comparison

    '''
    
    # load model
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    # model for activations
    if run_name.find('VGG')!= -1:
        model = proxy_VGG3(model, 
                           eval_mode=eval_mode,
                           device=device,
                           run_name=run_name,
                           noise_variance=0, 
                           layer_to_test=layer_to_test)
    elif run_name.find('ResNet') != -1: 
        model = proxy_ResNet(model, 
                           eval_mode=eval_mode,
                           device=device,
                           run_name=run_name,
                           noise_variance=0) 

    # set eval mode for inference 
    if eval_mode: model.eval()

    frequency_radius = [i for i in range(2,16)]

    if not isinstance(frequency_radius, list): frequency_radius = [frequency_radius]

    # retrieve batch from loader
    X, y = next(iter(test_loader))
    X, y = X.to(device), y.to(device) 

    if mse_comparison:
        lambdas = model.get_bn_parameters()['BN_' + str(layer_to_test)]
        num_channels = lambdas.size(0)
        low_f_comparison = np.zeros((num_channels, 14))
        high_f_comparison = np.zeros((num_channels, 14))
    
    for r, radius in enumerate(frequency_radius):

        print('Radius: ', r)
    
        # get high and low frequency images for the given batch
        low_f_img, high_f_img = generateDataWithDifferentFrequencies_3Channel(X.cpu().numpy(), radius)

        # convert images to appropriate format to feed into model
        low_f_img = torch.tensor(low_f_img, dtype=torch.float32, device=device)
        high_f_img = torch.tensor(high_f_img, dtype=torch.float32, device=device)

        # retrive max and min lambda channel idxs
        if run_name.find('ResNet') != -1: 
            layer_key = 'BN_' + str(layer_to_test)
            ch_max = torch.argsort(model.net.bn1.weight.cpu().detach(), descending=True)[0]
            ch_min = torch.argsort(model.net.bn1.weight.cpu().detach(), descending=False)[0]
        else: 
            lambdas = model.get_bn_parameters()['BN_' + str(layer_to_test)]
            ch_max = torch.argsort(lambdas, descending=True)[0]
            ch_min = torch.argsort(lambdas, descending=False)[0]
        chs = [ch_max, ch_min]

        if mse_comparison:
            print('MSE COMPARISON')

            # feed standard sample and get corresponding activations
            _ = model(X)
            if use_conv:
                activations = model.conv_frequency_activation
            else:
                activations = model.bn_frequency_activation

            # feed low-frequency sample and get corresponding activations 
            _ = model(low_f_img)
            if use_conv:
                activations_low = model.conv_frequency_activation
            else:
                activations_low = model.bn_frequency_activation

            # feed high-frequency sample and get corresponding activations 
            _ = model(high_f_img)
            if use_conv:
                activations_high = model.conv_frequency_activation
            else:
                activations_high = model.bn_frequency_activation

            if run_name.find('ResNet') != -1: 
                ordered_channels = torch.argsort(model.net.bn1.weight.cpu().detach(), descending=False)
            else: 
                lambdas = model.get_bn_parameters()['BN_' + str(layer_to_test)]
                ordered_channels = torch.argsort(lambdas, descending=False)

            chns = [0, 7, 15, 25, 35, 45, 55, 63]

            path_out = './results/' + get_model_name(run_name) + '/'

            if eval_mode: path_out += 'eval/'
            else: path_out += 'no_eval/' 
            
            path_out += 'PGD' + '/' + 'frequency_analysis/'
            if not os.path.isdir(path_out): os.mkdir(path_out)

            path_out += 'channel_frequency/'
            if not os.path.isdir(path_out): os.mkdir(path_out)

            path_out += run_name + '/'
            if not os.path.isdir(path_out): os.mkdir(path_out)

            if use_conv: path_out += 'use_conv' + '/'
            else: path_out += 'use_bn' + '/'
            if not os.path.isdir(path_out): os.mkdir(path_out)

            path_out += 'layer_' + str(layer_to_test) + '/'
            if not os.path.isdir(path_out): os.mkdir(path_out)

            for bb, ch_ in enumerate(ordered_channels):
                #print(mse_error(activations_low[:, ch_, :, :], activations[:, ch_, :, :]), mse_error(activations_high[:, ch_, :, :], activations[:, ch_, :, :]))
                low_f_comparison[bb, r] = mse_error(activations_low[:, ch_, :, :], activations[:, ch_, :, :])
                high_f_comparison[bb, r] = mse_error(activations_high[:, ch_, :, :], activations[:, ch_, :, :])

        np.save(path_out + 'low_f_comparison' + '.npy', low_f_comparison)
        np.save(path_out + 'high_f_comparison' + '.npy', high_f_comparison)

        if visualization:
            # iterate through the given batch
            for i, _ in enumerate(X):

                # feed standard sample and get corresponding activations
                _ = model(X[i, :, :, :].unsqueeze(0))
                activations = model.bn1.cpu().detach().numpy()

                # feed low-frequency sample and get corresponding activations 
                _ = model(low_f_img[i, :, :, :].unsqueeze(0))
                activations_low = model.bn1.cpu().detach().numpy()

                # feed high-frequency sample and get corresponding activations 
                _ = model(high_f_img[i, :, :, :].unsqueeze(0))
                activations_high = model.bn1.cpu().detach().numpy()
                
                # stop after getting 10 samples
                if i == 10: break

                # create path
                root_path = './results/' + get_model_name(run_name) + '/'

                if eval_mode: root_path += 'eval/'
                else: root_path += 'no_eval/' 
                
                root_path += 'PGD' + '/' + 'frequency_analysis/'
                if not os.path.isdir(root_path): os.mkdir(root_path)

                root_path += 'visualization/'
                if not os.path.isdir(root_path): os.mkdir(root_path)

                root_path += 'radius_' + str(frequency_radius) + '/'
                if not os.path.isdir(root_path): os.mkdir(root_path)

                root_path += run_name + '/'
                if not os.path.isdir(root_path): os.mkdir(root_path)

                root_path += 'img_' + str(i) + '/'
                if not os.path.isdir(root_path): os.mkdir(root_path)

                # inverse normalization 
                inv_normalize = transforms.Normalize(
                    mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
                    std=[1/0.2023, 1/0.1994, 1/0.2010])

                input_img = torch.clamp(inv_normalize(X[i, :, :, :]), min=0., max=0.9999)
                input_img_low = torch.clamp(inv_normalize(low_f_img[i, :, :, :]), min=0., max=0.9999)
                input_img_high = torch.clamp(inv_normalize(high_f_img[i, :, :, :]), min=0., max=0.9999)

                fig_C = plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(np.transpose(input_img.cpu().detach().numpy(), (1, 2, 0)))
                plt.xticks([])
                plt.yticks([])
                plt.title('Standard Sample')
                plt.subplot(1, 3, 2)
                plt.imshow(np.transpose(input_img_low.cpu().detach().numpy(), (1, 2, 0)))
                plt.xticks([])
                plt.yticks([])
                plt.title('Low Frequency')
                plt.subplot(1, 3, 3)
                plt.imshow(np.transpose(input_img_high.cpu().detach().numpy(), (1, 2, 0)))
                plt.xticks([])
                plt.yticks([])
                plt.title('High Frequency')
                #fig_C.savefig('./frequency_comparison' + '.jpg')
                fig_C.savefig(root_path + 'frequency_comparison' + '.jpg')
                plt.close()
            
                # for each channel save all 3 activations into the same image
                for ii in range(activations.shape[1]):
                    if ii in chs:
                        fig = plt.figure()
                        plt.subplot(1, 3, 1)
                        plt.imshow(activations[0, ii, :, :])
                        plt.xticks([])
                        plt.yticks([])
                        plt.title('Standard Activation')
                        plt.subplot(1, 3, 2)
                        plt.imshow(activations_low[0, ii, :, :])
                        plt.xticks([])
                        plt.yticks([])
                        plt.title('Low Frequency')
                        plt.subplot(1, 3, 3)
                        plt.imshow(activations_high[0, ii, :, :])
                        plt.xticks([])
                        plt.yticks([])
                        plt.title('High Frequency')
                        if ii == ch_max:
                            #fig.savefig('./ch_max' + '.jpg')
                            fig.savefig(root_path + 'ch_max' + '.jpg')
                        elif ii == ch_min:
                            fig.savefig(root_path + 'ch_min' + '.jpg')
                        plt.close()

                chns = [0, 7, 15, 25, 35, 45, 55, 63]
                temp_count = 0
                if run_name.find('ResNet') != -1: 
                    '''layer_key = 'BN_' + str(layer_to_test)
                    lambdas = model.get_bn_parameters()[layer_key]'''
                    ordered_channels = torch.argsort(model.net.bn1.weight.cpu().detach(), descending=False)
                else:
                    ordered_channels = torch.argsort(model.bn.weight.cpu().detach(), descending=False)
                fig_low, axs_low = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
                fig_high, axs_high = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
                axs_low = axs_low.ravel()
                axs_high = axs_high.ravel()
                
                for bb, ch_ in enumerate(ordered_channels):
                    if bb in chns:
                        axs_low[temp_count].imshow(activations_low[0, ch_, :, :])
                        axs_low[temp_count].set_xticks([])
                        axs_low[temp_count].set_yticks([])
                        axs_low[temp_count].set_title('Ordered Channel #' + str(bb))

                        axs_high[temp_count].imshow(activations_high[0, ch_, :, :])
                        axs_high[temp_count].set_xticks([])
                        axs_high[temp_count].set_yticks([])
                        axs_high[temp_count].set_title('Ordered Channel #' + str(bb))
                        temp_count+= 1

                        plt.close()

                fig_low.savefig(root_path + 'channels_low' + '.jpg')
                fig_high.savefig(root_path + 'channels_high' + '.jpg')

    if mse_comparison:
        
        root_path = './results/' + get_model_name(run_name) + '/'

        if eval_mode: root_path += 'eval/'
        else: root_path += 'no_eval/' 
        
        root_path += 'PGD' + '/' + 'frequency_analysis/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

        root_path += 'mse_comparison/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

        root_path += run_name + '/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

        if use_conv: root_path += 'use_conv' + '/'
        else: root_path += 'use_bn' + '/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

        root_path += 'layer_' + str(layer_to_test) + '/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

        max_low = torch.max(torch.tensor(low_f_comparison))
        max_high = torch.max(torch.tensor(high_f_comparison))
        max_overall = torch.max(torch.tensor([max_low, max_high])).numpy()

        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
        axs = axs.ravel()
        chns = [0, 7, 15, 25, 35, 45, 55, 63]
        for cnt in range(len(chns)):
            axs[cnt].plot(np.transpose(np.arange(2,16,1)), low_f_comparison[chns[cnt], :], label='Low-Frequency')
            axs[cnt].plot(np.transpose(np.arange(2,16,1)), high_f_comparison[chns[cnt], :], label='High-Frequency')
            axs[cnt].set_xlabel('Frequency Radius')
            axs[cnt].set_ylabel('MSE Error')
            axs[cnt].set_ylim([0, max_overall + 0.05])
            axs[cnt].set_title('Ordered Channel #' + str(chns[cnt]))
            axs[cnt].legend()
            plt.close()

        fig.savefig(root_path + 'mse_comparison' + '.jpg')

def IB_noise_calculation(model, 
                         model_path, 
                         test_loader, 
                         device, 
                         run_name,
                         eval_mode=True, 
                         layer_to_test=0, 
                         capacity_regularization=False, 
                         use_scaling=True):
    '''
    input(s):
        - model
        - samples
        - layer to test
    
    return(s):
        - noise array for each channel in a layer
        - array of robust/non-robust channels in a layer 
    '''
    
    print('layer: ', layer_to_test)

    # load model
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    # initiate model
    if run_name.find('VGG')!= -1:
        n_channels = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        model = proxy_VGG3(model, 
                           eval_mode=eval_mode,
                           device=device,
                           run_name=run_name,
                           noise_variance=0, 
                           IB_noise_calculation=True, 
                           layer_to_test=int(layer_to_test))

        # initiate noise tensor (specific to first channel)
        # for full-BN config we are going to test fo multiple layers
        # while for other configs we test only for the first layer
        if get_bn_int_from_name(run_name) in [100, 1]: 
            noise_length = model.get_bn_parameters()['BN_' + str(layer_to_test)].size(0)
        else: 
            noise_length = n_channels[int(layer_to_test)]
        model.noise_std = 0. + torch.zeros(noise_length, device=device, requires_grad=True) 

        # get channel variance
        if get_bn_int_from_name(run_name) in [100, 1]: 
            running_var = model.get_running_variance()['BN_' + str(layer_to_test)]
        else:
            # since in these configs we do not have the running variance parameter we feed a 
            # batch to the model and take the batch variance as "running variance".
            # RETROSPECTIVE WARNING: "get_running_var" is a wrong for this name since 
            # we are actually retrieving the test variance of the model (when not full-BN).
            running_var = 0
            X, y = next(iter(test_loader))
            X, y = X.to(device), y.to(device) 
            model.get_running_var = True
            _ = model(X)
            running_var = model.running_var
            model.get_running_var = False
        
        running_var = running_var.to(device)


    if run_name.find('ResNet')!= -1:
        model = proxy_ResNet(model, 
                             eval_mode=eval_mode,
                             device=device,
                             run_name=run_name,
                             noise_variance=0, 
                             IB_noise_calculation=True, 
                             layer_to_test=int(layer_to_test))
        
        # initiate noise tensor (specific to first channel)
        noise_length = model.get_bn_parameters()['BN_' + str(layer_to_test)].size(0)
        # model.noise_std  = torch.zeros(noise_length, device=device, requires_grad=True)
        if int(layer_to_test) == 0:
            noise_scaling = 0.5
        elif int(layer_to_test) == 49:
            noise_scaling = 0.4
        elif int(layer_to_test) == 1:
            noise_scaling = 0.4
        elif int(layer_to_test) < 30:
            noise_scaling = 0.2
        else:
            noise_scaling = 0.12
        print('NOISE SCALING: ', noise_scaling)
        model.noise_std  = noise_scaling * torch.ones(noise_length, device=device, requires_grad=True)
        # get channel variance
        running_var = model.get_running_variance()['BN_' + str(layer_to_test)]
    
    # create path
    if model_path.find('bitbucket') != -1:
        print(model_path)
        root_path = './gpucluster/CIFAR10/' + get_model_name(run_name) + '/'
    else:
        root_path = './results/' + get_model_name(run_name) + '/'

    if eval_mode: root_path += 'eval/'
    else: root_path += 'no_eval/' 
    
    root_path += 'PGD' + '/' 
    if not os.path.isdir(root_path): os.mkdir(root_path)

    if capacity_regularization: root_path += 'capacity_regularization' + '/'
    if not os.path.isdir(root_path): os.mkdir(root_path)

    root_path += 'IB_noise_calculation' + '/' 
    if not os.path.isdir(root_path): os.mkdir(root_path)

    root_path += run_name + '/'
    if not os.path.isdir(root_path): os.mkdir(root_path)

    root_path += 'layer_' + str(layer_to_test) + '/'
    if not os.path.isdir(root_path): os.mkdir(root_path)

    if use_scaling:
        print('Use **BN** scaling for ordering')
        root_path += 'BN_scaling' + '/'
        if not os.path.isdir(root_path): os.mkdir(root_path)
    else:
        print('Use **Lambda** scaling for ordering')
        root_path += 'lambda_scaling' + '/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

    # set eval mode
    if eval_mode: model.eval()

    # initiate hyperparam
    lr = 0.01
    beta_IB = 0.01
    iterations = 100

    accs = []

    fig = plt.figure()
    running_var = model.get_running_variance()['BN_' + str(layer_to_test)]
    lambdas = model.get_bn_parameters()['BN_' + str(layer_to_test)].cpu()
    plt.scatter(lambdas, lambdas/torch.sqrt(running_var.cpu()))
    plt.xlabel(r'$\lambda$' + '-values')
    plt.ylabel(r'$\lambda - \sigma$' + ' ratio')
    plt.title('Comparison of ' + r'$\lambda$' + ' and ' + r'$\sigma$' + ' scaling')
    fig.savefig(root_path + 'BN_scaling_comparison.jpg')
    plt.close()

    # calculate noise
    for j in range(iterations):
        correct = 0
        total = 0
        for i, data in enumerate(test_loader, 0):
            # get data
            X,y = data
            X,y = X.to(device), y.to(device)

            # model prediction
            yp = model(X)

            if i==0:
                _, predicted = torch.max(yp.data, 1)
                total = y.size(0)
                correct = (predicted == y).sum().item()
                acc = correct/total
                accs.append(acc)
                print('Accuracy: ', correct/total)
                fig = plt.figure()
                plt.scatter(np.arange(len(accs)), accs)
                plt.xlabel('Iteration')
                plt.ylabel('Accuracy')
                plt.title(run_name.split('_')[0] + ' ' + 'Layer-' + str(layer_to_test))
                fig.savefig(root_path + 'accuracy.jpg')
                fig.savefig('./z_keep_track_noise_IB/keep_track_acc.jpg')
                plt.close()

            # cross-entropy loss
            loss_ce = nn.CrossEntropyLoss()(yp,y)

            # K-L loss
            loss_kl = (1/2)*(torch.sum(running_var/nn.functional.softplus(model.noise_std)) \
                      - torch.sum(torch.log(nn.functional.softplus(model.noise_std)/running_var)) - noise_length)

            # total loss                        
            total_loss = loss_ce + beta_IB*loss_kl

            # get gradients
            total_loss.backward(retain_graph=True)

            # track noise evolution for 1st batch of each iteration
            if i==0:
                # get ordering of the channels depending on the mode chosen (use_scaling or not)
                if get_bn_int_from_name(run_name) in [100, 1]: 
                    lambdas = model.get_bn_parameters()['BN_' + str(layer_to_test)].cpu()
                    sorted_lambdas = torch.argsort(lambdas, descending=True)
                    sorted_lambdas_values, _ = torch.sort(lambdas, descending=True)
                    if use_scaling:
                        if capacity_regularization:
                            scaling = 1/torch.sqrt(running_var).cpu()
                            sorted_lambdas = torch.argsort(scaling, descending=True)
                            sorted_lambdas_values, _ = torch.sort(scaling, descending=True)
                        else:
                            scaling = lambdas/torch.sqrt(running_var).cpu()
                            sorted_lambdas = torch.argsort(scaling, descending=True)
                            sorted_lambdas_values, _ = torch.sort(scaling, descending=True)

                # get noise transformation function depending on model
                if run_name.find('ResNet')!= -1: noise_std = model.variance_function(model.noise_std) **2
                elif run_name.find('VGG')!= -1: noise_std = (nn.functional.softplus(model.noise_std)) **2
                # order channels depending on model config
                if get_bn_int_from_name(run_name) in [100, 1]: ordered_noise_std = noise_std[sorted_lambdas]
                else: ordered_noise_std = noise_std
                
                # plot admissible noise variation against scaling (be it lambdas only or BN scaling overall)
                # fig = plt.figure()
                # fig, ax1 = plt.subplots() 
                ###############################################################################################
                fig = plt.figure(figsize=(6,5))
                ax1 = fig.add_subplot(111)

                if get_bn_int_from_name(run_name) in [100, 1]: x_axis = np.arange(sorted_lambdas.size(0))
                else: x_axis = np.arange(len(n_channels[int(layer_to_test)]))

                plt.scatter(x_axis, ordered_noise_std.cpu().detach().numpy(), c="b")
                if get_bn_int_from_name(run_name) in [100, 1]:
                    ax2 = ax1.twinx()
                    ax2.scatter(x_axis, sorted_lambdas_values.cpu().detach().numpy(), c="orange")
                    if use_scaling:
                        ax2.set_ylabel('BatchNorm Net Scaling')
                    else:
                        ax2.set_ylabel(r'$\lambda$' + ' Scaling')
                ax1.set_xlabel('Ordered Channel Index')
                ax1.set_ylabel('Admissible Noise Variance')

                plt.title(run_name.split('_')[0] + ' ' + 'Layer-' + str(layer_to_test))
                plt.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.1)
                plt.tight_layout()

                fig.savefig(root_path + 'channel_noise_variance.jpg')
                if os.path.isfile(root_path + 'channel_noise_variance.jpg'):
                    print('Success: Figure saved')
                else:
                    print('SaveError: Unable to save Figure')
                fig.savefig('./z_keep_track_noise_IB/keep_track_noise.jpg')
                plt.close()

                fig = plt.figure()
                plt.scatter(np.arange(len(accs)), accs)
                plt.xlabel('Iteration')
                plt.ylabel('Accuracy')
                # plt.title(run_name.split('_')[0] + ' ' + 'Layer-' + str(layer_to_test))
                plt.title('Channel Noise Allowance - ' + run_name.split('_')[0] + ' ' + 'Layer-' + str(layer_to_test))
                fig.savefig(root_path + 'accuracy.jpg')
                fig.savefig('./z_keep_track_noise_IB/keep_track_acc.jpg')
                plt.close()

                ###############################################################################################
                fig = plt.figure(figsize=(6,5))
                ax1 = fig.add_subplot(111)

                avg_len = 8
                mean_val = []
                mean_lambda_val = []
                std_val = []
                std_lambda_val = []
                temporary = ordered_noise_std.cpu().detach().numpy()
                temporary2 = sorted_lambdas_values.cpu().detach().numpy()

                for i in range(0, noise_length, avg_len):
                    #print(ordered_noise_std.cpu().detach().numpy())
                    idx_1 = i
                    idx_2 = i + avg_len

                    mean_val.append(np.mean(temporary[idx_1:idx_2]))
                    std_val.append(np.std(temporary[idx_1:idx_2]))

                    mean_lambda_val.append(np.mean(temporary2[idx_1:idx_2]))
                    std_lambda_val.append(np.std(temporary2[idx_1:idx_2]))
                
                mean_val = np.array(mean_val)
                mean_lambda_val = np.array(mean_lambda_val)
                std_val = np.array(std_val)
                std_lambda_val = np.array(std_lambda_val)
                
                if get_bn_int_from_name(run_name) in [100, 1]: x_axis = np.arange(0, sorted_lambdas.size(0), avg_len)
                else: x_axis = np.arange(0, n_channels[int(layer_to_test)], avg_len)

                ax1.scatter(x_axis, mean_val, c='b')
                ax1.plot(x_axis, mean_val, c='b')
                ax1.fill_between(x_axis, mean_val-std_val, mean_val+std_val, alpha=0.5)
                ax1.set_xlabel('Ordered Channel Index')
                ax1.set_ylabel('Gaussian Noise Standard Deviation')

                if get_bn_int_from_name(run_name) in [100, 1]:
                    ax2 = ax1.twinx()
                    ax2.scatter(x_axis, mean_lambda_val, c="orange")
                    ax2.plot(x_axis, mean_lambda_val, c="orange")
                    #ax2.fill_between(x_axis, mean_lambda_val-std_lambda_val, mean_lambda_val+std_lambda_val, alpha=0.5)
                    if use_scaling:    
                        ax2.set_ylabel('BatchNorm Net Scaling')
                    else:
                        ax2.set_ylabel(r'$\lambda$' + ' Scaling')
                
                plt.title('Channel Noise Allowance - ' + run_name.split('_')[0] + ' ' + 'Layer-' + str(layer_to_test))
                plt.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.1)
                plt.tight_layout()

                fig.savefig(root_path + 'gaussian_noise_std_averaged.jpg')
                if os.path.isfile(root_path + 'gaussian_noise_std_averaged.jpg'):
                    print('Success: Figure saved')
                else:
                    print('SaveError: Unable to save Figure')
                fig.savefig('./z_keep_track_noise_IB/keep_track_IB_noise_av.jpg')
                plt.close()

            # noise update
            model.noise_std = model.noise_std - lr*model.noise_std.grad.detach()
        
        if run_name.find('VGG')!= -1: acc_th = 0.905
        elif run_name.find('ResNet')!= -1: acc_th = 0.925
        if acc > acc_th:
            break
    
    # determine for each channel whether or not it is "robust" or "non-robust"
    # get channel_variation and compare it to the calculated noise:
    if run_name.find('ResNet')!= -1: noise_std = model.variance_function(model.noise_std) ** 2
    elif run_name.find('VGG')!= -1: noise_std = (nn.functional.softplus(model.noise_std)) ** 2
    noise_ordered_channels = torch.argsort(noise_std, descending=False)

    # if the config has Batch Norm at the first layer then we save the noise ordered based on lambdas
    # but we also care to save the ordering of the noise itself to comapre it with that of lambda
    if get_bn_int_from_name(run_name) in [100, 1]: 
        lambdas = model.get_bn_parameters()['BN_' + str(layer_to_test)].cpu()
        if use_scaling:
            scaling = lambdas/torch.sqrt(running_var.cpu())
            sorted_lambdas = torch.argsort(scaling, descending=True)
            sorted_lambdas_values, _ = torch.sort(scaling, descending=True)
        else:
            sorted_lambdas = torch.argsort(lambdas, descending=True)
            sorted_lambdas_values, _ = torch.sort(lambdas, descending=True)

        ordered_noise_std = noise_std[sorted_lambdas]
        
    # else we save it in the given order
    else: ordered_noise_std = noise_std

    np.save(root_path + 'ordered_channel_noise_variance.npy', np.array(ordered_noise_std.cpu().detach()))
    np.save(root_path + 'noise_ordered_lambdas.npy', np.array(noise_ordered_channels.cpu().detach()))
    np.save(root_path + 'sorted_scaling_values.npy', np.array(sorted_lambdas_values.cpu().detach()))

    max_ch_var = torch.max(running_var).cpu().detach()
    noise_std = nn.functional.softplus(model.noise_std).cpu().detach()
    ch_robustness = noise_std**2 > max_ch_var*torch.ones_like(noise_std)
    np.save(root_path + 'ch_robustness.npy', np.array(ch_robustness))

def get_parametric_frequency(model, 
                             model_path, 
                             test_loader, 
                             device, 
                             run_name,
                             eval_mode=True, 
                             layer_to_test=0, 
                             get_parametric_frequency_MSE_only=False,
                             get_parametric_frequency_MSE_CE=False,
                             capacity_regularization=False, 
                             use_scaling=False):
    
    # load model
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    print('LAYER TO TEST: ', layer_to_test)

    # initiate model
    if run_name.find('VGG')!= -1:
        n_channels = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        model = proxy_VGG3(model, 
                           eval_mode=eval_mode,
                           device=device,
                           run_name=run_name,
                           noise_variance=0, 
                           IB_noise_calculation=False,
                           get_parametric_frequency_MSE_only=get_parametric_frequency_MSE_only,
                           get_parametric_frequency_MSE_CE=get_parametric_frequency_MSE_CE,
                           gaussian_std=[0*i for i in range(n_channels[int(layer_to_test)])],
                           layer_to_test=int(layer_to_test))

        
    elif run_name.find('ResNet')!= -1:
        model = proxy_ResNet(model, 
                             eval_mode=eval_mode,
                             device=device,
                             run_name=run_name,
                             noise_variance=0, 
                             IB_noise_calculation=True, 
                             layer_to_test=int(layer_to_test))
    
    # initiate noise tensor (specific to first channel)
    if get_bn_int_from_name(run_name) in [100, 1]: 
        noise_length = model.get_bn_parameters()['BN_' + str(layer_to_test)].size(0)
        print('NOISE LENGTH: ', noise_length)
    else: 
        noise_length = n_channels[int(layer_to_test)]
    
    # initial noise std
    if get_parametric_frequency_MSE_only: std_init = 0.75
    elif get_parametric_frequency_MSE_CE: 
        if get_bn_int_from_name(run_name) == 0:
            std_init = 0.75
        else:
            if int(layer_to_test) > 0:
                std_init = 0.75
            else: 
                std_init = 1

    # initialize noise standard deviation as a list of 2-dimensional tensors (sigma_x, sigma_y)_C
    model.gaussian_std = std_init*torch.ones(noise_length, device=device, requires_grad=True)
    
    # create path
    if model_path.find('bitbucket') != -1:
        root_path = './gpucluster/CIFAR10/' + get_model_name(run_name) + '/'
    else:
        root_path = './results/' + get_model_name(run_name) + '/'

    if eval_mode: root_path += 'eval/'
    else: root_path += 'no_eval/' 
    
    root_path += 'PGD' + '/' 
    if not os.path.isdir(root_path): os.mkdir(root_path)

    if capacity_regularization: root_path += 'capacity_regularization' + '/'
    if not os.path.isdir(root_path): os.mkdir(root_path)

    root_path += 'Gaussian_Parametric_frequency' + '/' 
    if not os.path.isdir(root_path): os.mkdir(root_path)

    if get_parametric_frequency_MSE_only: root_path += 'MSE' + '/'  
    elif get_parametric_frequency_MSE_CE: root_path += 'MSE_CE' + '/'
    if not os.path.isdir(root_path): os.mkdir(root_path)

    root_path += run_name + '/'
    if not os.path.isdir(root_path): os.mkdir(root_path)

    root_path += 'layer_' + str(layer_to_test) + '/'
    if not os.path.isdir(root_path): os.mkdir(root_path)

    if use_scaling:
        print('Use Bn scaling for ordering')
        root_path += 'BN_scaling' + '/'
        if not os.path.isdir(root_path): os.mkdir(root_path)
    else:
        print('Use lambda scaling for ordering')
        root_path += 'lambda_scaling' + '/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

    # set eval mode
    if eval_mode: model.eval()

    # initiate hyperparam
    if get_parametric_frequency_MSE_only: lr = 0.1
    elif get_parametric_frequency_MSE_CE: 
        if layer_to_test == 0:
            lr = 0.001
            beta_mse = 10
        else: 
            lr = 0.5
            beta_mse = 0.5

    iterations = 20

    accs = [] 

    fig = plt.figure()
    running_var = model.get_running_variance()['BN_' + str(layer_to_test)].cpu()
    lambdas = model.get_bn_parameters()['BN_' + str(layer_to_test)]
    plt.scatter(lambdas, lambdas/torch.sqrt(running_var))
    plt.xlabel(r'$\lambda$' + '-values')
    plt.ylabel(r'$\lambda - \sigma$' + ' ratio')
    plt.title('Comparison of ' + r'$\lambda$' + ' and ' + r'$\sigma$' + ' scaling')
    fig.savefig(root_path + 'BN_scaling_comparison.jpg')
    plt.close()

    # calculate std
    for j in range(iterations):
        correct = 0
        total = 0
        for i, data in enumerate(test_loader, 0):
            # get data
            X,y = data
            X,y = X.to(device), y.to(device)

            # model prediction
            yp = model(X)
            
            if get_parametric_frequency_MSE_only:
                gt = model.ground_truth_activations
                loss_mse = nn.MSELoss()(yp, gt)
                loss_mse.backward(retain_graph=True)

                if i == 0: accs.append(loss_mse.cpu().detach().numpy())

            elif get_parametric_frequency_MSE_CE:
                loss_ce = nn.CrossEntropyLoss()(yp,y)
                gt = model.ground_truth_activations
                pred_features = model.gaussian_activations
                loss_mse = nn.MSELoss()(pred_features, gt)
                #loss_mse = 0
                loss = loss_ce + beta_mse*loss_mse
                loss.backward(retain_graph=True)

                if i == 0:
                    _, predicted = torch.max(yp.data, 1)
                    total = y.size(0)
                    correct = (predicted == y).sum().item()
                    accs.append(correct/total)
                    print('ITERATION: ', j)
                    print('Accuracy: ', correct/total)

            # track noise evolution
            if i==0:
                if get_bn_int_from_name(run_name) in [100, 1]: 
                    lambdas = model.get_bn_parameters()['BN_' + str(layer_to_test)]
                    sorted_lambdas = torch.argsort(lambdas, descending=True)
                    sorted_lambdas_values, _ = torch.sort(lambdas, descending=True)  
                    if use_scaling:
                        running_var = model.get_running_variance()['BN_' + str(layer_to_test)]
                        if capacity_regularization:
                            scaling = 1/torch.sqrt(running_var).cpu()
                            sorted_lambdas = torch.argsort(scaling, descending=True)
                            sorted_lambdas_values, _ = torch.sort(scaling, descending=True)
                        else:
                            scaling = lambdas/torch.sqrt(running_var).cpu()
                            sorted_lambdas = torch.argsort(scaling, descending=True)
                            sorted_lambdas_values, _ = torch.sort(scaling, descending=True)
                else:
                    if model_path.find('bitbucket')!=-1:
                        sorted_lambdas = np.load('./gpucluster/CIFAR10/VGG19/eval/PGD/IB_noise_calculation/' + run_name + '/layer_0/noise_ordered_lambdas.npy')
                    else:
                        sorted_lambdas = np.load('./results/VGG19/eval/PGD/IB_noise_calculation/' + run_name + '/layer_0/noise_ordered_lambdas.npy')
                         
                noise_std = model.gaussian_std.detach()
                if get_bn_int_from_name(run_name) in [100, 1]: ordered_noise_std = noise_std[sorted_lambdas]
                else: ordered_noise_std = noise_std[sorted_lambdas]

                ###############################################################################################
                fig = plt.figure(figsize=(6,5))
                ax1 = fig.add_subplot(111)

                if get_bn_int_from_name(run_name) in [100, 1]: x_axis = np.arange(sorted_lambdas.size(0))
                else: x_axis = np.arange(len(n_channels[int(layer_to_test)]))

                plt.scatter(x_axis, ordered_noise_std.cpu().detach().numpy())
                if get_bn_int_from_name(run_name) in [100, 1]:
                    ax2 = ax1.twinx()
                    ax2.scatter(x_axis, sorted_lambdas_values.cpu().detach().numpy(), c="orange")
                    if use_scaling:
                        ax2.set_ylabel('BatchNorm Net Scaling')
                    else:
                        ax2.set_ylabel(r'$\lambda$' + ' Scaling')
                ax1.set_xlabel('Ordered Channel Index')
                ax1.set_ylabel('Gaussian Noise Standard Deviation')

                plt.title('Channel Frequency Allowance - ' + run_name.split('_')[0] + ' ' + 'Layer-' + str(layer_to_test))
                plt.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.1)

                fig.savefig(root_path + 'gaussian_noise_std.jpg')
                if os.path.isfile(root_path + 'gaussian_noise_std.jpg'):
                    print('Success: Figure saved')
                else:
                    print('SaveError: Unable to save Figure')
                if get_parametric_frequency_MSE_only:
                    fig.savefig('./z_keep_track_noise_IB/keep_track_MSE.jpg')
                elif get_parametric_frequency_MSE_CE:
                    fig.savefig('./z_keep_track_noise_IB/keep_track_MSE_CE.jpg')

                plt.close()
                ###############################################################################################

                ###############################################################################################
                fig = plt.figure(figsize=(6,5))
                ax1 = fig.add_subplot(111)

                avg_len = 8
                mean_val = []
                mean_lambda_val = []
                std_val = []
                std_lambda_val = []
                temporary = ordered_noise_std.cpu().detach().numpy()
                temporary2 = sorted_lambdas_values.cpu().detach().numpy()
                for i in range(0, n_channels[int(layer_to_test)], avg_len):
                    #print(ordered_noise_std.cpu().detach().numpy())
                    idx_1 = i
                    idx_2 = i + avg_len

                    mean_val.append(np.mean(temporary[idx_1:idx_2]))
                    std_val.append(np.std(temporary[idx_1:idx_2]))

                    mean_lambda_val.append(np.mean(temporary2[idx_1:idx_2]))
                    std_lambda_val.append(np.std(temporary2[idx_1:idx_2]))
                
                mean_val = np.array(mean_val)
                mean_lambda_val = np.array(mean_lambda_val)
                std_val = np.array(std_val)
                std_lambda_val = np.array(std_lambda_val)

                if get_bn_int_from_name(run_name) in [100, 1]: x_axis = np.arange(0, sorted_lambdas.size(0), avg_len)
                else: x_axis = np.arange(0, n_channels[int(layer_to_test)], avg_len)

                ax1.scatter(x_axis, mean_val, c='b')
                ax1.plot(x_axis, mean_val, c='b')
                ax1.fill_between(x_axis, mean_val-std_val, mean_val+std_val, alpha=0.5)
                ax1.set_xlabel('Ordered Channel Index')
                ax1.set_ylabel('Gaussian Noise Standard Deviation')

                if get_bn_int_from_name(run_name) in [100, 1]:
                    ax2 = ax1.twinx()
                    ax2.scatter(x_axis, mean_lambda_val, c="orange")
                    ax2.plot(x_axis, mean_lambda_val, c="orange")
                    #ax2.fill_between(x_axis, mean_lambda_val-std_lambda_val, mean_lambda_val+std_lambda_val, alpha=0.5)
                    if use_scaling:
                        ax2.set_ylabel('BatchNorm Net Scaling')
                    else:
                        ax2.set_ylabel(r'$\lambda$' + ' Scaling')
                
                plt.title('Channel Frequency Allowance - ' + run_name.split('_')[0] + ' ' + 'Layer-' + str(layer_to_test))
                plt.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.1)

                fig.savefig(root_path + 'gaussian_noise_std_averaged.jpg')
                if os.path.isfile(root_path + 'gaussian_noise_std_averaged.jpg'):
                    print('Success: Figure saved')
                else:
                    print('SaveError: Unable to save Figure')
                if get_parametric_frequency_MSE_only:
                    fig.savefig('./z_keep_track_noise_IB/keep_track_MSE_av.jpg')
                elif get_parametric_frequency_MSE_CE:
                    fig.savefig('./z_keep_track_noise_IB/keep_track_MSE_CE_av.jpg')

                plt.close()
                ###############################################################################################

                if get_parametric_frequency_MSE_only:
                    fig = plt.figure()
                    plt.scatter(np.arange(len(accs)), accs)
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.title(run_name.split('_')[0] + ' ' + 'layer' + str(layer_to_test))
                    fig.savefig(root_path + 'loss.jpg')
                    fig.savefig('./z_keep_track_noise_IB/keep_track_loss.jpg')
                    plt.close()
                elif get_parametric_frequency_MSE_CE:
                    fig = plt.figure()
                    plt.scatter(np.arange(len(accs)), accs)
                    plt.xlabel('Iteration')
                    plt.ylabel('Accuracy')
                    plt.title(run_name.split('_')[0] + ' ' + 'layer' + str(layer_to_test))
                    fig.savefig(root_path + 'accuracy.jpg')
                    fig.savefig('./z_keep_track_noise_IB/keep_track_accuracy.jpg')
                    plt.close()

            # noise update
            model.gaussian_std = model.gaussian_std - lr*model.gaussian_std.grad.detach()


    noise_std = model.gaussian_std

    frequency_ordered_channels = torch.argsort(noise_std, descending=False)

    # if the config has Batch Norm at the first layer then we save the noise ordered based on lambdas
    # but we also care to save the ordering of the noise itself to comapre it with that of lambda
    if get_bn_int_from_name(run_name) in [100, 1]: 
        lambdas = model.get_bn_parameters()['BN_' + str(layer_to_test)]
        sorted_lambdas = torch.argsort(lambdas, descending=True)
        sorted_lambdas_values, _ = torch.sort(lambdas, descending=True)
        if use_scaling:
            running_var = model.get_running_variance()['BN_' + str(layer_to_test)]
            if capacity_regularization:
                scaling = 1/torch.sqrt(running_var).cpu()
                sorted_lambdas = torch.argsort(scaling, descending=True)
                sorted_lambdas_values, _ = torch.sort(scaling, descending=True)
            else:
                scaling = lambdas/torch.sqrt(running_var).cpu()
                sorted_lambdas = torch.argsort(scaling, descending=True)
                sorted_lambdas_values, _ = torch.sort(scaling, descending=True)
 
        ordered_noise_std = noise_std[sorted_lambdas]
    
    # else we save it in the given order
    else: 
        if model_path.find('bitbucket')!=-1:
            sorted_lambdas = np.load('./gpucluster/CIFAR10/VGG19/eval/PGD/IB_noise_calculation/' + run_name + '/layer_0/noise_ordered_lambdas.npy')
        else:
            sorted_lambdas = np.load('./results/VGG19/eval/PGD/IB_noise_calculation/' + run_name + '/layer_0/noise_ordered_lambdas.npy')
        
        ordered_noise_std = noise_std[sorted_lambdas]
    
    np.save(root_path + 'ordered_channel_noise_variance.npy', np.array(ordered_noise_std.cpu().detach()))
    np.save(root_path + 'sorted_scaling_values.npy', np.array(sorted_lambdas_values.cpu().detach()))
    np.save(root_path + 'frequency_ordered_channels.npy', np.array(frequency_ordered_channels.cpu().detach()))
    
def test_low_pass_robustness(model, 
                             model_path, 
                             model_tag,
                             PATH_to_deltas_,
                             test_loader, 
                             device, 
                             run_name,
                             attack,
                             epsilon,
                             num_iter,
                             radius,
                             capacity_regularization=False,
                             regularization_mode='',
                             eval_mode=True):

    print('Epsilon: ', epsilon)
    print('Radius:', radius)

    # load model
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device) 

    if capacity_regularization:
        model = model_setup(model, 
                            model_path, 
                            device,
                            eval_mode, 
                            run_name, 
                            noise_variance=0, 
                            regularization_mode=regularization_mode)

    # set eval mode
    if eval_mode: model.eval()
    if run_name.find('VGG')!= -1: model.classifier.eval()
    
    # getting min and max pixel values to be used in PGD for clamping
    min_tensor, max_tensor = get_minmax(test_loader=test_loader, device=device)

    # helper variables
    total = 0
    correct_s = 0
    correct_clean = 0

    for i, data in enumerate(test_loader, 0):
        X, y = data
        X, y = X.to(device), y.to(device)

        # get path to existing perturbations
        path = get_path2delta(PATH_to_deltas_, model_tag, run_name, attack, epsilon)

        # perform PGD if not existent already
        name_out = 'adversarial_delta_' + str(i) + '.pth'
        
        if not os.path.isfile(path + name_out):
            # perform PGD
            delta = pgd_linf(model, 
                             X, 
                             y, 
                             epsilon, 
                             max_tensor, 
                             min_tensor,
                             alpha=epsilon/10, 
                             num_iter=num_iter, 
                             noise_injection=False) 

            # save perturbations
            torch.save(delta[0], path + name_out)
            # create adversarial examples
            adv_inputs = X + delta[0]
            input = adv_inputs

        else:
            delta = torch.load(path + name_out)
            if radius < 16:
                low_f_img, _ = generateDataWithDifferentFrequencies_3Channel(delta.cpu().numpy(), radius)
                low_f_img = torch.tensor(low_f_img, dtype=torch.float32, device=device)
                input = X + low_f_img
            else:
                input = X + delta
                
        with torch.no_grad():
            outputs_clean = model(X)
            outputs = model(input)
            
        _, predicted_clean = torch.max(outputs_clean.data, 1)
        _, predicted = torch.max(outputs.data, 1)
        
        #print('clean ------------------------------: ', (predicted_clean == y).sum().item())
        #print('adversarial ------------------------------: ', (predicted == y).sum().item())

        total += y.size(0)
        correct_clean += (predicted_clean == y).sum().item()
        correct_s += (torch.logical_and(predicted == y, predicted_clean == y)).sum().item()

    print(correct_s/total)

    return correct_s/total

def compare_frequency_domain(model, 
                             model_path, 
                             test_loader, 
                             device, 
                             run_name,
                             eval_mode=True, 
                             layer_to_test=0, 
                             get_HF_difference=False, 
                             layer_wise_analysis=False):
    
    # load model
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    print('LAYER TO TEST: ', layer_to_test)

    # initiate model
    if run_name.find('VGG')!= -1:
        model = proxy_VGG2(model, 
                           eval_mode=eval_mode,
                           device=device,
                           noise_variance=0,
                           run_name=run_name)

        
    elif run_name.find('ResNet')!= -1:
        model = proxy_ResNet(model, 
                             eval_mode=eval_mode,
                             device=device,
                             run_name=run_name,
                             noise_variance=0, 
                             layer_to_test=int(layer_to_test))
    
     # set eval mode for inference 
    if eval_mode: model.eval()

    # create path
    if model_path.find('bitbucket') != -1:
        root_path = './gpucluster/CIFAR10/' + get_model_name(run_name) + '/'
    else:
        root_path = './results/' + get_model_name(run_name) + '/'

    if eval_mode: root_path += 'eval/'
    else: root_path += 'no_eval/' 
    
    root_path += 'PGD' + '/' 
    if not os.path.isdir(root_path): os.mkdir(root_path)

    root_path += 'frequency_domain_comparison' + '/' 
    if not os.path.isdir(root_path): os.mkdir(root_path)

    root_path += run_name + '/'
    if not os.path.isdir(root_path): os.mkdir(root_path)

    if layer_wise_analysis:

        root_path += 'layer_' + str(layer_to_test) + '/'
        if not os.path.isdir(root_path): os.mkdir(root_path)

        # retrieve batch from loader
        X, y = next(iter(test_loader))
        X, y = X.to(device), y.to(device) 

        _ = model(X)
        
        # get activations from each of the 64 channels BEFORE and AFTER BN  
        before_bn = model.conv1.cpu().detach()
        after_bn = model.bn1.cpu().detach()

        # compute fft
        _, high_f_before = get_frequency_components(before_bn.numpy(), r=15)    
        _, high_f_after = get_frequency_components(after_bn.numpy(), r=15)

        if layer_to_test == 0:
            if run_name.find('VGG')!= -1:
                lambdas = model.bn.weight.detach()
            elif run_name.find('ResNet')!= -1:
                lambdas = model.net.bn1.weight.detach()
        else:
            if run_name.find('VGG')!= -1:
                lambdas = model.get_bn_parameters()['BN_' + str(layer_to_test)].detach()
                
            elif run_name.find('ResNet')!= -1:
                lambdas = model.get_bn_parameters()['BN_' + str(layer_to_test)].detach()
                running_variances = model.get_running_variance()['BN_' + str(layer_to_test)].detach()

        sorted_lambdas = torch.argsort(lambdas, descending=True)

        # save std layer distribution
        sns.displot(data=torch.sqrt(running_variances).cpu().numpy(), kind="kde")
        plt.savefig(root_path + 'std_layer_distribution.png')
        plt.close() 

        print(lambdas[sorted_lambdas].cpu().detach().numpy())

        # save lambda-std ratio ordered by lambda
        fig = plt.figure()
        plt.scatter(sorted_lambdas.cpu().detach().numpy(), \
            lambdas[sorted_lambdas].cpu().detach().numpy()/np.sqrt(running_variances[sorted_lambdas].cpu().detach().numpy()))
        plt.hlines(y=1, xmin=0, xmax=torch.max(sorted_lambdas).cpu().numpy(), color='red', linestyle ='dashed', linewidth = 2)
        plt.xlabel('Sorted ' + r'$\lambda$' + ' indexes')
        plt.ylabel(r'$\lambda - \sigma $' + ' ratio')
        plt.title('BatchNorm Net Scaling comparison with ' + r'$\lambda$')
        plt.savefig(root_path + 'BN_scaling_vs_lambda.png')
        plt.close()

        if get_HF_difference:
            for i, sample in enumerate(X):
                print(i)
                if i > 10: break
                path = root_path + 'img_' + str(i) + '/'
                if not os.path.isdir(path): os.mkdir(path)

                lamda_var_ratio = []
                var_before_and_after_ratio = []
                mean_before_and_after_ratio = []

                for num, j in enumerate(sorted_lambdas):
                    
                    if layer_to_test == 0:
                        if run_name.find('VGG')!= -1:
                            lamda_var_ratio.append(lambdas[j].cpu().detach().numpy()/np.sqrt(model.bn.running_var[j].cpu().detach().numpy()))
                        elif run_name.find('ResNet')!= -1:
                            lamda_var_ratio.append(lambdas[j].cpu().detach().numpy()/np.sqrt(model.net.bn1.running_var[j].cpu().detach().numpy()))
                    
                        frequency_activations_before = high_f_before[i]
                        frequency_activations_after = high_f_after[i]

                        path_out = path + 'ch_' + str(int(num)) + '/'
                        if not os.path.isdir(path_out): os.mkdir(path_out)

                        fig, (ax1, ax2) = plt.subplots(figsize=(10, 3), ncols=2)

                        ax1.set_axis_off()
                        im = ax1.imshow(frequency_activations_before[j, :, :], cmap='viridis')
                        
                        ax2.set_axis_off()
                        im = ax2.imshow(frequency_activations_after[j, :, :], cmap='viridis')
                    
                        fig.subplots_adjust(right=0.8)
                        fig.colorbar(im, ax=[ax1, ax2], shrink=0.95)
                        fig.savefig(path_out + 'frequency_component_comparison.png')
                        plt.close()

                        temp_before = get_flattend_frequency_components(frequency_activations_before[j, :, :], r=15)
                        temp_after = get_flattend_frequency_components(frequency_activations_after[j, :, :], r=15)
                        temp_dict = {'Before BN': temp_before, 'After BN': temp_after}
                        sns.displot(data=temp_dict, kind="kde")
                        plt.savefig(path_out + 'distribution.png')
                        plt.close()

                        var_before_and_after_ratio.append(np.var(temp_after)/np.var(temp_before))
                        mean_before_and_after_ratio.append(np.mean(np.abs(temp_after))/np.mean(np.abs(temp_before)))

                    else:
                        if run_name.find('VGG')!= -1:
                            lamda_var_ratio.append(lambdas[j].cpu().detach().numpy()/np.sqrt(running_variances[j].cpu().detach().numpy()))
                        elif run_name.find('ResNet')!= -1:
                            lamda_var_ratio.append(lambdas[j].cpu().detach().numpy()/np.sqrt(running_variances[j].cpu().detach().numpy()))

                if layer_to_test == 0:
                    fig = plt.figure()
                    plt.scatter(lamda_var_ratio, var_before_and_after_ratio)
                    plt.hlines(y=1, xmin=np.min(lamda_var_ratio), xmax=np.max(lamda_var_ratio), color='red', linestyle ='dashed', linewidth = 2)
                    plt.xlabel(r'$\lambda - \sigma $' + ' ratio')
                    plt.ylabel('Variance Ratio (After-Before) of HF')
                    plt.title('BatchNorm parameters comparison to HF behaviour')
                    fig.savefig(path + 'BN_params_vs_HF.png')
                    plt.close()

                    fig = plt.figure()
                    plt.scatter(lamda_var_ratio, mean_before_and_after_ratio)
                    plt.hlines(y=1, xmin=np.min(lamda_var_ratio), xmax=np.max(lamda_var_ratio), color='red', linestyle ='dashed', linewidth = 2)
                    plt.xlabel(r'$\lambda - \sigma $' + ' ratio')
                    plt.ylabel('Mean Ratio (After-Before) of HF')
                    plt.title('BatchNorm parameters comparison to HF behaviour')
                    fig.savefig(path + 'BN_params_vs_HF_mean.png')
                    plt.close()

                fig = plt.figure()
                to_plot = [lamda_var_ratio[i] for i in sorted_lambdas.cpu().detach().numpy()]
                plt.scatter(sorted_lambdas.cpu().detach().numpy(), to_plot)
                plt.hlines(y=1, xmin=0, xmax=np.max(sorted_lambdas.cpu().detach().numpy()), color='red', linestyle ='dashed', linewidth = 2)
                plt.xlabel('Sorted ' + r'$\lambda$' + ' indexes')
                plt.ylabel(r'$\lambda - \sigma $' + ' ratio')
                plt.title('BatchNorm Net Scaling comparison with ' + r'$\lambda$')
                fig.savefig(path + 'BN_scaling_vs_lambda.png')
                plt.close()

    else:
        root_path += 'layers_scaling' + '/'
        if not os.path.isdir(root_path): os.mkdir(root_path)
        lambdas = model.get_bn_parameters()
        running_variances = model.get_running_variance()
        np.save(root_path + 'lambdas.npy', lambdas)
        np.save(root_path + 'running_variances.npy', running_variances)

def test_SquareAttack(model, 
                      model_path, 
                      test_loader, 
                      device, 
                      run_name,
                      epsilon,
                      n_queries,
                      eval_mode=True):
    # load model
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    # set eval mode for inference 
    if eval_mode: model.eval()

    print('EVAL MODE: ', not model.training)

    # getting min and max pixel values to be used in PGD for clamping
    min_tensor, max_tensor = get_minmax(test_loader=test_loader, device=device)

    version = 'custom'
    adversary = AutoAttack(model, 
                           norm='Linf', 
                           eps=epsilon, 
                           version=version,
                           device=device,
                           verbose=False,
                           min_tensor=min_tensor, 
                           max_tensor=max_tensor, 
                           n_queries=n_queries)

    adversary.attacks_to_run = ['square']

    total = 0
    correct_s = 0
    for _, data in enumerate(test_loader, 0):
        X, y = data
        X, y = X.to(device), y.to(device)

        advimg = adversary.run_standard_evaluation(X, y.type(torch.LongTensor).to(device), 
                 bs=X.size(0))
        
        outputs = model(advimg)
        outputs_clean = model(X)

        _, predicted_clean = torch.max(outputs_clean.data, 1)
        _, predicted = torch.max(outputs.data, 1)

        print('adversarial ------------------------------: ', (predicted == y).sum().item())

        total += y.size(0)
        correct_s += (torch.logical_and(predicted == y, predicted_clean == y)).sum().item()

    return correct_s/total

def HF_attenuate(model, 
                 model_path, 
                 test_loader, 
                 device, 
                 run_name,
                 epsilon,
                 num_iter,
                 radius=15,
                 layer_to_test=0,
                 attenuate_HF=True,
                 eval_mode=True):

     # load model
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)       

    # initiate model
    if run_name.find('VGG')!= -1:
        model = proxy_VGG3(model, 
                           eval_mode=eval_mode,
                           device=device,
                           run_name=run_name,
                           noise_variance=0, 
                           attenuate_HF=attenuate_HF,
                           layer_to_test=int(layer_to_test))

    # set eval mode
    if eval_mode: model.eval()
    if run_name.find('VGG')!= -1:
        model.classifier.eval()
    
    # getting min and max pixel values to be used in PGD for clamping
    min_tensor, max_tensor = get_minmax(test_loader=test_loader, device=device)

    # helper variables
    correct_s = 0
    total = 0

    for i, data in enumerate(test_loader, 0):
        X, y = data
        X, y = X.to(device), y.to(device)

        delta = pgd_linf(model, 
                         X, 
                         y, 
                         epsilon, 
                         max_tensor, 
                         min_tensor,
                         alpha=epsilon/10, 
                         num_iter=num_iter)
        
        adv_inputs = X + delta[0]

        outputs = model(adv_inputs)
        outputs_clean = model(X)

        _, predicted_clean = torch.max(outputs_clean.data, 1)
        _, predicted = torch.max(outputs.data, 1)

        total += y.size(0)
        correct_s += (torch.logical_and(predicted == y, predicted_clean == y)).sum().item()
    
    return correct_s/total

def adversarial_transferrability(model, 
                                 model_path, 
                                 model_2, 
                                 model_path_2,
                                 model_tag,
                                 PATH_to_deltas_,
                                 test_loader, 
                                 device, 
                                 run_name,
                                 run_name_2,
                                 attack,
                                 epsilon,
                                 num_iter,
                                 eval_mode=True):

    print('Attack: ', attack)
    print('Epsilon: ', epsilon)
    print('--------------------------')
    print('Model ATTACKING: ', run_name.split('_')[0] + '_' + run_name.split('_')[1])
    print('Model being ATTACKED: ', run_name_2.split('_')[0] + '_' + run_name_2.split('_')[1])

    # load model from which attacks are transferred
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)     
    
    # set eval mode
    if eval_mode: model.eval()
    if run_name.find('VGG')!= -1: model.classifier.eval()
    print('EVAL MODE: ', not model.training)
    
    # getting min and max pixel values to be used in PGD for clamping
    min_tensor, max_tensor = get_minmax(test_loader=test_loader, device=device)

    # helper variables
    total = 0
    correct_s = 0
    correct_clean = 0

    # attack model and save pertrubations
    for i, data in enumerate(test_loader, 0):
        X, y = data
        X, y = X.to(device), y.to(device)

        # get path to existing perturbations
        path = get_path2delta(PATH_to_deltas_, model_tag, run_name, attack, epsilon)

        # perform PGD if not existent already
        name_out = 'adversarial_delta_' + str(i) + '.pth'
        
        if not os.path.isfile(path + name_out):
            # perform PGD
            delta = pgd_linf(model, 
                             X, 
                             y, 
                             epsilon, 
                             max_tensor, 
                             min_tensor,
                             alpha=epsilon/10, 
                             num_iter=num_iter, 
                             noise_injection=False) 

            # save perturbations
            torch.save(delta[0], path + name_out)
            # create adversarial examples
            adv_inputs = X + delta[0].to(device)
            input = adv_inputs

        # if the perturbations of the model already exist in memory just load the, 
        else:
            delta = torch.load(path + name_out)
            input = X + delta.to(device)
        
        # now test the obtained perturbations on the second model
        # load model from which attacks are transferred
        model_2.load_state_dict(torch.load(model_path_2, map_location='cpu'))
        model_2.to(device)     
        
        # set eval mode
        if eval_mode: model_2.eval()
        if run_name_2.find('VGG')!= -1: model_2.classifier.eval()
        # print('EVAL MODE: ', not model_2.training)

        with torch.no_grad():
            outputs_clean = model_2(X)
            outputs = model_2(input)
            
        _, predicted_clean = torch.max(outputs_clean.data, 1)
        _, predicted = torch.max(outputs.data, 1)
        
        #print('clean ------------------------------: ', (predicted_clean == y).sum().item())
        #print('adversarial ------------------------------: ', (predicted == y).sum().item())

        total += y.size(0)
        correct_clean += (predicted_clean == y).sum().item()
        correct_s += (torch.logical_and(predicted == y, predicted_clean == y)).sum().item()

    print('Adversarial transferred accuracy: ', correct_s/total)

    return correct_s/total