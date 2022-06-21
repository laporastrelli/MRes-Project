from tkinter.tix import Tree
from bleach import clean
import torch.nn as nn
import torch
import os
import csv
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils_.adversarial_attack import fgsm, pgd_linf, pgd_linf_capacity, pgd_linf_total_capacity
from utils_ import get_model
from utils_.miscellaneous import get_minmax, get_path2delta, get_bn_int_from_name, CKA, cosine_similarity, get_model_name
from utils_.log_utils import get_csv_path, get_csv_keys

from advertorch.attacks import LinfPGDAttack

from models.proxy_VGG import proxy_VGG
from models.proxy_VGG2 import proxy_VGG2
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
                noise_variance):

    net.load_state_dict(torch.load(model_path))
    net.to(device)

    ################## MODEL SELECTION ##################
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
        capacity=0,
        get_logits=False):

    # net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    net.load_state_dict(torch.load(model_path))
    net.to(device)

    ################## TEMPORARY ##################
    if run_name.find('ResNet')!= -1:
        net = proxy_ResNet(net, 
                           eval_mode=eval_mode,
                           device=device,
                           run_name=run_name,
                           noise_variance=noise_variance)
    ################## TEMPORARY ################## 

    if inject_noise:
        if run_name.find('VGG') != -1:
            noisy_net = noisy_VGG(net, 
                                  eval_mode=eval_mode,
                                  noise_variance=noise_variance, 
                                  device=device,
                                  capacity_=capacity,
                                  noise_capacity_constraint=noise_capacity_constraint, 
                                  run_name=run_name)

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
            noisy_net = noisy_VGG(net, 
                                  eval_mode=eval_mode,
                                  noise_variance=noise_variance, 
                                  device=device,
                                  capacity_=capacity,
                                  noise_capacity_constraint=noise_capacity_constraint, 
                                  run_name=run_name)
            outputs = noisy_net(X)
            temp = outputs
        elif get_logits:
            print('GET LOGITS')
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

        _, predicted = torch.max(outputs.data, 1)

        total += y.size(0)
        correct += (predicted == y).sum().item()

    print('Test Accuracy: %d %%' % (100 * correct / total))
       
    acc = correct / total
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
                       capacity_mode='lambda_based',
                       use_pop_stats=True,
                       batch=0,
                       noise_variance=0,
                       capacity_regularization=False,
                       beta=0,
                       regularization_mode='euclidean',
                       save_analysis=True):
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
        path_out += get_model_name(run_name) +'/'
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
        if save_analysis:
            path_out += run_name + '/'
            if not os.path.isdir(path_out): 
                os.mkdir(path_out)
            if capacity_regularization:
                path_out += regularization_mode + '/'
                if not os.path.isdir(path_out): 
                    os.mkdir(path_out)
                path_out += str(beta).replace('.','') + '/'
                if not os.path.isdir(path_out): 
                    os.mkdir(path_out)
            path_out += 'lambdas.npy'
            if not os.path.isfile(path_out):
                np.save(path_out, lambdas)

    # variance-based capacity calculation
    elif capacity_mode == 'variance_based':
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
                     transfer_mode='variance_based',
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

    print('Channel being transferred: ', layer_key)

    # get number of channels in layer
    # print(net.get_bn_parameters())
    channel_size = net.get_bn_parameters()[layer_key[0]].size(0)

    # get number of channels to transfer
    if channel_transfer in ['largest', 'smallest']: transfers = np.arange(9)
    elif channel_transfer == 'individual': transfers = np.arange(0, channel_size, int(channel_size/64))
    print('channels to transfer: ', len(transfers))

    # create directory
    dir_path = './results/' + get_model_name(run_name) + '/eval/PGD/channel_transfer/' \
               + run_name.split('_')[0]  + '_' +  run_name.split('_')[1] + '/'  
    if not os.path.isdir(dir_path): os.mkdir(dir_path)
    dir_path += layer_key[0] + '/'
    if not os.path.isdir(dir_path): os.mkdir(dir_path)
    dir_path += transfer_mode + '/'
    if not os.path.isdir(dir_path): os.mkdir(dir_path)

    # get min and max tensor
    min_tensor, max_tensor = get_minmax(test_loader=test_loader, device=device)
    
    # counters
    correct_clean = 0
    correct_s = 0
    total = 0

    for n_channels in transfers:
        for eps in epsilon_list:
            epsilon = float(eps)
            for i, data in enumerate(test_loader, 0):
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
                        (len(csv_dict[curr_key]) == channel_size and channel_transfer=='individual'): 
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

                # select channels (i.e. channels-corresponding channels) to transfer
                tot_num_channels = tmp_capacity_idx.size(0)
                if channel_transfer in ['smallest', 'largest']:
                    if int(n_channels) != 0:
                        capacity_ch = tmp_capacity_idx[0:int(n_channels)*int((tot_num_channels/8))].cpu().detach().numpy()
                    else:
                        capacity_ch = tmp_capacity_idx[0].cpu().detach().numpy()
                elif channel_transfer == 'individual':
                    capacity_ch = tmp_capacity_idx[int(n_channels)].cpu().detach().numpy()

                capacity_activations = adv_activations[layer_key[0]][-1][:, capacity_ch, :, :]
                print(capacity_ch, int(layer_key[0][-1]), capacity_activations.shape)
                transfer_activation = [capacity_ch, int(layer_key[0].split('_')[-1]), capacity_activations]

                # subsitute clean activation with adversarial activations 
                with torch.no_grad():
                    outputs = net(X, transfer_activation)
                    outputs_clean = net(X) 

                _, predicted_clean = torch.max(outputs_clean.data, 1)
                _, predicted = torch.max(outputs.data, 1)

                correct_clean += (predicted_clean == y).sum().item()

                print('clean ------------------------------: ', (predicted_clean == y).sum().item())
                print('adversarial ------------------------------: ', (predicted == y).sum().item())

                total += y.size(0)
                correct_s += (torch.logical_and(predicted == y, predicted_clean == y)).sum().item()

                # save result 
                if (channel_transfer in ['smallest', 'largest'] and i == 20) or (channel_transfer=='individual' and i == 6):
                    fname = dir_path + channel_transfer + '.npy'
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
    net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    net.to(device)

    ################## MODEL SELECTION ##################
    if inject_noise:
        print('Adversarial Test With Noise ...')
        net = noisy_VGG(net, 
                        eval_mode=use_pop_stats,
                        noise_variance=noise_variance, 
                        device=device,
                        capacity_=capacity,
                        noise_capacity_constraint=noise_capacity_constraint,
                        run_name=run_name, 
                        scaled_noise=scaled_noise, 
                        scaled_noise_norm=scaled_noise_norm, 
                        scaled_noise_total=scaled_noise_total)
        
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
    #####################################################

    ################## EVAL MODE ##################
    if use_pop_stats:
        net.eval()
    # setting dropout to eval mode (in VGG)
    if run_name.find('VGG')!= -1:
        net.classifier.eval()
    ###############################################


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
                
                # ------ Advertorch PGD function
                else:
                    adversary = LinfPGDAttack(
                                    net, loss_fn=nn.CrossEntropyLoss(), eps=epsilon,
                                    nb_iter=40, eps_iter=epsilon/10, rand_init=False, clip_min=min_tensor, clip_max=max_tensor,
                                    targeted=False)
                    adv_inputs = adversary.perturb(X, y)
                    delta = [adv_inputs-X]
            
            if len(delta) == 1:
                if save:     
                    path = get_path2delta(PATH_to_deltas_, model_tag, run_name, attack)
                    eps_ = 'eps_' + str(epsilon).replace('.', '')
                    if not os.path.isdir(path + '/' + eps_ + '/'):
                        os.mkdir(path + '/' + eps_ + '/')
                    torch.save(delta[0], path + '/' + eps_ + "/adversarial_delta_" + str(i) + ".pth") 
                
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
    
    print('Saliency Map')
        
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    if run_name.find('VGG')!= -1:
        net = proxy_VGG2(model, 
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

        ch_max = torch.argsort(net.bn.weight, descending=True)[0]
        ch_min = torch.argsort(net.bn.weight, descending=False)[0]
        
        chs = [ch_max, ch_min]
        modes = ['max', 'min']
        
        inv_normalize = transforms.Normalize(
            mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
            std=[1/0.2023, 1/0.1994, 1/0.2010])

        input_img = inv_normalize(X[j])

        map_min = torch.min(saliency_map)    
        map_max = torch.max(saliency_map)

        for jj in range(saliency_map.size(1)):
            if jj in chs:
                temp = saliency_map[0, jj, :, :]
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
                plt.imshow(net.bn1[0, jj, :, :].cpu().detach().numpy())
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

        ordered_channels = torch.argsort(net.bn.weight, descending=False)
        for bb, ch_ in enumerate(ordered_channels):
            fig = plt.figure()
            plt.imshow(net.bn1[0, ch_, :, :].cpu().detach().numpy())
            plt.xticks([])
            plt.yticks([])
            plt.title('Ordered Channel #' + str(bb))
            plt.savefig('./results/VGG19/eval/PGD/saliency_maps/channels/' + 'ch_' + str(bb) + '.jpg')
            plt.close()
        
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

