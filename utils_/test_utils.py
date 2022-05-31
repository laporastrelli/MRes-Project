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
from utils_.adversarial_attack import fgsm, pgd_linf, pgd_linf_capacity
from utils_ import get_model
from utils_.miscellaneous import get_minmax, get_path2delta, get_bn_int_from_name, CKA, cosine_similarity
from utils_.log_utils import get_csv_path, get_csv_keys

#### check ####
from advertorch.attacks import LinfPGDAttack
#### check ####

#### temporary ####
from models.proxy_VGG import proxy_VGG
from models.proxy_VGG2 import proxy_VGG2
from models.noisy_VGG import noisy_VGG
from models.test_VGG import test_VGG
#### temporary ####


# TODO: ##################################################################################
# - Remember to change max and min of clamp in PGD (adversarial_attack.py) when using SVHN
# - BEWARE OF net.eval ---> CRUCIAL
# - Remember to save delta tensors if needed (it is now disabled)
# TODO: ##################################################################################


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
    if inject_noise:
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
            print('STANDARD TEST')
            outputs = net(X)

        _, predicted = torch.max(outputs.data, 1)

        total += y.size(0)
        correct += (predicted == y).sum().item()
        print((predicted == y).sum().item())

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
                     get_similarity=False, 
                     eval=True, 
                     custom=True,
                     save=False, 
                     save_analysis=False,
                     get_max_indexes=False, 
                     channel_transfer='', 
                     n_channels=0):

    net.load_state_dict(torch.load(model_path))
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
                        scaled_noise_norm=scaled_noise_norm)

    if capacity_calculation or len(get_similarity)>0 or channel_transfer:
        net = proxy_VGG(net, 
                        eval_mode=use_pop_stats,
                        device=device,
                        run_name=run_name,
                        noise_variance=noise_variance)
    
    ################## EVAL MODE ##################
    if use_pop_stats:
        net.eval()
    # setting dropout to eval mode (in VGG)
    if run_name.find('VGG')!= -1:
        net.classifier.eval()

    ################## VERBOSE ##################
    # printing info
    print('eval MODE:                       ', use_pop_stats)
    print('inject noise MODE:               ', inject_noise)
    print('test on clean batch stats MODE:  ', no_eval_clean)
    print('---------------------------------')

    if run_name.find('VGG')!= -1:
        print('features training MODE:          ', net.features.training)
        print('average pooling training MODE:   ', net.avgpool.training)
        print('classifier training MODE:        ', net.classifier.training)

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
                    if i==0 and (capacity_calculation or len(get_similarity) > 0 or get_max_indexes) and get_bn_int_from_name(run_name)!=0:
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
                            layer_key = ['BN_0']
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
                        '''# ------------ TEMP ------------:
                        bn_dict = net.get_bn_parameters()
                        test_var = net.get_test_variance()
                        fig = plt.figure()
                        # plt.scatter(bn_dict[layer_key[0]][1].cpu().detach().numpy(), capacities[layer_key[0]][0])
                        plt.scatter(test_var[layer_key[0]].cpu().detach().numpy(), capacities[layer_key[0]][0])
                        plt.xlabel('Population Variance')
                        plt.ylabel('Capacity')
                        plt.savefig('./temp.jpg')
                        # ------------ TEMP -------------'''
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
                                    print(folder_name)
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
                        elif len(channel_transfer)>0 and get_bn_int_from_name(run_name)==1:
                            
                            fname = './results/VGG19/eval/PGD/channel_transfer/' + channel_transfer + '.npy'
                            curr_key = str(run_name + '_' + str(epsilon))
                            csv_dict = np.load(fname, allow_pickle='TRUE').item()
                            if curr_key in csv_dict.keys() and len(csv_dict[curr_key]) == 9:
                                correct_s = 100
                                total = 100
                                correct_clean = 100
                                print('------ KEY ALREADY FULL ------')
                                break

                            print('FEATURE TRANSFER MODE')
                            net.set_verbose(verbose=True)
                            if run_name.find('bn')!= -1:
                                layer_key = ['BN_' + str(i) for i in range(16)]
                            else:
                                layer_key = ['BN_0']
                            _, capacities, adv_activations = pgd_linf_capacity(net, 
                                                                                X, 
                                                                                y, 
                                                                                epsilon, 
                                                                                max_tensor, 
                                                                                min_tensor, 
                                                                                alpha=epsilon/10, 
                                                                                num_iter=num_iter, 
                                                                                layer_key=layer_key)         
                            bn_variance = net.get_bn_parameters()['BN_0']       

                            
                            net.set_verbose(verbose=False)
                            if channel_transfer == 'largest':
                                descending = True
                            elif channel_transfer == 'smallest':
                                descending = False

                            tmp_capacity_idx = torch.argsort(torch.Tensor(np.array(capacities[layer_key[0]][-1]) \
                                               - np.array(capacities[layer_key[0]][0])), descending=descending)
                            if int(n_channels) != 0:
                                print('MULTIPLE FEATURES TRANSFER')
                                capacity_ch = tmp_capacity_idx[0:(int(n_channels)*8)-1].cpu().detach().numpy()
                            else:
                                print('SINGLE FEATURE TRANSFER')
                                capacity_ch = tmp_capacity_idx[0].cpu().detach().numpy()
                            capacity_activations = adv_activations[layer_key[0]][-1][:, capacity_ch, :, :]
                            print(capacity_ch, int(layer_key[0][-1]), capacity_activations.size())
                            transfer_activation = [capacity_ch, int(layer_key[0][-1]), capacity_activations]
                            delta = [torch.zeros_like(X).detach()]
                        
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
                    outputs_clean = net(X)

                _, predicted_clean = torch.max(outputs_clean.data, 1)
                _, predicted = torch.max(outputs.data, 1)
                
                correct_clean += (predicted_clean == y).sum().item()

                print('clean ------------------------------: ', (predicted_clean == y).sum().item())
                print('adversarial ------------------------------: ', (predicted == y).sum().item())

                total += y.size(0)
                correct_s += (torch.logical_and(predicted == y, predicted_clean == y)).sum().item()

                if len(channel_transfer) > 0 and i==20:
                    fname = './results/VGG19/eval/PGD/channel_transfer/' + channel_transfer + '.npy'
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
                 epsilon=0.0392):
    
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    net = proxy_VGG2(model, 
                     eval_mode=eval_mode,
                     device=device,
                     run_name=run_name,
                     noise_variance=0)

    if eval_mode:
        net.eval()

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

    score = net(X)
    pred_score, predicted = torch.max(score, 1)
    max_score = torch.max(pred_score)

    for j, _ in enumerate(X): 
        get_gradient = False
        if (not adversarial and predicted[j] == y[j]) or (adversarial and predicted[j] != y[j]):
            score = net(X[j].unsqueeze(0))
            pred_score = torch.max(score)
            norm_score = pred_score/max_score
            norm_score.backward(retain_graph=True)
            saliency_map = net.bn1.grad
            print("Are grads None: ", saliency_map is None)
            saliency_map.zero_()

        root_path = '.results/VGG19/'
        if eval_mode:
            root_path += 'eval/'
        else:   
            root_path += 'no_eval/' 
            
        root_path += 'PGD/saliency_maps/'

        if adversarial:
            root_path += 'adversarial/'
        else:
            root_path += 'clean/'

        if not os.path.isdir(root_path):
            os.mkdir(root_path)

        if adversarial:
            root_path += str(epsilon).replace('.', '') + '/'
        if not os.path.isdir(root_path):
            os.mkdir(root_path)

        root_path += run_name + '/'
        if not os.path.isdir(root_path):
            os.mkdir(root_path)

        root_path += 'img_' + str(j) + '/'
        if not os.path.isdir(root_path):
            os.mkdir(root_path)
            
        if j > 10:
            save_image(X[j].numpy(), root_path + 'sample_' + str(j) + 'jpg')
            break

        for jj in range(saliency_map.size(1)):
            save_image(saliency_map[0, jj, :, :].numpy(), \
                root_path + 'ch' + str(jj) + 'jpg')

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


