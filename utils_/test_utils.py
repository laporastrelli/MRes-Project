from bleach import clean
import torch.nn as nn
import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from utils_.adversarial_attack import fgsm, pgd_linf, pgd_linf_capacity
from utils_ import get_model
from utils_.miscellaneous import get_minmax, get_path2delta, get_bn_int_from_name

#### check ####
from advertorch.attacks import LinfPGDAttack
#### check ####

#### temporary ####
from models.proxy_VGG import proxy_VGG
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

    net.load_state_dict(torch.load(model_path))
    net.to(device)
    if eval_mode:
        net.eval()
    correct = 0
    total = 0

    for i, data in enumerate(test_loader, 0):  
        X, y = data
        X, y = X.to(device), y.to(device)

        if random_resizing:
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
            print('STANDARD TEST')
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
                     eval=True, 
                     custom=True,
                     save=False):

    net.load_state_dict(torch.load(model_path))
    net.to(device)

    ################## IMPORTANT ##################
    if inject_noise:
        print('Adversarial Test With Noise ...')
        net = noisy_VGG(net, 
                        eval_mode=use_pop_stats,
                        noise_variance=noise_variance, 
                        device=device,
                        capacity_=capacity,
                        noise_capacity_constraint=noise_capacity_constraint,
                        run_name=run_name, 
                        scaled_noise=scaled_noise)

    elif capacity_calculation:
        print('Calculating Capacity')
        net = proxy_VGG(net, 
                        eval_mode=use_pop_stats,
                        device=device,
                        run_name=run_name,
                        noise_variance=noise_variance)
    
    ################## IMPORTANT ##################
    if use_pop_stats:
        net.eval()

    # setting dropout to eval mode (in VGG)
    if run_name.find('VGG')!= -1:
        net.classifier.eval()

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

    # adversarial evaluation
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
                if custom:
                    if i==1 and capacity_calculation and get_bn_int_from_name(run_name)!=0:
                        net.set_verbose(verbose=True)
                        if run_name.find('bn')!= -1:
                            layer_key = ['BN_' + str(i) for i in range(16)]
                        else:
                            layer_key = 'BN_1'
                        delta, capacities = pgd_linf_capacity(net, 
                                                              X, 
                                                              y, 
                                                              epsilon, 
                                                              max_tensor, 
                                                              min_tensor, 
                                                              alpha=epsilon/10, 
                                                              num_iter=num_iter, 
                                                              layer_key=layer_key)
                        net.set_verbose(verbose=False) 
                        model_name = run_name.split('_')[0]

                        if use_pop_stats:
                            eval_mode_str = 'eval'
                        else:
                            eval_mode_str = 'no_eval'
                        
                        path_out = './results/' + model_name + '/' + eval_mode_str + '/'\
                                 + attack + '/capacity/'

                        if len(layer_key)==1:
                            if layer_key[0] == 'BN_0':
                                path_out += 'first_layer/'
                            else:
                                path_out += 'second_layer/'
                        else:
                            path_out += 'all_layers/'
                        
                        print('MODE: ', net.get_noisy_mode())

                        if net.get_noisy_mode():
                            path_out += 'noisy_test/'
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

                        for folder_name in folder_names:
                            if not os.path.isdir(path_out + folder_name):
                                os.mkdir(path_out + folder_name)
                            sub_folder_name = str(epsilon).replace('.', '')
                            if not os.path.isdir(path_out + folder_name + '/' + sub_folder_name):
                                os.mkdir(path_out + folder_name + '/' + sub_folder_name)
                            t = 0
                            fig = plt.figure()
                            for temp in capacities[folder_name]:
                                x_axis = [t]
                                for x_, y_ in zip(x_axis, [temp]):
                                    plt.scatter([x_] * len(y_), y_)
                                    plt.xlabel('PGD steps')
                                    plt.ylabel('Capacity Estimate')
                                    fig.savefig(path_out + folder_name + '/' \
                                        + sub_folder_name + '/' + run_name + '_capacity.png')
                                t+=1
                            plt.xticks(np.arange(0, t))

                    else:     
                        if capacity_calculation:
                            delta = [torch.zeros_like(X)]
                        else:                       
                            delta = pgd_linf(net, X, y, epsilon, max_tensor, min_tensor, alpha=epsilon/10, num_iter=num_iter)
                    adv_inputs = X + delta[0]
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
                    outputs = net(adv_inputs)
                    outputs_clean = net(X)
        
                _, predicted_clean = torch.max(outputs_clean.data, 1)
                _, predicted = torch.max(outputs.data, 1)
                
                correct_clean += (predicted_clean == y).sum().item()

                total += y.size(0)
                correct_s += (torch.logical_and(predicted == y, predicted_clean == y)).sum().item()

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

    acc = correct_s / correct_clean
    print('Adversarial Test Accuracy: ', acc)

    if isinstance(epsilon, list):
        return acc.tolist()
    else:
        return acc

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


