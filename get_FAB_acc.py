############ IMPORTS ############
from itertools import count
from statistics import mean
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
from advertorch.attacks import FABAttack
from utils_.get_model import get_model
from utils_.adversarial_attack import pgd_linf

def get_FAB_acc(run_name):

    FLAGS = flags.FLAGS

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

    # load model
    net.load_state_dict(torch.load(PATH_to_model))
    net.eval()
    net.to(device)

    # initialize helper vars
    correct = 0
    counter = 0
    correct_samples = np.zeros((len(test_loader), FLAGS.batch_size))

    # set useful paths
    path_to_deltas = '/data2/users/lr4617/deltas/' + model_name + '/' + run_name + '/' + 'PGD' + '/eps_01/'
    deltas = os.listdir(path_to_deltas)

    # First run PGD and identify for each  batch the correctly classified samples
    for i, data in enumerate(test_loader, 0):
        X, y = data
        X, y = X.to(device), y.to(device)
        
        print('Attacking -- PGD')
        current_time = time.time()
        
        # if len(os.listdir('./sample_delta/')) == i:
        if len(os.listdir('./sample_delta/')) == len(test_loader) or len(os.listdir('./sample_delta/'))== i:
            delta = pgd_linf(net, X, y, float(FLAGS.epsilon), alpha=1e-2, num_iter=int(FLAGS.PGD_iterations))
            delta_name = './sample_delta/' + 'perturbation_' + str(i) + '.pt'
            torch.save(delta[0], delta_name)

            next_time = time.time()
            print("Time taken to attack {num_samples} samples (min): {time}" 
                .format(num_samples=str(FLAGS.batch_size), 
                time=str((next_time-current_time)/60)))

        else:
            print('Loading delta ...')
            delta = torch.load('./sample_delta/' + 'perturbation_' + str(i) + '.pt')

        # predict and identify correctly classified samples
        with torch.no_grad():
            outputs = net(X+delta[0].to(device))

        _, predicted = torch.max(outputs.data, 1)

        # one-hot encoded vector for correctly classified samples in the batch 
        correct_samples[i, 0:X.size()[0]] = (predicted != y).cpu().numpy()
        correct_samples.astype(int)

        counter += np.sum(correct_samples[i, :])
    
    # for each iterable batch we create a smaller batch of length equal to the number 
    # of correctly classified samples in that batch and calculate the L-infinity norm
    # between the original sample and the adversarial sample. The ideea is that if FAB
    # is not able to find an adversarial examples with smaller perturbation than PGD does,
    # then this might imply that the selected samples are more globally robust, not just 
    # based on gradient measures.

    for j, datapoints in enumerate(test_loader, 0):
        X, y = datapoints
        X, y = X.to(device), y.to(device)

        # initialize smaller batches
        X_positive = torch.zeros((int(np.sum(correct_samples[j, :])), X.size()[1], X.size()[2], X.size()[3]))
        Y_positive = torch.zeros((int(np.sum(correct_samples[j, :]))))

        # populate smaller batches
        cnt = 0
        for idx, x in enumerate(X,0):
            if correct_samples[j, idx] == 1:
               X_positive[cnt] = x
               Y_positive[cnt] = y[idx]
               cnt+=1

        # FAB attack 
        current_time = time.time()
        print('Attacking -- FAB')

        adversary = FABAttack(net, 
                              norm='Linf',
                              n_restarts=10,
                              n_iter=150,
                              eps=0.1,
                              alpha_max=0.1,
                              eta=1.05,
                              beta=0.9,)

        X_positive = X_positive.to(device)
        Y_positive = Y_positive.to(device)

        advimg = adversary.perturb(X_positive, Y_positive)

        with torch.no_grad():
            outputs = net(advimg)
        
        _, predicted = torch.max(outputs.data, 1)
        
        correct += (predicted == Y_positive).sum().item()
        
        next_time = time.time()
        print('Attacked...')
        print("Batch Accuracy: ", correct/np.sum(correct_samples[j, :]))
        print("Time taken to attack {num_samples} samples (min): {time}" 
                .format(num_samples=str(np.sum(correct_samples[j, :])), 
                time=str((next_time-current_time)/60)))    

        dist = []
        dist_ = []
        pred_vector = (predicted == Y_positive)

        for h, _ in enumerate(pred_vector):
            
            np_X = X_positive[h, :, :, :].cpu().numpy()
            plt.imshow(np.transpose(np_X, (1, 2, 0)))
            plt.savefig('original_sample.png')

            adv_sample = advimg[h,:, :, :]
            np_adv_sample = adv_sample.cpu().numpy()
            plt.imshow(np.transpose(np_adv_sample, (1, 2, 0)))
            plt.savefig('adversarial_example.png')

            # calculated l-infinity distance between original and adversarial image
            dist.append(torch.abs((X_positive[h, :, :, :] - advimg[h, :, :, :]).view(x.size(0), -1).max(dim=1)[0]))
            dist_.append(torch.norm((X_positive[h] - advimg[h]), p=float('inf')))

        mean_dist = sum(dist)/len(dist)
        mean_dist_ = sum(dist_)/len(dist_)
        print("Mean L-Infinity distance: ", mean_dist, mean_dist_)

    return correct/counter, mean_dist
