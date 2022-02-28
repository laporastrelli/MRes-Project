############ IMPORTS ############
from foolbox.criteria import Misclassification
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
import foolbox as fb
import foolbox.attacks as fa
import eagerpy as ep

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils_ import utils_flags, load_data, test_utils
from absl import app
from absl import flags
from foolbox.devutils import flatten

from utils_.get_model import get_model

def get_DBA_acc(run_name):

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

    net.load_state_dict(torch.load(PATH_to_model))
    net.to(device)
    net.eval()

    preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(net, bounds=bounds, preprocessing=preprocessing)
    fmodel = fmodel.transform_bounds((0, 1))
    assert fmodel.bounds == (0, 1), "Model bounds should be between 0 and 1."

    acc = 0

    for i, data in enumerate(test_loader, 0):
        X, y = data
        X, y = X.to(device), y.to(device)

        attack =  fa.LinfinityBrendelBethgeAttack(steps=10, init_attack=fa.LinearSearchBlendedUniformNoiseAttack(steps=1000))

        print('Attacking ...')
        
        current_time = time.time()
        advs = attack.run(fmodel, X, Misclassification(y))
        next_time = time.time()

        print('Attacked...')

        print("Time taken to attack 128 samples (min): ", (next_time-current_time)/60)

        print(advs[0, :, :, :])
        print(X[0, :, :, :])

        # adv_sample = X[0,:, :, :] + advs[0,:, :, :]
        adv_sample = advs[0,:, :, :]
        np_adv_sample = adv_sample.cpu().numpy()
        np_X = X[0,:, :, :].cpu().numpy()

        plt.imshow(np.transpose(np_X, (1, 2, 0)))
        plt.savefig('original_sample.png')
        plt.imshow(np.transpose(np_adv_sample, (1, 2, 0)))
        plt.savefig('adversarial_example.png')
        
        acc += fb.accuracy(fmodel, advs, y)
        norms = torch.norm(torch.flatten(torch.abs(X-advs), start_dim=1), float('inf'), dim=1)
        
        print(norms)
        print(torch.median(norms))
        print(acc)
    
    return acc/len(test_loader)

'''
X = ep.astensor(X)
y = ep.astensor(y)

init_attack = fa.DatasetAttack()
init_attack.feed(fmodel, X)
init_advs = init_attack.run(fmodel, X, y)

# norms = ep.norms.lp(flatten(advs - X), p=ep.inf, axis=-1)

attack = fa.BoundaryAttack(steps=500,
                            init_attack=fa.LinearSearchBlendedUniformNoiseAttack(steps=1000),
                            update_stats_every_k=1)

# error = is_adv.double().float().mean().item()
'''