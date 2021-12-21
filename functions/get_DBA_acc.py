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
import foolbox as fb
import foolbox.attacks as fa
import eagerpy as ep

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import utils_flags, load_data, test_utils
from absl import app
from absl import flags

from utils.get_model import get_model

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
    net.cuda()
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

        print(type(X), X.shape)
        print(type(y), y.shape)

        X = ep.astensor(X)
        y = ep.astensor(y)

        init_attack = fa.DatasetAttack()
        init_attack.feed(fmodel, X)
        init_advs = init_attack.run(fmodel, X, y)

        attack = fb.attacks.LinfinityBrendelBethgeAttack(steps=15)
        print('Attacking ...')
        advs = attack.run(fmodel, X, y, starting_points=init_advs)

        acc += fb.accuracy(fmodel, advs, y)
        print(acc)

        # error = is_adv.double().float().mean().item()
    
    return acc/len(test_loader)

