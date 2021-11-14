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

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from models import ResNet, ResNet_bn
from utils import utils_flags, load_data, test_utils
from absl import app
from absl import flags

from utils.get_model import get_model

from utils.adversarial_attack import fgsm, pgd_linf_rand
from utils import get_model

def main(argv):
    
    del argv
    FLAGS = flags.FLAGS

    # choose device
    device = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")

    # load dataset
    _, test_loader = load_data.get_data(FLAGS.dataset_path, FLAGS.dataset)

    # get model
    PATH_TO_MODEL = '/data2/users/lr4617/models/VGG19/VGG19_01_11_2021_18_59_09.pth'
    net = get_model.get_model('VGG19', batch_norm=False)
    net.load_state_dict(torch.load(PATH_TO_MODEL))
    net.cuda()

    total = 0
    correct_s = 0
    for i, data in enumerate(test_loader, 0):
        X, y = data
        X, y = X.to(device), y.to(device)    

        delta = fgsm(net, X, y, epsilon=0.1)

        outputs = net(X+delta)
            
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct_s += (predicted == y).sum().item()

    print(correct_s/total)

if __name__ == '__main__':
    app.run(main)