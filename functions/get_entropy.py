from pickle import TRUE
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

from utils_ import utils_flags

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from scipy.stats import entropy
from absl import flags


FLAGS = flags.FLAGS

def probability(sequence, bins_cntr):
    counts = np.zeros((len(bins_cntr), 1))
    for elem in sequence:
        temp = elem*np.ones((len(bins_cntr), ))

        diff = np.abs(temp - bins_cntr)
        idx = diff.tolist().index(np.min(diff))

        counts[idx] += 1
    
    return counts/len(sequence)


def get_layer_entropy(layer_outputs):
    if not isinstance(layer_outputs, np.ndarray):
        layer_outputs = np.asarray(layer_outputs)
    
    max_batch_num = 5
    layer_entropy = np.zeros((layer_outputs.shape[1], max_batch_num*FLAGS.batch_size))

    for batch_n, batch_output in enumerate(layer_outputs):
        print(batch_n)
        if batch_n >= max_batch_num:
            break
        
        for i, layer_output in enumerate(batch_output):
            for j, single_test_out in enumerate(layer_output):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                n, bins, _ = ax.hist(single_test_out, bins='fd', density=True, stacked=True)

                prob_vector = probability(single_test_out, bins)
                layer_entropy[i, j+(batch_n*FLAGS.batch_size)] = entropy(prob_vector)

    return layer_entropy




