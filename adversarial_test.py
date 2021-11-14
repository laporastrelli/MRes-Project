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
from utils import utils_flags, load_data, test_utils
from absl import app
from absl import flags

from utils.get_model import get_model

'''
    Given the model we want to:
    - generate adversarial examples
    - test each model on adversarial examples
    - carry out cross-mode testing
'''

def main(argv):
    
    del argv
    FLAGS = flags.FLAGS

    # choose device
    device = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")

    # load dataset
    _, test_loader = load_data.get_data(FLAGS.dataset_path, FLAGS.dataset)

    attack = 'FGSM'

    model_names = os.listdir('./logs/adv/')
    for model_name in model_names:
        print(model_name)
        if model_name != 'ResNet50' and model_name != 'ResNet101':

            model_runs = os.listdir('./logs/adv/' + str(model_name) + '/')
            for filename in model_runs:

                if filename.split('.')[1] != 'txt':
                    continue 

                filename = filename.split('.')[0]

                # get run serial
                FLAGS.model_name = filename.split('_', 1)[0]
                run_name = filename.split('_', 1)[1]

                # set path to logs
                path_to_logs = './logs/adv/' + FLAGS.model_name + '/'

                # generate adversarial exmaples and test on them
                modes = ['', 'bn']
                for mode in modes:
                    # retieve model
                    model_name = FLAGS.model_name + mode
                    print('MODEL NAME: ', model_name)
                    
                    PATH_to_model = FLAGS.root_path + '/models/' + model_name + '/' + model_name + '_' + run_name + '.pth'
                    PATH_to_deltas = FLAGS.root_path + '/deltas/'

                    if mode == '': batch_norm=False
                    else: batch_norm=True

                    net = get_model(FLAGS.model_name, batch_norm)

                    adv_test_acc = test_utils.adversarial_test(net, 
                                                            PATH_to_model, 
                                                            model_name, 
                                                            filename, 
                                                            test_loader, 
                                                            PATH_to_deltas, 
                                                            device, 
                                                            attack=attack, 
                                                            epsilon=FLAGS.epsilon)
                    # save results    
                    if isinstance(adv_test_acc, list):
                        np.save(path_to_logs + mode + '_' + run_name + '.npy', np.array(adv_test_acc))

                    elif isinstance(adv_test_acc, float) or (isinstance(adv_test_acc, list) and len(adv_test_acc)==1):
                        f = open(path_to_logs + filename + ".txt", "a")
                        f.write(attack)
                        f.write("Adversarial Test Accuracy " + mode + " : " + str(adv_test_acc) + "\n")
                
                ########################################################################################################################

                # cross-mode test (BN on Standard)
                adv_test_acc = test_utils.cross_model_testing(FLAGS.model_name + '_' + run_name, 
                                                            'BN_on_STD', 
                                                            FLAGS.root_path, 
                                                            test_loader, 
                                                            device, 
                                                            attack=attack, 
                                                            epsilon=FLAGS.epsilon)

                if isinstance(adv_test_acc, float):
                    # write results to file
                    f = open(path_to_logs + filename + ".txt", "a")
                    f.write("Adversarial Test Accuracy (BN on Standard): " + str(adv_test_acc) + "\n")
                    f.close()  
                else:
                    np.save(path_to_logs + 'BN_on_STD_' + run_name + '.npy', np.array(adv_test_acc))

                # cross-mode test (Standard on BN)
                adv_test_acc = test_utils.cross_model_testing(FLAGS.model_name + '_' + run_name, 
                                                            'STD_ON_BN', 
                                                            FLAGS.root_path, 
                                                            test_loader, 
                                                            device, 
                                                            attack=attack, 
                                                            epsilon=FLAGS.epsilon)
                if isinstance(adv_test_acc, float):
                    # write results to file 
                    f = open(path_to_logs + filename + ".txt", "a")
                    f.write("Adversarial Test Accuracy (Standard on BN): " + str(adv_test_acc) + "\n")
                    f.close() 
                else:
                    np.save(path_to_logs + 'STD_on BN' + run_name + '.npy', np.array(adv_test_acc))

if __name__ == '__main__':
    app.run(main)