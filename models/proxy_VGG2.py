from os import stat
from tabnanny import verbose
from cv2 import mean
from numpy import int32
import torch
import torch.nn as nn
from zmq import device

from utils_ import utils_flags
from absl import flags
from collections import namedtuple

FLAGS = flags.FLAGS

class proxy_VGG2(nn.Module):

    def __init__(self, 
                 net, 
                 eval_mode,
                 device, 
                 noise_variance=0., 
                 verbose=False, 
                 run_name=''):

        super(proxy_VGG2, self).__init__()

        self.noise_variance = float(noise_variance)
        self.device = device
        self.temp_net = net
        self.capacity = {}
        self.activations = {}
        self.verbose = verbose
        self.run_name = run_name
        self.bn_parameters = {}
        self.test_variance = {}

        ############################
        self.conv = net.features[0]
        self.bn = net.features[1]
        self.activation = net.features[2]
        ############################

        ############################
        self.bn1 = 0
        ############################

        features = list(net.features[3:])
        
        # ---- WATCH OUT w/ .eval() ---- #
        ###########################################
        if eval_mode:
            net.eval()
            self.features = nn.ModuleList(features).eval()
            self.avgpool = net.avgpool.eval()
            self.classifier = net.classifier.eval()
        else:
            self.features = nn.ModuleList(features)
            self.avgpool = net.avgpool
            self.classifier = net.classifier
        ###########################################

    def forward(self, x, ch_activation=[], saliency_layer=''):
        bn_count = 0

        conv1 = self.conv(x)
        self.bn1 = self.bn(conv1)
        self.bn1.retain_grad()
        x = self.activation(self.bn1)

        for ii, model in enumerate(self.features):
            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), \
                       "Previous module should be Conv2d"

                if self.verbose:
                    var_test = x.var([0, 2, 3], unbiased=False).to(self.device)
                    self.capacity['BN_' + str(bn_count)] = (var_test * model.weight**2)/model.running_var
                    self.activations['BN_' + str(bn_count)] = x 
                    self.bn_parameters['BN_' + str(bn_count)] = [model.weight, model.running_var]
                    self.test_variance['BN_' + str(bn_count)] = var_test
                
                if len(ch_activation)> 0:
                    ch, bn_idx, activation = ch_activation
                    if bn_count == bn_idx: 
                        print('TRANSFERRING CHANNEL')
                        if isinstance(ch, list):
                            for idx_ in ch:
                                x[:, idx_, :, :] = activation[idx_]
                        else:
                            x[:, ch, :, :] = activation
                

                bn_count += 1

            x = model(x)

            if self.noise_variance != float(0):
                x = x + torch.normal(0, float(self.noise_variance), size=x.size()).to(self.device)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)

        return out

    def get_capacity(self):
        return self.capacity
    
    def get_activations(self):
        return self.activations
    
    def get_bn_parameters(self):
        return self.bn_parameters
    
    def get_test_variance(self):
        return self.test_variance

    def get_noisy_mode(self):
        if self.noise_variance == float(0):
            noisy_mode = False
        else:
            noisy_mode = True
        return noisy_mode

    def set_verbose(self, verbose):
        self.verbose = verbose
    
