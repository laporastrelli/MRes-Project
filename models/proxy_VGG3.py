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

class proxy_VGG3(nn.Module):
    '''
    This proxy VGG model is used for the calculation of noise
    using the **Information Bottleneck** approach. Alternatively, 
    it could also be used for the channel transfer experiment.
    '''

    def __init__(self, 
                 net, 
                 eval_mode,
                 device, 
                 noise_variance=0., 
                 verbose=False, 
                 run_name='', 
                 IB_noise_calculation=False,
                 IB_noise_std=0, 
                 layer_to_test=0):

        super(proxy_VGG3, self).__init__()

        self.noise_variance = float(noise_variance)
        self.device = device
        self.temp_net = net
        self.capacity = {}
        self.activations = {}
        self.verbose = verbose
        self.run_name = run_name

        # IB noise calculation
        self.IB_noise_calculation = IB_noise_calculation
        self.layer_to_test = layer_to_test

        self.bn_parameters = {}
        self.test_variance = {}

        ############################
        self.noise_std = IB_noise_std
        ############################

        ############################
        features = list(net.features)
        ############################
        
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

    def forward(self, x, ch_activation=[]):

        # set bn counter
        bn_count = 0

        # unpack rest of the model
        for ii, model in enumerate(self.features):
            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"

                if self.verbose:
                    var_test = x.var([0, 2, 3], unbiased=False).to(self.device)
                    self.capacity['BN_' + str(bn_count)] = (var_test * model.weight**2)/model.running_var
                    self.activations['BN_' + str(bn_count)] = x 
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
                
                if self.IB_noise_calculation and self.layer_to_test==bn_count:
                    noise = torch.zeros_like(x, device=self.device)
                    for dim in range(self.noise_std.size(0)):
                        noise[:, dim, :, :] = nn.functional.softplus(self.noise_std[dim])\
                                              *torch.normal(0, 1, size=x[:, dim, :, :].size(), device=self.device)                
                    x = x + noise.to(self.device)
                    self.noise_std.retain_grad()
                
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
    
    def get_bn_parameters(self):
        bn_count = 0
        for ii, model in enumerate(self.features): # BatchNorm layers are only present in the encoder of VGG19
            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                self.bn_parameters['BN_' + str(bn_count)] = model.weight.cpu().detach()
                bn_count += 1
        return self.bn_parameters
    
    def get_running_variance(self):
        bn_count = 0
        for ii, model in enumerate(self.features): # BatchNorm layers are only present in the encoder of VGG19
            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                self.bn_parameters['BN_' + str(bn_count)] = model.running_var
                bn_count += 1
        return self.bn_parameters
