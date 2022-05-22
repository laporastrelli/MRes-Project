from os import stat
from tabnanny import verbose
from cv2 import mean
import torch
import torch.nn as nn
from zmq import device

from utils_ import utils_flags
from absl import flags
from collections import namedtuple

FLAGS = flags.FLAGS

class proxy_VGG(nn.Module):

    def __init__(self, 
                 net, 
                 eval_mode,
                 device, 
                 noise_variance=0., 
                 verbose=False, 
                 run_name=''):

        super(proxy_VGG, self).__init__()

        self.noise_variance = float(noise_variance)
        self.device = device
        self.temp_net = net
        self.capacity = {}
        self.activations = {}
        self.verbose = verbose
        self.run_name = run_name

        features = list(net.features)
        
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

    def forward(self, x):
        bn_count = 0
        for ii, model in enumerate(self.features):
            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                var_test = x.var([0, 2, 3], unbiased=False).to(self.device)
                if self.verbose:
                    self.capacity['BN_' + str(bn_count)] = (var_test * model.weight**2)/model.running_var
                    self.activations['BN_' + str(bn_count)] = x

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

    def get_noisy_mode(self):
        if self.noise_variance == float(0):
            noisy_mode = False
        else:
            noisy_mode = True
        return noisy_mode

    def set_verbose(self, verbose):
        self.verbose = verbose
    
