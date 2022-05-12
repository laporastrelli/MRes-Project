from os import stat
from tabnanny import verbose
from cv2 import mean
import torch
import torch.nn as nn

from utils_ import utils_flags
from absl import flags
from collections import namedtuple

FLAGS = flags.FLAGS

class proxy_VGG(nn.Module):

    def __init__(self, 
                 net, 
                 eval_mode,
                 device, 
                 noise_variance=0, 
                 verbose=False, 
                 run_name=''):

        super(proxy_VGG, self).__init__()

        self.noise_variance = noise_variance
        self.device = device
        self.temp_net = net
        self.capacity = {}
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
                var_train = model.running_var
                lambda_ = model.weight

                if self.verbose:
                    self.capacity['BN_' + str(bn_count)] = (var_test * lambda_**2)/var_train

                bn_count += 1
            x = model(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)

        return out

    def get_capacity(self):
        return self.capacity

    def set_verbose(self, verbose):
        self.verbose = verbose