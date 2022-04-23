from os import stat
from tabnanny import verbose
from cv2 import mean
import torch
import torch.nn as nn

from utils_ import utils_flags
from absl import flags

FLAGS = flags.FLAGS

from collections import namedtuple

# VGG contains the following class variable:
# features --> avgpool --> flatten --> classifier

# ---OBSERVATION---: Models are more robust if (without using net.eval)
#                    BatchNorm makes use of test mini-batch statatsics
#                    at evaluation time. 

# ---METHOD---: Inject noise to BN layers in "features":
#               for each BN layer that we find save running_mean
#               and running_var and unject Gaussian Noise with
#               mean equal zero and variance equal to alpha*running_var 
#               where alpha is tuned to keep performance invariant (or almost).

# ---GOAL---: Observe whether by injecting noise (and using net.eval())
#             the model is more robust to adversarial attacks. 

class noisy_VGG(nn.Module):

    def __init__(self, 
                 net, 
                 eval_mode,
                 noise_variance, 
                 device, 
                 capacity_,
                 noise_capacity_constraint, 
                 mode=''):

        super(noisy_VGG, self).__init__()

        self.noise_variance = noise_variance
        self.device = device
        self.capacity_ = capacity_
        self.noise_capacity_constraint = noise_capacity_constraint
        self.mode = mode
        self.temp_net = net
        self.capacity = {}

        features = list(net.features)
        
        # ---- WATCH OUT w/ .eval() ---- #
        ###########################################
        if eval_mode:
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
                if self.noise_capacity_constraint:
                    if self.mode == 'standard':
                        self.capacity_ = 3.32*torch.ones(x.shape[1], dtype=torch.float).to(self.device)
                        # get channel mean and variance
                        mean_ch = x.mean([0, 2, 3]).to(self.device)
                        var_ch = x.var([0, 2, 3], unbiased=False).to(self.device)
                        max_var = torch.max(var_ch)*torch.ones(x.shape[1], dtype=torch.float).to(self.device)
                        mean_var_ch = torch.mean(var_ch).to(self.device)
                        # construct noise term for each image (pixel)
                        noise = ((torch.square(x - mean_ch[None, :, None, None]))/(2*self.capacity_[None, :, None, None]*\
                            (1/mean_var_ch)*(var_ch[None, :, None, None]))) \
                            - var_ch[None, :, None, None]
                        self.capacity_ = None
                    else: 
                        noise = torch.square(model.weight) - model.running_var
                        noise = noise.to(self.device)
                        if self.verbose:
                            self.capacity['BN_' + str(bn_count)] = (torch.square(x - model.running_mean[None, :, None, None]))\
                                                                    /(2*torch.sqrt(noise[None, :, None, None] + model.running_var[None, :, None, None])).detach()
                else:
                    noise = torch.normal(0, float(self.noise_variance), size=x.size())
                if not FLAGS.noise_after_BN:
                    if self.noise_capacity_constraint:
                        running_var = model.state_dict()['running_var']
                        updated_running_var = running_var + noise
                        running_var.copy_(updated_running_var)
                    else:
                        x = x + noise[None, :, None, None].to(self.device)
                
                bn_count += 1

            x = model(x)

            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d) and FLAGS.noise_after_BN:
                x = x + noise.to(self.device)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)

        return out


    def get_capacity(self):
        return self.capacity

    def set_verbose(self, verbose):
        self.verbose = verbose