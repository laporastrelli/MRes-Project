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
                 run_name='',
                 mode='', 
                 verbose=False, 
                 scaled_noise=False, 
                 scaled_noise_norm=False,
                 scaled_noise_total=False):

        super(noisy_VGG, self).__init__()

        self.noise_variance = noise_variance
        self.device = device
        self.capacity_ = capacity_
        self.noise_capacity_constraint = noise_capacity_constraint
        self.run_name = run_name
        self.mode = mode
        self.temp_net = net
        self.capacity = {}
        self.activations = {}
        self.verbose = verbose
        self.scaled_noise = scaled_noise
        self.scaled_noise_norm = scaled_noise_norm
        self.scaled_noise_total = scaled_noise_total
        self.noise_out = {}
        self.init_capacity = {}
        self.pgd_steps = 0
        self.init_var = 0

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

    def forward(self, x, total_capacity=0, noise_in=0):
        if self.run_name.find('no_bn') != -1:
            for _, model in enumerate(self.features):
                if isinstance(model, torch.nn.modules.conv.Conv2d):
                    x = model(x)
                    if self.noise_capacity_constraint:
                        pass
                    else:
                        noise = torch.normal(0, float(self.noise_variance), size=x.size())
                    x = x + noise.to(self.device)
                else:
                    x = model(x)
            
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            out = self.classifier(x)                
        else:
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
                                self.capacity['BN_' + str(bn_count)] = (torch.square(x - model.running_mean[None, :, None, None])) \
                                                                        /(2*torch.sqrt(noise[None, :, None, None] \
                                                                        + model.running_var[None, :, None, None])).detach()
                                self.activations['BN_' + str(bn_count)] = x 
                    elif noise_in != 0:
                        curr_key = 'BN_' + str(bn_count)
                        if curr_key in list(noise_in.keys()):
                            noise = noise_in[curr_key]
                        else:
                            noise = torch.zeros_like(x)
                    else:
                        capacity = (x.var([0,2,3], unbiased=False)*model.weight**2)/(model.running_var*self.noise_variance) 
                        if self.verbose:
                            self.capacity['BN_' + str(bn_count)] = capacity
                        if self.scaled_noise or self.scaled_noise_norm:
                            if self.pgd_steps == 0:
                                self.init_capacity['BN_' + str(bn_count)] = capacity 
                                noise = torch.zeros_like(x)
                            else:
                                capacity_diff = capacity - self.init_capacity['BN_' + str(bn_count)]
                                if self.scaled_noise_norm:
                                    capacity_diff = torch.nn.functional.relu(capacity_diff)
                                if torch.nonzero(capacity_diff).size(0) > 0:
                                    noise_variance_d = ((capacity_diff - torch.min(capacity_diff))/(torch.max(capacity_diff) \
                                                       - torch.min(capacity_diff))) * self.noise_variance*torch.ones_like(capacity_diff)
                                    noise = torch.zeros_like(x)
                                    for d in range(x.size(1)):
                                        noise[:, d, :, :] = torch.normal(0, noise_variance_d[d].item(), size=x[:, d, :, :].size())
                                else:
                                    noise = torch.zeros_like(x)
                        elif total_capacity!=0 and self.scaled_noise_total:
                            if self.pgd_steps == 0:
                                self.init_capacity['BN_' + str(bn_count)] = capacity 
                                noise = torch.zeros_like(x)
                            else:
                                capacity_diff = capacity - self.init_capacity['BN_' + str(bn_count)]
                                temp = torch.nn.functional.softmax(capacity_diff)
                                scaled_total_capacity = temp*(torch.sum(total_capacity['BN_' + str(bn_count)].to(self.device))\
                                                        * torch.ones_like(temp))
                                noise_variance_d = ((x.var([0,2,3], unbiased=False)*(model.weight**2)))/(scaled_total_capacity*model.running_var)
                                noise = torch.zeros_like(x)
                                for d in range(x.size(1)):
                                    noise[:, d, :, :] = torch.normal(0, noise_variance_d[d].item(), size=x[:, d, :, :].size())
                                self.noise_out['BN_' + str(bn_count)] = noise
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
    
    def get_activations(self):
        return self.activations
    
    def get_noise(self):
        return self.noise_out

    def set_PGD_steps(self, steps):
        self.pgd_steps = steps
    
    def set_verbose(self, verbose):
        self.verbose = verbose
    
    def set_noise_injection_mode(self, mode):
        if mode == 'scaled_norm':
            self.scaled_noise_norm = True
            self.scaled_noise_total =  False
        elif mode == 'scaled_total': 
            self.scaled_noise_norm = False
            self.scaled_noise_total =  True


