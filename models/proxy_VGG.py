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

class proxy_VGG(nn.Module):

    def __init__(self, 
                 net, 
                 eval_mode,
                 device, 
                 noise_variance=0., 
                 verbose=False, 
                 run_name='',
                 train_mode=False, 
                 regularization_mode='', 
                 prune_mode='', 
                 prune_percentage=1., 
                 layer_to_test=0):

        super(proxy_VGG, self).__init__()

        self.noise_variance = float(noise_variance)
        self.device = device
        self.temp_net = net
        self.capacity = {}
        self.activations = {}
        self.verbose = verbose
        self.run_name = run_name
        self.bn_parameters = {}
        self.bn_biases = {}
        self.test_variance = {}
        self.test_mean = {}
        self.running_variances = {}
        self.running_means = {}
        self.gradients = 0
        self.train_mode = train_mode
        self.regularization_mode = regularization_mode
        self.prune_mode = prune_mode
        self.prune_percentage = prune_percentage
        self.layer_to_prune = layer_to_test
        self.num_iterations = 0

        features = list(net.features)
        
        # ---- WATCH OUT w/ .eval() ---- #
        ###########################################
        if not self.train_mode:
            if eval_mode:
                net.eval()
                self.features = nn.ModuleList(features).eval()
                self.avgpool = net.avgpool.eval()
                self.classifier = net.classifier.eval()
            else:
                self.features = nn.ModuleList(features)
                self.avgpool = net.avgpool
                self.classifier = net.classifier
            
            if self.regularization_mode == 'BN_once':
                self.set_statistics()
        
        elif self.train_mode:
            self.features = nn.ModuleList(features)
            self.avgpool = net.avgpool
            self.classifier = net.classifier
            if self.regularization_mode in ['uniform_lambda', 'BN_once']:
                self.set_gradient_mode(which='lambda')
            elif self.regularization_mode == 'lambda_and_beta':
                self.set_gradient_mode(which='lambda_and_beta')
            elif self.regularization_mode == 'uniform_beta':
                self.set_gradient_mode(which='uniform_beta')
        ###########################################
    
    def set_grad(self, var):
        def hook(grad):
            var.grad = grad
        return hook
    
    def set_gradient_mode(self, which='lambda'):
        for ii, model in enumerate(self.features): # BatchNorm layers are only present in the encoder of VGG19
            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                if which in ['lambda', 'lambda_and_beta']:
                    model.weight.requires_grad = False
                    if which == 'lambda_and_beta':
                        model.bias.requires_grad = False
                elif which == 'uniform_beta':
                    model.bias.requires_grad = False
                elif which == 'running_stats':
                    model.running_var.requires_grad = False
                    model.running_mean.requires_grad = False
    
    def set_statistics(self):
        bn_count = 0
        for ii, model in enumerate(self.features): # BatchNorm layers are only present in the encoder of VGG19
            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                model.running_mean = 0 * model.running_mean
                model.running_var = 1 * model.running_var
                bn_count += 1
    
    def set_running_stats(self, layer, mode='zero'):
        if mode == 'zero':
            layer.running_mean = 0 * layer.running_mean
            layer.running_var = 1 * layer.running_var
        return layer
        
    def set_iteration_num(self, iterations, epoch):
        if epoch == 0:
            self.num_iterations = iterations
        else:
            self.num_iterations = 1
    
    def forward(self, x, ch_activation=[], saliency_layer=''):
        bn_count = 0
        conv_count = 0

        for ii, model in enumerate(self.features):

            if isinstance(model, torch.nn.modules.conv.Conv2d):
                x = model(x)
                if self.verbose:
                    #self.activations['BN_' + str(conv_count)] = x 
                    pass
                
                if len(ch_activation)> 0 and self.get_bn_int_from_name() not in [100, 1]:
                    ch, bn_idx, activation = ch_activation
                    if conv_count == bn_idx: 
                        print('TRANSFERRING CHANNEL')
                        if isinstance(ch, list):
                            for idx_ in ch:
                                x[:, idx_, :, :] = activation[idx_]
                        else:
                            x[:, ch, :, :] = activation

                conv_count += 1

            elif isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"

                if self.verbose and self.get_bn_int_from_name() in [100, 1]:
                    #self.activations['BN_' + str(bn_count)] = x
                    var_test = x.var([0, 2, 3], unbiased=False).to(self.device).detach()
                    self.capacity['BN_' + str(bn_count)] = (var_test * (model.weight**2))/model.running_var
                    self.test_variance['BN_' + str(bn_count)] = var_test
                    self.test_mean['BN_' + str(bn_count)] = x.mean([0, 2, 3]).detach()
                    '''if bn_count in [0,1,2,5,8,10,12,15]:
                        print(((var_test * (model.weight**2))/model.running_var).mean())'''
                    
                if len(ch_activation)> 0 and self.get_bn_int_from_name() in [100, 1]:
                    ch, bn_idx, activation = ch_activation
                    if bn_count == bn_idx: 
                        print('TRANSFERRING CHANNEL')
                        if isinstance(ch, list):
                            for idx_ in ch:
                                x[:, idx_, :, :] = activation[idx_]
                        else:
                            x[:, ch, :, :] = activation
                
                if len(saliency_layer)> 0 and bn_count == 0:
                    # self.gradients = torch.zeros_like(x, requires_grad=True)
                    temp = x 
                    temp.retain_grad()
                    temp.register_hook(self.set_grad(temp))
                    self.set_gradients(temp)

                if self.regularization_mode == 'wandb_only':
                    model = self.set_running_stats(model, mode='zero')
                
                if self.regularization_mode == 'BN_once' and self.num_iterations > 0:
                    model = self.set_running_stats(model, mode='zero')

                if len(self.prune_mode)>0 and bn_count == self.layer_to_prune:
                    if self.prune_mode == 'lambda':
                        lambdas = model.weight.detach().clone()
                        how_many_to_zero = int(lambdas.size(0) * (1-self.prune_percentage))
                        ordered_idxs = torch.argsort(lambdas, descending=False)

                        model.weight.data = model.weight.data * torch.where(lambdas > lambdas[ordered_idxs[how_many_to_zero]], 1., 0.)

                        x = model(x)

                        model.weight.data = lambdas

                    elif self.prune_mode == 'lambda_inverse':
                        lambdas = model.weight.detach().clone()
                        how_many_to_zero = int(lambdas.size(0) * (1-self.prune_percentage))
                        ordered_idxs = torch.argsort(lambdas, descending=True)

                        model.weight.data = model.weight.data * torch.where(lambdas < lambdas[ordered_idxs[how_many_to_zero]], 1., 0.)

                        x = model(x)

                        model.weight.data = lambdas
                
                else:
                    x = model(x)

                bn_count += 1

            else:
                x = model(x)

            if self.noise_variance != float(0):
                x = x + torch.normal(0, float(self.noise_variance), size=x.size()).to(self.device)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)

        return out

    def get_bn_parameters(self, get_variance=False, get_mean=False):
        bn_count = 0
        for ii, model in enumerate(self.features):
            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), \
                       "Previous module should be Conv2d"
                self.bn_parameters['BN_' + str(bn_count)] = model.weight.cpu().detach()
                if get_variance:
                    self.running_variances['BN_' + str(bn_count)] = model.running_var.detach()
                if get_mean:
                    self.running_means['BN_' + str(bn_count)] = model.running_mean.detach()
                bn_count +=1
        return self.bn_parameters
    
    def get_bn_biases(self, get_variance=False):
        bn_count = 0
        for ii, model in enumerate(self.features):
            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), \
                       "Previous module should be Conv2d"
                self.bn_biases['BN_' + str(bn_count)] = model.bias.cpu().detach()
                if get_variance:
                    self.running_variances['BN_' + str(bn_count)] = model.running_var.detach()
                bn_count +=1
        return self.bn_biases
    
    def get_running_variance(self):
        self.get_bn_parameters(get_variance=True)
        return self.running_variances
    
    def get_running_mean(self):
        self.get_bn_parameters(get_mean=True)
        return self.running_means

    def get_capacity(self):
        return self.capacity
    
    def get_activations(self):
        return self.activations
    
    def get_test_variance(self):
        return self.test_variance
    
    def set_gradients(self, gradients):
        self.gradients = gradients
    
    def get_gradients(self):
        return self.gradients

    def get_noisy_mode(self):
        if self.noise_variance == float(0):
            noisy_mode = False
        else:
            noisy_mode = True
        return noisy_mode

    def get_bn_int_from_name(self):
        temp = self.run_name.split('_')[1]
        if temp == 'bn':
            bn_locations = 100
        elif temp == 'no':
            bn_locations = 0
        else:
            # add 1 for consistency with name 
            bn_locations = int(temp) + 1
        
        return bn_locations

    def set_verbose(self, verbose):
        self.verbose = verbose
    
