from statistics import variance
from numpy import identity
import torch
import torch.nn as nn
import torch.nn.functional as F
from zmq import device
import math

class proxy_ResNet(nn.Module):
    def __init__(self, 
                net, 
                eval_mode,
                device, 
                noise_variance=0., 
                verbose=False, 
                run_name='', 
                IB_noise_calculation=False,
                IB_noise_std=0,
                get_parametric_frequency_MSE_CE=False,
                gaussian_std=[0*i for i in range(64)],
                layer_to_test=0,
                saliency_map=False, 
                train_mode=False, 
                regularization_mode='', 
                bounded_lambda=False,
                prune_mode='', 
                prune_percentage=1.):

        super(proxy_ResNet, self).__init__()

        self.noise_variance = float(noise_variance)
        self.device = device
        self.net = net
        self.verbose = verbose
        self.run_name = run_name
        self.capacity = {}
        self.activations = {}
        self.bn_parameters = {}
        self.test_variance = {}
        self.running_variances = {}
        self.BN_names = []
        self.gradients = 0
        self.eval_mode = eval_mode
        self.train_mode = train_mode
        self.regularization_mode = regularization_mode
        self.bounded_lambda = bounded_lambda
        self.num_iterations = 0
        self.prune_mode = prune_mode
        self.prune_percentage = prune_percentage

        print('Prune Percentage: ', self.prune_percentage)

        # saliency map mode
        self.saliency_map = saliency_map

        # for rank preserving init
        self.last_layer = 0

        # IB noise calculation mode
        self.IB_noise_calculation = IB_noise_calculation
        self.IB_noise_std = IB_noise_std
        self.layer_to_test = layer_to_test

        # parametric frequency MSE-CE
        self.get_parametric_frequency_MSE_CE = get_parametric_frequency_MSE_CE

        ############################
        self.channels = len(gaussian_std)
        self.gaussian_std = torch.FloatTensor(gaussian_std)
        self.gaussian_std.requires_grad = True
        self.ground_truth_activations = 0
        self.gaussian_activations = 0
        # self.conv_gaussian = nn.Conv2d(self.channels, self.channels, kernel_size=5, stride=1, padding=2, bias=False, groups=self.channels)
        # self.conv_gaussian.weight.requires_grad = False
        ############################

        ############################
        self.bn1_act = 0
        self.conv1_act = 0
        ############################

        ############################
        self.conv_frequency_activation = 0
        self.bn_frequency_activation = 0
        ############################

        if self.eval_mode:
            net.eval()    
        
        if self.train_mode:
            if self.regularization_mode == 'uniform_lambda':
                self.set_gradient_mode(which='lambda')  
            elif self.regularization_mode == 'uniform_lambda_beta':
                self.set_gradient_mode(which='lambda_and_beta')  
            elif self.regularization_mode == 'uniform beta':
                self.set_gradient_mode(which='beta')  
        
        elif not self.train_mode:
            if self.regularization_mode == 'uniform_lambda':
                self.set_gradient_mode(which='lambda')  
            if self.regularization_mode == 'BN_once':
                self.set_running_stats()
                self.num_iterations = 1
    
    def set_iteration_num(self, iterations, epoch):
        if epoch == 0:
            self.num_iterations = iterations
        else:
            self.num_iterations = 1
    
    def set_running_stats(self):
        # first BN layer (if existent)
        net_vars = [i for i in dir(self.net) if not callable(i)]
        if 'bn1' in net_vars: 
            self.net.bn1.running_mean = torch.zeros_like(self.net.bn1.running_mean, device=self.device)
            self.net.bn1.running_var = torch.ones_like(self.net.bn1.running_var, device=self.device)
        # four consecutive layers each containing blocks made up from sublocks
        layers = [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
        # unpack each of the four layers
        for _, layer in enumerate(layers):
            temp_layer = list(layer)
            # unpack each sublock
            for _, block in enumerate(temp_layer):
                block_vars = [j for j in dir(block) if not callable(j)]
                if 'bn1' in block_vars: bn_mode = True
                else: bn_mode = False
                
                # re-construct block
                if bn_mode: 
                    block.bn1.running_mean = torch.zeros_like(block.bn1.running_mean, device=self.device)
                    block.bn1.running_var = torch.ones_like(block.bn1.running_var, device=self.device)
                    
                if bn_mode: 
                    block.bn2.running_mean = torch.zeros_like(block.bn2.running_mean, device=self.device)
                    block.bn2.running_var = torch.ones_like(block.bn2.running_var, device=self.device)

                if bn_mode: 
                    block.bn3.running_mean = torch.zeros_like(block.bn3.running_mean, device=self.device)
                    block.bn3.running_var = torch.ones_like(block.bn3.running_var, device=self.device)
                
                shortcut = list(block.shortcut)
                if len(shortcut) > 0:
                    for shortcut_layer in shortcut:
                        if isinstance(shortcut_layer, torch.nn.modules.batchnorm.BatchNorm2d):
                            shortcut_layer.running_mean = torch.zeros_like(shortcut_layer.running_mean, device=self.device)
                            shortcut_layer.running_var = torch.ones_like(shortcut_layer.running_var, device=self.device)

    def set_gradient_mode(self, which='lambda'):
        # first BN layer (if existent)
        net_vars = [i for i in dir(self.net) if not callable(i)]
        if 'bn1' in net_vars: 
            if which in ['lambda_and_beta', 'lambda']:
                self.net.bn1.weight.requires_grad = False
            if which in ['lambda_and_beta', 'beta']:
                self.net.bn1.bias.requires_grad = False
        # four consecutive layers each containing blocks made up from sublocks
        layers = [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
        # unpack each of the four layers
        for _, layer in enumerate(layers):
            temp_layer = list(layer)
            # unpack each sublock
            for _, block in enumerate(temp_layer):
                block_vars = [j for j in dir(block) if not callable(j)]
                if 'bn1' in block_vars: bn_mode = True
                else: bn_mode = False
                
                # re-construct block
                if bn_mode: 
                    if which in ['lambda_and_beta', 'lambda']:
                        block.bn1.weight.requires_grad = False
                    if which in ['lambda_and_beta', 'beta']:
                        block.bn1.bias.requires_grad = False
                    
                if bn_mode: 
                    if which in ['lambda_and_beta', 'lambda']:
                        block.bn2.weight.requires_grad = False
                    if which in ['lambda_and_beta', 'beta']:
                        block.bn2.bias.requires_grad = False

                if bn_mode: 
                    if which in ['lambda_and_beta', 'lambda']:
                        block.bn3.weight.requires_grad = False
                    if which in ['lambda_and_beta', 'beta']:
                        block.bn3.bias.requires_grad = False
                
                shortcut = list(block.shortcut)
                if len(shortcut) > 0:
                    for shortcut_layer in shortcut:
                        if isinstance(shortcut_layer, torch.nn.modules.batchnorm.BatchNorm2d):
                            if which in ['lambda_and_beta', 'lambda']:
                                shortcut_layer.weight.requires_grad = False
                            if which in ['lambda_and_beta', 'beta']:
                                shortcut_layer.bias.requires_grad = False

    def replace_activation(self, x, ch_activation, bn_count):
        ch, bn_idx, activation = ch_activation
        if bn_count == bn_idx: 
            if isinstance(ch, list):
                for idx_ in ch:
                    x[:, idx_, :, :] = torch.from_numpy(activation[idx_]).to(self.device)
            else:
                x[:, ch, :, :] = torch.from_numpy(activation).to(self.device)
            return x
        else:
            return x
    
    def variance_function(self, noise_std):
        return 0.01 + nn.functional.relu(noise_std)

    def inject_IB_noise(self, activation, bn_count):
        if self.layer_to_test == bn_count:
            noise = torch.zeros_like(activation, device=self.device)
            for dim in range(self.noise_std.size(0)):
                # temp = nn.functional.softplus(self.noise_std[dim])
                temp = self.variance_function(self.noise_std[dim])
                noise[:, dim, :, :] = temp*torch.normal(0, 1, size=activation[:, dim, :, :].size(), device=self.device)
            activation = activation + noise.to(self.device)
            self.noise_std.retain_grad()
            return activation
        else:
            return activation

    def prune(self, bn_layer, x, mode='lambda'):
        lambdas = bn_layer.weight.detach().clone()
        how_many_to_zero = int(lambdas.size(0) * (1-self.prune_percentage))
        print('ZERO-CHANNELS: ', how_many_to_zero)
        if self.prune_mode == 'lambda':
            ordered_idxs = torch.argsort(lambdas, descending=False)
            x = x * (torch.where(lambdas > lambdas[ordered_idxs[how_many_to_zero]], 1., 0.).view(1, -1, 1, 1).expand(x.size()))
        elif self.prune_mode == 'lambda_inverse':
            ordered_idxs = torch.argsort(lambdas, descending=True)
            x = x * (torch.where(lambdas < lambdas[ordered_idxs[how_many_to_zero]], 1., 0.).view(1, -1, 1, 1).expand(x.size()))
        
        return x

    def get_gaussian_kernel(self, kernel_size=5, channels=1):
        channels = self.channels
        conv_gaussian_kernel = torch.zeros(channels, 1, kernel_size, kernel_size).to(self.device)

        for channel in range(channels):
            # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
            x_coord = torch.arange(kernel_size)
            x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1).float().to(self.device)

            mean = (kernel_size - 1)/2.

            # Calculate the 2-dimensional gaussian kernel which is
            # the product of two gaussian distributions for two different
            # variables (in this case called x and y)
            gaussian_kernel = (1./(2.*math.pi*self.gaussian_std[channel]**2)) *\
                                torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) /\
                                (2*self.gaussian_std[channel]**2))

            # Make sure sum of values in gaussian kernel equals 1.
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

            # Reshape to 2d depthwise convolutional weight
            conv_gaussian_kernel[channel, :, :, :] = gaussian_kernel.view(1, kernel_size, kernel_size)
            conv_gaussian_kernel[channel, :, :, :] = gaussian_kernel.view(1, kernel_size, kernel_size)
        
        return conv_gaussian_kernel.to(self.device)

    def forward(self, x, ch_activation=[]):
        bn_count = 0
        # first conv layer 
        x = self.net.conv1(x)

        if bn_count == self.layer_to_test:
            self.conv_frequency_activation = x.detach().cpu()

        self.conv1_act = x.detach().clone()
        
        if self.get_parametric_frequency_MSE_CE and bn_count==self.layer_to_test:
            # construct channel-wise gaussian-based convolutional layer
            self.ground_truth_activations = x.detach().clone()
            
            # apply it
            if self.layer_to_test == 0:
                kernel_size = 5
                padding = 2
            else:
                kernel_size = 3
                padding = 1

            self.gaussian_activations = F.conv2d(x, self.get_gaussian_kernel(), padding=2, groups=self.channels)

            # calculate standard deviation gradient
            self.gaussian_std.retain_grad()

            # calculate convolution output gradient
            self.gaussian_activations.retain_grad()

        # first BN layer (if existent)
        net_vars = [i for i in dir(self.net) if not callable(i)]
        if 'bn1' in net_vars: 
            if self.verbose:
                var_test = x.var([0, 2, 3], unbiased=False).to(self.device)
                self.capacity['BN_' + str(bn_count)] = ((var_test * (self.net.bn1.weight**2))/self.net.bn1.running_var)
                self.activations['BN_' + str(bn_count)] = (x).cpu().detach().numpy()
            if len(ch_activation)> 0: x = self.replace_activation(x, ch_activation, bn_count)
            if self.IB_noise_calculation: x = self.inject_IB_noise(x, bn_count)
            if int(self.num_iterations) > 0 and self.regularization_mode == 'BN_once': self.net.bn1 = nn.Sequential()
            
            if len(self.prune_mode) > 0 and bn_count == self.layer_to_test:              
                x = self.prune(self.net.bn1, x, self.prune_mode)
            if self.bounded_lambda:
                self.net.bn1.weight.data = self.net.bn1.weight.data.clamp(-torch.sqrt(self.net.bn1.running_var), torch.sqrt(self.net.bn1.running_var))
            
            if self.get_parametric_frequency_MSE_CE and bn_count==self.layer_to_test:
                # apply batch norm
                self.bn1_act = self.net.bn1(self.gaussian_activations)
            else:
                self.bn1_act = self.net.bn1(x)
                if bn_count == self.layer_to_test:
                    self.bn_frequency_activation = x.detach().cpu()

            bn_count += 1  
            #print(self.net.bn1.weight.requires_grad)

            if self.saliency_map:
                self.bn1_act.retain_grad()
            # first activation function layer 
            x = self.net.activation_fn(self.bn1_act)
        else:
            x = self.net.activation_fn(x)
        # four consecutive layers each containing blocks made up from sublocks
        layers = [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
        # unpack each of the four layers
        for _, layer in enumerate(layers):
            temp_layer = list(layer)
            # unpack each sublock
            for _, block in enumerate(temp_layer):
                block_vars = [j for j in dir(block) if not callable(j)]
                if 'bn1' in block_vars: bn_mode = True
                else: bn_mode = False
                
                # placeholder for shortcut
                temp = x 
                # re-construct block
                x = block.conv1(x)

                if bn_count == self.layer_to_test:
                    self.conv_frequency_activation = x.detach().cpu()

                if bn_mode: 
                    if self.verbose:
                        var_test = x.var([0, 2, 3], unbiased=False).to(self.device)
                        self.capacity['BN_' + str(bn_count)] = ((var_test * (block.bn1.weight**2))/block.bn1.running_var)
                        self.activations['BN_' + str(bn_count)] = (x).cpu().detach().numpy()
                    if len(ch_activation)> 0: x = self.replace_activation(x, ch_activation, bn_count)
                    if self.IB_noise_calculation: x = self.inject_IB_noise(x, bn_count)
                    if int(self.num_iterations) > 0 and self.regularization_mode == 'BN_once': block.bn1 = nn.Sequential()
                    if len(self.prune_mode) > 0 and bn_count == self.layer_to_test:              
                        x = self.prune(block.bn1, x, self.prune_mode)
                    if self.bounded_lambda:
                        block.bn1.weight.data = block.bn1.weight.data.clamp(-torch.sqrt(block.bn1.running_var), torch.sqrt(block.bn1.running_var))
                    x = block.bn1(x)
                    if bn_count == self.layer_to_test:
                        self.bn_frequency_activation = x.detach().cpu()

                    bn_count += 1

                x = block.activation_fn(x)

                x = block.conv2(x)

                if bn_count == self.layer_to_test:
                    self.conv_frequency_activation = x.detach().cpu()

                if bn_mode: 
                    if self.verbose:
                        var_test = x.var([0, 2, 3], unbiased=False).to(self.device)
                        self.capacity['BN_' + str(bn_count)] = ((var_test * (block.bn2.weight**2))/block.bn2.running_var)
                        self.activations['BN_' + str(bn_count)] = (x).cpu().detach().numpy()
                    if len(ch_activation)> 0: x = self.replace_activation(x, ch_activation, bn_count)
                    if self.IB_noise_calculation: x = self.inject_IB_noise(x, bn_count)
                    if int(self.num_iterations) > 0 and self.regularization_mode == 'BN_once': block.bn2 = nn.Sequential()
                    
                    if len(self.prune_mode) > 0 and bn_count == self.layer_to_test:              
                        x = self.prune(block.bn2, x, self.prune_mode)
                    if self.bounded_lambda:
                        block.bn2.weight.data = block.bn2.weight.data.clamp(-torch.sqrt(block.bn2.running_var), torch.sqrt(block.bn2.running_var))
                    x = block.bn2(x)

                    if bn_count == self.layer_to_test:
                        self.bn_frequency_activation = x.detach().cpu()
                    
                    bn_count += 1

                x = block.activation_fn(x)

                x = block.conv3(x)

                if bn_count == self.layer_to_test:
                    self.conv_frequency_activation = x.detach().cpu()
                if bn_mode: 
                    if self.verbose:
                        var_test = x.var([0, 2, 3], unbiased=False).to(self.device)
                        self.capacity['BN_' + str(bn_count)] = ((var_test * (block.bn3.weight**2))/block.bn3.running_var)
                        self.activations['BN_' + str(bn_count)] = (x).cpu().detach().numpy()
                    if len(ch_activation)> 0: x = self.replace_activation(x, ch_activation, bn_count)
                    if self.IB_noise_calculation: x = self.inject_IB_noise(x, bn_count)
                    if int(self.num_iterations) > 0 and self.regularization_mode == 'BN_once': block.bn3 = nn.Sequential()
                    
                    if len(self.prune_mode) > 0 and bn_count == self.layer_to_test:              
                        x = self.prune(block.bn3, x, self.prune_mode)
                    if self.bounded_lambda:
                        block.bn3.weight.data = block.bn3.weight.data.clamp(-torch.sqrt(block.bn3.running_var), torch.sqrt(block.bn3.running_var))
                    x = block.bn3(x)

                    if bn_count == self.layer_to_test:
                        self.bn_frequency_activation = x.detach().cpu()

                    bn_count += 1

                shortcut = list(block.shortcut)
                if len(shortcut) > 0:
                    for shortcut_layer in shortcut:
                        if isinstance(shortcut_layer, torch.nn.modules.batchnorm.BatchNorm2d):
                            if self.verbose:
                                var_test = temp.var([0, 2, 3], unbiased=False).to(self.device)
                                self.capacity['BN_' + str(bn_count)] = ((var_test * (shortcut_layer.weight**2))/shortcut_layer.running_var)
                                self.activations['BN_' + str(bn_count)] = (temp).cpu().detach().numpy()
                            if len(ch_activation)> 0: x = self.replace_activation(x, ch_activation, bn_count)
                            if self.IB_noise_calculation: x = self.inject_IB_noise(x, bn_count)
                            if int(self.num_iterations) > 0 and self.regularization_mode == 'BN_once': shortcut_layer = nn.Sequential()
                            if len(self.prune_mode) > 0 and bn_count == self.layer_to_test:              
                                x = self.prune(shortcut_layer, x, self.prune_mode)
                            if self.bounded_lambda:
                                shortcut_layer.weight.data = shortcut_layer.weight.data.clamp(-torch.sqrt(shortcut_layer.running_var),\
                                                             torch.sqrt(shortcut_layer.running_var))
                        temp = shortcut_layer(temp)
                        if isinstance(shortcut_layer, torch.nn.modules.conv.Conv2d):
                            if bn_count == self.layer_to_test:
                                self.conv_frequency_activation = temp.detach().cpu()
                        elif isinstance(shortcut_layer, torch.nn.modules.batchnorm.BatchNorm2d):
                            if bn_count == self.layer_to_test:
                                self.bn_frequency_activation = temp.detach().cpu()
                            bn_count += 1
                        
                x = x + temp
                
                x = block.activation_fn(x)
        
        x = F.avg_pool2d(x, 4)
        pre_out = x.view(x.size(0), -1)
        self.last_layer = pre_out.detach().clone()
        final = self.net.linear(pre_out)

        return final

    def set_verbose(self, verbose):
        self.verbose = verbose
    
    def get_capacity(self):
        return self.capacity
    
    def get_activations(self):
        return self.activations
    
    def get_noisy_mode(self):
        return False
    
    def get_bn_parameters(self, get_names=False, get_variance=False):
        bn_count = 0
        # first BN layer (if existent)
        net_vars = [i for i in dir(self.net) if not callable(i)]
        if 'bn1' in net_vars: 
            if self.regularization_mode == 'euclidean_total':
                self.bn_parameters['BN_' + str(bn_count)] = self.net.bn1.weight
            else:
                self.bn_parameters['BN_' + str(bn_count)] = self.net.bn1.weight.detach()
            if get_variance:
                self.running_variances['BN_' + str(bn_count)] = self.net.bn1.running_var.detach()
            bn_count += 1
            if get_names:
                self.BN_names.append('BN_0')
            
        # four consecutive layers each containing blocks made up from sublocks
        layers = [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
        # unpack each of the four layers
        for layer_count, layer in enumerate(layers):
            temp_layer = list(layer)
            bn_count_internal = 0
            # unpack each sublock
            for _, block in enumerate(temp_layer):
                block_vars = [j for j in dir(block) if not callable(j)]
                if 'bn1' in block_vars: bn_mode = True
                else: bn_mode = False
                
                # re-construct block
                if bn_mode: 
                    if self.regularization_mode == 'euclidean_total':
                        self.bn_parameters['BN_' + str(bn_count)] = block.bn1.weight
                    else:
                        self.bn_parameters['BN_' + str(bn_count)] = block.bn1.weight.detach()
                    if get_variance:
                        self.running_variances['BN_' + str(bn_count)] = block.bn1.running_var.detach()
                    bn_count += 1
                    if get_names:
                        self.BN_names.append('BN_' + str(layer_count) + '_' + str(bn_count_internal))
                        bn_count_internal += 1

                if bn_mode: 
                    if self.regularization_mode == 'euclidean_total':
                        self.bn_parameters['BN_' + str(bn_count)] = block.bn2.weight
                    else:
                        self.bn_parameters['BN_' + str(bn_count)] = block.bn2.weight.detach()
                    if get_variance:
                        self.running_variances['BN_' + str(bn_count)] = block.bn2.running_var.detach()
                    bn_count += 1
                    if get_names:
                        self.BN_names.append('BN_' + str(layer_count) + '_' + str(bn_count_internal))
                        bn_count_internal += 1

                if bn_mode and (self.run_name.find('ResNet18') == -1 and self.run_name.find('ResNet34')) == -1: 
                    if self.regularization_mode == 'euclidean_total':
                        self.bn_parameters['BN_' + str(bn_count)] = block.bn3.weight
                    else:
                        self.bn_parameters['BN_' + str(bn_count)] = block.bn3.weight.detach()
                    if get_variance:
                        self.running_variances['BN_' + str(bn_count)] = block.bn3.running_var.detach()
                    bn_count += 1
                    if get_names:
                        self.BN_names.append('BN_' + str(layer_count) + '_' + str(bn_count_internal))
                        bn_count_internal += 1

                shortcut = list(block.shortcut)
                if len(shortcut) > 0:
                    for shortcut_layer in shortcut:
                        if isinstance(shortcut_layer, torch.nn.modules.batchnorm.BatchNorm2d):
                            if self.regularization_mode == 'euclidean_total':
                                self.bn_parameters['BN_' + str(bn_count)] = shortcut_layer.weight
                            else:
                                self.bn_parameters['BN_' + str(bn_count)] = shortcut_layer.weight.detach()
                            if get_variance:
                                self.running_variances['BN_' + str(bn_count)] = shortcut_layer.running_var.detach()
                            bn_count += 1
                            if get_names:
                                self.BN_names.append('BN_' + str(layer_count) + '_' + str(bn_count_internal) + '_skipcon')
                                bn_count_internal += 1

        return self.bn_parameters   
    
    def get_BN_names(self): 
        _ = self.get_bn_parameters(get_names=True)
    
    def get_running_variance(self):
        _ = self.get_bn_parameters(get_variance=True)
        return self.running_variances












