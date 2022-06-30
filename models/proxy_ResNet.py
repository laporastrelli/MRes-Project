import torch
import torch.nn as nn
import torch.nn.functional as F
from zmq import device

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
                layer_to_test=0,
                saliency_map=False):

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
        self.BN_names = []
        self.gradients = 0
        self.eval_mode = eval_mode

        # saliency map mode
        self.saliency_map = saliency_map

        # IB noise calculation mode
        self.IB_noise_calculation = IB_noise_calculation
        self.IB_noise_std = IB_noise_std
        self.layer_to_test = layer_to_test

        ############################
        self.bn1 = 0
        ############################

        if self.eval_mode:
            net.eval()    
    
    def set_verbose(self, verbose):
        self.verbose = verbose
    
    def get_capacity(self):
        return self.capacity
    
    def get_activations(self):
        return self.activations
    
    def get_noisy_mode(self):
        return False
    
    def get_bn_parameters(self, get_names=False):
        bn_count = 0
        # first BN layer (if existent)
        net_vars = [i for i in dir(self.net) if not callable(i)]
        if 'bn1' in net_vars: 
            self.bn_parameters['BN_' + str(bn_count)] = self.net.bn1.weight
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
                    self.bn_parameters['BN_' + str(bn_count)] = block.bn1.weight
                    bn_count += 1
                    if get_names:
                        self.BN_names.append('BN_' + str(layer_count) + '_' + str(bn_count_internal))
                        bn_count_internal += 1

                if bn_mode: 
                    self.bn_parameters['BN_' + str(bn_count)] = block.bn2.weight
                    bn_count += 1
                    if get_names:
                        self.BN_names.append('BN_' + str(layer_count) + '_' + str(bn_count_internal))
                        bn_count_internal += 1

                if bn_mode: 
                    self.bn_parameters['BN_' + str(bn_count)] = block.bn3.weight
                    bn_count += 1
                    if get_names:
                        self.BN_names.append('BN_' + str(layer_count) + '_' + str(bn_count_internal))
                        bn_count_internal += 1

                shortcut = list(block.shortcut)
                if len(shortcut) > 0:
                    for shortcut_layer in shortcut:
                        if isinstance(shortcut_layer, torch.nn.modules.batchnorm.BatchNorm2d):
                            self.bn_parameters['BN_' + str(bn_count)] = shortcut_layer.weight
                            bn_count += 1
                            if get_names:
                                self.BN_names.append('BN_' + str(layer_count) + '_' + str(bn_count_internal) + '_skipcon')
                                bn_count_internal += 1

        return self.bn_parameters   
    
    def get_BN_names(self): 
        _ = self.get_bn_parameters(get_names=True)
    
    def replace_activation(self, x, ch_activation, bn_count):
        ch, bn_idx, activation = ch_activation
        if bn_count == bn_idx: 
            # print('TRANSFERRING CHANNEL')
            if isinstance(ch, list):
                for idx_ in ch:
                    x[:, idx_, :, :] = torch.from_numpy(activation[idx_]).to(self.device)
            else:
                x[:, ch, :, :] = torch.from_numpy(activation).to(self.device)
            return x
        else:
            return x
    
    def inject_IB_noise(self, activation, bn_count):
        if self.layer_to_test == bn_count:
            noise = torch.zeros_like(activation, device=self.device)
            for dim in range(self.noise_std.size(0)):
                noise[:, dim, :, :] = nn.functional.softplus(self.noise_std[dim])\
                                      *torch.normal(0, 1, size=activation[:, dim, :, :].size(), device=self.device)

            activation = activation + noise.to(self.device)
            self.noise_std.retain_grad()
            return activation
        else:
            return activation
        
    def forward(self, x, ch_activation=[]):
        bn_count = 0
        # first conv layer 
        x = self.net.conv1(x)
        # first BN layer (if existent)
        net_vars = [i for i in dir(self.net) if not callable(i)]
        if 'bn1' in net_vars: 
            if self.verbose:
                var_test = x.var([0, 2, 3], unbiased=False).to(self.device)
                self.capacity['BN_' + str(bn_count)] = ((var_test * (self.net.bn1.weight**2))/self.net.bn1.running_var).cpu().detach().numpy()
                self.activations['BN_' + str(bn_count)] = (x).cpu().detach().numpy()
            if len(ch_activation)> 0: x = self.replace_activation(x, ch_activation, bn_count)
            if self.IB_noise_calculation: x = self.inject_IB_noise(x, bn_count)
            bn_count += 1
            self.bn1 = self.net.bn1(x)
            if self.saliency_map:
                self.bn1.retain_grad()
        # first activation function layer 
        x = self.net.activation_fn(self.bn1)
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
                if bn_mode: 
                    if self.verbose:
                        var_test = x.var([0, 2, 3], unbiased=False).to(self.device)
                        self.capacity['BN_' + str(bn_count)] = ((var_test * (block.bn1.weight**2))/block.bn1.running_var).cpu().detach().numpy()
                        self.activations['BN_' + str(bn_count)] = (x).cpu().detach().numpy()
                    if len(ch_activation)> 0: x = self.replace_activation(x, ch_activation, bn_count)
                    bn_count += 1
                    x = block.bn1(x)
                x = block.activation_fn(x)

                x = block.conv2(x)
                if bn_mode: 
                    if self.verbose:
                        var_test = x.var([0, 2, 3], unbiased=False).to(self.device)
                        self.capacity['BN_' + str(bn_count)] = ((var_test * (block.bn2.weight**2))/block.bn2.running_var).cpu().detach().numpy()
                        self.activations['BN_' + str(bn_count)] = (x).cpu().detach().numpy()
                    if len(ch_activation)> 0: x = self.replace_activation(x, ch_activation, bn_count)
                    bn_count += 1
                    x = block.bn2(x)
                x = block.activation_fn(x)

                x = block.conv3(x)
                if bn_mode: 
                    if self.verbose:
                        var_test = x.var([0, 2, 3], unbiased=False).to(self.device)
                        self.capacity['BN_' + str(bn_count)] = ((var_test * (block.bn3.weight**2))/block.bn3.running_var).cpu().detach().numpy()
                        self.activations['BN_' + str(bn_count)] = (x).cpu().detach().numpy()
                    if len(ch_activation)> 0: x = self.replace_activation(x, ch_activation, bn_count)
                    bn_count += 1
                    x = block.bn3(x)

                shortcut = list(block.shortcut)
                if len(shortcut) > 0:
                    for shortcut_layer in shortcut:
                        if isinstance(shortcut_layer, torch.nn.modules.batchnorm.BatchNorm2d):
                            if self.verbose:
                                var_test = temp.var([0, 2, 3], unbiased=False).to(self.device)
                                self.capacity['BN_' + str(bn_count)] = ((var_test * (shortcut_layer.weight**2))/shortcut_layer.running_var).cpu().detach().numpy()
                                self.activations['BN_' + str(bn_count)] = (temp).cpu().detach().numpy()
                            if len(ch_activation)> 0: x = self.replace_activation(x, ch_activation, bn_count)
                            bn_count += 1
                        temp = shortcut_layer(temp)
                x = x + temp
                x = block.activation_fn(x)
        
        x = F.avg_pool2d(x, 4)
        pre_out = x.view(x.size(0), -1)
        final = self.net.linear(pre_out)

        return final













