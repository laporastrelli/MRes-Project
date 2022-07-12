from os import stat
from tabnanny import verbose
from cv2 import mean
from numpy import int32
import torch
import torch.nn as nn
from torch.nn import functional as F
from zmq import device
import math
import numbers

from utils_ import utils_flags
from absl import flags
from collections import namedtuple
from torchvision import datasets, transforms


FLAGS = flags.FLAGS

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        print(kernel.size())

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        #############################################################
        self.register_buffer('weight', kernel)
        #############################################################

        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

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
                 get_parametric_frequency_MSE_only=False,
                 get_parametric_frequency_MSE_CE=False,
                 IB_noise_std=0, 
                 gaussian_std=[0*i for i in range(64)],
                 layer_to_test=0):

        super(proxy_VGG3, self).__init__()

        self.noise_variance = float(noise_variance)
        self.device = device
        self.temp_net = net
        self.capacity = {}
        self.activations = {}
        self.verbose = verbose
        self.run_name = run_name
        self.get_running_var = False
        self.running_var = 0 

        self.bn_parameters = {}
        self.test_variance = {}

        # IB noise calculation
        self.IB_noise_calculation = IB_noise_calculation
        self.layer_to_test = layer_to_test

        ############################
        self.noise_std = IB_noise_std
        ############################

        # parametric frequency calculation
        self.get_parametric_frequency_MSE_only = get_parametric_frequency_MSE_only
        self.get_parametric_frequency_MSE_CE = get_parametric_frequency_MSE_CE

        ############################
        self.gaussian_std = torch.FloatTensor(gaussian_std)
        self.gaussian_std.requires_grad = True
        self.ground_truth_activations = 0
        self.gaussian_activations = 0
        self.channels = 64
        # self.conv_gaussian = nn.Conv2d(self.channels, self.channels, kernel_size=5, stride=1, padding=2, bias=False, groups=self.channels)
        # self.conv_gaussian.weight.requires_grad = False
        self.conv1 = net.features[0]
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

        if self.get_parametric_frequency_MSE_only:
            # carry out initial convolution
            x = self.conv1(x)

            # construct channel-wise gaussian-based convolutional layer
            self.ground_truth_activations = x.detach().clone()

            # apply it
            self.gaussian_activations = F.conv2d(x, self.get_gaussian_kernel(), padding=2, groups=self.channels)

            # calculate standard deviation gradient
            self.gaussian_std.retain_grad()

            # calculate convolution output gradient
            self.gaussian_activations.retain_grad()

            return self.gaussian_activations

        else:
            # set bn counter
            bn_count = 0

            # set conv counter for non-BN models
            conv_count = 0

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
                    
                    if self.get_parametric_frequency_MSE_CE and self.layer_to_test==bn_count:
                        # construct channel-wise gaussian-based convolutional layer
                        self.ground_truth_activations = x.detach().clone()
                        
                        # apply it
                        self.gaussian_activations = F.conv2d(x, self.get_gaussian_kernel(), padding=2, groups=self.channels)

                        # calculate standard deviation gradient
                        self.gaussian_std.retain_grad()

                        # calculate convolution output gradient
                        self.gaussian_activations.retain_grad()
                    
                    if self.get_parametric_frequency_MSE_CE and self.layer_to_test==bn_count:
                        x = model(self.gaussian_activations)
                    else:
                        x = model(x)

                    bn_count += 1

                else:
                    x = model(x)

                # get IB noise variance for un-normalized models
                if self.get_bn_int_from_name() not in [100, 1]:
                    if isinstance(model, torch.nn.modules.conv.Conv2d):    
                        if self.get_running_var and self.layer_to_test==conv_count:
                            print('LAYER: ', conv_count)
                            self.running_var = x.var([0, 2, 3], unbiased=False).detach()
                        if self.IB_noise_calculation and self.layer_to_test==conv_count:
                            noise = torch.zeros_like(x, device=self.device)
                            for dim in range(self.noise_std.size(0)):
                                noise[:, dim, :, :] = nn.functional.softplus(self.noise_std[dim])\
                                                        *torch.normal(0, 1, size=x[:, dim, :, :].size(), device=self.device)                
                            x = x + noise.to(self.device)
                            self.noise_std.retain_grad()

                        conv_count += 1

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
    
    