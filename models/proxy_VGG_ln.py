from os import stat
from time import time
from tkinter import Image
from matplotlib import scale
import numpy as np
from tabnanny import verbose
from cv2 import mean
from numpy import int32
import torch
import torch.nn as nn
from torch.nn import functional as F
from zmq import device
import math
import numbers
import time

from utils_ import utils_flags
from absl import flags
from collections import namedtuple
from torchvision import datasets, transforms

FLAGS = flags.FLAGS

class Tanh_ScaleLayer(nn.Module):

   def __init__(self, n_channels, device, init_value=1, var_scaling=1.0):
       super().__init__()
       self.scale = nn.Parameter(init_value*torch.ones(n_channels, device=device))
       self.var_scaling = var_scaling

   def forward(self, input):
       return (self.var_scaling*nn.functional.tanh(self.scale)).view(1, -1, 1, 1).expand(input.size()) * input

class proxy_VGG_ln(nn.Module):
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
                 attenuate_HF=False,
                 IB_noise_std=0, 
                 gaussian_std=[0*i for i in range(64)],
                 layer_to_test=0, 
                 bounded_lambda=False, 
                 dropout_bn=False,
                 nonlinear_lambda=False,
                 free_lambda=False,
                 uniform_lambda=False,
                 train_mode=False):

        super(proxy_VGG_ln, self).__init__()

        self.noise_variance = float(noise_variance)
        self.device = device
        self.temp_net = net
        self.capacity = {}
        self.activations = {}
        self.verbose = verbose
        self.run_name = run_name
        self.get_running_var = False
        self.running_var = 0 
        self.bounded_lambda = bounded_lambda
        self.dropout_bn = dropout_bn
        self.nonlinear_lambda = nonlinear_lambda
        self.free_lambda = free_lambda
        self.uniform_lambda = uniform_lambda
        self.train_mode = train_mode

        self.attenuate_HF = attenuate_HF

        self.bn_parameters = {}
        self.running_variances = {}
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
        self.channels = len(gaussian_std)
        self.gaussian_std = torch.FloatTensor(gaussian_std)
        self.gaussian_std.requires_grad = True
        self.ground_truth_activations = 0
        self.gaussian_activations = 0
        # self.conv_gaussian = nn.Conv2d(self.channels, self.channels, kernel_size=5, stride=1, padding=2, bias=False, groups=self.channels)
        # self.conv_gaussian.weight.requires_grad = False
        self.conv1 = net.features[0]
        ############################

        ############################
        self.last_layer = 0
        ############################

        ############################
        features = list(net.features)
        classifier = list(net.classifier)
        ############################
        
        # ---- WATCH OUT w/ .eval() ---- #
        ###########################################
        if eval_mode:
            net.eval()
            self.features = nn.ModuleList(features).eval()
            self.avgpool = net.avgpool.eval()
            # self.classifier = net.classifier.eval()
            self.classifier = nn.ModuleList(classifier).eval()
        else:
            self.features = nn.ModuleList(features)
            self.avgpool = net.avgpool
            self.classifier = nn.ModuleList(classifier)
        ###########################################

        ###########################################
        if self.nonlinear_lambda and self.train_mode:
            self.init_scale_layers()
            self.set_gradient_mode()
        
        if self.dropout_bn:
            self.init_dropout_layers()
        
        if (self.bounded_lambda and self.free_lambda) or self.uniform_lambda:
            print('Training with gradient free adaptive lambda')
            self.set_gradient_mode()
        ###########################################
    
    def init_scale_layers(self):
        self.scale_layers = []
        for ii, model in enumerate(self.features): 
                if isinstance(model, torch.nn.modules.normalization.LayerNorm):
                    assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                    self.scale_layers.append(Tanh_ScaleLayer(n_channels=model.weight.size(), device=self.device, init_value=0.5))
    
    def init_dropout_layers(self):
        self.dropout_layers = []
        bn_c = 0
        for ii, model in enumerate(self.features): 
                if isinstance(model, torch.nn.modules.normalization.LayerNorm):
                    assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                    if bn_c == 0:
                        p = 0.85
                    else:
                        p = 0.85
                    self.dropout_layers.append(nn.Dropout(p=p))
                    bn_c +=1

    def set_gradient_mode(self):
        for ii, model in enumerate(self.features): # BatchNorm layers are only present in the encoder of VGG19
            if isinstance(model, torch.nn.modules.normalization.LayerNorm):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                model.weight.requires_grad = False
    
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
            self.gaussian_activations = F.conv2d(x, self.get_gaussian_kernel(kernel_size=kernel_size), padding=padding, groups=self.channels)

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

                if isinstance(model, torch.nn.modules.normalization.LayerNorm):
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
                    
                    if self.get_parametric_frequency_MSE_CE and bn_count==self.layer_to_test:
                        # construct channel-wise gaussian-based convolutional layer
                        self.ground_truth_activations = x.detach().clone()
                        
                        # apply it
                        if self.layer_to_test == 0:
                            kernel_size = 5
                            padding=2
                        else:
                            kernel_size = 3
                            padding=1
                        self.gaussian_activations = F.conv2d(x, self.get_gaussian_kernel(), padding=2, groups=self.channels)

                        # calculate standard deviation gradient
                        self.gaussian_std.retain_grad()

                        # calculate convolution output gradient
                        self.gaussian_activations.retain_grad()

                        # apply batch norm
                        x = model(self.gaussian_activations)

                    if self.attenuate_HF and bn_count==self.layer_to_test:
                        # apply BN
                        x = model(x)
                        # create copy of BN-activated layer
                        x_HF = x.clone()
                        # transform activations to have attenuated HF 
                        scale = model.weight / torch.sqrt(model.running_var)
                        scale = torch.where(scale > 1, scale, torch.Tensor([1.]).to(self.device))
                        x = self.HF_manipulation(x_HF, r=15, scale=scale)

                    if self.bounded_lambda:
                        # model.weight = torch.nn.Parameter((torch.sqrt(model.running_var)/model.weight) * nn.functional.softmax(model.weight))
                        # model.weight.data = model.weight.data.clamp(-torch.sqrt(model.running_var)/model.weight, torch.sqrt(model.running_var)/model.weight)
                        if self.free_lambda:
                            '''adaptive_scaling = torch.ones_like(model.weight)
                            for i in range(model.weight.size(0)):
                                adaptive_scaling[i] = torch.normal(model.running_var[i], 0.5)
                                #print(adaptive_scaling)'''
                            adaptive_scaling = 0.1*torch.ones_like(model.weight)
                            model.weight.data = adaptive_scaling*model.weight.data
                        else:
                            model.weight.data = model.weight.data.clamp(-torch.sqrt(model.running_var), torch.sqrt(model.running_var))
                        x = model(x)

                    if self.nonlinear_lambda:
                        lambda_scaling = self.scale_layers[bn_count]
                        lambda_scaling.var_scaling = torch.sqrt(model.running_var)
                        x = lambda_scaling(x)
                        x = model(x)

                    if self.dropout_bn:
                        bn_dropout = self.dropout_layers[bn_count]
                        x = bn_dropout(x)
                        x = model(x)

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
                        if self.get_parametric_frequency_MSE_CE and self.layer_to_test==conv_count:
                            # construct channel-wise gaussian-based convolutional layer
                            self.ground_truth_activations = x.detach().clone()
                            # apply it
                            x = F.conv2d(x, self.get_gaussian_kernel(), padding=2, groups=self.channels)

                            self.gaussian_activations = x.clone()

                            # calculate standard deviation gradient
                            self.gaussian_std.retain_grad()

                            # calculate convolution output gradient
                            self.gaussian_activations.retain_grad()

                            conv_count += 1
                            bn_count += 1

                if self.noise_variance != float(0):
                    x = x + torch.normal(0, float(self.noise_variance), size=x.size()).to(self.device)
            
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            
            # unpack classifier module list to get last hidden layer activations
            for jj, classifier_layer in enumerate(self.classifier):
                if jj == 5:
                    self.last_layer = x.clone()
                x = classifier_layer(x)

            return x

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

    def get_bn_parameters(self, get_variance=False):
        bn_count = 0
        for ii, model in enumerate(self.features):
            if isinstance(model, torch.nn.modules.normalization.LayerNorm):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), \
                       "Previous module should be Conv2d"
                self.bn_parameters['BN_' + str(bn_count)] = model.weight.cpu().detach()
                if get_variance:
                    self.running_variances['BN_' + str(bn_count)] = model.running_var.cpu().detach()
                bn_count +=1
        return self.bn_parameters
    
    def get_running_variance(self):
        self.get_bn_parameters(get_variance=True)
        return self.running_variances
    
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
    
    def distance(self, i, j, imageSize, r):
        dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
        if dis < r:
            return 1.0
        else:
            return 0

    def mask_radial(self, img, r):
        rows, cols = img.shape
        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = self.distance(i, j, imageSize=rows, r=r)
        return mask

    def fft(self, img):
        return torch.fft.fft2(img)

    def fftshift(self, img):
        return torch.fft.fftshift(self.fft(img))

    def ifft(self, img):
        return torch.fft.ifft2(img)

    def ifftshift(self, img):
        return self.ifft(torch.fft.ifftshift(img))
    
    def scaled_distance(self, i, j, imageSize, r, scale=1):
        dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
        if dis < r:
            return 0.0
        else:
            if scale > 1.0:
                return 1/scale
            else:
                return 1.0

    def scaled_mask_radial(self, img, r, scale):
        # first create location-based mask based on distance from the center of the image
        mask =  torch.zeros_like(img[0, 0, :, :])
        for i in range(img.size(2)):
            for j in range(img.size(3)):
                mask[i, j] = self.scaled_distance(i, j, imageSize=img.size(2), r=r, scale=1)
        
        # reshape mask to have the same dimension as img
        mask = mask.view(1, 1, mask.size(0), mask.size(1)).expand(img.size())
        scaling = scale.view(1,-1, 1, 1).expand(img.size())

        # scale mask based on 0/1 value
        scaled_mask = mask*(1/scaling)
        scaled_mask = scaled_mask + torch.where(scaled_mask == 0, 1., 0.)

        return scaled_mask
    
    def HF_manipulation(self, Images, r, scale):
        # apply FFT
        fd = self.fftshift(Images)
        #### TEST ####
        #print(torch.equal(fd[0, 0, :, :], self.fftshift(Images[0, 0, :, :])))
        # get mask and apply it
        mask = self.scaled_mask_radial(Images, r, scale)
        fd = fd * mask

        # apply IFFT
        img = self.ifftshift(fd)

        return torch.real(img)


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
