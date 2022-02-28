import torch 
import torch.nn as nn

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(BatchNorm2d, self).__init__()

        """
        An implementation of a Batch Normalization over a mini-batch of 2D inputs.

        The mean and standard-deviation are calculated per-dimension over the
        mini-batches and gamma and beta are learnable parameter vectors of
        size num_features.

        Parameters:
        - num_features: C from an expected input of size (N, C, H, W).
        - eps: a value added to the denominator for numerical stability. Default: 1e-5
        - momentum: momentum – the value used for the running_mean and running_var
        computation. Default: 0.1
        - gamma: the learnable weights of shape (num_features).
        - beta: the learnable bias of the module of shape (num_features).
        """

        # TODO: Define the parameters used in the forward pass                 #
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # self.register_parameter is not used as it was mentioned on piazza
        # that this will be overridden
        self.gamma = torch.ones(self.num_features)
        self.beta = torch.zeros(self.num_features)

        self.running_mean = torch.zeros(self.num_features)
        self.running_variance = torch.ones(self.num_features)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        """
        During training this layer keeps running estimates of its computed mean and
        variance, which are then used for normalization during evaluation.
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data of shape (N, C, H, W) (same shape as input)
        """
        # TODO: Implement the forward pass                                     #
        #       (be aware of the difference for training and testing)          #
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mean = x.mean([0, 2, 3])
        variance = x.var([0, 2, 3], unbiased=False)
        n = x.numel() / x.size(1)

        if self.training:
          self.running_mean = (1-self.momentum)*self.running_mean + (self.momentum*mean)
          self.running_variance = (1-self.momentum)*self.running_variance + (self.momentum*variance*(n/(n-1)))

        else:
          mean = self.running_mean
          variance = self.running_variance

        out_ = (x - mean[None, :, None, None]) / (torch.sqrt(variance[None, :, None, None] + self.eps))
        out = out_*self.gamma[None, :, None, None] +  self.beta[None, :, None, None]

        x = out

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return x