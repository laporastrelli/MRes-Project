from os import stat
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

    def __init__(self, net, noise_variance, device):
        super(noisy_VGG, self).__init__()

        self.noise_variance = noise_variance
        self.device = device

        features = list(net.features)
        
        # ---- WATCH OUT w/ .eval() ---- #
        ###########################################
        self.features = nn.ModuleList(features).eval()
        self.avgpool = net.avgpool.eval()
        self.classifier = net.classifier.eval()
        ###########################################

        stats = []
        for jj, layer in enumerate(self.features):
            if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[jj-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                temp_running_mean = layer.running_mean
                temp_running_var = layer.running_var
                stats.append([temp_running_mean, temp_running_var])

    
    def forward(self, x):
        for ii, model in enumerate(self.features):
            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                noise = torch.normal(0, float(self.noise_variance), size=x.size())
                if not FLAGS.noise_after_BN:
                    x = x + noise.to(self.device)

            x = model(x)

            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d) and FLAGS.noise_after_BN:
                x = x + noise.to(self.device)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)

        return out


