from os import stat
import torch
import torch.nn as nn

from utils_ import utils_flags
from absl import flags

FLAGS = flags.FLAGS

from collections import namedtuple

class noisy_VGG_train(nn.Module):

    def __init__(self, net, train_noise_variance, device):
        super(noisy_VGG_train, self).__init__()

        self.noise_variance = train_noise_variance
        self.device = device

        features = list(net.features)
        self.features = nn.ModuleList(features)
        self.avgpool = net.avgpool
        self.classifier = net.classifier
 
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


