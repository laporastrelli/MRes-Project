from os import stat
from numpy import outer
import torch
import torch.nn as nn

from collections import namedtuple
from models.Batch_Norm_custom import BatchNorm2d

class test_VGG(nn.Module):

    def __init__(self, net, batch):
        super(test_VGG, self).__init__()

        self.batch = batch
        features = list(net.features)
        
        # ---- WATCH OUT w/ .eval() ---- #
        ###########################################
        self.features = nn.ModuleList(features).eval()
        self.avgpool = net.avgpool.eval()
        self.classifier = net.classifier.eval()
        ###########################################

    def forward(self, x_adv):
        batch_ = self.batch
        for ii, model in enumerate(self.features):
            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                # calculating clean batch stats
                temp = batch_
                running_mean = temp.mean([0, 2, 3])
                running_var = temp.var([0, 2, 3], unbiased=False)

                # setting up batch norm replacing running stats w/ batch stats
                bn_custom = BatchNorm2d(batch_.shape[1])
                bn_custom.training = False
                bn_custom.running_mean = running_mean
                bn_custom.running_variance = running_var
                bn_custom.gamma = model.weight
                bn_custom.beta = model.bias

                # apply BN w/ batch stats 
                x_adv = bn_custom(x_adv)
                batch_ = bn_custom(batch_)

            else:
                batch_ = model(batch_)
                x_adv = model(x_adv)

        x_adv = self.avgpool(x_adv)
        x_adv = torch.flatten(x_adv, 1)
        out = self.classifier(x_adv)

        return out



