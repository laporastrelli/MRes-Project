from os import stat
import torch
import torch.nn as nn

from collections import namedtuple

# VGG contains the following class variable:
# features --> avgpool --> flatten --> classifier

# ---OBSERVATION---: Models are more robust if (without using net.eval)
#                    BatchNorm makes use of test mini-batch statatsics
#                    at evaluation time. 

# ---AIM---: Inject noise to BN layers in "features":
#            for each BN layer that we find save running_mean
#            and running_var and unject Gaussian Noise with
#            mean equal zero and variance equal to alpha*running_var 
#            where alpha is tuned to keep performance invariant (or almost).

# ---GOAL---: Observe whether by injecting noise (and using net.eval())
#             the model is more robust to adversarial attacks. 

class proxy_VGG(nn.Module):

    def __init__(self, net):
        super(proxy_VGG, self).__init__()

        features = list(net.features)[:-1]
        
        # ---- WATCH OUT w/ .eval() ---- #
        ###########################################
        self.features = nn.ModuleList(features).eval()
        ###########################################

        stats = []
        for jj, layer in enumerate(self.features):
            if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[jj-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                temp_running_mean = layer.running_mean
                temp_running_var = layer.running_var
                stats.append([temp_running_mean, temp_running_var])

    def forward(self, x):
        results = []
        cnt_bn = 0 
        for ii, model in enumerate(self.features):
            if isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                assert isinstance(self.features[ii-1], torch.nn.modules.conv.Conv2d), "Previous module should be Conv2d"
                cnt_bn += 1
                temp = x
                temp_mean = temp.mean([0, 2, 3])
                temp_var = temp.var([0, 2, 3], unbiased=False)

                if cnt_bn == 1:
                    temp_running_mean = model.running_mean
                    temp_running_var = model.running_var

            x = model(x)
            if isinstance(model, torch.nn.modules.conv.Conv2d):
                if isinstance(self.features[ii+1], torch.nn.modules.batchnorm.BatchNorm2d):
                    continue
                else: 
                    results.append(x.mean([2,3]).cpu().numpy())

            elif isinstance(model, torch.nn.modules.batchnorm.BatchNorm2d):
                gamma = model.weight
                beta = model.bias
                eps = model.eps
                running_mean = model.running_mean.detach().clone()
                running_var = model.running_var.detach().clone()

                out = (temp - running_mean[None, :, None, None]) / (torch.sqrt(running_var[None, :, None, None] + eps))
                custom_batch_norm = (out*gamma[None, :, None, None]) + beta[None, :, None, None]

                print('RUNNING STATISTICS: ', (custom_batch_norm - x).abs().max())

                out_ = (temp - temp_mean[None, :, None, None]) / (torch.sqrt(temp_var[None, :, None, None] + eps))
                custom_batch_norm_ = (out_*gamma[None, :, None, None]) + beta[None, :, None, None]
                
                print('mini-BATCH STATISTICS:', (custom_batch_norm_ - x).abs().max())

                out__ = (temp - temp_running_mean[None, :, None, None]) / (torch.sqrt(temp_running_var[None, :, None, None] + eps))
                custom_batch_norm__ = (out__*gamma[None, :, None, None]) + beta[None, :, None, None]

                # print('PREVIOUS RUNNING STATISTICS: ', (custom_batch_norm__ - x).abs().max())

                results.append(x.mean([2,3]).cpu().numpy())

        return results


