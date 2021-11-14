from torch._C import _enable_minidumps
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import Bottleneck 

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet50(pretrained=False, progress=True)

    def get_model_architecture(self):
        temp = nn.Sequential(*list(self.net.children()))
        print(self.net.children())
        prev_modules_ = []
        modules_ = []
        add = True

        dims = [64, 128, 256, 512]
        layers = [3, 4, 6, 3]
        cnt = -1

        for module in self.net.modules():
            print(module)
            print('cacca')

        '''for child in self.net.children():
            print(child)
            print("\n")
            if (isinstance(child, nn.BatchNorm2d)):
                add = False
            if (isinstance(child, nn.Sequential)):
                cnt += 1
                sub_modules = []
                sequential = nn.Sequential()
                for nephew in child.children():
                    print(nephew)
                    if isinstance(nephew, models.resnet.Bottleneck):
                        sub_modules = []
                        bottleneck = models.resnet.Bottleneck(dims[cnt], layers[cnt])
                        bottleneck_ = []
                        for grand_nephew in nephew.children():
                            if not (isinstance(grand_nephew, nn.BatchNorm2d)) and not (isinstance(grand_nephew, nn.Sequential)):
                                sub_modules.append(grand_nephew)
                            elif (isinstance(grand_nephew, nn.Sequential)):
                                sub_sub_modules = []
                                sequential_ = nn.Sequential()
                                for grand_grand_nephew in grand_nephew.children():
                                    if not (isinstance(grand_grand_nephew, nn.BatchNorm2d)):
                                        print('cacca')
                                        sub_sub_modules.append(grand_grand_nephew)
                                sequential_ = nn.Sequential(*sub_sub_modules)
                        bottleneck_ = sub_modules + list(sequential_.children())
                        b = bottleneck(*bottleneck_)
                sequential = nn.Sequential(*list(b)) 
                modules_.append(*list(sequential))
                
            elif add:
                modules_.append(child)

            add = True
            a = set(modules_)
            b = set(prev_modules_)

        print(modules_)
        sequential = nn.Sequential(*modules_)'''
        

