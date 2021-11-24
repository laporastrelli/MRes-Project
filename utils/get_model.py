from absl.flags import FLAGS
import torchvision.models as models
from models import ResNet, ResNet_bn, VGG, resnet_

def get_model(model_name, where_bn):
    if model_name == 'ResNet50':
        if sum(where_bn)>1:
            print('BN')
            net = ResNet_bn.resnet50()
        else:
            print('no BN')
            net = ResNet.resnet50(where_bn=where_bn)
    
    if model_name == 'ResNet101':
        if sum(where_bn)>1:
            net = ResNet_bn.resnet101()
        else:
            net = ResNet.resnet101(where_bn=where_bn)

    elif model_name == 'VGG19':
        if sum(where_bn)==0:
            net = VGG.vgg19(where_bn=where_bn)
        else:
            net = VGG.vgg19_bn(where_bn=where_bn)

    elif model_name == 'VGG16':
        if sum(where_bn)==0:
            net = VGG.vgg16(where_bn=where_bn)
        else:
            net = VGG.vgg16_bn(where_bn=where_bn)
        
    return net