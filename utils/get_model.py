from absl.flags import FLAGS
import torchvision.models as models
from models import ResNet, ResNet_bn, resnet

def get_model(model_name, batch_norm):
    if model_name == 'ResNet50':
        if batch_norm:
            net = ResNet_bn.resnet50()
        else:
            net = ResNet.resnet50(where_bn=FLAGS.where_bn)
    
    elif model_name == 'ResNet101':
        if batch_norm:
            net = ResNet_bn.resnet101()
        else:
            net = ResNet.resnet101(where_bn=FLAGS.where_bn)

    elif model_name == 'VGG19':
        if batch_norm:
            net = models.vgg19_bn()
        else:
            net = models.vgg19()
    
    elif model_name == 'VGG16':
        if batch_norm:
            net = models.vgg16_bn()
        else:
            net = models.vgg16()

    elif model_name == 'VGG13':
        if batch_norm:
            net = models.vgg13_bn()
        else:
            net = models.vgg13()
    
    elif model_name == 'VGG11':
        if batch_norm:
            net = models.vgg11_bn()
        else:
            net = models.vgg11()
    return net