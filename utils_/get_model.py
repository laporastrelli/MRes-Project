from absl.flags import FLAGS
import torchvision.models as models
from models import ResNet_v1, ResNet_v2, ResNet_v3, VGG

def get_model(model_name, where_bn):
    if model_name == 'ResNet50':
        if sum(where_bn)>1:
            print('BN')
            if int(FLAGS.version) == 2:
                net = ResNet_v2.ResNet50(where_bn=where_bn)
            elif int(FLAGS.version) == 3:
                net = ResNet_v3.ResNet50(where_bn=where_bn)
            else:
                print("Custom ResNet")
                net = ResNet_v1.ResNet50(where_bn=where_bn)
        else:
            print('no BN')
            print(where_bn)
            if int(FLAGS.version) == 2:
                net = ResNet_v2.ResNet50(where_bn=where_bn)
            elif int(FLAGS.version) == 3:
                net = ResNet_v3.ResNet50(where_bn=where_bn)
            else:
                net = ResNet_v1.ResNet50(where_bn=where_bn)
    
    if model_name == 'ResNet101':
        if sum(where_bn)>1:
            if int(FLAGS.version) == 2:
                net = ResNet_v2.Resnet101(where_bn=where_bn)
            else:
                net = ResNet_v1.ResNet101(where_bn=where_bn)
        else:
            if int(FLAGS.version) == 2:
                net = ResNet_v2.resnet101(where_bn=where_bn)
            else:
                net = ResNet_v1.ResNet101(where_bn=where_bn)

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