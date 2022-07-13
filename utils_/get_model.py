from absl.flags import FLAGS
import torchvision.models as models
from models import ResNet_v1, ResNet_v2, ResNet_v3, VGG, noisy_VGG_train, proxy_VGG, proxy_ResNet
from models.proxy_VGG3 import proxy_VGG3
from utils_ import utils_flags

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
        
        if FLAGS.capacity_regularization or FLAGS.rank_init:
            print('Training/Testing with capacity regularization ...')
            net = proxy_ResNet.proxy_ResNet(net, 
                                            eval_mode=FLAGS.use_pop_stats,
                                            device=FLAGS.device,
                                            noise_variance=FLAGS.noise_variance, 
                                            train_mode=FLAGS.train,
                                            regularization_mode=FLAGS.regularization_mode)
    
    if model_name == 'ResNet34':
        if sum(where_bn)>1:
            if int(FLAGS.version) == 2:
                net = ResNet_v2.Resnet101(where_bn=where_bn)
            else:
                net = ResNet_v1.ResNet34(where_bn=where_bn)
        else:
            if int(FLAGS.version) == 2:
                net = ResNet_v2.resnet101(where_bn=where_bn)
            else:
                net = ResNet_v1.ResNet34(where_bn=where_bn)

    elif model_name == 'VGG19':
        if sum(where_bn)==0:
            net = VGG.vgg19(where_bn=where_bn, normalization=FLAGS.normalization)
        else:
            net = VGG.vgg19_bn(where_bn=where_bn, normalization=FLAGS.normalization)

        if FLAGS.normalization == 'ln':
            print(net)

        if FLAGS.train_noisy:
            print("Noisy Training ...")
            net = noisy_VGG_train.noisy_VGG_train(net, FLAGS.train_noise_variance, FLAGS.device)
        
        if FLAGS.capacity_regularization:
            print('Training/Testing with capacity regularization ...')
            if model_name.find('VGG')!= -1:
                net = proxy_VGG.proxy_VGG(net, 
                                          eval_mode=FLAGS.use_pop_stats,
                                          device=FLAGS.device,
                                          noise_variance=FLAGS.noise_variance, 
                                          train_mode=FLAGS.train,
                                          regularization_mode=FLAGS.regularization_mode)
        
        if FLAGS.rank_init:
            print('Training/Testing with rank-preserving initialization ...')
            if model_name.find('VGG')!= -1:
                net = proxy_VGG3(net, 
                                eval_mode=FLAGS.use_pop_stats,
                                device=FLAGS.device,
                                noise_variance=FLAGS.noise_variance)

            

    elif model_name == 'VGG16':
        if sum(where_bn)==0:
            net = VGG.vgg16(where_bn=where_bn)
        else:
            net = VGG.vgg16_bn(where_bn=where_bn)
        
    return net