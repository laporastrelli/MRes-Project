from cProfile import run
import torch 
import torch.nn as nn
import os 
import math as mt

from zmq import device

def get_minmax(test_loader, device):
    max_ = [-100, -100, -100]
    min_ = [100, 100, 100]
    for _, temp_ in enumerate(test_loader, 0):
        X, y = temp_
        for channel in range(X.size(1)):
            if torch.max(X[:, channel, :, :]) > max_[channel]:
                max_[channel] = torch.max(X[:, channel, :, :])
            if torch.min(X[:, channel, :, :]) < min_[channel]:
                min_[channel] = torch.min(X[:, channel, :, :])

    max_tensor = torch.FloatTensor(max_)[:, None, None] * torch.ones_like(X[0, :, :, :])
    max_tensor = max_tensor.to(device)
    min_tensor = torch.FloatTensor(min_)[:, None, None] * torch.ones_like(X[0, :, :, :])
    min_tensor = min_tensor.to(device)

    return min_tensor, max_tensor

def get_path2delta(PATH_to_deltas_,
                   model_tag, 
                   run_name, 
                   attack):
                   
    PATH_to_deltas = PATH_to_deltas_ + model_tag
    if not os.path.isdir(PATH_to_deltas):
        os.mkdir(PATH_to_deltas)
    
    # create delta model-run folder if not existent
    if not os.path.isdir(PATH_to_deltas + '/' + run_name):
        os.mkdir(PATH_to_deltas + '/' + run_name )

    # create delta model-run folder if not existent
    if not os.path.isdir(PATH_to_deltas + '/' + run_name + '/' + attack + '/'):
        os.mkdir(PATH_to_deltas + '/' + run_name + '/' + attack + '/')

    path = PATH_to_deltas + '/' + run_name + '/' + attack + '/'
    
    return path

def get_model_path(root_path, model_name, run_name):
    PATH_to_model = root_path + '/models/' + model_name + '/' + run_name + '.pth'
    return PATH_to_model

def get_model_specs(run_name):
    model_name = run_name.split('_')[0]
    if run_name.split('_')[1] == 'no':
        if model_name.find('ResNet')!= -1:
            where_bn = [0,0,0,0]
        else:
            where_bn = [0,0,0,0,0]
    elif run_name.split('_')[1] == 'bn':
        if model_name.find('ResNet')!= -1:
            where_bn = [1,1,1,1]
        else:
            where_bn = [1,1,1,1,1]
    else:
        where_bn_idx = int(run_name.split('_')[1])
        if model_name.find('ResNet')!= -1:
            where_bn = [i*0 for i in range(4)]
        elif model_name.find('VGG')!= -1:
            where_bn = [i*0 for i in range(5)]
        where_bn[where_bn_idx] = 1
    
    return model_name, where_bn

def get_model_name(run_name):
    return run_name.split('_')[0]

def get_bn_config(run_name):
    model_name = run_name.split('_')[0]
    if run_name.split('_')[1] == 'no':
        if model_name.find('ResNet')!= -1:
            where_bn = [0,0,0,0]
        else:
            where_bn = [0,0,0,0,0]
    elif run_name.split('_')[1] == 'bn':
        if model_name.find('ResNet')!= -1:
            where_bn = [1,1,1,1]
        else:
            where_bn = [1,1,1,1,1]
    else:
        where_bn_idx = int(run_name.split('_')[1])
        if model_name.find('ResNet')!= -1:
            where_bn = [i*0 for i in range(4)]
        elif model_name.find('VGG')!= -1:
            where_bn = [i*0 for i in range(5)]
        where_bn[where_bn_idx] = 1
    
    return where_bn

def set_eval_mode(net, use_pop_stats):
    if use_pop_stats:
        net.eval()

def get_epsilon_budget(dataset):

    if dataset == 'CIFAR10':
        # 2/255, 5/255, 8/255, 10/255, 12/255, 16/255, 0.1, 0.2
        epsilon_in = [0.0392, 0.0980, 0.1565, 0.1961, 0.2352, 0.3137, 0.5, 1]

    # epsilon budget for SVHN
    if dataset == 'SVHN':
        # 2/255, 5/255, 8/255, 10/255, 12/255, 16/255, 0.1, 0.2
        epsilon_in = [0.0157, 0.0392, 0.0626, 0.0784, 0.0941, 0.125, 0.2, 0.4]  
    
    return epsilon_in

def get_bn_int_from_name(run_name):

    temp = run_name.split('_')[1]
    if temp == 'bn':
        bn_locations = 100
    elif temp == 'no':
        bn_locations = 0
    else:
        # add 1 for consistency with name 
        bn_locations = int(temp) + 1
    
    return bn_locations
    
def get_bn_config_train(model_name, bn_int):
    if model_name.find('VGG') != -1:
        if bn_int==100:
            bn_locations = [1,1,1,1,1]
        elif bn_int==0:
            bn_locations = [0,0,0,0,0]
        else:
            bn_locations = [i*0 for i in range(5)]
            bn_locations[int(bn_int-1)] = 1

    elif model_name.find('ResNet')!= -1:
        if bn_int==100:
            bn_locations = [1,1,1,1]
        elif bn_int==0:
            bn_locations = [0,0,0,0]
        else:
            bn_locations = [i*0 for i in range(4)]
            bn_locations[int(bn_int-1)] = 1
    
    return bn_locations

def set_load_pretrained(train, test_run):
    if train:
        load_pretrained = False
    elif not train and not test_run:
        load_pretrained = True

    return load_pretrained

def resize(X):
    # (ch, batchsize, height, width) --> (ch, batchsize, height x width)
    return X.view((X.size(1), -1)) 

def get_gram(X, device):
    # (ch, batchsize, height x width) --> (ch, batchsize, batchsize)
    return torch.matmul(X, torch.permute(X, (0, 2, 1))).to(device)

def CKA(X_clean, X_adv, device):
    X_clean = resize(X_clean)
    X_adv = resize(X_adv)
    K = get_gram(X_clean, device)
    L = get_gram(X_adv, device)
    CKA = HSIC(K,L,X_clean.size(0),device)/(torch.sqrt(HSIC(K,K,X_clean.size(0),device)\
          *HSIC(L,L,X_clean.size(0),device))) # (ch, 1)
    return CKA.cpu().detach().numpy().tolist()

def HSIC(gram1, gram2, m, device):
    return (center(gram1, device).view((gram1.size(0), -1))\
           *center(gram2, device).view((gram2.size(0), -1))).sum(axis=1)/((m-1)^2)

def center(matrix, device):
    centering = torch.eye(matrix.size(1)) - (1/matrix.size(1))*torch.ones(matrix.size(1), matrix.size(1))
    # centering = centering.unsqueeze(0).repeat(matrix.size(0), 1, 1, 1).to(device)
    centering = centering.to(device)
    centered_matrix = torch.matmul(centering, matrix)
    centered_matrix = torch.matmul(centered_matrix, centering)
    return centered_matrix

def cosine_similarity(X_clean, X_adv, device):
    X_clean = resize(X_clean).to(device)
    X_adv = resize(X_adv).to(device)
    return torch.nn.functional.cosine_similarity(X_clean, X_adv).cpu().detach().numpy().tolist()

def get_bn_layer_idx(model, model_name):
    bn_idx = []
    num_channels = []
    if model_name.find('VGG')!= -1:
        features = nn.ModuleList(list(model.features))
        for ii, layer in enumerate(features):
            if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
                bn_idx.append(ii)
                num_channels.append(layer.weight.size(0))

    return bn_idx


