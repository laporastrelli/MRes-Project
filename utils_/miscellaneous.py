from cProfile import run
import torch 
import os 

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

