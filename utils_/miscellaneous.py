from cProfile import run
import torch 
import torch.nn as nn
import os 
import math as mt
import numpy as np

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
                   attack, 
                   epsilon):
    
    # create delta model-type folder if not existent
    PATH_to_deltas = PATH_to_deltas_ + '/' + model_tag + '/'
    if not os.path.isdir(PATH_to_deltas):
        os.mkdir(PATH_to_deltas)
    
    # create delta model-run folder if not existent
    PATH_to_deltas +=  run_name + '/'
    if not os.path.isdir(PATH_to_deltas):
        os.mkdir(PATH_to_deltas)

    # create delta model-run folder if not existent
    PATH_to_deltas += attack + '/'
    if not os.path.isdir(PATH_to_deltas):
        os.mkdir(PATH_to_deltas)
    
    PATH_to_deltas += 'eps_' + str(epsilon).replace('.', '') + '/'
    if not os.path.isdir(PATH_to_deltas):
        os.mkdir(PATH_to_deltas)
    
    return PATH_to_deltas

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

def get_epsilon_budget(dataset, large_epsilon=False):

    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        # 2/255, 5/255, 8/255, 10/255, 12/255, 16/255, 0.1, 0.2
        epsilon_in = [0.0392, 0.0980, 0.1565, 0.1961, 0.2352, 0.3137, 0.5, 1]

        if large_epsilon:
            epsilon_in = [5]
            # epsilon_in = [1.25, 1.5, 1.75, 2, 2.5, 3.0, 3.5, 4.0, 4.5, 4.99]
            # epsilon_in = [100, 10, 20, 30, 40, 50, 60, 70]

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
           *center(gram2, device).view((gram2.size(0), -1))).sum(axis=1)/((m-1)**2)

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

def nonzero_idx(tensor,axis,invalid_item =-1):
    mask = tensor!=0
    indices = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_item)
    return indices

def entropy(iid_sample_in, which='K-L', device='cpu'):
    '''Kozachenko-Leonenko estimator for entropy
       based on a nearest-neighbour estimate. In 
       this function only 1-d vectors are considered.'''

    if len(list(iid_sample_in.size())) == 4:
        
        print(iid_sample_in)

        n_samples = iid_sample_in.size(0)
        dim = iid_sample_in.size(1)
        n_samples_size = int(n_samples/2)
        dim_size = int(dim/2)
        idx_samples = torch.randint(n_samples, size=(n_samples_size,))
        idx_dim = torch.randint(n_samples, size=(dim_size,))
        
        iid_sample = iid_sample_in.view(n_samples, -1)
        iid_sample = iid_sample[idx_samples, idx_dim]

    elif iid_sample_in.size(0) == 1:
        iid_sample = iid_sample_in.reshape(n_samples, 1)

    if which == 'K-L':
        repeated_sample = iid_sample.unsqueeze(0).repeat(n_samples_size, 1, 1)
        to_substract = iid_sample.unsqueeze(1).repeat(1, n_samples_size, 1)
        diff = repeated_sample - to_substract
        distance = torch.linalg.norm(diff, dim=2)
        sorted_distance, _ = torch.sort(distance, descending=False)
        print(sorted_distance)
        distance_vector = sorted_distance[:, 1]
        print(distance_vector)
        volume_unit = torch.Tensor([mt.pi**(dim_size)/mt.gamma(dim_size + 1)]).to(device)
        h = (1/n_samples_size)*torch.sum(torch.log(int(n_samples_size) * (distance_vector**(n_samples_size))) + torch.log(volume_unit) + 0.57721566490153286060)
        print(h)

        to_substract = iid_sample.reshape([iid_sample.size(0), -1]).repeat(1, iid_sample.size(0))
        diff = torch.sqrt(torch.pow(repeated_sample - to_substract, 2))
        sorted_distances, _ = torch.sort(diff, dim=1)
        positive_idxs = nonzero_idx(sorted_distances.numpy(), axis=1)
        min_distances = sorted_distances[:, positive_idxs]
        print('MIN: ', torch.min(min_distances))
        print('SUM: ', min_distances.mean())
        print('Internal: ', torch.sum(iid_sample.size(0)*(torch.tensor(mt.pi)**(2))*min_distances))
    
    elif which == 'gaussian':
        h = 1/2 * torch.log2(2*mt.pi*torch.var(iid_sample, unbiased=True)) + 1/2

    return h

    

    '''
    if len(list(iid_sample_in.size())) == 4:
        
        print(iid_sample_in)

        n_samples = iid_sample_in.size(0)
        dim = iid_sample_in.size(1)
        
        
        dim_size = int(dim/2)
        rand_idx = torch.randint(n_samples, size=(dim_size,))
        
        iid_sample = iid_sample_in.view(dim, -1)
        iid_sample = iid_sample[rand_idx, :]

    elif iid_sample_in.size(0) == 1:
        iid_sample = iid_sample_in.reshape(n_samples, 1)

    if which == 'K-L':
        repeated_sample = iid_sample.unsqueeze(0).repeat(dim_size, 1, 1)
        to_substract = iid_sample.unsqueeze(1).repeat(1, dim_size, 1)
        diff = repeated_sample - to_substract
        distance = torch.linalg.norm(diff, dim=2)
        sorted_distance, _ = torch.sort(distance, descending=False)
        print(sorted_distance)
        distance_vector = sorted_distance[:, 1]
        print(distance_vector)
        volume_unit = torch.Tensor([mt.pi**(dim/2)/mt.gamma(dim/2 + 1)]).to(device)
        h = (1/dim_size)*torch.sum(torch.log(int(dim_size) * (distance_vector**(dim_size))) + torch.log(volume_unit) + 0.57721566490153286060)
        print(h)

        to_substract = iid_sample.reshape([iid_sample.size(0), -1]).repeat(1, iid_sample.size(0))
        diff = torch.sqrt(torch.pow(repeated_sample - to_substract, 2))
        sorted_distances, _ = torch.sort(diff, dim=1)
        positive_idxs = nonzero_idx(sorted_distances.numpy(), axis=1)
        min_distances = sorted_distances[:, positive_idxs]
        print('MIN: ', torch.min(min_distances))
        print('SUM: ', min_distances.mean())
        print('Internal: ', torch.sum(iid_sample.size(0)*(torch.tensor(mt.pi)**(2))*min_distances))
        '''
        # h = (1/iid_sample.size(0))*torch.sum(torch.log(iid_sample.size(0)) (torch.tensor(mt.pi)**(2))*min_distances) + 0.57721566490153286060)'''
    
    