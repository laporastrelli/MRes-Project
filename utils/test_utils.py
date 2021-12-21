import torch.nn as nn
import torch
import os
import numpy as np

from utils.adversarial_attack import fgsm, pgd_linf
from utils import get_model

def test(net, 
        model_path, 
        test_loader, 
        device):

    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):  
            X, y = data
            X,y = X.to(device), y.to(device)
            
            outputs = net(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print('Test Accuracy: %d %%' % (100 * correct / total))

    acc = correct / total

    return acc

def adversarial_test(net, 
                    model_path, 
                    model_tag, 
                    run_name, 
                    test_loader, 
                    PATH_to_deltas_, 
                    device, 
                    attack,
                    epsilon, 
                    eval=True):

    net.load_state_dict(torch.load(model_path))
    net.cuda()

    if len(epsilon) > 1:
        correct_s = np.zeros((1, len(epsilon)))
    else:
        correct_s = 0
    total = 0

    if eval:
        
        for i, data in enumerate(test_loader, 0):
            X, y = data
            X, y = X.to(device), y.to(device)

            if attack == 'FGSM':
                delta = fgsm(net, X, y, epsilon)
                    
            elif attack == 'PGD':
                delta = pgd_linf(net, X, y, epsilon, alpha=1e-2, num_iter=40)

            # create delta model folder if not existent
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

            # save deltas and test model on adversaries
            if len(delta) == 1:
                eps_ = 'eps_' + str(epsilon[0]).replace('.', '')
                if not os.path.isdir(path + '/' + eps_ + '/'):
                    os.mkdir(path + '/' + eps_ + '/')

                torch.save(delta[0], path + '/' + eps_ + "/adversarial_delta_" + str(i) + ".pth") 
                
                with torch.no_grad():
                    outputs = net(X+delta[0])
                
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct_s += (predicted == y).sum().item()

            # if multiples epsilons are used save each of them and test model on adversaries
            else:

                for k, idv_delta in enumerate(delta):
                    num = epsilon[k]
                    eps_ = 'eps_' + str(num).replace('.', '')
                    if not os.path.isdir(path + '/' + eps_ + '/'):
                        os.mkdir(path + '/' + eps_ + '/')

                    torch.save(idv_delta, path + '/' + eps_ + '/' 
                                + "/adversarial_delta_" + str(i) + '.pth') 

                    with torch.no_grad():
                        outputs = net(X+idv_delta)

                    _, predicted = torch.max(outputs.data, 1)
                    correct_s[0, k] += (predicted == y).sum().item()

            total += y.size(0)

    else:
        if len(epsilon) > 1:
            correct_s = np.zeros((1, len(epsilon)))
        else:
            correct_s = 0
        total = 0
        PATH_to_deltas = PATH_to_deltas_ + model_tag

        
        for i, data in enumerate(test_loader, 0):
            X, y = data
            X, y = X.to(device), y.to(device)

            # load deltas 
            for h, eps in enumerate(epsilon):
                
                eps_ = 'eps_' + str(eps).replace('.', '')
                path = PATH_to_deltas + '/' + run_name + '/' + attack + '/' + eps_ + '/'
                deltas = os.listdir(path)
                delta = torch.load(path + "/adversarial_delta_" + str(i) + ".pth")
                
                with torch.no_grad():
                    outputs = net(X+delta)

                _, predicted = torch.max(outputs.data, 1)
                
                if len(epsilon) > 1:
                    correct_s[0, h] += (predicted == y).sum().item()
                else:
                    correct_s += (predicted == y).sum().item()

            total += y.size(0)

    acc = correct_s / total

    print('Adversarial Test Accuracy: ', acc)

    if len(epsilon) > 1:
        return acc.tolist()
    else:
        return acc

def cross_model_testing(file_name, 
                        mode, 
                        root_path,
                        test_loader, 
                        device, 
                        attack, 
                        epsilon):

    model_name = file_name.split('_', 1)[0]
    og_model_name = model_name
    dt_string = file_name.split('_', 1)[1]

    PATH_to_deltas = root_path + '/deltas/'

    if mode=='BN_on_STD':
        # retrieve model object and get the trained model weights path
        batch_norm = False
        run_name = og_model_name + '_' + dt_string
        net = get_model.get_model(og_model_name, batch_norm)   
        PATH_to_model = root_path + '/models/' + og_model_name + '/' + run_name + '.pth'

        # model name of the deltas to retrieve
        model_name = og_model_name + 'bn'
        run_name = og_model_name + '_' + dt_string

        # adversarial test
        adv_test_acc = adversarial_test(net, 
                                        PATH_to_model, 
                                        model_name, 
                                        run_name, 
                                        test_loader, 
                                        PATH_to_deltas, 
                                        device, 
                                        attack, 
                                        epsilon, 
                                        eval=False)

    else:
        # retrieve model object and get the trained model weights path
        batch_norm = True
        run_name = og_model_name + 'bn' + '_' + dt_string
        net = get_model.get_model(og_model_name, batch_norm) 
        PATH_to_model = root_path + '/models/' + og_model_name + 'bn' + '/' + run_name + '.pth'

        # model name of the deltas to retrieve
        model_name = og_model_name
        run_name = model_name + '_' + dt_string

        # adversarial test
        adv_test_acc = adversarial_test(net, 
                                        PATH_to_model, 
                                        model_name, 
                                        run_name, 
                                        test_loader, 
                                        PATH_to_deltas, 
                                        device, 
                                        attack, 
                                        epsilon, 
                                        eval=False)

    return adv_test_acc


def cross_model_testing_(test_loader, models_path, deltas_path, model_tag, device):

    saved_models = os.listdir(models_path)
    deltas = os.listdir(deltas_path)

    # load desired model
    for model_path in saved_models:
        model = torch.load(models_path + model_path)
        model_tag = model_path.split('_')[1]

        # for the same model - test for each of the adversarial produced by all models (including model itself)
        for delta_path in deltas:
            spec_deltas = os.listdir(deltas_path + delta_path)
            delta_model_name = delta_path

            # load iterator of test image on which adversarial variation are then added
            test_iter = iter(test_loader)
            
            # load the model-specific delta and evaluate the selectd model
            correct = 0
            total = 0
            for i, model_delta in enumerate(spec_deltas):
                batch_delta = torch.load(deltas_path + delta_path + '/' + model_delta)
                X, y = test_iter.next()
                X, y = X.to(device), y.to(device)
                model.cuda()
                
                outputs = model(X+batch_delta)

                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            print('ACCURACY ' + model_tag + ' trained on ' + delta_model_name + ' ADVERSARIAL test images: %d %%' % (100 * correct / total))


