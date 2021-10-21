import torch.nn as nn
import torch
import os

from utils.adversarial_attack import fgsm

def test(model_path, test_loader, device):
    # net.load_state_dict(torch.load(PATH))
    net = torch.load(model_path)
    net.cuda()

    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            X, y = data
            X,y = X.to(device), y.to(device)

            outputs = net(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print('Test Accuracy: %d %%' % (100 * correct / total))

def adversarial_test(model_path, model_tag, test_loader, dir_name, device):
    net = (torch.load(model_path))
    net.cuda()

    correct = 0
    total = 0

    save_delta = False

    for i, data in enumerate(test_loader):
        X, y = data
        X,y = X.to(device), y.to(device)
        delta = fgsm(net, X, y)

        dir_name = './deltas_partioned/' + model_tag + '/'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        if len(os.listdir(dir_name)) == 0 or save_delta==True:
            save_delta = True
            torch.save(delta, dir_name + "adversarial_delta_" + str(i) + ".pth") 
            
        outputs = net(X+delta)

        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        
    print('ADVERSARIAL Test accuracy: %d %%' % (100 * correct / total))

    
def cross_model_testing(test_loader, models_path, deltas_path, model_tag, device):

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