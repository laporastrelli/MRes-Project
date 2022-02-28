import torch.nn as nn
import torch.optim as optim
import torch 
import csv

from absl import flags
from absl.flags import FLAGS
from torch._C import _propagate_and_assign_input_shapes
from utils_ import utils_flags

def train (train_loader, val_loader, model, device, model_name, batch_norm, writer, run_name):
    
    # parse inputs 
    FLAGS = flags.FLAGS

    # cuda settings
    torch.cuda.empty_cache()

    ################ Modes ################

    # option for training models at the best of their performance (1)
    if FLAGS.mode == 'optimum': 
        if batch_norm :
            opt_func = optim.Adam
            lr_scheduler = optim.lr_scheduler.OneCycleLR
            max_lr = 2e-03
            # opt = optim.SGD(model.parameters(), lr=max_lr,  momentum=0.9, weight_decay=1e-4)
            opt = opt_func(model.parameters(), lr=max_lr, weight_decay=1e-4)
            n_epochs = 75
            scheduler = lr_scheduler(opt, max_lr, epochs=n_epochs, steps_per_epoch=len(train_loader))
            grad_clip = False
            grad_clip_val = 0.1

        else:
            max_lr = 2e-04
            opt = optim.SGD(model.parameters(), lr=max_lr,  momentum=0.9, weight_decay=1e-4)
            n_epochs = 125
            scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr, epochs=n_epochs, steps_per_epoch=len(train_loader))
            grad_clip = True
            grad_clip_val = 0.01
            
    # option for training models at the best of their performance (2) -- ResNet (only)
    if FLAGS.mode == 'standard': 
        if FLAGS.dataset == 'CIFAR10':
            # option for full-BN training
            if batch_norm and sum(FLAGS.where_bn)>1:
                lr_scheduler = optim.lr_scheduler.MultiStepLR
                lr_ = 0.1
                opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=5e-4)
                n_epochs = 150
                grad_clip = False
                scheduler = optim.lr_scheduler.MultiStepLR(opt, [50, 100], gamma=0.1) 
            # option for partial (or none)-BN training
            else: 
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau
                val = 1
                if val in FLAGS.where_bn and FLAGS.where_bn.index(1) == 2:
                    lr_ = 0.03
                else:
                    lr_ = 0.01
                opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=5e-4)
                n_epochs = 150
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=10)
                grad_clip = True
                grad_clip_val = 0.1

        elif FLAGS.dataset == 'SVHN':
            # option for full-BN training
            if batch_norm and sum(FLAGS.where_bn)>1:
                lr_scheduler = optim.lr_scheduler.MultiStepLR
                lr_ = 0.1
                opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=5e-4)
                n_epochs = 75
                grad_clip = False
                scheduler = optim.lr_scheduler.MultiStepLR(opt, [25, 45], gamma=0.1) 
            # option for partial (or none)-BN training
            else: 
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau
                val = 1
                if val in FLAGS.where_bn and FLAGS.where_bn.index(1) == 2:
                    lr_ = 0.03
                elif sum(FLAGS.where_bn)==0:
                    lr_ = 0.001
                else:
                    lr_ = 0.01
                opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=5e-4)
                n_epochs = 75
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=10)
                grad_clip = True
                grad_clip_val = 0.1
                
    # option for training models at the best of their performance (3) -- VGG or ResNet
    else:
        if model_name.find('ResNet')!= -1 :
            print("Training ResNet50 ...")
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau

            interval = []
            if batch_norm:
                opt_func = optim.SGD
                lr_ = 0.1
                n_epochs = 200
                momentum = 0.9
                weight_decay = 5e-4
                grad_clip = False
            else:
                opt_func = optim.SGD
                lr_ = 0.01
                n_epochs = 200
                momentum = 0.9
                weight_decay = 5e-4
                grad_clip = True
                grad_clip_val=0.1

            opt = opt_func(model.parameters(), lr=lr_,  momentum=momentum, weight_decay=weight_decay)
            scheduler = lr_scheduler(opt, 'min', patience=15)
        
        if model_name.find('VGG')!= -1:
            print("Training VGG ...")
            opt_func = optim.SGD
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau
            interval = []
            n_epochs = 65
            if FLAGS.dataset=='SVHN':
                n_epochs=35
            momentum = 0.9
            weight_decay=5e-4
            grad_clip = False
            lr_ = 0.01

            if FLAGS.dataset == 'SVHN' and FLAGS.where_bn[4]==1 and sum(FLAGS.where_bn)==1:
                lr_= 0.0075

            opt = opt_func(model.parameters(), lr=lr_,  momentum=momentum, weight_decay=weight_decay)
            scheduler = lr_scheduler(opt, 'min', patience=6)

    print('LR scheduler: ', lr_scheduler)
    print('Starting LR: ', lr_)

    ################ Training ################
    for epoch_num in range(n_epochs):
        model.train()
        total_loss, total_err = 0.,0.
        for i, data in enumerate(train_loader, 0):
            X,y = data
            X,y = X.to(device), y.to(device)

            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)
            opt.zero_grad()
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=grad_clip_val)

            opt.step()

            if lr_scheduler == optim.lr_scheduler.OneCycleLR:
                scheduler.step()

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item()

        print(*("{:.6f}".format(i) for i in (int(epoch_num), total_err/len(train_loader.dataset), total_loss/len(train_loader.dataset))), sep="\t")

        writer.add_scalar('Training accuracy',
                        1 - (total_err/len(train_loader.dataset)),
                        epoch_num* len(train_loader) + i)

        writer.add_scalar('Training loss',
                        total_loss/len(train_loader.dataset),
                        epoch_num* len(train_loader) + i)

        ################ Validation ################
        valid_loss = 0.0
        model.eval()
      
        correct = 0
        total = 0

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data
                X,y = X.to(device), y.to(device)

                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)

                loss = nn.CrossEntropyLoss()(outputs,y)
                valid_loss += loss.item()

                total += y.size(0)
                correct += (predicted == y).sum().item()

            print('Validation Accuracy: %d %%' % (100 * correct / total))

            val_acc = correct / total

            writer.add_scalar('Validation accuracy',
                            val_acc,
                            epoch_num* len(val_loader) + i)
            writer.add_scalar('Validation loss',
                            valid_loss/len(val_loader.dataset),
                            epoch_num* len(val_loader) + i)

        if lr_scheduler == optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(valid_loss/len(val_loader.dataset))
        else:
            scheduler.step()

    metric_dict = {'tes_acc': str(val_acc), 'test_loss': str(valid_loss)}
    # writer.add_hparams(hparam_dict, metric_dict, run_name=run_name)

    return model














    '''if lr_scheduler != optim.lr_scheduler.MultiStepLR and lr_scheduler != optim.lr_scheduler.StepLR: interval = 'n.a.'

    str_interval = ''
    for i, el in enumerate(interval):
        if i > 0:
            str_interval += ' - '
        str_interval += str(el)

    hparam_dict = {'model name': model_name,
                  'batch-norm': batch_norm,
                  'scheduler': str(lr_scheduler), 
                  'intervals': str_interval,
                  'lr': lr_,
                  'momentum': momentum, 
                  'weight decay': weight_decay, 
                  'optimizer': str(opt_func),  
                  'grad_clip': grad_clip, 
                  'epochs': n_epochs}
    '''
