import torch.nn as nn
import torch.optim as optim
import torch 
import csv
import math
import wandb

from absl import flags
from absl.flags import FLAGS
from torch._C import _propagate_and_assign_input_shapes
from utils_ import utils_flags
from utils_.miscellaneous import get_bn_layer_idx



def train (train_loader, 
           val_loader, 
           model, 
           device, 
           model_name, 
           batch_norm, 
           writer, 
           run_name):

    # parse inputs 
    FLAGS = flags.FLAGS

    # init server for wandb
    if FLAGS.save_to_wandb:
        run = wandb.init(project="capacity-regularization", 
                         entity="information-theoretic-view-of-bn", 
                         name=run_name)
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

    # wandb config
    if FLAGS.save_to_wandb:
        if sum(FLAGS.where_bn)==0:
            bn_string = 'No'
        elif sum(FLAGS.where_bn)>1:
            bn_string = 'Yes - ' + 'all'
        else:
            bn_string = 'Yes - ' + str(FLAGS.where_bn.index(1) + 1) + ' of ' + str(len(FLAGS.where_bn))
        config = {
            "run_name": run_name,
            "bn_config": bn_string,
            "dataset": FLAGS.dataset,
            "batch_size": FLAGS.batch_size,
            "optimizer": opt_func,
            "capacity_regularization": FLAGS.capacity_regularization,
            "regularization_mode": FLAGS.regularization_mode,
            "beta": FLAGS.beta,
            "learning_rate": lr_,
            "learning_rate_scheduler": lr_scheduler,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "gradient_clipping": grad_clip, 
            "epochs": n_epochs}
        run.config.update(config)

    print('LR scheduler: ', lr_scheduler)
    print('Starting LR: ', lr_)

    ################ Training ################
    for epoch_num in range(n_epochs):
        model.train()
        total_loss, total_err = 0.,0.
        total_regularizer = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            loss = 0
            X,y = data
            X,y = X.to(device), y.to(device)

            if FLAGS.capacity_regularization:
                model.set_verbose(verbose=True)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)

            if FLAGS.capacity_regularization:
                if FLAGS.regularization_mode == 'gauss_entropy':
                    regularizer = 1
                    bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                    num_channels = 0
                    for _, idx in enumerate(bn_idx):
                        if i > 0:
                            regularizer *= torch.prod(2*math.pi*(model.features[idx].weight**2))
                        num_channels += model.features[idx].weight.size(0) 
                    loss += FLAGS.beta*(((1/2*math.log2(regularizer)) + num_channels/2)/num_channels)
                elif FLAGS.regularization_mode == 'capacity': 
                    regularizer = 1 
                    layer_key = ['BN_0', 'BN_1']
                    bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                    for mm, idx in enumerate(bn_idx):
                        if i == 0 and epoch_num == 0:
                            regularizer =  1
                        else:
                            test_var = model.get_test_variance()[layer_key[mm]].cpu().detach().numpy().tolist()
                            model.set_verbose(verbose=False)
                            for ll in range(len(test_var)):
                                regularizer *=  1 + ((test_var[ll] * model.features[idx].weight[ll].cpu().detach().numpy()**2) / \
                                                (model.features[idx].running_var[ll].cpu().detach().numpy()))
                                
                               
                    loss += FLAGS.beta*(1/2*math.log2(regularizer))

            opt.zero_grad()
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=grad_clip_val)

            opt.step()

            if lr_scheduler == optim.lr_scheduler.OneCycleLR:
                scheduler.step()

            total += y.size(0)
            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item()
            total_regularizer += FLAGS.beta*(1/2*math.log2(regularizer))

        print(*("{:.6f}".format(i) for i in (int(epoch_num), total_err/len(train_loader.dataset), total_loss/len(train_loader.dataset))), sep="\t")
        
        if FLAGS.save_to_wandb:
            run.log({"loss_train": total_loss/len(train_loader)})
            run.log({"regularizer_train": total_regularizer/len(train_loader)})
            run.log({"acc_train": (total - total_err)/total})

        ################ Validation ################
        valid_loss = 0.0
        valid_regularizer = 0.0
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

                if FLAGS.capacity_regularization:
                    if FLAGS.regularization_mode == 'gauss_entropy':
                        regularizer = 1
                        bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                        num_channels = 0
                        for _, idx in enumerate(bn_idx):
                            if i > 0:
                                regularizer *= torch.prod(2*math.pi*(model.features[idx].weight**2))
                            num_channels += model.features[idx].weight.size(0) 
                        loss += FLAGS.beta*(((1/2*math.log2(regularizer)) + num_channels/2)/num_channels)
                    elif FLAGS.regularization_mode == 'capacity': 
                        regularizer = 1 
                        layer_key = ['BN_0', 'BN_1']
                        bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                        for mm, idx in enumerate(bn_idx):
                            if i == 0 and epoch_num == 0:
                                regularizer =  1
                            else:
                                test_var = model.get_test_variance()[layer_key[mm]].cpu().detach().numpy().tolist()
                                model.set_verbose(verbose=False)
                                for ll in range(len(test_var)):
                                    regularizer *=  1 + ((test_var[ll] * model.features[idx].weight[ll].cpu().detach().numpy()**2) / \
                                                    (model.features[idx].running_var[ll].cpu().detach().numpy()))
        
                        loss += FLAGS.beta*(1/2*math.log2(regularizer))

                valid_loss += loss.item()
                valid_regularizer += regularizer

                total += y.size(0)
                correct += (predicted == y).sum().item()

            print('Validation Accuracy: %d %%' % (100 * correct / total))

            val_acc = correct / total
            
            if FLAGS.save_to_wandb:
                run.log({"loss_validation": valid_loss/len(val_loader)})
                run.log({"regularizer_validation": valid_regularizer/len(val_loader)})
                run.log({"acc_validation": correct / total})

        if lr_scheduler == optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(valid_loss/len(val_loader.dataset))
        else:
            scheduler.step()

    if FLAGS.save_to_wandb:
        return model, run 
    else:
        return model









    '''if lr_scheduler != optim.lr_scheduler.MultiStepLR and lr_scheduler != optim.lr_scheduler.StepLR: interval = 'n.a.'


    metric_dict = {'tes_acc': str(val_acc), 'test_loss': str(valid_loss)}
    # writer.add_hparams(hparam_dict, metric_dict, run_name=run_name)

    writer.add_scalar('Validation accuracy',
                            val_acc,
                            epoch_num* len(val_loader) + i)
    writer.add_scalar('Validation loss',
                    valid_loss/len(val_loader.dataset),
                    epoch_num* len(val_loader) + i)

    writer.add_scalar('Training accuracy',
                        1 - (total_err/len(train_loader.dataset)),
                        epoch_num* len(train_loader) + i)

    writer.add_scalar('Training loss',
                        total_loss/len(train_loader.dataset),
                        epoch_num* len(train_loader) + i)

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
