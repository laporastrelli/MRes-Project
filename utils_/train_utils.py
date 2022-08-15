from sqlite3 import enable_shared_cache
import torch.nn as nn
import torch.optim as optim
import torch 
import csv
import math
import wandb
import matplotlib.pyplot as plt
import numpy as np


from absl import flags
from absl.flags import FLAGS
from torch._C import _propagate_and_assign_input_shapes
from utils_ import utils_flags
from utils_.miscellaneous import get_bn_layer_idx, entropy
from torch import linalg as LA



def train(train_loader, 
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
            opt_func = optim.SGD
            momentum=0.9
            weight_decay=5e-4
            # option for full-BN training
            if batch_norm and sum(FLAGS.where_bn)>1:
                lr_scheduler = optim.lr_scheduler.MultiStepLR
                lr_ = 0.1
                if  FLAGS.normalization == 'bn' and FLAGS.train_small_lr:
                    lr_ = 0.001
                opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=5e-4)
                n_epochs = 150
                grad_clip = False
                scheduler = optim.lr_scheduler.MultiStepLR(opt, [50, 100], gamma=0.1)

            elif FLAGS.use_SkipInit:
                print('Using SkipInit with large Learning Rate ...')
                lr_scheduler = optim.lr_scheduler.MultiStepLR
                lr_ = 0.1
                opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=5e-4)
                n_epochs = 150
                grad_clip = False
                scheduler = optim.lr_scheduler.MultiStepLR(opt, [50, 100], gamma=0.1) 
            
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
            
        
        if FLAGS.dataset == 'CIFAR100':
            opt_func = optim.SGD
            momentum=0.9
            weight_decay=5e-4
            # option for full-BN training
            if batch_norm and sum(FLAGS.where_bn)>1:
                lr_scheduler = optim.lr_scheduler.MultiStepLR
                lr_ = 0.1
                if  FLAGS.normalization == 'bn' and FLAGS.train_small_lr:
                    lr_ = 0.001
                opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=5e-4)
                n_epochs = 200
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
                opt_func = optim.SGD
                lr_ = 0.1
                momentum=0.9
                weight_decay=5e-4
                opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=5e-4)
                n_epochs = 75
                grad_clip = False
                scheduler = optim.lr_scheduler.MultiStepLR(opt, [25, 45], gamma=0.1) 
            # option for partial (or none)-BN training
            else: 
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau
                opt_func = optim.SGD
                val = 1
                if val in FLAGS.where_bn and FLAGS.where_bn.index(1) == 2:
                    lr_ = 0.03
                elif sum(FLAGS.where_bn)==0:
                    lr_ = 0.01
                else:
                    lr_ = 0.01
                momentum=0.9
                weight_decay=5e-4
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
            if FLAGS.dataset=='CIFAR100':
                n_epochs=150
            momentum = 0.9
            weight_decay=5e-4
            grad_clip = False
            lr_ = 0.01
            if model_name.find('VGG16')!= -1 and (batch_norm and sum(FLAGS.where_bn)>1):
                lr_ = 0.05
            if FLAGS.normalization == 'ln' or FLAGS.nonlinear_lambda or FLAGS.dropout_lambda:
                lr_ = 0.001
            elif  FLAGS.normalization == 'bn' and FLAGS.train_small_lr:
                lr_ = 0.001

            if FLAGS.dataset == 'SVHN' and FLAGS.where_bn[4]==1 and sum(FLAGS.where_bn)==1:
                lr_= 0.0075

            opt = opt_func(model.parameters(), lr=lr_,  momentum=momentum, weight_decay=weight_decay)
            scheduler = lr_scheduler(opt, 'min', patience=6)

    print('LR scheduler: ', lr_scheduler)
    print('Starting LR: ', lr_)

    if FLAGS.rank_init:
        negative_rank_ = torch.tensor(0.0)
        negative_rank_.requires_grad = True
        optimizer_init = optim.SGD(model.parameters(), lr=0.1)
        for iteration in range(int(FLAGS.pre_training_steps)):
            model.train()
            print(iteration)
            total_rank = 0
            for i, data in enumerate(train_loader, 0):
                X,y = data
                X,y = X.to(device), y.to(device)

                # zero the gradients
                optimizer_init.zero_grad()

                # forward pass of the batch
                _ = model(X)

                # get last layer activations
                activations = model.last_layer

                # compute rank on last layer activations 
                activations_t = torch.transpose(activations, 0, 1)
                activations_g = torch.matmul(activations_t, activations)/X.size(0)

                activations_n = torch.linalg.matrix_norm(activations_g, ord='fro')
                numerator = (torch.trace(activations_g))**2
                denominator = (activations_n)**2
                rank_ = numerator/denominator
                negative_rank_ = -rank_

                print(negative_rank_.requires_grad)

                total_rank += rank_

                # compute gradients
                negative_rank_.backward()

                # optimize weights
                optimizer_init.step()

            if FLAGS.save_to_wandb:
                run.log({"rank_pre_train": total_rank/len(train_loader)})

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
            "rank-preserving init": FLAGS.rank_init,
            "pre-training steps": FLAGS.pre_training_steps,
            "beta": FLAGS.beta,
            "learning_rate": lr_,
            "learning_rate_scheduler": lr_scheduler,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "gradient_clipping": grad_clip, 
            "epochs": n_epochs}
        run.config.update(config)

    ################ Training ################
    layers = [0,1,2,5,8,10,12,15]
    layers_keys = ['BN_' + str(i) for i in layers]
    capacities_clipped = dict.fromkeys(layers_keys, [])
    capacities = dict.fromkeys(layers_keys, [])
    steps = [0, 260]

    for epoch_num in range(n_epochs):
        model.train()
        total_loss, total_err = 0.,0.
        total_regularizer = 0
        total = 0
        regularizer = 0.0

        for i, data in enumerate(train_loader, 0):

            loss = 0
            X,y = data
            X,y = X.to(device), y.to(device)

            if FLAGS.capacity_regularization:
                if model_name.find('VGG')!=-1:
                    model.set_verbose(verbose=True)
                if FLAGS.regularization_mode == 'BN_once':
                    model.set_iteration_num(iterations=i, epoch=epoch_num)
                    regularizer = 0

            if FLAGS.track_capacity and FLAGS.bounded_lambda and i in steps:
                fig, axs = plt.subplots(nrows=2, ncols=4, sharey=False, figsize=(13,7))
                axs = axs.ravel()
                
                # get lambdas for each layer that we want to monitor:
                layers = [0,1,2,5,8,10,12,15]
                titles = ['Layer - ' + str(k) for k in layers]
                for b, key in enumerate(layers_keys):
                    if epoch_num == 0 and i == steps[0]:
                        if model_name.find('ResNet')!= -1 :
                            capacities[key] = model.get_bn_parameters()[key].cpu()
                        else:
                            capacities[key] = model.get_bn_parameters()[key]
                    else:
                        to_add = []
                        if epoch_num == 0 and i == steps[1]:
                            exists = [capacities[key]]
                        else:
                            exists = capacities[key]
                        for c in range(len(exists) + 1):
                            if c < len(exists):
                                to_add.append(exists[c])
                            else:
                                if model_name.find('ResNet')!= -1 :
                                    to_add.append(model.get_bn_parameters()[key].cpu())
                                else:
                                    to_add.append(model.get_bn_parameters()[key])

                        capacities[key] = to_add
    
                        
                    if epoch_num == 0 and i == steps[0]:
                        to_plot = [capacities[key]]
                    else:
                        to_plot = capacities[key]
                    pos = np.arange(len(to_plot))
                    axs[b].violinplot(to_plot, pos, points=60, widths=0.6, showmeans=True,
                        showextrema=True, showmedians=True, bw_method=0.5)
                    axs[b].set_title(titles[b], fontsize=11)
                    #axs[b].set_xticks([d for d in range(len(capacities[key]))])
                    #axs[b].set_xticklabels([120*f for f in range(len(capacities[key]))], fontsize=11)
                if model_name.find('VGG')!=-1:
                    plt.suptitle(r'$\lambda$' + ' Training Distibution VGG19', fontsize=16)
                else:
                    plt.suptitle(r'$\lambda$' + ' Training Distibution ResNet50', fontsize=16)
                fig.tight_layout(pad=2, rect=[0.03, 0.05, 1, 0.98])
                fig.text(0.48, 0.03, 'Training Step', va='center', fontsize=14)
                fig.text(0.02, 0.5, r'$\lambda$' + ' Distribution', va='center', rotation='vertical', fontsize=14)
                path_out = FLAGS.csv_path.replace('.csv', '_' + run_name + '_' + 'lambda_dist' + '.jpg')
                fig.savefig(path_out)
                plt.close()

            yp = model(X)

            if FLAGS.track_capacity and FLAGS.bounded_lambda and i in steps:
                fig_c, axs_c = plt.subplots(nrows=2, ncols=4, sharey=False, figsize=(13,7))
                axs_c = axs_c.ravel()

                # get lambdas for each layer that we want to monitor:
                layers = [0,1,2,5,8,10,12,15]
                if model_name.find('ResNet')!= -1 :
                    layers = [0,1,2,10,20,30,39,49]
                titles = ['Layer - ' + str(k) for k in layers]
                for v, key in enumerate(layers_keys):
                    if epoch_num == 0 and i == steps[0]:
                        if model_name.find('ResNet')!= -1 :
                            capacities_clipped[key] = model.get_bn_parameters()[key].cpu()
                        else:
                            capacities_clipped[key] = model.get_bn_parameters()[key]
                    else:
                        to_add = []
                        if epoch_num == 0 and i == steps[1]:
                            exists = [capacities_clipped[key]]
                        else:
                            exists = capacities_clipped[key]
                        for c in range(len(exists) + 1):
                            if c < len(exists):
                                to_add.append(exists[c])
                            else:
                                if model_name.find('ResNet')!= -1 :
                                    to_add.append(model.get_bn_parameters()[key].cpu())
                                else:
                                    to_add.append(model.get_bn_parameters()[key])

                        capacities_clipped[key] = to_add

                    if epoch_num == 0 and i == steps[0]:
                        to_plot = [capacities_clipped[key]]
                    else:
                        to_plot = capacities_clipped[key]
                    pos = np.arange(len(to_plot))
                    axs_c[v].violinplot(to_plot, pos, points=60, widths=0.6, showmeans=True,
                        showextrema=True, showmedians=True, bw_method=0.5)
                    axs_c[v].set_title(titles[v], fontsize=11)
                if model_name.find('VGG')!=-1:
                    plt.suptitle(r'$\lambda$' + ' Training Distibution VGG19 (clipped ' + r'$\lambda$' + ')', fontsize=16)
                else:
                    plt.suptitle(r'$\lambda$' + ' Training Distibution ResNet50 (clipped ' + r'$\lambda$' + ')', fontsize=16)
                fig_c.tight_layout(pad=2, rect=[0.03, 0.05, 1, 0.98])
                fig_c.text(0.48, 0.03, 'Training Step', va='center', fontsize=14)
                fig_c.text(0.02, 0.5, r'$\lambda$' + ' Distribution', va='center', rotation='vertical', fontsize=14)
                path_out = FLAGS.csv_path.replace('.csv', '_' + run_name + '_' + 'clipped_lambda_dist' + '.jpg')
                fig_c.savefig(path_out)
                plt.close()

            if FLAGS.track_rank:
                if i == 0:
                    print(torch.transpose(model.last_layer, 0, 1).size())
                    rank = torch.matrix_rank(torch.transpose(model.last_layer, 0, 1))
                    
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
                elif FLAGS.regularization_mode == 'capacity_norm': 
                    regularizer = 1 
                    bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                    if run_name.find('bn')!= -1:
                        layer_key = ['BN_' + str(i) for i in range(16)]
                    else:
                        layer_key = ['BN_' +  str(i) for i in range(len(bn_idx))]
                    for mm, idx in enumerate(bn_idx):
                        if i == 0 and epoch_num == 0:
                            regularizer =  1
                        else:
                            test_var = model.get_test_variance()[layer_key[mm]].cpu().detach().numpy().tolist()
                            model.set_verbose(verbose=False)
                            capacity_arg = 1 + ((test_var * model.features[idx].weight.cpu().detach().numpy()**2)/ \
                                                (model.features[idx].running_var.cpu().detach()))
                            regularizer = torch.prod((capacity_arg -  capacity_arg.mean() + 1.5))                    
                    temp_ = FLAGS.beta*(1/2*math.log(regularizer))
                    # temp_ = FLAGS.beta*(1/2*(regularizer-1))
                    loss += temp_
                elif FLAGS.regularization_mode == 'lambda_entropy':
                    regularizer = 0
                    bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                    if i == 0 and epoch_num == 0:
                        regularizer = 0
                    elif epoch_num > 0:
                        if run_name.find('bn')!= -1:
                            layer_key = ['BN_' + str(i) for i in range(16)]
                        else:
                            layer_key = ['BN_' +  str(i) for i in range(len(bn_idx))]
                        for mm, idx in enumerate(bn_idx):
                            regularizer += entropy(model.features[idx].weight.cpu().detach(), which='K-L')
                    temp_ = FLAGS.beta*(regularizer/len(bn_idx))
                    loss += temp_
                elif FLAGS.regularization_mode == 'lambda_entropy_gaussian':
                    regularizer = 0
                    bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                    if i == 0 and epoch_num == 0:
                        regularizer = 0
                    else:
                        if run_name.find('bn')!= -1:
                            layer_key = ['BN_' + str(i) for i in range(16)]
                        else:
                            layer_key = ['BN_' +  str(i) for i in range(len(bn_idx))]
                        for mm, idx in enumerate(bn_idx):
                            regularizer += entropy(model.features[idx].weight.cpu().detach(), which='gaussian')
                    temp_ = FLAGS.beta*(regularizer/len(bn_idx))
                    loss += temp_
                elif FLAGS.regularization_mode == 'infinity':
                    regularizer = 0
                    bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])

                    if run_name.find('bn')!= -1:
                        layer_key = ['BN_' + str(i) for i in range(16)]
                    else:
                        layer_key = ['BN_' +  str(i) for i in range(len(bn_idx))]

                    if i == 0 and epoch_num == 0:
                        regularizer = 0
                    else:
                        for _, idx in enumerate(bn_idx):
                            weights = model.features[idx].weight.cpu().detach()
                            regularizer += torch.norm(weights - weights.mean(), float('inf'))
                            # regularizer += torch.norm(weights, float('inf'))
                    temp_ = FLAGS.beta*(regularizer)
                    loss += temp_
                elif FLAGS.regularization_mode == 'euclidean_total':
                    # regularizer = torch.tensor(0., device=device, requires_grad=True)
                    regularizer = None
                    bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                    if run_name.find('bn')!= -1:
                        layer_key = ['BN_' + str(i) for i in range(16)]
                    else:
                        layer_key = ['BN_' +  str(i) for i in range(len(bn_idx))]
                    for _, idx in enumerate(bn_idx):
                        if model_name.find('VGG')!= -1:
                            weights = model.features[idx].weight
                        if i == 0 and epoch_num == 0:
                            regularizer = LA.norm(weights, 2)*0
                        else:
                            if regularizer == None:
                                regularizer = LA.norm(weights, 2)
                            else:
                                regularizer = regularizer + LA.norm(weights, 2)
                    regularizer = FLAGS.beta*(regularizer)
                    loss += regularizer
                elif FLAGS.regularization_mode == 'euclidean_first_layer':
                    # regularizer = torch.tensor(0., device=device, requires_grad=True)
                    regularizer = None
                    bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                    if run_name.find('bn')!= -1:
                        layer_key = ['BN_' + str(i) for i in range(16)]
                        layer_key = ['BN_0']
                    else:
                        layer_key = ['BN_' +  str(i) for i in range(len(bn_idx))]
                    for _, idx in enumerate(bn_idx):
                        weights = model.features[idx].weight
                        if i == 0 and epoch_num == 0:
                            regularizer = LA.norm(weights, 2)*0
                        else:
                            if regularizer == None:
                                regularizer = LA.norm(weights, 2)
                            else:
                                regularizer = regularizer + LA.norm(weights, 2)
                    regularizer = FLAGS.beta*(regularizer)
                    loss += regularizer

                elif FLAGS.regularization_mode == 'uniform_lambda':
                    regularizer = 0
                elif FLAGS.regularization_mode == 'wandb_only':
                    regularizer = 0

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
            total_regularizer += regularizer

        print(*("{:.6f}".format(i) for i in (int(epoch_num), total_err/len(train_loader.dataset), total_loss/len(train_loader.dataset))), sep="\t")
        
        if FLAGS.save_to_wandb:
            run.log({"loss_train": total_loss/len(train_loader)})
            run.log({"regularizer_train": total_regularizer/len(train_loader)})
            run.log({"acc_train": (total - total_err)/total})
            if FLAGS.track_rank:
                run.log({"rank train (last layer)": rank})

        ################ Validation ################
        valid_loss = 0.0
        valid_regularizer = 0.0
        regularizer_val = 0.0
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
                    elif FLAGS.regularization_mode == 'capacity_norm': 
                        regularizer = 1 
                        bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                        if run_name.find('bn')!= -1:
                            layer_key = ['BN_' + str(i) for i in range(16)]
                        else:
                            layer_key = ['BN_' +  str(i) for i in range(len(bn_idx))]
                        for mm, idx in enumerate(bn_idx):
                            if i == 0 and epoch_num == 0:
                                regularizer =  1
                            else:
                                test_var = model.get_test_variance()[layer_key[mm]].cpu().detach().numpy().tolist()
                                model.set_verbose(verbose=False)
                                capacity_arg = 1 + ((test_var * model.features[idx].weight.cpu().detach().numpy()**2)/ \
                                                    (model.features[idx].running_var.cpu().detach()))
                                regularizer = torch.prod((capacity_arg -  capacity_arg.mean() + 1.5))                    
                        temp = FLAGS.beta*(1/2*math.log(regularizer))
                        loss += temp      
                    elif FLAGS.regularization_mode == 'lambda_entropy':
                        regularizer = 0
                        if run_name.find('bn')!= -1:
                            layer_key = ['BN_' + str(i) for i in range(16)]
                        else:
                            layer_key = ['BN_0', 'BN_1']
                        bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                        for mm, idx in enumerate(bn_idx):
                            regularizer += entropy(model.features[idx].weight.cpu().detach(), which='K-L')
                        temp = FLAGS.beta*(regularizer/len(bn_idx))
                        loss += temp
                    elif FLAGS.regularization_mode == 'lambda_entropy_gaussian':
                        regularizer = 0
                        bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                        if i == 0 and epoch_num == 0:
                            regularizer = 0
                        else:
                            if run_name.find('bn')!= -1:
                                layer_key = ['BN_' + str(i) for i in range(16)]
                            else:
                                layer_key = ['BN_' +  str(i) for i in range(len(bn_idx))]
                            for mm, idx in enumerate(bn_idx):
                                regularizer += entropy(model.features[idx].weight.cpu().detach(), which='gaussian')
                        temp = FLAGS.beta*(regularizer/len(bn_idx))
                        loss += temp
                    elif FLAGS.regularization_mode == 'infinity':
                        regularizer = 0
                        bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])

                        if run_name.find('bn')!= -1:
                            layer_key = ['BN_' + str(i) for i in range(16)]
                            layer_key = ['BN_0', 'BN_1']
                            bn_idx = bn_idx[0:len(layer_key)]
                        else:
                            layer_key = ['BN_' +  str(i) for i in range(len(bn_idx))]

                        if i == 0 and epoch_num == 0:
                            regularizer = 0
                        else:
                            for _, idx in enumerate(bn_idx):
                                weights = model.features[idx].weight.cpu().detach()
                                regularizer += torch.norm(weights - weights.mean(), float('inf'))
                                # regularizer += torch.norm(weights, float('inf'))
                        temp = FLAGS.beta*(regularizer)
                        loss += temp
                    elif FLAGS.regularization_mode == 'euclidean_total':
                        regularizer_val = None
                        bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                        if run_name.find('bn')!= -1:
                            layer_key = ['BN_' + str(i) for i in range(16)]
                        else:
                            layer_key = ['BN_' +  str(i) for i in range(len(bn_idx))]
                        for _, idx in enumerate(bn_idx):
                            weights = model.features[idx].weight
                            if i == 0 and epoch_num == 0:
                                regularizer_val = LA.norm(weights, 2)*0
                            else:
                                if regularizer_val == None:
                                    regularizer_val = LA.norm(weights, 2)
                                else:
                                    regularizer_val = regularizer_val + LA.norm(weights, 2)
                        regularizer_val = FLAGS.beta*(regularizer_val)
                        loss += regularizer_val
                    elif FLAGS.regularization_mode == 'euclidean_first_layer':
                        # regularizer = torch.tensor(0., device=device, requires_grad=True)
                        regularizer_val = None
                        bn_idx = get_bn_layer_idx(model, run_name.split('_')[0])
                        if run_name.find('bn')!= -1:
                            layer_key = ['BN_' + str(i) for i in range(16)]
                            layer_key = ['BN_0']
                        else:
                            layer_key = ['BN_' +  str(i) for i in range(len(bn_idx))]
                        for _, idx in enumerate(bn_idx):
                            weights = model.features[idx].weight
                            if i == 0 and epoch_num == 0:
                                regularizer_val = LA.norm(weights, 2)*0
                            else:
                                if regularizer_val == None:
                                    regularizer_val = LA.norm(weights, 2)
                                else:
                                    regularizer_val = regularizer_val + LA.norm(weights, 2)
                        regularizer_val = FLAGS.beta*(regularizer_val)
                        loss += regularizer_val
                    elif FLAGS.regularization_mode == 'uniform_lambda':
                        regularizer_val = 0
                    elif FLAGS.regularization_mode == 'wandb_only':
                        regularizer_val = 0
                    elif FLAGS.regularization_mode == 'BN_once':
                        model.set_iteration_num(iterations=i, epoch=1)
                        regularizer_val = 0

                valid_loss += loss.item()
                valid_regularizer += regularizer_val

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
