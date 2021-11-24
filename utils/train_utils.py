from absl.flags import FLAGS
from torch._C import _propagate_and_assign_input_shapes
import torch.nn as nn
import torch.optim as optim
import torch 


def train (train_loader, val_loader, model, device, model_name, batch_norm, writer, run_name):
    
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
            scheduler = lr_scheduler(opt, max_lr, epochs=n_epochs, 
                                                            steps_per_epoch=len(train_loader))
            grad_clip = False
            grad_clip_val = 0.1

        else:
            max_lr = 2e-04
            opt = optim.SGD(model.parameters(), lr=max_lr,  momentum=0.9, weight_decay=1e-4)
            n_epochs = 125
            scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr, epochs=n_epochs, 
                                                            steps_per_epoch=len(train_loader))
            grad_clip = True
            grad_clip_val = 0.01
            
    # option for training models at the best of their performance (2)
    if FLAGS.mode == 'standard': 
        if batch_norm :
            lr_ = 0.05
            opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=5e-4)
            n_epochs = 200
            # scheduler = optim.lr_scheduler.MultiStepLR(opt, [150, 250], gamma=0.1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
            # scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1) 
        else: 
            lr_ = 0.01
            opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=1e-4)
            n_epochs = 150
            scheduler = optim.lr_scheduler.MultiStepLR(opt, [60, 90, 120], gamma=0.1)
    
    # option for training BN and non BN trained models under the same hyperparameters setup
    else:
        if model_name.find('ResNet')!= -1 :
            print("Training ResNet50 ...")

            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR
            # lr_scheduler = optim.lr_scheduler.StepLR

            interval = []
            if batch_norm:
                opt_func = optim.SGD
                lr_ = 0.08
                n_epochs = 200
                momentum = 0.9
                weight_decay = 5e-4
                grad_clip = False
            else:
                opt_func = optim.SGD
                lr_ = 0.01
                n_epochs = 300
                momentum = 0.9
                weight_decay = 5e-4
                grad_clip = False

            opt = opt_func(model.parameters(), lr=lr_,  momentum=momentum, weight_decay=weight_decay)
            scheduler = lr_scheduler(opt, T_max=200)
            # scheduler = lr_scheduler(opt, step_size=50, gamma=0.1)
        
        if model_name.find('VGG')!= -1:
            print("Training VGG ...")
            opt_func = optim.SGD
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau
            # lr_scheduler = optim.lr_scheduler.MultiStepLR
            interval = []
            n_epochs = 100
            momentum=0.9
            weight_decay=5e-4
            grad_clip = False

            if batch_norm:
                lr_ = 0.01
            else:
                lr_ = 0.01

            opt = opt_func(model.parameters(), lr=lr_,  momentum=momentum, weight_decay=weight_decay)
            scheduler = lr_scheduler(opt, 'min', patience=6)
            # scheduler = lr_scheduler(opt, [40, 60, 80], gamma=0.1)

    if lr_scheduler != optim.lr_scheduler.MultiStepLR and lr_scheduler != optim.lr_scheduler.StepLR: interval = 'n.a.'

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

    ################ Training ################
    for epoch_num in range(n_epochs):
        model.train()
        total_loss, total_err = 0.,0.
        for i, data in enumerate(train_loader, 0):
            X,y = data
            X,y = X.to(device), y.to(device)

            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=grad_clip_val)

            opt.step()
            opt.zero_grad()

            if FLAGS.mode == 'optimum':
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

            # custom early stopping
            '''if val_acc > 0.915 and not batch_norm:
                break
            if val_acc > 0.915 and batch_norm:
                break'''

        if FLAGS.mode != 'optimum':
            if model_name.find('VGG')!= -1: 
                scheduler.step(valid_loss/len(val_loader.dataset))
            else:
                scheduler.step()
        
    metric_dict = {'tes_acc': str(val_acc), 'test_loss': str(valid_loss)}
    # writer.add_hparams(hparam_dict, metric_dict, run_name=run_name)

    return model