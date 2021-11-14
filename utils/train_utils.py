from torch._C import _propagate_and_assign_input_shapes
import torch.nn as nn
import torch.optim as optim
import torch 


def train (train_loader, val_loader, model, device, model_name, n_epochs, batch_norm, writer, optimum=False):

    #### Learning rate schedulers 

    # option for training models at the best of their performance
    if optimum: 
        if batch_norm:
            lr_ = 0.1
            opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=5e-4)
            n_epochs = 75
            scheduler = optim.lr_scheduler.MultiStepLR(opt, [35], gamma=0.1)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        else:
            lr_ = 0.01
            opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=1e-4)
            n_epochs = 150
            scheduler = optim.lr_scheduler.MultiStepLR(opt, [120, 180], gamma=0.1)

    # option for training BN and non BN trained models under the same hyperparameters setup
    else:
        if model_name.find('ResNet')!= -1:
            lr_ = 0.1
            opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=1e-4)
            n_epochs = 100
            scheduler = optim.lr_scheduler.MultiStepLR(opt, [40, 60, 80], gamma=0.1)
        
        if model_name.find('VGG')!= -1:
            lr_ = 0.01
            opt = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, weight_decay=5e-4)
            n_epochs = 100
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=6)

    #### Training 

    for epoch_num in range(n_epochs):
        model.train()
        total_loss, total_err = 0.,0.
        for i, data in enumerate(train_loader, 0):
            X,y = data
            X,y = X.to(device), y.to(device)

            opt.zero_grad()
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)
            loss.backward()

            if not batch_norm:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=0.01)

            opt.step()

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item()

        print(*("{:.6f}".format(i) for i in (int(epoch_num), total_err/len(train_loader.dataset), total_loss/len(train_loader.dataset))), sep="\t")

        writer.add_scalar('training accuracy',
                        1 - (total_err/len(train_loader.dataset)),
                        epoch_num* len(train_loader) + i)
        writer.add_scalar('training loss',
                        total_loss/len(train_loader.dataset),
                        epoch_num* len(train_loader) + i)

        valid_loss = 0.0
        model.eval()
      
        correct = 0
        total = 0

        #### Evaluation

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

            writer.add_scalar('validation error',
                            val_acc,
                            epoch_num* len(val_loader) + i)
            writer.add_scalar('validation loss',
                            valid_loss/len(val_loader.dataset),
                            epoch_num* len(val_loader) + i)

            # custom early stopping
            if val_acc > 0.93 and batch_norm==False:
                break
            if val_acc > 0.93 and batch_norm==True:
                break
            
            # update scheduler
            if model_name.find('VGG')!= -1: 
                scheduler.step(valid_loss/len(val_loader.dataset))
            else:
                scheduler.step()

    return model