import torch.nn as nn
import torch.optim as optim


def train (loader, model, device, n_epochs=30):
    opt = optim.SGD(model.parameters(), lr=1e-1)

    for epoch_num in range(n_epochs):
        total_loss, total_err = 0.,0.
        for i, data in enumerate(loader, 0):
            X,y = data
            X,y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item()
            
        print(*("{:.6f}".format(i) for i in (int(epoch_num), total_err/len(loader.dataset), total_loss/len(loader.dataset))), sep="\t")
    
    return model