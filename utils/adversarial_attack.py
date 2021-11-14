import torch 
import torch.nn as nn

def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    deltas = []
    for eps in epsilon:
        delta = torch.zeros_like(X, requires_grad=True)
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        deltas.append(eps * delta.grad.detach().sign())

    return deltas 

def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    deltas = []
    for eps in epsilon:
        delta = torch.zeros_like(X, requires_grad=True)
        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-eps,eps)
            delta.grad.zero_()
        deltas.append(delta.detach())
    return deltas

def pgd_linf_rand(model, X, y, epsilon, alpha, num_iter, restarts):
    """ Construct PGD adversarial examples on the samples X, with random restarts"""
    deltas = []
    for eps in epsilon:
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        max_delta = torch.zeros_like(X)
        
        for i in range(restarts):
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 2 * epsilon - epsilon
            
            for t in range(num_iter):
                loss = nn.CrossEntropyLoss()(model(X + delta), y)
                loss.backward()
                delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
                delta.grad.zero_()
            
            all_loss = nn.CrossEntropyLoss(reduction='none')(model(X+delta),y)
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)
        
        deltas.append(max_delta)
    
    return deltas
