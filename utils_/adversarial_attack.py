from tkinter.tix import Tree
import torch 
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    deltas = []
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    deltas.append(epsilon * delta.grad.detach().sign())

    return deltas 

def pgd_linf(model, X, y,  epsilon, max_, min_, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    deltas = []
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = torch.clamp(X.data + delta.data, min=min_, max=max_) - X.data
        delta.grad.zero_()
    deltas.append(delta.detach())
    return deltas

def pgd_linf_analysis(model, X, y,  epsilon, max_, min_, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    deltas = []
    loss_variation = []
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = torch.clamp(X.data + delta.data, min=min_, max=max_) - X.data
        
        temp_grad = delta.grad.detach()
        temp_loss = []
        prev_X = torch.zeros_like(delta.data)
        for step in range(250, 10, -10):
            temp_X = torch.clamp(X + (1/step)*temp_grad, min=min_, max=max_) 
            #print(torch.equal(temp_X, prev_X))
            prev_X = temp_X
            temp_loss.append(nn.CrossEntropyLoss()(model(X + (temp_X-X)), y).detach().cpu().numpy())
        loss_variation += np.var(temp_loss)

        delta.grad.zero_()
        
    deltas.append(delta.detach())
    return deltas, loss_variation

def pgd_linf_capacity(model, X, y,  epsilon, max_, min_, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    deltas = []
    capacities = []
    delta = torch.zeros_like(X, requires_grad=True)
    last_ = 0
    for t in range(num_iter): 
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = torch.clamp(X.data + delta.data, min=min_, max=max_) - X.data
        delta.grad.zero_()

        # getting capacity
        temp = model.get_capacity()['BN_0']
        capacities.append(temp)

        if t>0:
            diff = temp - last_
            diff = torch.flatten(diff, start_dim=2, end_dim=3)
            diff, _ = torch.max(diff, dim=2, keepdim=True)
            diff = torch.squeeze(diff, dim=2)
            diff = torch.argmax(diff, dim=1)

            print(diff.unsqueeze(1).transpose(0,1))

        last_ = temp

        x_axis = [t]
        temp = temp.mean([0,2,3])
        for x_, y_ in zip(x_axis, [temp.cpu().detach().numpy()]):
            plt.scatter([x_] * len(y_), y_)
            plt.xticks([1, t+1])
            plt.savefig('./test_capacity__.png')

    deltas.append(delta.detach())

    return deltas, capacities

def pgd_linf_rand(model, X, y, epsilon, alpha, num_iter, restarts):
    """ Construct PGD adversarial examples on the samples X, with random restarts"""
    deltas = []
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
