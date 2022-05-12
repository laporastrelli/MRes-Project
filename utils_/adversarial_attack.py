from tkinter.tix import Tree
from turtle import clear
import torch 
import torch.nn as nn
import torch.linalg as la

import numpy as np
import matplotlib.pyplot as plt

from .miscellaneous import get_bn_int_from_name

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

def pgd_linf_loss_analysis(model, X, y,  epsilon, max_, min_, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    deltas = []
    delta = torch.zeros_like(X, requires_grad=True)
    temp_delta = torch.zeros_like(X, requires_grad=True)
    orig_delta = torch.zeros_like(X, requires_grad=True)
    temp_grad = torch.zeros_like(X, requires_grad=True)
    start, end = 90, 1
    second_range = np.arange(1.0, 0.1, -0.012)
    step_loss = np.zeros((num_iter, len(range(start, end, -1)) + len(second_range)))
    full_range = np.concatenate([range(start, end, -1), second_range])

    model.classifier.eval()

    for t in range(num_iter):
        print('PGD step: ', t)

        # compute gradients
        orig_delta = delta.detach()
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()

        # compute perturbations
        temp_grad = delta.grad.detach().sign()
        delta.data = (delta + alpha*temp_grad).clamp(-epsilon,epsilon)
        delta.data = torch.clamp(X.data + delta.data, min=min_, max=max_) - X.data

        print('Loss at step {t}: {l}'.format(t=t, l=nn.CrossEntropyLoss()(model(X + delta), y)))
        
        for k, step in enumerate(full_range):
            temp_delta.data = (orig_delta + (1/step)*alpha*temp_grad).clamp(-epsilon, epsilon)
            temp_delta.data = torch.clamp(X.data + temp_delta.data, min=min_, max=max_) - X.data
            step_loss[t, k] = nn.CrossEntropyLoss()(model(X + temp_delta), y).detach().cpu().numpy()
        
        print('Max Loss along gradient step {step}: {loss_}' \
              .format(step=step_loss[t, :].tolist().index(np.max(step_loss[t, :])), \
              loss_=step_loss[t, step_loss[t, :].tolist().index(np.max(step_loss[t, :]))]))

        delta.grad.zero_()
        
    deltas.append(delta.detach())
    return deltas, step_loss

def pgd_linf_grad_analysis(model, X, y,  epsilon, max_, min_, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    deltas = []
    delta = torch.zeros_like(X, requires_grad=True)
    temp_delta = torch.zeros_like(X, requires_grad=True)
    orig_delta = torch.zeros_like(X, requires_grad=True)
    prev_delta = torch.zeros_like(X, requires_grad=True)
    temp_grad = torch.zeros_like(X, requires_grad=True)
    start, end = 90, 1
    second_range = np.arange(1.0, 0.1, -0.012)
    step_norm = np.zeros((num_iter, len(range(start, end, -1)) + len(second_range)))
    full_range = np.concatenate([range(start, end, -1), second_range])

    model.classifier.eval()

    for t in range(num_iter):
        print('PGD step: ', t)

        orig_delta = delta.detach()
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()

        temp_grad = delta.grad.detach().sign()
        delta.data = (delta + alpha*temp_grad).clamp(-epsilon, epsilon)
        delta.data = torch.clamp(X.data + delta.data, min=min_, max=max_) - X.data

        for k, step in enumerate(full_range):
            temp_delta.data = (orig_delta + (1/step)*alpha*temp_grad).clamp(-epsilon, epsilon)
            temp_delta.data = torch.clamp(X.data + temp_delta.data, min=min_, max=max_) - X.data

            step_loss = nn.CrossEntropyLoss()(model(X + temp_delta), y)
            step_loss.backward() 
            local_grad = temp_delta.grad.detach().sign()

            step_norm[t, k] = la.norm(local_grad - temp_grad)

            temp_delta.grad.zero_()

        delta.grad.zero_()  

    deltas.append(delta.detach())
    return deltas, step_norm

def pgd_linf_capacity_(model, X, y,  epsilon, max_, min_, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    deltas = []
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

    return deltas

def pgd_linf_capacity(model, run_name, X, y,  epsilon, max_, min_, alpha, num_iter):
    """ Construct PGD adversarial examples on the examples X"""
    deltas = []
    capacities = []
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter): 
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = torch.clamp(X.data + delta.data, min=min_, max=max_) - X.data
        delta.grad.zero_()

        # getting capacity
        temp = model.get_capacity()['BN_0']
        capacities.append(temp.cpu().detach().numpy())

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
