import torch
from torch import nn
import os
import numpy as np
import random,math

def compute_metrics(logits,true_labels):
    _, predicted = torch.max(logits, 1)  # Get the index of the maximum value in each row
    correct = (predicted == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total
    return accuracy

def train_one_epoch(model, train_loader, optimizer, criterion,scheduler, device):
    model.train()
    avg_loss = 0.0
    avg_accuracy = 0.0
    n = 0
    for data,labels in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        labels = labels.to(device)
        logits = model(data)
        loss = criterion(logits,labels)
        out = compute_metrics(logits,labels)
        avg_accuracy += out*data.size(0)
        avg_loss += loss.item()*data.size(0)
        n+= data.size(0)
        loss.backward()
        optimizer.step()
    scheduler.step()
    return avg_loss / n,avg_accuracy / n
    
    
@torch.no_grad()
def valid_one_epoch(model, valid_loader,criterion,device):
    model.eval()
    avg_loss = 0.0
    avg_accuracy = 0.0
    n = 0
    for data,labels in valid_loader:
        data = data.to(device)
        labels = labels.to(device)
        logits = model(data)
        loss = criterion(logits,labels)
        out = compute_metrics(logits,labels)
        avg_accuracy += out*data.size(0)
        avg_loss += loss.item()*data.size(0)
        n+= data.size(0)
    return avg_loss /n,avg_accuracy / n
