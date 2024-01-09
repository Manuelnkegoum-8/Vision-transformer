import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim
from model import*
import warmup_scheduler
import torchvision.transforms as transforms
from colorama import Fore, Style
from torchinfo import summary
import os
import torch.optim.lr_scheduler as lr_schedule
import argparse
from utils.loader import load_data
from utils.train import train_one_epoch,valid_one_epoch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

# Data args
parser.add_argument('--data_path', default='./dataset', type=str, help='dataset path')
parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'T-IMNET', 'SVHN'], type=str, help='Image Net dataset path')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--freq', default=10, type=int, metavar='N', help='log frequency (by iteration)')

# Model parameters
parser.add_argument('--height', default=32, type=int, metavar='N', help='image height')
parser.add_argument('--width', default=32, type=int, metavar='N', help='image width')
parser.add_argument('--channel', default=3, type=int, help='disable cuda')
parser.add_argument('--heads', default=12, type=int, help='number oftransformer heads')
parser.add_argument('--depth', default=9, type=int, help='number of transformer blocks')
parser.add_argument('--patch_size', default=4, type=int, help='patch size')
parser.add_argument('--dim', default=192, type=int, help='embedding dim of patch')
parser.add_argument('--mlp_dim', default=384, type=int, help='feed forward hidden_dim for a transformer block')

# Optimization hyperparams
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
parser.add_argument('--lr', default=0.003, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-2, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default=False, help='Version')

args = parser.parse_args()
lr = args.lr
weight_decay = args.weight_decay
height, width, n_channels = args.height, args.width, args.channel
patch_size, dim, n_head = args.patch_size, args.dim, args.heads
feed_forward, num_blocks = args.mlp_dim, args.depth
batch_size = args.batch_size
warmup = args.warmup


device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

policy = transforms.autoaugment.AutoAugmentPolicy.CIFAR10

def train():
    train_loader, val_loader, num_classes = load_data(policy,height,width,batch_size)
    model = vit(height,width,n_channels,patch_size,batch_size,dim,n_head,feed_forward,num_blocks,num_classes)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay = weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    num_epochs = args.epochs
    base_scheduler = lr_schedule.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=warmup, after_scheduler=base_scheduler)
    # Train the model
    best_loss = float('inf')
    torch.autograd.set_detect_anomaly(True)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        final_epoch = args.epochs
        num_epochs = final_epoch - (checkpoint['epoch'] + 1)

    print(Fore.LIGHTGREEN_EX+'='*100)
    print("[INFO] Begin training for {0} epochs".format(num_epochs))
    summary(model,input_size = (n_channels,height,width),batch_dim = 0)
    print('='*100+Style.RESET_ALL)


    for epoch in tqdm(range(num_epochs)):
        train_loss,train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion,scheduler,device)
        valid_loss,valid_accuracy= valid_one_epoch(model, val_loader,criterion,device)
        if epoch%args.freq==0:
            print(Fore.YELLOW+'='*100)
            print(f"epoch: {epoch}\t train_loss: {train_loss:.4f}\t valid_loss: {valid_loss:.4f}")
            print(f"train_accuracy: {train_accuracy:.4f}\t valid_accuracy: {valid_accuracy:.4f}")
            print('*'*80+Style.RESET_ALL)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, f"vit_model.pt")
            
    print(Fore.GREEN+'='*100)
    print("[INFO] End training")
    print('='*100+Style.RESET_ALL)

if __name__ == '__main__':
    train()