import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torchvision.datasets as datasets


def load_data(policy,height,width,batch_size):
    mean = torch.tensor((0.4914, 0.4822, 0.4465))
    std = torch.tensor((0.2023, 0.1994, 0.2010))
    data_transforms = transforms.Compose([
        transforms.Resize((height,width)),  # Resize the images to a specific size
        transforms.AutoAugment(policy=policy),
        transforms.ToTensor(),           # Convert images to tensors
        transforms.Normalize(mean,std),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((height,width)),  # Resize the images to a specific size
        transforms.ToTensor(),           # Convert images to tensors
        transforms.Normalize(mean,std),
    ])

    train_data = datasets.CIFAR10(root='/home/cytech/dataset',transform=data_transforms,train=True,download=True)
    val_data = datasets.CIFAR10(root='/home/cytech/dataset',transform=val_transforms,train=False,download=True)
    num_classes = len(train_data.classes)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=batch_size,shuffle=False)
    
    
    return train_loader, val_loader, num_classes
