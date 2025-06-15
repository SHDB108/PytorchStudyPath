import torch
import torchvision
from torch.utils.data import DataLoader

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=transform, download=True)

data_loader = DataLoader(train_set, batch_size=64, shuffle=True)