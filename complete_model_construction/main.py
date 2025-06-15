import torchvision
import torch
from torch import nn, device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

trainset = torchvision.datasets.CIFAR10("../dataset", True, transform=torchvision.transforms.ToTensor(), download=True)
testset = torchvision.datasets.CIFAR10("../dataset", False, transform=torchvision.transforms.ToTensor(), download=True)


trainloader = DataLoader(trainset, batch_size=64)
testloader = DataLoader(testset, batch_size=64)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("MPS device not available.")

model = MyModel().to(device)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
epochs = 1

total_steps = 0

writer = SummaryWriter("logs")

for epoch in range(epochs):
    model.train()
    for data in trainloader:
        imgs, target = data
        imgs,target = imgs.to(device), target.to(device)
        output = model(imgs)

        loss = loss_fn(output, target).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if total_steps % 100 == 0:
            print('Train step: {}, Loss: {:.6f}'.format(total_steps, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_steps)

        total_steps += 1

total_test_steps = 0
total_test_loss = 0
total_test_accuracy = 0
with torch.no_grad():
    model.eval()
    for data in testloader:
        imgs, target = data
        imgs,target = imgs.to(device), target.to(device)
        output = model(imgs)

        loss = loss_fn(output, target).to(device)
        total_test_loss += loss.item()

        accuracy = (output.argmax(1) == target).sum()
        total_test_accuracy += accuracy.item()

        writer.add_scalar('test_loss', loss.item(), total_test_steps)
        writer.add_scalar('test_accuracy', accuracy.item(), total_test_steps)

        total_test_steps += 1