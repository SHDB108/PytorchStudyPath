import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10(root='dataset', train=False, download=True, transform=transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class RNNModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(out_features=64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

my_model = RNNModule()
print(my_model)

writer = SummaryWriter("logs")
step = 0
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)

for data in dataloader:
    imgs, targets = data
    inputs = imgs
    outputs = my_model(inputs)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)


    # writer.add_images("inputs", inputs, step)
    # writer.add_images("outputs", outputs, step)

    step += 1

writer.close()