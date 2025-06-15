import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10("dataset", True, transform=transform, download=True)
testset = torchvision.datasets.CIFAR10("dataset", False, transform=transform, download=True)

for i in range(10):
    img,label = trainset[i]
    writer.add_image("train_set", img, i)