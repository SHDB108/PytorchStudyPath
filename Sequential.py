from torch import nn

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
    nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2),
    nn.Sigmoid()
)

