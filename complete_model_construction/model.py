from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

