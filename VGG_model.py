import torch.utils.data
import torchvision.datasets
from torch import nn

model = torchvision.models.vgg16(pretrained=True)

print(model)
model.classifier.add_module("new_layer", nn.Linear(in_features=1000, out_features=10))
model.classifier[6] = nn.Linear(in_features=1000, out_features=10)