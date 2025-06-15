import torch
import torchvision

model = torchvision.models.vgg16(pretrained=False)

# 方式1
torch.save(model, "model_all.pth")

model_load = torch.load("model_all.pth")

# 方式2
torch.save(model.state_dict(), "model_weight.pth")

model = torchvision.models.vgg16(pretrained=False)
model.load_state_dict(model_load)