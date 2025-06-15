from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from read_data import writer

writer = SummaryWriter("logs")
img = Image.open("images/example.jpg")
print(img)

# To tensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image("img_tensor", img_tensor, 1)

# Normalize
trans_norm = transforms.Normalize(img_tensor[0,0,0], img_tensor[0,0,0])
img_norm = trans_norm(img_tensor)
writer.add_image("img_norm", img_norm, 2)

# Resize
trans_resize = transforms.Resize((10,10))
img_resize = trans_resize(img)
print(img_resize)
writer.add_image("img_resize", trans_tensor(img_resize), 3)

# Compose
trans_resize_2 = transforms.Resize(500)
trans_compose = transforms.Compose([trans_resize_2, trans_tensor])
img_resize_2 = trans_compose(img)
writer.add_image("img_resize_2", img_resize_2, 3)

writer.close()
