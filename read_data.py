import os
from PIL import Image
from torch.utils import data


class MyDataset(data.Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_path = os.path.join(self.path, img_name)
        img  = Image.open(img_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)