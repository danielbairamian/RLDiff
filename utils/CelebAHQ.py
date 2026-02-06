import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


'''https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256'''
class CelebAHQ(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, return_labels=False):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            return_labels (bool): Whether to return dummy labels with the images.
        """
        self.root_dir = root
        self.transform = transform
        self.return_labels = return_labels
        self.image_files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        if self.return_labels:
            return image, 0  # Return image and dummy label 0
        else:
            return image

