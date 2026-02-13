import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from utils.CelebAHQ import CelebAHQ

def get_denorm_fn(mean, std):
    """Generates a denormalization function for a specific mean and std."""
    # We use .view(1, 3, 1, 1) to make it broadcastable with [Batch, Channel, H, W]
    m = torch.tensor(mean).view(1, -1, 1, 1)
    s = torch.tensor(std).view(1, -1, 1, 1)
    
    def denormalize(batch_tensor):
        # Ensure tensors are on the same device as the input data
        return (batch_tensor * s.to(batch_tensor.device)) + m.to(batch_tensor.device)
    
    return denormalize

def CIFAR_dataloader(dataset_path, batch_size, num_workers=4, shuffle=True, drop_last=True, train=True):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    dataset = CIFAR10(root=dataset_path, train=train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                             num_workers=num_workers, drop_last=drop_last, pin_memory=True)
    
    denorm_fn = get_denorm_fn(mean, std)
    info_dict = {"H": 32, "W": 32, "C": 3}
    return dataloader, info_dict, denorm_fn

def MNIST_dataloader(dataset_path, batch_size, num_workers=4, shuffle=True, drop_last=True, train=True):
    mean, std = (0.1307,), (0.3081,)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),  # Pad to 32x32
        transforms.Normalize(mean, std)
    ])
    dataset = MNIST(root=dataset_path, train=train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                             num_workers=num_workers, drop_last=drop_last, pin_memory=True)
    
    denorm_fn = get_denorm_fn(mean, std)
    info_dict = {"H": 32, "W": 32, "C": 1}
    return dataloader, info_dict, denorm_fn


def CelebAHQ_dataloader(dataset_path, batch_size, num_workers=4, shuffle=True, drop_last=True):
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    dataset = CelebAHQ(root=dataset_path, transform=transform, return_labels=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers , drop_last=drop_last, pin_memory=True)
    
    denorm_fn = get_denorm_fn(mean, std)
    info_dict = {"H":256, "W": 256, "C":3}
    return dataloader, info_dict, denorm_fn