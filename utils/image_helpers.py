import torch
import torchvision
def tensorboard_image_process(img, num_images=64):
    if not torch.is_tensor(img):
        img = torch.as_tensor(img, dtype=torch.float32)
    return torchvision.utils.make_grid(img[:num_images].detach().cpu())