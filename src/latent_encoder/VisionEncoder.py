import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class VisionEncoder(nn.Module):
    def __init__(self, input_W:int, input_H:int, input_channels:int, latent_channels:List[int], latent_dim:int, kernel_size:int=3, stride:int=2, padding=1):
        super(VisionEncoder, self).__init__()

        self.encoder_layers = nn.ModuleList()

        self.input_W = input_W
        self.input_H = input_H
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        in_dim = input_channels
        for i in range(len(latent_channels)):
            out_dim = latent_channels[i]  

            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.SiLU(),
                )
            )

            in_dim = out_dim

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_W, input_H)

            for layer in self.encoder_layers:
                dummy_input = layer(dummy_input)
            
            self.shape_before_flatten = dummy_input.shape[1:] # [C, H, W]
            self.flattened_size = dummy_input.numel()            


        self.projection_layer = nn.Linear(self.flattened_size, latent_dim)
        self.flatten = nn.Flatten()

    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.projection_layer(x)
        x = F.tanh(x)
        return x
