import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple



class AutoEncoder(nn.Module):
    def __init__(self, input_W:int, input_H:int, input_channels:int, latent_channels:List[int], latent_dim:int, kernel_size:int=3, stride:int=2, padding=1):
        super(AutoEncoder, self).__init__()

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

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
        self.un_project_layer = nn.Linear(latent_dim,  self.flattened_size)
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, self.shape_before_flatten)
        
        self.decoder_layers = nn.ModuleList()
        out_channels = input_channels
        for i in range(len(latent_channels)):
            in_channels = latent_channels[i]
            self.decoder_layers.insert(0, 
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.SiLU() if i != 0 else nn.Identity() # No activation on last layer (layers are inserted in reverse order so i==0 is last layer)
                )
            )
        
            out_channels = in_channels
        
        self.refine_layer = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.decoder_layers.append(self.refine_layer)

    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.projection_layer(x)
        x = F.tanh(x) # Bound the latent space to [-1, 1]

        return x
    
    def decode(self, x):
        x = self.un_project_layer(x)
        x = F.silu(x)
        x = self.unflatten(x)
        for layer in self.decoder_layers:
            x = layer(x)
        return x



if  __name__ == "__main__":
    # Example usage
    batch_size = 16
    input_W, input_H = 32, 32
    input_channels = 3
    # 32 x 32 --> 16 x 16 --> 8 x 8 --> 4x4 --> 2x2 
    latent_dim = 512
    latent_channels = [32, 64, 128, 256]


    ae = AutoEncoder(input_W=input_W, input_H=input_H, input_channels=input_channels, latent_channels=latent_channels, latent_dim=latent_dim)
    input = torch.zeros(batch_size, input_channels, input_W, input_H)


    print(ae)

    encoded = ae.encode(input)
    decoded = ae.decode(encoded)

