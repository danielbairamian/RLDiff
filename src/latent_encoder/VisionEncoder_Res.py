import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

        # nn.init.orthogonal_(self.conv1.weight, gain=nn.init.calculate_gain("relu"))
        # nn.init.orthogonal_(self.conv2.weight, gain=0.1)  # near-identity at init

    def forward(self, x):
        return x + self.conv2(F.silu(self.conv1(F.silu(x))))


class IMPALABlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2, kernel_size: int = 3, stride: int = 2, padding: int = 1):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.res_blocks = nn.Sequential(*[ResBlock(out_channels) for _ in range(num_res_blocks)])

        nn.init.orthogonal_(self.downsample.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        return self.res_blocks(self.downsample(x))


class VisionEncoder(nn.Module):
    def __init__(
        self,
        input_W: int,
        input_H: int,
        input_channels: int,
        latent_channels: List[int],
        latent_dim: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        num_res_blocks: int = 2,
    ):
        """
        IMPALA-style vision encoder for reinforcement learning.

        Args:
            input_W:         Input image width.
            input_H:         Input image height.
            input_channels:  Number of input image channels (e.g. 3 for RGB, 1 for grayscale).
            latent_channels: Channel widths for each IMPALA block. Each block downsamples
                             spatial resolution via a strided conv before applying
                             `num_res_blocks` residual blocks.
                             Example: [32, 64, 64] matches the original IMPALA config.
            latent_dim:      Dimensionality of the output state vector.
            kernel_size:     Kernel size for the downsampling conv in each IMPALA block.
            stride:          Stride for the downsampling conv in each IMPALA block.
            padding:         Padding for the downsampling conv in each IMPALA block.
            num_res_blocks:  Number of residual blocks inside each IMPALA block.
        """
        super().__init__()

        blocks = []
        in_ch = input_channels
        for out_ch in latent_channels:
            blocks.append(IMPALABlock(in_ch, out_ch, num_res_blocks, kernel_size, stride, padding))
            in_ch = out_ch

        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(latent_channels[-1], latent_dim)

        # small init on final projection — keeps early policy/value outputs near zero,
        # which avoids large initial gradients from arbitrary action preferences
        # nn.init.orthogonal_(self.proj.weight, gain=0.01)
        # nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor, values in [0, 1] or [-1, 1].
        Returns:
            z: (B, latent_dim) state vector.
        """
        x = self.encoder(x)
        x = F.silu(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


if __name__ == "__main__":
    encoder = VisionEncoder(
        input_W=96,
        input_H=96,
        input_channels=3,
        latent_channels=[32, 64, 64],  # original IMPALA config
        latent_dim=256,
        num_res_blocks=2,
    )

    x = torch.randn(4, 3, 96, 96)
    z = encoder(x)

    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(z.shape)}")
    print(f"Params: {sum(p.numel() for p in encoder.parameters()):,}")