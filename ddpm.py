from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module, Linear, Conv2d, MSELoss, Sigmoid, LeakyReLU
from torch.nn import Sequential, ModuleList
import math

from modules import DownBlock, UpBlock


class Encoder(Module):
    """Receives images 1x28x28 as (N,C,H,W)."""
    def __init__(self):
        super().__init__()
        self._modules = ModuleList([
            DownBlock(1,8,3,1),  # 14x14
            DownBlock(8,16,3,1), # 7x7
            DownBlock(16,32,7),  # 1x1
        ])

    def forward(self, x: Tensor) -> Tuple[Tensor,...]:
        res = [x]
        for module in self._modules:
            res.append(module(res[-1]))
        return tuple(res)


class Decoder(Sequential):
    """Receives embeddings as (N,C,H,W)."""
    def __init__(self):
        super().__init__()
        self._modules = ModuleList([
            UpBlock(32,16,7),    # 7x7
            UpBlock(16,8,3,1,1), # 14x14
            UpBlock(8,1,3,1,1),  # 28x28
        ])
        self._composers = ModuleList([
            CatConvComposer(16),
            CatConvComposer(8),
            CatConvComposer(1),
        ])

    def forward(self, xs: Tuple[Tensor,...]) -> Tensor:
        xs = list(xs)
        x = xs.pop()
        for skip, module, composer in reversed(zip(
            xs, self._modules, self._composers
        )):
            x = composer(module(x), skip)
        return x


class CatConvComposer(Module):
    def __init__(self, channels):
        super.__init__()
        self._conv = Conv2d(2*channels, channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        return self._conv(torch.cat([x1,x2],dim=1))


class SinusoidalPositionEmbedding(Module):
    "Started from https://huggingface.co/blog/annotated-diffusion"
    def __init__(self, channels, edge):
        super().__init__()
        self._channels = channels
        self._edge = edge

    def forward(self, time):
        """
        Expected time.shape == (N,) with N even.
        """
        device = time.device
        half_dim = self._channels // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings[:,:,None,None].repeat(1,1,self._edge,self._edge)


class Unet(Sequential):
    def __init__(self):
        super().__init__(
            Encoder(), 
            Decoder(), 
        )

