from typing import Tuple, Iterable, Optional, List

import torch
from torch import Tensor, tensor
from torch.nn import Sequential, Module, Linear, Conv2d, Sigmoid, LeakyReLU
from torch.nn import ModuleList, MSELoss, BatchNorm2d
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import math
import numpy as np
from modules import DownBlock, UpBlock
from tqdm.notebook import tqdm
from training import MetricsCollector, save_model


def normalize_image(image: Tensor) -> Tensor:
    return image * 2. - 1.

def denormalize_image(image: Tensor) -> Tensor:
    return torch.clip((image + 1.) / 2., 0., 1.)


class Corruptor:
    def __init__(self, betas: Iterable[float]):
        self._betas = tensor(betas).type(torch.float32)
        self._alphas = 1 - self._betas
        self._alphasc = torch.cumprod(self._alphas, dim=0)

    def __len__(self) -> int:
        return len(self._betas)

    def corrupt(self, x_0: Tensor, t: int) -> Tuple[Tensor,Tensor]:
        alphac = self._alphasc[t].to(x_0.device)
        epsilon = torch.randn(x_0.shape, device=x_0.device)
        x_t = torch.sqrt(alphac) * x_0 + epsilon * torch.sqrt(1 - alphac)
        return x_t, epsilon

    def uncorrupt(self, x_t: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        eps = self.scale_noise(noise, t)
        alphac = self._alphasc[t]
        return (x_t - eps) / torch.sqrt(alphac)

    def scale_noise(self, epsilon: Tensor, t: Tensor) -> Tensor:
        alphac = self._alphasc[t]
        return epsilon * torch.sqrt(1 - alphac)

    def uncorrupt_step(self, x_t: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        alpha = self._alphas[t]
        alphac = self._alphasc[t]
        eps = noise * ((1 - alpha) / torch.sqrt(1 - alphac))
        x_t_1 = (x_t - eps) / torch.sqrt(alpha)
        if t > 0:
            z = torch.randn(x_t.shape, device=x_t.device)
            x_t_1 += torch.sqrt(1 - alpha) * z
        return x_t_1


class DdpmDataset(Dataset):
    def __init__(self, data: Dataset, corruptor: Corruptor):
        self._data = data
        self._corruptor = corruptor

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, int]:
        x_0 = normalize_image(self._data[index])
        t = torch.randint(1, len(self._corruptor), (1,)).to(self._data.device)
        x_t, epsilon = self._corruptor.corrupt(x_0, t)
        return x_t, epsilon, t


class Unet(Module):
    def __init__(self):
        super().__init__()
        self._encoder = Encoder()
        self._decoder = Decoder()

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        skips_bneck = self._encoder(x_t, t)
        return self._decoder(skips_bneck, t)
        


class Encoder(Module):
    """Receives images 1x28x28 as (N,C,H,W)."""
    def __init__(self):
        super().__init__()
        self._blocks = ModuleList([
            DownBlock(1,8,3,1),   # 14x14
            DownBlock(8,16,3,1), # 7x7
            DownBlock(16,32,7),  # 1x1
        ])
        self._residual = ModuleList([
            ResidualBlock(8),
            ResidualBlock(16),
            ResidualBlock(32),
        ])
        self._time_embeddings = ModuleList([
            SinusoidalPositionEmbedding(8),
            SinusoidalPositionEmbedding(16),
            SinusoidalPositionEmbedding(32),
        ])

    def forward(self, x: Tensor, t: Tensor) -> Tuple[Tensor,...]:
        res = [x]
        for block, residual, emb in zip(
            self._blocks, self._residual, self._time_embeddings
        ):
            x = res[-1]
            x = block(x)
            x = residual(x, t)
            x = emb(x, t)
            res.append(x)
        return tuple(res)


class Decoder(Module):
    """Receives embeddings as (N,C,H,W)."""
    def __init__(self):
        super().__init__()
        self._blocks = ModuleList([
            DecoderBlock(32,16,7),    # 7x7
            DecoderBlock(16,8,3,1,1), # 14x14
            DecoderBlock(8,1,3,1,1),  # 28x28
        ])

    def forward(self, xs: Tuple[Tensor,...], t: Tensor) -> Tensor:
        xs = list(xs)
        x = xs.pop()
        for skip, block in zip(reversed(xs), self._blocks):
            x = block(x, skip, t)
        return x


class DecoderBlock(Module):
    def __init__(self, ch_in, ch_out, kern, pad = 0, opad = 0):
        super().__init__()
        self._residual = ResidualBlock(ch_in)
        self._embedding = SinusoidalPositionEmbedding(ch_in)
        self._up_block = UpBlock(ch_in, ch_out, kern, pad, opad)
        self._composer = CatConvComposer(ch_out)

    def forward(self, x: Tensor, skip: Tensor, t: tensor) -> Tensor:
        x = self._residual(x, t)
        x = self._embedding(x, t)
        x = self._up_block(x)
        x = self._composer(x, skip)
        return x


class ResidualBlock(Module):
    def __init__(self, channels: int):
        super().__init__()
        self._emb = SinusoidalPositionEmbedding(channels)
        self._block1 = Sequential(
            Conv2d(channels, channels, kernel_size=3, padding=1),
            BatchNorm2d(channels),
            LeakyReLU(),
        )
        self._block2 = Sequential(
            Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2),
            BatchNorm2d(channels),
            LeakyReLU(),
        )
        self._compose = CatConvComposer(channels)

    def forward(self, x: Tensor, t) -> Tensor:
        x = self._emb(x, t)
        return self._compose(self._block1(x), self._block2(x)) + x


class CatConvComposer(Module):
    def __init__(self, channels):
        super().__init__()
        self._conv = Conv2d(2*channels, channels, kernel_size=1)

    def forward(self, x1, x2):
        return self._conv(torch.cat([x1,x2],dim=1))


class SinusoidalPositionEmbedding(Module):
    def __init__(self, channels: int):
        super().__init__()
        self._channels = channels
        self._mlp = Sequential(
            Conv2d(channels, 2*channels, kernel_size=1),
            BatchNorm2d(2*channels),
            LeakyReLU(),

            Conv2d(2*channels, 2*channels, kernel_size=1),
            BatchNorm2d(2*channels),
            LeakyReLU(),

            Conv2d(2*channels, 2*channels, kernel_size=1),
            BatchNorm2d(2*channels),
            LeakyReLU(),
        )

    def forward(self, x: Tensor, t: Tensor):
        """
        Expected t.shape == (N,) with N even.
        """
        assert(x.shape[1] == self._channels)
        device = x.device
        half_dim = self._channels // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = self._mlp(embeddings[:,:,None,None])
        embeddings = embeddings.repeat(1,1,x.shape[2], x.shape[3])

        scale, shift = embeddings.chunk(2, dim=1);
        res = x * (scale + 1) + shift
        return res


def train(
        dataset: Dataset, 
        corruptor = Corruptor,
        model: Optional[Unet] = None,
        batch_size: int = 64,
        epochs=500,
) -> Tuple[Corruptor, Module, List[float]]:
    # The dataset:
    data = DataLoader(
        dataset=DdpmDataset(dataset, corruptor),
        batch_size=batch_size,
        shuffle=True,
    )

    # The optimization problem
    if model is None:
        model = Unet()
        model.to(dataset.device)
    loss = MSELoss()
    loss.to(dataset.device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(
        optimizer=optimizer, 
        step_size=100,
        gamma=0.5,
    )
    metrics = MetricsCollector()

    # The training loop:
    for epoch in tqdm(range(epochs)):
        losses = []
        for x_t, eps, t in data:
            # One optimizer step:
            optimizer.zero_grad()
            eps_est = model(x_t, t)
            l = metrics.extract_loss("train", loss(eps, eps_est))
            l.backward()
            optimizer.step()
            losses.append(l.item())

        # Storing the loss:
        metrics.epoch()
        metrics.print_last()

        # Sporadic operations:
        if (epoch + 1) % 100 == 0:
            # Saving the current model:
            save_model(f"models/ddpm{epoch+1:04}", model, metrics)

        # ready for another epoch:
        scheduler.step()

    # Returning the trained model:
    return model, metrics


def sample(
    corruptor: Corruptor,
    model: Module,
    N: int = 10,
    stop_at: int = 0,
) -> Tuple[Tensor,...]:
    with torch.no_grad():
        # Generate the noise:
        device = next(model.parameters()).device
        x_t = torch.randn((N,1,28,28), device=device)
        res = [x_t[:,0,:,:]]

        # Denoising:
        T = len(corruptor)
        for t in reversed(tuple(range(T))):
            if t <= stop_at:
                break
            epsilon = model(x_t, t)
            x_t = corruptor.uncorrupt_step(x_t, epsilon, t)
            res.append(x_t[:,0,:,:])

        return tuple(res)

