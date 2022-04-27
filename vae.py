from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Sequential, Linear, Conv2d, MSELoss, Sigmoid, LeakyReLU

from modules import DownBlock, UpBlock


class VariationalAutoencoder(Module):
    def __init__(self, dim: int = 2, stochastic = True):
        super().__init__()
        # Here the shape depends on the encoder:
        encoder = Sequential(
            DownBlock(1,32,3,1), # 14x14
            DownBlock(32,16,3,1), # 7x7
            DownBlock(16,8,3), # 3x3
        )
        enc_shape = encoder(torch.zeros(1,1,28,28)).shape[1:]
        n = np.prod(enc_shape)
        self._encoder = Sequential(
            encoder,
            Flatten(),
            Linear(n, n//4), LeakyReLU(),
            ReparameterizationTrick(n//4, dim, stochastic),
        )
        self._decoder = Sequential(
            Linear(dim, n//4), LeakyReLU(),
            Reshaper(n//4, enc_shape),
            UpBlock(8,16,3), # 7x7
            UpBlock(16,32,3,1,1), # 14x14
            UpBlock(32,1,3,1,1, Sigmoid), # 28x28
        )
    
    def forward(self, x: Tensor) -> Tuple[Tensor,Tensor,Tensor]:
        z, mu, logvar = self._encoder(x)
        y_hat = self._decoder(z)

        return y_hat, mu, logvar


class Flatten(Module):
    def forward(self, theta):
        return torch.flatten(theta, start_dim=1)
    

class ReparameterizationTrick(Module):
    def __init__(self, in_dim: int, dim: int, stochastic):
        super().__init__()
        self._mu = Linear(in_dim, dim)
        self._logvar = Linear(in_dim, dim) if stochastic else None
    
    def forward(self, theta: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu = self._mu(theta)

        if self._logvar is None:
            return mu, mu, mu

        logvar = self._logvar(theta)
        epsilon = torch.randn(*mu.shape, requires_grad = False)
        sigma = torch.exp(0.5 * logvar)
        z = epsilon * sigma + mu

        return z, mu, logvar


class Reshaper(Module):
    def __init__(self, dim: int, enc_shape: Tuple[int, ...]):
        super().__init__()
        self._resize = Sequential(
            Linear(dim, np.prod(enc_shape)),
            LeakyReLU(),
        )
        self._enc_shape = enc_shape

    def forward(self, z: Tensor) -> Tensor:
        return self._resize(z).reshape(-1,*self._enc_shape)


class ElboLoss(Module):
    def __init__(self, reconstr = MSELoss, beta = 1.0):
        super().__init__()
        self._beta = beta
        self._reconstr = reconstr()

    def forward(
        self, 
        outcome: Tuple[Tensor,Tensor,Tensor],
        gt: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        pred, mu, logvar = outcome
        rec = self._reconstr(pred, gt)
        klds = 1 + logvar - mu.pow(2) - logvar.exp()
        kld = - 0.5 * torch.sum(klds)
        return rec + self._beta * kld, rec, kld


class MmdLoss(Module):
    def __init__(self, reconstr = MSELoss, beta = 1.0):
        super().__init__()
        self._beta = beta
        self._reconstr = reconstr()

    def forward(self, outcome, gt):
        pred, mu, _ = outcome
        rec = self._reconstr(pred, gt)
        mmd = self._mmd(torch.randn(*mu.shape[0:2], requires_grad = False), mu)
        return rec + self._beta * mmd, rec, mmd

    def _mmd(self, a, b):
        return (
            self._gaussian_kernel(a, a).mean() +
            self._gaussian_kernel(b, b).mean() -
            2*self._gaussian_kernel(a, b).mean()
        )

    def _gaussian_kernel(self, a, b):
        # Info:
        batch_a, batch_b = a.shape[0], b.shape[0]
        depth = a.shape[1]

        # Final shape with repetitions:
        a = a.view(batch_a, 1, depth)
        b = b.view(1, batch_b, depth)
        a_core = a.expand(batch_a, batch_b, depth)
        b_core = b.expand(batch_a, batch_b, depth)

        # Matrix with all point2point RBF values:
        numerator = (a_core - b_core).pow(2).mean(2)/depth
        return torch.exp(-numerator)

