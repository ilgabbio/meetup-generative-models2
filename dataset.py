from typing import Tuple

from torchvision.datasets import MNIST
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np


def load_mnist(train: bool) -> Tensor:
    return MNIST(root='./data', train=train, download=True).data / 255


class PairDataset(Dataset[Tuple[Tensor,Tensor]]):
    def __init__(self, xs: Dataset[Tensor], ys: Dataset[Tensor]):
        self._xs, self._ys = xs, ys

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self._xs[index][None,:,:], self._ys[index][None,:,:]

    def __len__(self) -> int:
        return self._xs.shape[0]


def corrupt(data: Tensor, sigma = 0.3) -> Tensor:
    """Corrupting data with white noise."""
    noise = np.random.randn(*data.shape)*sigma
    return data + noise.astype(np.float32)

