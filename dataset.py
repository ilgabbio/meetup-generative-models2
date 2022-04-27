from typing import Tuple, Optional

from torchvision.datasets import MNIST
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np


def load_mnist(train: bool) -> Tuple[Tensor,Tensor]:
    mnist = MNIST(root='./data', train=train, download=True)
    return mnist.data / 255, mnist.targets


class PairDataset(Dataset[Tuple[Tensor,Tensor]]):
    def __init__(self, xs: Dataset[Tensor], ys: Optional[Dataset[Tensor]] = None):
        self._xs, self._ys = xs, ys if ys is not None else xs

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        x = self._xs[index]
        if len(x.shape) == 2:
            x = x[None,:,:]
        y = self._ys[index]
        if len(y.shape) == 2:
            y = y[None,:,:]
        return x, y

    def __len__(self) -> int:
        return self._xs.shape[0]


def corrupt(data: Tensor, sigma = 0.3) -> Tensor:
    """Corrupting data with white noise."""
    noise = np.random.randn(*data.shape)*sigma
    return data + noise.astype(np.float32)

