from torch.nn import Sequential
from modules import DownBlock, UpBlock

class Encoder(Sequential):
    """Receives images 1x28x28 as (N,C,H,W)."""
    def __init__(self):
        super().__init__(
            DownBlock(1,8,3,1),  # 14x14
            DownBlock(8,16,3,1), # 7x7
            DownBlock(16,32,7),  # 1x1
        )

class Decoder(Sequential):
    """Receives embeddings as (N,C,H,W)."""
    def __init__(self):
        super().__init__(
            UpBlock(32,16,7),    # 7x7
            UpBlock(16,8,3,1,1), # 14x14
            UpBlock(8,1,3,1,1),  # 28x28
        )

class Autoencoder(Sequential):
    def __init__(self):
        super().__init__(
            Encoder(), 
            Decoder(), 
        )

