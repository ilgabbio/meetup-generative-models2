from torch.nn import Module, Sequential, Conv2d, LeakyReLU, ConvTranspose2d


def num_params(model: Module):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


class DownBlock(Sequential):
    def __init__(self, ch_in, ch_out, kern, pad=0, sep=False):
        super().__init__(
            self._conv_sep(ch_in, ch_out, kern, pad) if sep
            else self._conv(ch_in, ch_out, kern, pad),
            LeakyReLU(),
        )
        
    def _conv(self, ch_in, ch_out, kern, pad) -> Module:
        return Conv2d(ch_in, ch_out, kernel_size=kern, stride=2, padding=pad)
        
    def _conv_sep(self, ch_in, ch_out, kern, pad) -> Module:
        return Sequential(
            Conv2d(ch_in, ch_in, kernel_size=kern, stride=2, padding=pad, groups=ch_in),
            Conv2d(ch_in, ch_out, kernel_size=1),
        )


class UpBlock(Sequential):
    def __init__(self, ch_in, ch_out, kern, pad = 0, opad = 0, sep=False):
        super().__init__(
            self._conv_sep(ch_in, ch_out, kern, pad, opad) if sep
            else self._conv(ch_in, ch_out, kern, pad, opad),
            LeakyReLU(),
        )

    def _conv(self, ch_in, ch_out, kern, pad, opad):
        return ConvTranspose2d(
            ch_in, ch_out, kernel_size=kern, stride=2,
            padding=pad, output_padding=opad,
        )

    def _conv_sep(self, ch_in, ch_out, kern, pad, opad):
        return Sequential(
            ConvTranspose2d(
                ch_in, ch_in, kernel_size=kern, stride=2,
                padding=pad, output_padding=opad, groups=ch_in,
            ),
            Conv2d(ch_in, ch_out, kernel_size=1),
        )

