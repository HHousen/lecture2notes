import torch
import torch.nn as nn


class AdaptiveConcatPool2d(nn.Module):
    """
    Layer that concats AdaptiveAvgPool2d and AdaptiveMaxPool2d
    https://docs.fast.ai/layers.html#AdaptiveConcatPool2d
    """

    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)
