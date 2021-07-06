import torch
import torch.nn as nn
import torch.nn.functional as F


# implement mish activation function
def f_mish(input, inplace=False):
    """
    Applies the mish function element-wise:
    :math:`mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`
    """
    return input * torch.tanh(F.softplus(input))


# implement class wrapper for mish activation function
class mish(nn.Module):
    """
    Applies the mish function element-wise:
    :math:`mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`

    Shape:
        - Input: ``(N, *)`` where ``*`` means, any number of additional
          dimensions
        - Output: ``(N, *)``, same shape as the input

    Examples:
        >>> m = mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    """

    def __init__(self, inplace=False):
        """
        Init method.
        """
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return f_mish(input, inplace=self.inplace)
