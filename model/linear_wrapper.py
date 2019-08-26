
import torch
from torch import nn


class LinearWrapper(nn.Module):
    """
    Wraps a features extractor module with a linear layer at its output.
    """

    def __init__(self, extractor: nn.Module, embedded_dim, out_dim, bias=True):
        super().__init__()

        self.module = extractor
        self.fc = nn.Linear(embedded_dim, out_dim, bias=bias)

    def forward(self, *inputs, **k_inputs):
        x = self.module(*inputs, **k_inputs)
        x = self.fc(x)

        return x
