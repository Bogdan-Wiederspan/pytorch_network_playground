from __future__ import annotations

import torch


class EmptyLayer(torch.nn.Module):
    """
    Empty Layer that return always an empty tensor.
    This kind of layer is useful, when one wants an optional tensor in concatenation operations.
    """
    def __init__(self):
        super().__init__()
        self.ndim = 0

    def forward(self, *args, **kwargs):
        return torch.tensor([])
