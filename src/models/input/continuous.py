from __future__ import annotations

import torch

from models.utils.layer_utils import optional_layer


class ContinuousInputLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        continuous_inputs: tuple[str],
        std_layer: torch.nn.Module | None = None,
        rotation_layer: torch.nn.Module | None = None,
        padding_continuous_layer: torch.nn.Module | None = None,
        *args,
        **kwargs,
    ):
        """
        Enables the use of categorical and continuous features in a single model.
        A tokenizer and embedding layer are created  is created using and an embedding layer.
        The continuous features are passed through a linear layer and then concatenated with the
        categorical features.
        """
        super().__init__()
        self.ndim = len(continuous_inputs)

        self.rotation_layer = optional_layer(rotation_layer)
        self.std_layer = optional_layer(std_layer)
        self.padding_continuous_layer = optional_layer(padding_continuous_layer)

    def forward(self, x):
        x = self.padding_continuous_layer(x)
        x = self.rotation_layer(x)
        x = self.std_layer(x)
        return x
