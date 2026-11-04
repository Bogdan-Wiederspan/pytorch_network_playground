from __future__ import annotations

import torch


class InputLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        continuous_layer_inst: torch.nn.Module | None = None,
        categorical_layer_inst: torch.nn.Module | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.continuous_layer = continuous_layer_inst
        self.categorical_layer = categorical_layer_inst
        self.ndim = self.continuous_layer.ndim + self.categorical_layer.ndim

    def forward(self, *, categorical_inputs, continuous_inputs):
        x = torch.cat(
            [
                self.continuous_layer(continuous_inputs),
                self.categorical_layer(categorical_inputs),
            ],
            dim=1,
        )
        return x
