from __future__ import annotations

import torch

from ..utils.layer_utils import optional_layer


class ContinuousInputLayer(torch.nn.Module):
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
        Preprocessing branch for continuous features performed in this order:
            padding → rotation → standardization.

        The order is fixed:
            - padding replaces dummy markers with raw values first.
            - rotation needs unstandardized raw (px, py) components
            - standardization runs last on the already padded, rotated values using extern statstics (mean, std) to avoid padding bias the data.

        A layer passed as None is replaced by an identity.

        Args:
            continuous_inputs (tuple[str]): Names of the continuous features, defines the output dimension ``ndim``.
            std_layer (torch.nn.Module | None, optional): Standardization layer. Defaults to None.
            rotation_layer (torch.nn.Module | None, optional): Phi rotation layer. Defaults to None.
            padding_continuous_layer (torch.nn.Module | None, optional): Padding layer for dummy values. Defaults to None.
        """
        super().__init__()
        self.ndim = len(continuous_inputs)

        self.rotation_layer = optional_layer(rotation_layer)
        self.std_layer = optional_layer(std_layer)
        self.padding_continuous_layer = optional_layer(padding_continuous_layer)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Continuous features of shape (batch, ndim).

        Returns:
            torch.Tensor: Preprocessed features of shape (batch, ndim).
        """
        x = self.padding_continuous_layer(x)
        x = self.rotation_layer(x)
        x = self.std_layer(x)
        return x
