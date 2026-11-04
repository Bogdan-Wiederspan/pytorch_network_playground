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
    ):
        """
        Preprocessing branch for continuous features performed in this order:
            padding → rotation → standardization.

        The order is fixed:
            - padding replaces dummy markers with raw values first.
            - rotation needs unstandardized raw (px, py) components
            - standardization runs last on the already padded, rotated values using extern statstics (mean, std) to avoid padding bias the data.

        Specialcase:
        Rotation and standardization passed as None are replaced by an identity. The padding layer stays None,
        Since ``sanitize`` needs its mask and skips the padding steps entirely without it.

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

        # kept as None instead of an identity, the pipeline needs the mask of the padding layer
        self.padding_continuous_layer = padding_continuous_layer

    def sanitize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Padding and rotation on raw values. Missing entries are masked first, so the replacement can be re-applied
        after the rotation; otherwise a replacement that is no fixed point of the rotation would be rotated away.

        Args:
            x (torch.Tensor): Raw continuous features of shape (batch, ndim).

        Returns:
            torch.Tensor: Padded and rotated features, still unstandardized.
        """
        # when no padding layer is used, the rotation can be applied directly
        if self.padding_continuous_layer is None:
            return self.rotation_layer(x)

        mask = self.padding_continuous_layer.mask(x)
        x = self.padding_continuous_layer.replace(x, mask)
        x = self.rotation_layer(x)
        x = self.padding_continuous_layer.replace(x, mask)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Raw Continuous features of shape (batch, ndim).

        Returns:
            torch.Tensor: Preprocessed features of shape (batch, ndim).
        """
        x = self.sanitize(x)
        x = self.std_layer(x)
        return x
