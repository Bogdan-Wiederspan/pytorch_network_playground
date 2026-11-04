from __future__ import annotations

from typing import Sequence, Union

import torch

ValueLike = Union[float, int, Sequence[float], torch.Tensor]


class PaddingLayer(torch.nn.Module):
    def __init__(
        self,
        target_value: ValueLike,
        padding_value: ValueLike,
        target_dtype: torch.dtype = torch.float32,
        active: bool = True,
    ):
        """
        Replaces entries of the input that equal *target_value* with *padding_value*.

        Padding and Target values can be adjusted in 2 manner:
            1: as scalar -> is upcasted to same value for every feature
            2: as sequence -> individual treatment, with default value being used as dummy [num_features] (one value per feature).
        Resulting masking tensor is stored as buffer, so that they are part of the state dict and exported together with the model.

        Args:
            target_value (ValueLike): Value(s) that mark a missing entry. Defaults to 0.
            padding_value (ValueLike): Value(s) inserted for a missing entry. Defaults to EMPTY_FLOAT.
            target_dtype (torch.dtype, optional): dtype the input is cast to before the comparison. Defaults to torch.float32.
            active (bool, optional): If False the layer is the identity. Defaults to True.

        Raises:
            ValueError: If target and padding values are both per-feature but of different length.

        """
        super().__init__()
        target = torch.as_tensor(target_value, dtype=torch.float32)
        padding = torch.as_tensor(padding_value, dtype=torch.float32)
        if target.ndim > 1 or padding.ndim > 1:
            raise ValueError(f"target and padding values need to be scalar or 1D, got {target.shape} and {padding.shape}")
        if target.ndim == 1 and padding.ndim == 1 and target.shape != padding.shape:
            raise ValueError(f"per-feature target and padding values differ in length: {target.shape} vs {padding.shape}")

        self.target_value = torch.nn.Buffer(target, persistent=True)
        self.padding_value = torch.nn.Buffer(padding, persistent=True)
        self.target_dtype = target_dtype
        self.active = active

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Locate the entries that carry the missing-value marker.

        Args:
            x (torch.Tensor): Input of shape [batch, num_features].

        Returns:
            torch.Tensor: Boolean mask of the same shape, True where *x* equals *target_value*. All False if inactive.
        """
        if not self.active:
            return torch.zeros_like(x, dtype=torch.bool)

        # broadcast over the batch dimension: scalar or [num_features] against [batch, num_features]
        return x.to(self.target_dtype) == self.target_value

    def replace(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Insert the replacement values at the masked positions. Separating this from ``mask`` lets a pipeline
        re-apply the replacement after transformations (e.g. a rotation) that would otherwise alter it.

        Args:
            x (torch.Tensor): Input of shape [batch, num_features].
            mask (torch.Tensor): Boolean mask as returned by ``mask``.

        Returns:
            torch.Tensor: Copy of *x* in *target_dtype* with the masked entries set to *padding_value*.
        """
        if not self.active:
            return x

        x = x.to(self.target_dtype)
        return torch.where(mask, self.padding_value.to(self.target_dtype), x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shortcut for ``replace(x, mask(x))``.

        Args:
            x (torch.Tensor): Input of shape [batch, num_features].

        Returns:
            torch.Tensor: Copy of *x* in *target_dtype* with the marked entries replaced.
        """
        if not self.active:
            return x
        return self.replace(x, self.mask(x))
