from __future__ import annotations

from typing import Sequence, Union

import torch

ValueLike = Union[float, int, Sequence[float], torch.Tensor]


class PaddingLayer(torch.nn.Module):  # noqa: F811
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape [batch, num_features].

        Returns:
            torch.Tensor: Copy of *x* in *target_dtype* with the marked entries replaced.
        """
        if not self.active:
            return x

        x = x.to(self.target_dtype)
        # broadcast over the batch dimension: scalar or [num_features] against [batch, num_features]

        mask = (x == self.target_value)
        return torch.where(mask, self.padding_value.to(self.target_dtype), x)
