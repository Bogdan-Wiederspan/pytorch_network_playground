from __future__ import annotations

import torch

from utils import EMPTY_FLOAT, logger

logger_inst = logger.get_logger(__name__)


class PaddingLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        target_value: float | int = 0,
        padding_value: float | int = EMPTY_FLOAT,
        target_dtype: torch.dtype = torch.float32,
        active=True,
    ):
        """
        Pads input tensor on indices which equals *mask_value* with given *padding_value*.

        Args:
            target_value (int, optional): Target value to mask. Defaults to 0.
            padding_value (float, optional): Value which is used to pad the target. Defaults to utils.EMPTY_FLOAT.
        """
        super().__init__()

        self.target_value = torch.nn.Buffer(
            torch.tensor(target_value).to(torch.float32),
            persistent=True
            )
        self.padding_value = torch.nn.Buffer(
            torch.tensor(padding_value).to(torch.float32),
            persistent=True
            )
        self.target_dtype = target_dtype
        self.active = active

    def forward(self, x):
        if not self.active:
            return x
        x = x.to(self.target_dtype)
        mask = x == self.target_value
        x[mask] = self.padding_value
        return x
