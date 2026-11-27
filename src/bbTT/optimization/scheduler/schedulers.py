
from __future__ import annotations

import math

import torch


class HalfCosineLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Cosine decay over half a wavelength, constant afterwards:

            lr(k) = eta_min + (base_lr - eta_min) * (1 + cos(pi * min(k, T_max) / T_max)) / 2

        Unlike torch's ``CosineAnnealingLR``, which is periodic and rises again after ``T_max``, the learning rate
        stays at ``eta_min`` for every step ``k >= T_max``.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer whose learning rate is scheduled.
            T_max (int): Number of steps of the half wave, after which ``eta_min`` is reached.
            eta_min (float): Absolute learning rate at the end of the half wave. Defaults to 0.0.
            last_epoch (int): Index of the last step, -1 starts a fresh schedule. Defaults to -1.

        Raises:
            ValueError: If ``T_max`` is not positive.
        """
        if T_max <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        # set before super().__init__: the base class runs one step, which calls get_lr
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """
        Returns:
            list[float]: Learning rate of every parameter group at the current step.
        """
        progress = min(self.last_epoch, self.T_max) / self.T_max
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.eta_min + (base_lr - self.eta_min) * factor for base_lr in self.base_lrs]
