from __future__ import annotations

import math
from typing import Any, Callable, Optional, get_args

import torch

from bbTT.configs.scheduler_config import WARMUP_CHOICE

# A warm-up function maps the normalized progress t in [0, 1] and the start factor s to a learning-rate factor.
# Contract for every entry in WARMUP_REGISTRY: f(0, s) = s and f(1, s) = 1, so the warm-up always ends exactly on
# the base learning rate and the main scheduler can take over without any correction. WarmupScheduler checks f(1, s).
WarmupFn = Callable[[float, float], float]


def linear_warmup(progress: float, start_factor: float) -> float:
    """
    Linear ramp: f(t) = s + (1 - s) * t.

    Args:
        progress (float): Normalized progress t in [0, 1].
        start_factor (float): Factor s at t = 0.

    Returns:
        float: Learning-rate factor.
    """
    return start_factor + (1.0 - start_factor) * progress


def cosine_warmup(progress: float, start_factor: float) -> float:
    """
    Cosine ramp, slow at both ends: f(t) = 1 - (1 - s) * (1 + cos(pi * t)) / 2.

    Args:
        progress (float): Normalized progress t in [0, 1].
        start_factor (float): Factor s at t = 0.

    Returns:
        float: Learning-rate factor.
    """
    return 1.0 - (1.0 - start_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


WARMUP_REGISTRY: dict[str, WarmupFn] = {
    "linear": linear_warmup,
    "cosine": cosine_warmup,
}

# the config only knows the names, the functions live here: keep both in sync
if set(WARMUP_REGISTRY) != set(get_args(WARMUP_CHOICE)):
    raise RuntimeError(f"WARMUP_REGISTRY {set(WARMUP_REGISTRY)} and WARMUP_CHOICE {set(get_args(WARMUP_CHOICE))} differ")


class WarmupScheduler:
    def __init__(
        self,
        optimizer_inst: torch.optim.Optimizer,
        main_inst: torch.optim.lr_scheduler.LRScheduler,
        warmup_fn: Optional[WarmupFn],
        warmup_iterations: int,
        start_factor: float,
    ):
        """
        Two-phase learning-rate schedule: an iteration-driven warm-up followed by the main scheduler, which may be
        iteration-driven (e.g. CosineAnnealingLR) or metric-driven (ReduceLROnPlateau).

        The warm-up is a torch ``LambdaLR`` evaluating ``warmup_fn`` on the normalized progress
        ``t = iteration / warmup_iterations``; after ``warmup_iterations`` batch steps the factor is 1 (base learning
        rate) and the main scheduler takes over. Its counters (``T_max``, ``patience``, ...) only advance from then on.
        While warming up, metric steps are ignored.

        ``main_inst`` must be constructed BEFORE this wrapper: torch schedulers run one step in ``__init__`` that
        writes the base learning rate into the optimizer, the warm-up built here then sets ``base * f(0)`` on top.

        Args:
            optimizer_inst (torch.optim.Optimizer): Optimizer whose learning rate is scheduled.
            main_inst (torch.optim.lr_scheduler.LRScheduler): Already constructed main scheduler.
            warmup_fn (WarmupFn | None): Warm-up curve from ``WARMUP_REGISTRY``, None disables the warm-up.
            warmup_iterations (int): Number of batch steps until the warm-up factor reaches 1.
            start_factor (float): Factor of the base learning rate at the first iteration.

        Raises:
            ValueError: If a warm-up is requested with ``warmup_iterations <= 0`` or ``warmup_fn(1, s) != 1``.
        """
        self.main_inst = main_inst
        self.needs_metric = isinstance(main_inst, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.warmup_iterations = warmup_iterations
        self.warmup_inst: Optional[torch.optim.lr_scheduler.LambdaLR] = None

        if warmup_fn is None:
            return

        if warmup_iterations <= 0:
            raise ValueError(f"warmup_iterations must be positive when a warm-up is used, got {warmup_iterations}")
        end_factor = warmup_fn(1.0, start_factor)
        if not math.isclose(end_factor, 1.0):
            raise ValueError(f"Warm-up function must end at factor 1, {warmup_fn.__name__} ends at {end_factor}")

        # a plain closure on purpose: LambdaLR stores function objects as None in its state_dict,
        # so the checkpoint stays loadable with weights_only=True; the curve is rebuilt from the config on resume
        def lr_lambda(iteration: int) -> float:
            progress = min(iteration / warmup_iterations, 1.0)
            return warmup_fn(progress, start_factor)

        self.warmup_inst = torch.optim.lr_scheduler.LambdaLR(optimizer_inst, lr_lambda=lr_lambda)

    @property
    def in_warmup(self) -> bool:
        """
        Returns:
            bool: True while the warm-up has not yet done ``warmup_iterations`` steps.
        """
        return self.warmup_inst is not None and self.warmup_inst.last_epoch < self.warmup_iterations

    def step_batch(self) -> None:
        """
        Call once after every optimizer step. Advances the warm-up, or the main scheduler if it is iteration-driven.
        """
        if self.in_warmup:
            self.warmup_inst.step()
        elif not self.needs_metric:
            self.main_inst.step()

    def step_metric(self, metric: float) -> bool:
        """
        Call once per validation. Only a metric-driven main scheduler outside the warm-up is stepped.

        Args:
            metric (float): Metric the main scheduler judges, e.g. the confirmed best validation loss.

        Returns:
            bool: Whether the main scheduler was stepped.
        """
        if self.in_warmup or not self.needs_metric:
            return False
        self.main_inst.step(metric)
        return True

    def state_dict(self) -> dict[str, Any]:
        """
        Returns:
            dict[str, Any]: State dicts of both phases, ``"warmup"`` is None without a warm-up.
        """
        return {
            "warmup": None if self.warmup_inst is None else self.warmup_inst.state_dict(),
            "main": self.main_inst.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Args:
            state_dict (dict[str, Any]): Output of ``state_dict`` from a run with the same scheduler config.
        """
        if (self.warmup_inst is None) != (state_dict["warmup"] is None):
            raise ValueError("Warm-up configuration differs between checkpoint and current config")
        if self.warmup_inst is not None:
            self.warmup_inst.load_state_dict(state_dict["warmup"])
        self.main_inst.load_state_dict(state_dict["main"])
