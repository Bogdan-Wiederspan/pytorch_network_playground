from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from bbTT.configs.utils import choice_check

SCHEDULER_CHOICE = Literal["linear", "cosine_annealing", "reduce_on_plateau", "step", "exponential"]


@dataclass
class StepLRConfig:  # used by marcel
    step_size: int = 500  # number of iterations between two learning rate reductions
    gamma: float = 0.5  # learning rate reduction factor


@dataclass
class CosineAnnealingLRConfig:
    T_max: int = 5000  # maximum number of iterations
    eta_min: float = 1e-9  # minimum learning rate


@dataclass
class ReduceLROnPlateauConfig:
    # The metric this scheduler sees is NOT the raw validation loss but the checkpoint's best confirmed loss
    # (see SchedulerHandler and CheckpointConfig). It only changes when a checkpoint is confirmed, so `patience`
    # counts validations without a confirmed improvement and `threshold` is applied on top of CheckpointConfig.delta.
    mode: str = "min"  # min or max. min = lr reduces when the quantity monitored has stopped decreasing
    patience: int = 3  # wait x number of checks without confirmed improvement - Marcel: 10
    threshold_mode: str = "abs"  # type of min_delta - Marcel: abs
    factor: float = 0.5  # LR reduce by factor
    cooldown: int = 0  # number of iterations to wait after a learning rate reduction before resuming normal operation
    min_lr: float = 1e-8  # lower bound on the learning rate
    eps: float = 1e-08  # minimal decay applied to lr, if it is smaller than this value, it is set to this value
    threshold: float = 1e-4  # minimum change of the confirmed best loss to count as improvement (torch default)


@dataclass
class LinearLRConfig:
    start_factor: float = 1e-4  # the initial learning rate will be the start_factor times the base learning rate
    end_factor: float = 1.0  # the final learning rate will be the end_factor times the base learning rate
    total_iters: int = 1000  # number of iterations over which the multiplier increases from start_factor to end_factor


@dataclass
class ExponentialLRConfig:
    gamma: float = 0.9  # Multiplicative factor of learning rate decay.
    last_epoch: int = -1  # The index of last epoch. Default: -1.


SCHEDULER_REGISTRY = {
    "cosine_annealing": (CosineAnnealingLRConfig, "CosineAnnealingLR"),
    "reduce_on_plateau": (ReduceLROnPlateauConfig, "ReduceLROnPlateau"),
    "linear": (LinearLRConfig, "LinearLR"),
    "step": (StepLRConfig, "StepLR"),
    "exponential": (ExponentialLRConfig, "ExponentialLR"),
}


@dataclass
class SchedulerConfig:
    scheduler_chain: tuple[SCHEDULER_CHOICE, ...] = ("reduce_on_plateau",) # schedulers used in chain
    # scheduler_chain: tuple[SCHEDULER_CHOICE, ...] = ("linear", "reduce_on_plateau")  # schedulers used in chain
    milestones: tuple[int, ...] = (1000,)  # intervals after which the LR scheduler is swapped

    @property
    def config_chain(self) -> tuple[dict[str, Any], ...]:
        """
        Returns:
            tuple[dict[str, Any], ...]: Default keyword arguments of each scheduler in ``scheduler_chain``, in order.
        """
        return tuple(asdict(SCHEDULER_REGISTRY[choice][0]()) for choice in self.scheduler_chain)

    @property
    def scheduler_cls_chain(self) -> tuple[str, ...]:
        """
        Returns:
            tuple[str, ...]: Class name of each scheduler in ``scheduler_chain``, in order; instantiated in the training.
        """
        return tuple(SCHEDULER_REGISTRY[choice][1] for choice in self.scheduler_chain)

    def __post_init__(self):
        for choice in self.scheduler_chain:
            choice_check(choice, SCHEDULER_CHOICE)
