from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from bbTT.configs.utils import choice_check

OPTIMIZER_CHOICE = Literal["adamw"]


@dataclass
class ADAMWConfig:
    # weight decay lambda as in Loshchilov & Hutter (AdamW): per-step shrink at full learning rate
    weight_decay: float = 1.83e-5
    decouple_weight_decay_from_lr: bool = False  # if true, decoupled weight decay with lr, as in AdamW

    # decay rates (beta1, beta2) of the running averages of the gradient and of its square - Marcel: (0.9, 0.98)
    # the default is torch's; a smaller beta2 lets the step size adapt faster to a changing gradient scale
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class OptimizerConfig:
    lr: float = 1e-2  # start learning rate
    optimizer_choice: OPTIMIZER_CHOICE = "adamw"

    # upper bound of the L2 norm of the whole gradient (all parameters as one vector) - Marcel: 1.5
    # a larger gradient is scaled down to this length, its direction is kept. None disables the clipping.
    # Optional[...] instead of "| None": from_dict evaluates the hints
    max_grad_norm: Optional[float] = None

    # one sub config per OPTIMIZER_CHOICE, field name equals the choice, so it is overridable
    # (--override optimizer_config.adamw.weight_decay=1e-4) and saved with the checkpoint
    adamw: ADAMWConfig = field(default_factory=ADAMWConfig)

    def __post_init__(self):
        choice_check(self.optimizer_choice, OPTIMIZER_CHOICE)
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive or None, got {self.max_grad_norm}")

    @property
    def active_config(self) -> ADAMWConfig:
        """
        Returns:
            ADAMWConfig: Sub config of the chosen optimizer. Checkpoints written before the sub config was a
                field lack the attribute (from_dict skips __init__), then the code default is used.
        """
        return getattr(self, self.optimizer_choice, None) or ADAMWConfig()
