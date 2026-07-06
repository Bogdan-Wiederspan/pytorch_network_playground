from __future__ import annotations

from collections.abc import Iterable

import torch

from utils import logger

logger_inst = logger.get_logger(__name__)

def do_scheduler_step(
    loss: torch.tensor,
    logger_inst: logger.logging.Logger,
    scheduler_inst: torch.optim.lr_scheduler,
    model_inst: torch.nn.Module,
    optimizer_inst: torch.optim.Optimizer,
    checkpoint_inst: dict[torch.tensor]) -> None:
    """
    Perform with given *scheduler_inst* a scheduler step, under the conditions of the given *scheduler_inst*.
    If a step is performed be vebose using *logger_inst* and reload the last best state for *model_inst* and *optimizer_inst* from
    *checkpoint_inst*.

    Args:
        loss (torch.tensor): The current validation loss.
        logger_inst (logger.logging.Logger): Instance of the used Logger
        scheduler_inst (torch.optim.lr_scheduler): Instance of a Learning Rate Scheduler
        model_inst (torch.nn.Module): Instance of the Model
        optimizer_inst (torch.optim.Optimizer): Instance of an Optimizer
        checkpoint_inst (dict[torch.tensor]): Dictionary containing the learning rate, model state dict and optimizer state dict.
    """
    previous_lr =  optimizer_inst.param_groups[0]["lr"]
    scheduler_inst.step(loss)
    current_lr = optimizer_inst.param_groups[0]["lr"]
    if previous_lr != current_lr:
        logger_inst.info(
            f"{previous_lr} -> {current_lr}\n"
            f"Reload model and optimizer from iteration "
            )
        model_inst.load_state_dict(checkpoint_inst.last_checkpoint["model_state_dict"])
        optimizer_inst.load_state_dict(checkpoint_inst.last_checkpoint["optimizer_state_dict"])


def clip_gradients(parameters: Iterable[torch.nn.Parameter], clip_value: float = 1.0):
    for p in parameters:
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
