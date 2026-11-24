import torch

from bbTT.monitoring.logger.logger import get_logger
from bbTT.optimization.optimizer import weight_decay

logger_inst = get_logger(__name__)

def init_optimizer(full_config, model_inst: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build the optimizer chosen in the config with two parameter groups:
    linear-layer weights with weight decay, all remaining trainable parameters without.

    Args:
        full_config: Merged config, ``optimizer_config`` is read.
        model_inst (torch.nn.Module): Model whose parameters are optimized.

    Returns:
        torch.optim.Optimizer: Optimizer instance covering all trainable parameters.
    """
    optimizer_cfg = full_config.optimizer_config
    logger_inst.info("Initialize Optimizer")
    logger_inst.debug("Optimizer settings:\n"
        f"name: {optimizer_cfg.optimizer_choice}\n"
        f"lr: {optimizer_cfg.lr}\n"
        f"weight-decay: {optimizer_cfg.active_config.weight_decay}\n"
        f"decouple lr: {optimizer_cfg.active_config.decouple_weight_decay_from_lr}\n"
        f"betas: {optimizer_cfg.active_config.betas}\n"
        )

    no_decay_group, decay_group = weight_decay.weight_decay_parameter_groups(
        model = model_inst,
        weight_decay = optimizer_cfg.active_config.weight_decay,
        lr = optimizer_cfg.lr,
        decouple_weight_decay_from_lr = optimizer_cfg.active_config.decouple_weight_decay_from_lr,
        betas=tuple(optimizer_cfg.active_config.betas),
    )
    # each group carries its own weight_decay, the optimizer default below is never used
    parameter_groups = [no_decay_group, decay_group]
    name = optimizer_cfg.optimizer_choice

    global_weight_decay_default = 0.0

    # pick correct optimizer
    if name == "adamw":
        optimizer_inst = torch.optim.AdamW(
            parameter_groups,
            lr=optimizer_cfg.lr,
            weight_decay=global_weight_decay_default,
        )
    else:
        raise ValueError(f"Chosen Optimizer {name} does not exist")
    return optimizer_inst
