import torch

from bbTT.monitoring.logger.logger import get_logger
from bbTT.optimizer import weight_decay

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
    logger_inst.info("Initialize Optimizer")
    logger_inst.debug("Optimizer settings:"
        f"name: {full_config.optimizer_config.optimizer_choice}\n"
        f"decay-factor: {full_config.optimizer_config.active_config.decay_factor}\n"
        f"normalize: {full_config.optimizer_config.active_config.normalize,}\n"
        )
    no_decay_group, decay_group = weight_decay.normalized_weight_decay(
        model_inst,
        full_config.optimizer_config.active_config.decay_factor,
        full_config.optimizer_config.active_config.normalize,
    )
    # each group carries its own weight_decay, the optimizer default below is never used
    parameter_groups = [no_decay_group, decay_group]
    name = full_config.optimizer_config.optimizer_choice

    global_weight_decay_default = 0.0

    # pick correct optimizer
    if name == "adamw":
        optimizer_inst = torch.optim.AdamW(
            parameter_groups,
            lr=full_config.optimizer_config.active_config.lr,
            weight_decay=global_weight_decay_default,
        )
    else:
        raise ValueError(f"Chosen Optimizer {name} does not exist")
    return optimizer_inst


def init_scheduler(full_config, optimizer_inst):
    s_cfg = full_config.scheduler_config
    scheduler_instances = []
    logger_inst.info("Initialize Scheduler")
    for scheduler_cfg, scheduler_name in zip(s_cfg.config_chain, s_cfg.scheduler_cls_chain):
        logger_inst.debug(f"Name: {scheduler_name}, cfg: {scheduler_cfg}")
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler_instances.append(scheduler_cls(optimizer_inst, **scheduler_cfg))

    # sequential LR does not enable adaptive scheduler
    if len(scheduler_instances) == 1:
        scheduler_inst = scheduler_instances[0]
    else:
        scheduler_inst = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer_inst, schedulers=scheduler_instances, milestones=s_cfg.milestones
        )
    return scheduler_inst
