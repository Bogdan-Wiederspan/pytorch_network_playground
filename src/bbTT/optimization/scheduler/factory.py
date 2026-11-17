from __future__ import annotations

import torch

from bbTT.monitoring.logger.logger import get_logger
from bbTT.optimization.scheduler import schedulers
from bbTT.optimization.scheduler.warmup import WARMUP_REGISTRY, WarmupScheduler

logger_inst = get_logger(__name__)

def resolve_scheduler_cls(name: str) -> type[torch.optim.lr_scheduler.LRScheduler]:
    """
    Find a scheduler class by its name.
    Own custom schedulers take precedence over torch's, thus a naming overlap is possible.

    Args:
        name (str): Class name as given in ``SCHEDULER_REGISTRY`` in ``scheduler_config.py``.

    Returns:
        type[torch.optim.lr_scheduler.LRScheduler]: The scheduler class.

    Raises:
        ValueError: If neither ``schedulers`` nor ``torch.optim.lr_scheduler`` has a class of that name.
    """
    scheduler_cls = getattr(schedulers, name, None) or getattr(torch.optim.lr_scheduler, name, None)
    if scheduler_cls is None:
        raise ValueError(f"Unknown scheduler class {name}")
    return scheduler_cls


def init_scheduler(full_config, optimizer_inst: torch.optim.Optimizer) -> WarmupScheduler:
    """
    Build the main scheduler chosen in the config and wrap it with the configured warm-up.

    Args:
        full_config: Merged config, ``scheduler_config`` is read.
        optimizer_inst (torch.optim.Optimizer): Optimizer whose learning rate is scheduled.

    Returns:
        WarmupScheduler: Warm-up followed by the main scheduler.
    """
    s_cfg = full_config.scheduler_config
    logger_inst.info("Initialize Scheduler")

    logger_inst.debug(f"Name: {s_cfg.scheduler_cls}, cfg: {s_cfg.scheduler_kwargs}")
    logger_inst.debug(
        f"Warm-up: {s_cfg.warmup_choice}, iterations: {s_cfg.warmup_iterations}, start factor: {s_cfg.warmup_start_factor}"
    )

    # the main scheduler must exist before the warm-up, see WarmupScheduler
    main_inst = resolve_scheduler_cls(s_cfg.scheduler_cls)(optimizer_inst, **s_cfg.scheduler_kwargs)

    warmup_fn = None if s_cfg.warmup_choice is None else WARMUP_REGISTRY[s_cfg.warmup_choice]
    return WarmupScheduler(
        optimizer_inst=optimizer_inst,
        main_inst=main_inst,
        warmup_fn=warmup_fn,
        warmup_iterations=s_cfg.warmup_iterations,
        start_factor=s_cfg.warmup_start_factor,
    )
