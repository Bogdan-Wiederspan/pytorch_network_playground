from __future__ import annotations

from collections.abc import Iterable

import torch

from bbTT.data_handling.sampling.sampler import ProcessSampler
from bbTT.monitoring.logger.logger import get_logger
from bbTT.utils.utils import CPU_DEVICE

logger_inst = get_logger(__name__)

def clip_gradients(parameters: Iterable[torch.nn.Parameter], clip_value: float = 1.0):
    for p in parameters:
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

def calibrate_batch_norms(
    model_inst: torch.nn.Module,
    sampler: ProcessSampler,
    sample_columns: tuple[str, ...],
    num_batches: int,
    device: torch.device = CPU_DEVICE,
) -> None:
    """
    Fill the running statistics of every BatchNorm with real data before the first optimizer step.

    A fresh BatchNorm starts with running mean 0 and variance 1 and moves towards the data statistics
    only by an exponential moving average, so the first full passes in eval mode normalize with
    numbers that do not fit the data at all. This is severe for the LBN batch norm, which sees
    unstandardized GeV scales. The calibration runs *num_batches* forward passes in train mode without
    gradients and without an optimizer step; the momentum of every BatchNorm is set to None meanwhile,
    which turns the running statistics into the exact cumulative average over all seen batches:

        mu_k = mu_{k-1} + (mu_batch - mu_{k-1}) / k

    The sampler is restored to its state before the calibration, so the first training batch is the
    same with and without it. Parameters are not touched. Does nothing for num_batches <= 0.

    Args:
        model_inst (torch.nn.Module): Model whose BatchNorm modules are calibrated.
        sampler (ProcessSampler): Training sampler, batches are drawn with its sample_batch.
        sample_columns (tuple[str, ...]): Columns to sample, same as in the training loop.
        num_batches (int): Number of batches the statistics are averaged over.
        device (torch.device, optional): Device the batches are moved to. Defaults to CPU_DEVICE.
    """
    if num_batches <= 0:
        return

    batch_norms = [m for m in model_inst.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not batch_norms:
        logger_inst.info("No BatchNorm in the model, skipping calibration")
        return

    logger_inst.info(f"Calibrating {len(batch_norms)} BatchNorm layers over {num_batches} batches")
    momenta = [bn.momentum for bn in batch_norms]
    for bn in batch_norms:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average instead of EMA

    sampler_state = sampler.state_dict()
    was_training = model_inst.training
    model_inst.train()  # BatchNorm updates its statistics only in train mode
    with torch.no_grad():
        for _ in range(num_batches):
            events = sampler.sample_batch(sample_from=sample_columns, device=device)
            model_inst(categorical_inputs=events.get("categorical"), continuous_inputs=events.get("continuous"))

    model_inst.train(was_training)
    sampler.load_state_dict(sampler_state)
    for bn, momentum in zip(batch_norms, momenta):
        bn.momentum = momentum
