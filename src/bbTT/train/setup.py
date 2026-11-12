from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import bbTT.data_handling.sampling.sampler as sampler
from bbTT.checkpoint.checkpoint import CheckPoint
from bbTT.data_handling.io import get_data
from bbTT.data_handling.preprocessing.k_fold import FoldAndSplitCoordinator
from bbTT.data_handling.preprocessing.standardization import FeatureStatisticCache
from bbTT.data_handling.sampling.weight import WeightAggregator
from bbTT.loss import init_loss
from bbTT.models.utils import init_model
from bbTT.monitoring import TrainingMonitor, setup_monitoring
from bbTT.monitoring.logger.logger import get_logger
from bbTT.optimizer.scheduler_handler import SchedulerHandler
from bbTT.optimizer.utils import init_optimizer, init_scheduler
from bbTT.train.loops import TrainingLoop, ValidationLoop
from bbTT.utils.utils import DEVICE

if TYPE_CHECKING:
    from bbTT.configs.full_config import FullConfig
    from bbTT.data_handling.sampling.sampler import ProcessSampler

logger_inst = get_logger(__name__)

# columns that are split per fold into training and validation
COLUMNS_TO_SPLIT = (
    "continuous",
    "categorical",
    "event_id",
    "normalization_weights",
    "product_of_weights",
    "evaluation_mask",
)


@dataclass
class FoldData:
    """
    Data of one fold, ready for training.

    Args:
        training_sampler (ProcessSampler): Sampler over the training split, owns the process weights.
        validation_sampler (ProcessSampler): Sampler over the validation split, process weights loaded from training.
        feature_statistic_cache (FeatureStatisticCache): Mean and std of the training split for standardization.
    """

    training_sampler: ProcessSampler
    validation_sampler: ProcessSampler
    feature_statistic_cache: FeatureStatisticCache


@dataclass
class TrainingComponents:
    """
    Everything that is trained or steers the training of one fold.

    Args:
        model_inst (torch.nn.Module): Model on DEVICE in train mode.
        training_loop (TrainingLoop): One optimization step over a batch.
        validation_loop (ValidationLoop): One full pass over a sampler without optimization.
        optimizer_inst (torch.optim.Optimizer): Optimizer bound to the model parameters.
        scheduler_inst (torch.optim.lr_scheduler.LRScheduler | None): Scheduler bound to the optimizer, None disables it.
        training_loss_inst (torch.nn.Module): Loss used for the optimization step.
        validation_loss_inst (torch.nn.Module): Loss used for the full passes.
        checkpoint_inst (CheckPoint): Decides on and writes checkpoints from the validation loss.
        scheduler_handler_inst (SchedulerHandler): Steps the scheduler and reloads the best state when needed.
        training_monitor_inst (TrainingMonitor): Captures tensors and gradients during the loops via hooks.
    """

    model_inst: torch.nn.Module
    training_loop: TrainingLoop
    validation_loop: ValidationLoop
    optimizer_inst: torch.optim.Optimizer
    scheduler_inst: torch.optim.lr_scheduler.LRScheduler | None
    training_loss_inst: torch.nn.Module
    validation_loss_inst: torch.nn.Module
    checkpoint_inst: CheckPoint
    scheduler_handler_inst: SchedulerHandler
    training_monitor_inst: TrainingMonitor


def prepare_fold_data(
    full_config: FullConfig,
    current_fold: int,
    rebuild_cache: bool,
    save_cache: bool,
) -> FoldData:
    """
    Load the events, split them for the fold and wrap the splits into samplers.

    The order inside is fixed by memory: the weight statistics need all events, so the aggregator
    runs before the split, and the split releases the full event dict on the fly.

    Args:
        full_config (FullConfig): Complete configuration of the run.
        current_fold (int): Fold whose test part is held out.
        rebuild_cache (bool): Rebuild the event cache instead of reading it.
        save_cache (bool): Write a rebuilt event cache to disk.

    Returns:
        FoldData: Samplers of the training and validation split and the feature statistics of the training split.
    """
    # events is of form : {uid : {"COLUMN_NAME": torch.Tensor}}
    events = get_data(full_config.dataset_config, rebuild_cache=rebuild_cache, save_cache=save_cache)

    fold_split_coordinator = FoldAndSplitCoordinator(
        events=events,
        c_fold=current_fold,
        k_fold=full_config.training_config.k_fold,
        seed=full_config.training_config.seed,
        training_percentage=full_config.training_config.train_ratio,
        randomize=full_config.training_config.randomize,
    )

    # weight statistics need all events, so this runs before the split consumes them
    weight_aggregator = WeightAggregator(events, fold_split_coordinator.indices)
    split_events = fold_split_coordinator.split_and_free(
        events, kinds=("training", "validation"), columns=COLUMNS_TO_SPLIT
    )
    train_events, validation_events = split_events["training"], split_events["validation"]
    del events

    logger_inst.info("Create Sampler")
    training_sampler = sampler.create_sampler(
        train_events, train=True, weight_aggregator_inst=weight_aggregator, full_config=full_config
    )
    del train_events
    validation_sampler = sampler.create_sampler(
        validation_events, train=False, weight_aggregator_inst=weight_aggregator, full_config=full_config
    )
    del validation_events

    training_sampler.print_epoch_statistics()
    validation_sampler.load_process_weights_from(training_sampler)

    feature_statistic_cache = FeatureStatisticCache(
        dataset_config=full_config.dataset_config,
        sampler=training_sampler,
        split_fingerprint=fold_split_coordinator.fingerprint(),
        return_dummy=full_config.debug_config.get_batch_statistic_return_dummy,
        verbose=True,
    )

    return FoldData(
        training_sampler=training_sampler,
        validation_sampler=validation_sampler,
        feature_statistic_cache=feature_statistic_cache,
    )


def build_training_components(
    full_config: FullConfig,
    current_fold: int,
    training_sampler: ProcessSampler,
) -> TrainingComponents:
    """
    Build model, optimizer, losses, scheduler, checkpoint and monitor hooks for one fold.

    Expects the standardization statistics to be set in `full_config` already, the model reads them.

    Args:
        full_config (FullConfig): Complete configuration of the run, with fold statistics applied.
        current_fold (int): Fold the checkpoint is named after.
        training_sampler (ProcessSampler): Needed by the loss initialisation for the class weights.

    Returns:
        TrainingComponents: All components wired together and ready for the loop.
    """
    model_inst = init_model(full_config=full_config).to(DEVICE).train()
    training_loop, validation_loop = TrainingLoop(full_config), ValidationLoop(full_config)

    optimizer_inst = init_optimizer(full_config=full_config, model_inst=model_inst)
    training_loss_inst, validation_loss_inst = init_loss(
        full_config=full_config, device=DEVICE, training_sampler=training_sampler
    )
    scheduler_inst = init_scheduler(full_config=full_config, optimizer_inst=optimizer_inst)
    checkpoint_inst = CheckPoint(
        checkpoint_name=full_config.training_config.save_model_name,
        checkpoint_fold=current_fold,
        patience=full_config.record_config.checkpoint_patience,
        delta=full_config.record_config.checkpoint_delta,
        confirm=full_config.record_config.checkpoint_confirm,
    )
    scheduler_handler_inst = SchedulerHandler(
        scheduler_inst=scheduler_inst, checkpoint_inst=checkpoint_inst, logger_inst=logger_inst
    )

    training_monitor_inst = TrainingMonitor(to_cpu=True, non_blocking=True)
    setup_monitoring(training_monitor_inst, model_inst, training_loss_inst, validation_loss_inst)

    return TrainingComponents(
        model_inst=model_inst,
        training_loop=training_loop,
        validation_loop=validation_loop,
        optimizer_inst=optimizer_inst,
        scheduler_inst=scheduler_inst,
        training_loss_inst=training_loss_inst,
        validation_loss_inst=validation_loss_inst,
        checkpoint_inst=checkpoint_inst,
        scheduler_handler_inst=scheduler_handler_inst,
        training_monitor_inst=training_monitor_inst,
    )
