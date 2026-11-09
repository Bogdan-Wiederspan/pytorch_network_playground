from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecordConfig:
    batch_metrics: bool = False  # if batch metrics should be tracked, this is only for debugging and not used in production
    log_metrics: bool = True  # log metrics to tensorboard during training
    verbose_interval: int = 5  # interval between two logger training status print outs
    validation_interval: int = 200  # interval between two validation passes / plots are done during validation

    # Checkpoint settings, evaluated once per validation pass
    # A checkpoint is written when the validation loss beats the best confirmed loss by more than `checkpoint_delta`.
    # The best confirmed loss is also the metric a ReduceLROnPlateau scheduler sees (see SchedulerHandler), so this
    # is the finer of the two thresholds: the scheduler's `threshold` decides on top of it what still counts as
    # progress. Keep checkpoint_delta <= scheduler threshold, otherwise checkpoints stop before the scheduler reacts.
    checkpoint_delta: float = 0.0  # minimum improvement of the validation loss to count as new best
    checkpoint_patience: int = 0  # number of validations to skip after a written checkpoint before looking again
    # Hold a new best as candidate and write it only if the next validation also beats the previous best.
    # Protects checkpoint and scheduler from a single validation spike; costs one validation_interval of delay.
    checkpoint_confirm: bool = True

    # full pass over the training data is pure monitoring but costs ~4x a validation pass.
    # as compromise, the evaluation of training data is only done every `training_evaluation_interval` validation passes.
    training_evaluation_interval: int = 5 # 1 means that every validation pass evaluates the training data, 2 means every second validation pass, etc.
