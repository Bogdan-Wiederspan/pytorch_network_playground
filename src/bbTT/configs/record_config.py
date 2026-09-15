
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecordConfig:
    batch_metrics: bool = True # if batch metrics should be tracked
    log_metrics: bool = True # whether to log metrics to tensorboard during training, if false only validation loss is logged
    verbose_interval: int = 5 # interval between two logger outputs of training loss
    validation_interval: int = 30 # interval between two validation passes / plots are done during validation
