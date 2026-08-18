from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple

from bbTT.utils.utils import choice_check, multiply_sub_process_rates

MODEL_CHOICE = Literal["residual", "dense", "lbn_dense", "binned_lbn_dense"]
TRAINING_LOOP_CHOICE = Literal["cross_entropy", "sam", "signal_efficiency"]
VALIDATION_LOOP_CHOICE = Literal["signal_efficiency", "cross_entropy"]


@dataclass
class TrainingConfig:
    save_model_name: str = "delete2" # name of the model used to save
    log_metrics: bool = True # whether to log metrics to tensorboard during training, if false only validation loss is logged
    model_choice: MODEL_CHOICE = "lbn_dense"
    training_fn: TRAINING_LOOP_CHOICE = "cross_entropy" # name of the training loop
    validation_fn: VALIDATION_LOOP_CHOICE = "cross_entropy" # name of the validation loop
    max_train_iteration: int = 15000 # max number of batches
    verbose_interval: int = 5 # interval between two logger outputs of training loss
    validation_interval: int = 30 # interval between two validation passes / plots are done during validation
    gamma: float = 0.5
    label_smoothing: float = 0.0
    train_folds: Tuple[int, ...] = (0,) # which training folds to use
    k_fold: int = 5
    seed: int = 100 # set torch and numpy seed for reproducibility
    train_ratio: float = 0.75 # split ratio for k-fold data into train and validation
    t_batch_size: int = 4096 * 10
    v_batch_size: int = -1 # validation batch size, -1 = full set,

    # Sampler Settings
    sample_ratio: Dict[str, float] = field(default_factory=lambda:{"dy": 1 / 4, "tt": 1 / 4, "hh": 1 / 2}) # decide the ratio of tt, dy and hh within a batch
    sub_process_ratios: Dict[str, float] = field(default_factory=lambda:{ # decide the
        # HINT: each rate is multiplied together to final rate, e.g. if process id exist 2x with rate 2, final rate is 4
        "signal":{}, # empty categorizes are set to 1 by default
        "tt":{(1100,1200):1, 1300:1}, # groups are possible, and mixes are allowed
        "dy":{51667: 1, 51683: 1, 51664: 1, 51680: 1, 51720: 1, 51723: 1, 51726: 1,
        51729: 1, 51732: 1, 51735: 1, 51674: 1, 51690: 1, 51665: 1, 51681: 1,
        51661: 1, 51677: 1, 51670: 1, 51671: 1, 51672: 1, 51673: 1, 51675: 1,
        51686: 1, 51687: 1, 51688: 1, 51689: 1, 51691: 1, 51666: 1, 51682: 1,
        51699: 1, 51702: 1, 51705: 1, 51708: 1, 51711: 1, 51714: 1, 51693: 1,
        51668: 1, 51684: 1, 51663: 1, 51679: 1,
        },
    })
    use_sub_process_ratios: Tuple[str] = ("signal", "tt", "dy")
    sample_attributes: Tuple[str, ...] = (
        "continuous",
        "categorical",
        "targets",
        "product_of_weights",
        "evaluation_space_mask",
    ) # what to sample from sampler
    min_events_in_batch: int = 1 # sampler setting: minimal number of events of a subprocess in a batch

    def __post_init__(self):
        necessary_field = ("continuous", "categorical", "targets")
        if not set(necessary_field).issubset(set(self.sample_attributes)):
            raise ValueError(f"Sample Attributes need to contain: {necessary_field}" )
        choice_check(self.training_fn, TRAINING_LOOP_CHOICE)
        choice_check(self.validation_fn, VALIDATION_LOOP_CHOICE)
        choice_check(self.model_choice, MODEL_CHOICE)
        self.sub_process_ratios = multiply_sub_process_rates(self.use_sub_process_ratios, self.sub_process_ratios)
