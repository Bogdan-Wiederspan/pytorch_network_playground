from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Any

import torch

from data import features, load_data
from utils.utils import choice_check

# some values only accept certain values, not runtime check is performed!
ERAS_CHOICE = Literal["22pre", "22post", "23pre", "23post"]
LAST_ACTIVATION_CHOICE = Literal["Softmax", "Sigmoid", None]
KERNEL_CHOICE = Literal["GaussianKernelV3"]
OPTIMIZER_CHOICE = Literal["adamw", "sam"]

MODEL_CHOICE = Literal["binned_lbn", "bnet_lbn"]
LOSS_CHOICE = Literal["cross_entropy", "signal_efficiency", "signal_efficiency_binning_aware"]
TRAINING_LOOP_CHOICE = Literal["default", "sam"]
VALIDATION_LOOP_CHOICE = Literal["signal_efficiency"]
SIGNAL_EFFICIENCY_LOSS_MODE = Literal["full", "no_unc", "approximation"]



@dataclass
class DataConfig:
    # changes in this config will create a NEW hash of the data
    target_map: Dict[str, int] = field(default_factory=lambda: {"hh": 0, "tt": 1, "dy": 2}) # node: index
    continuous_features: List[str] = features.continuous_features
    categorical_features: List[str] = features.categorical_features
    dataset_pattern: Tuple[str] = (
        "dy_*",
        "tt_*",
        "hh_ggf_hbb_htt_kl1_kt1*",
        # "hh_ggf_hbb_htt_kl0_kt1*",
        )
    eras: Tuple[ERAS_CHOICE] = ("22pre")
    datasets: Optional[List[str]] = None
    cuts: Optional[Any] = None

    def __post_init__(self):
        # a dictionary of all files corresponding to a certain dataset
        self.datasets = load_data.find_datasets(self.dataset_pattern, self.eras, file_type="root", verbose=False)
    # TODO HASH ?



@dataclass
class ModelBuildingConfig:
    # Rotation Layer
    enable_rotation: bool = False # enable rotation layer based on two reference objects, TODO: this breaks export to torch script
    ref_phi_columns: Tuple[str, str] = ("res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2") # reference column to calculate rotation angle
    rotate_columns: Tuple[str, ...] = (
        "res_dnn_pnet_bjet1",
        "res_dnn_pnet_bjet2",
        "res_dnn_pnet_fatjet",
        "res_dnn_pnet_vis_tau1",
        "res_dnn_pnet_vis_tau2",
    ) # which columns should be rotated
    # Padding Layers
    categorical_padding_value: Optional[float] = None # missing values in categorical inputs are replaced by this
    continuous_padding_value: Optional[float] = None # missing values in continuous inputs are replaced by this

    # DenseNet/LBN config
    nodes: int = 128 # base number of nodes of the dense blocks, due to DenseNet connection, increases with more layers
    activation_functions: str = "elu" # string of the activation function of DenseBlocks - Marcel DNN: elu
    skip_connection_init: float = 1 # init value of the skip connection, 1 = exact copy
    freeze_skip_connection: bool = True # True = non-learnable skip connection value
    batch_norm_eps: float = 0.001 # epsilon denominator of batch norm - increase stability Marcel: 0.001
    LBN_M: int = 10 # number of particles of the lbn network Marcel: 10
    last_activation_fn: LAST_ACTIVATION_CHOICE = "Softmax" # add activation function after last layer
    #STD Layers
    mean: Optional[Any] = None
    std: Optional[Any] = None

    embedding_dim: int = 10 # dimension of embedding layer - Marcel 10
    expected_embedding_inputs: Optional[dict[Tuple[int]]] = field(init=False)

    eps_batchnorm: int = 0.001 # epsilon of batch norm layers
    normalize_linear: bool = False # activate weight normalization of linear layer, TODO currently BUGGED, leave at False


    def __post_init__(self):
        self.expected_embedding_inputs = features.expected_embedding_inputs()
        choice_check(self.last_activation_fn, LAST_ACTIVATION_CHOICE)


@dataclass
class BinningConfig:
    num_bins: int = 25
    lower_edge: float = 0.0
    upper_edge: float = 1.0
    binning_fn: Any = torch.linspace  # keep as a callable, if needed
    kernel_cls: KERNEL_CHOICE = "GaussianKernelV3"
    kernel_config: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "GaussianKernelV3": {
                "abs_mode": False, # all values are interpreted as absolute values, recommended to be relative
                "smoothing_width": 0.3, # width of gaussian where it goes from 100% to 10%
                "left_notch": 0.1, # shift of gaussian into linear part from left
                "right_notch": 0.1, # shift of gaussian into linear part from right
            }
        }
    )
    def __post_init__(self):
        choice_check(self.kernel_cls, KERNEL_CHOICE)

@dataclass
class TrainingConfig:
    model_choice: MODEL_CHOICE = "bnet_lbn"
    loss_fn: LOSS_CHOICE = "signal_efficiency_binning_aware"
    max_train_iteration: int = 60000 # max number of batches
    verbose_interval: int = 5 # interval between two logger outputs of training loss
    validation_interval: int = 100 # interval between two validation passes / plots are done during validation
    gamma: float = 0.5
    label_smoothing: float = 0.0
    train_folds: Tuple[int, ...] = (0,) # which training folds to use
    k_fold: int = 5
    seed: int = 100 # set torch and numpy seed for reproducibility
    train_ratio: float = 0.75 # split ratio for k-fold data into train and validation
    t_batch_size: int = 4096 * 10
    v_batch_size: int = -1 # validation batch size, -1 = full set,
    save_model_name: str = "test_binning" # name of the model used to save
    training_fn: TRAINING_LOOP_CHOICE = "default" # name of the training and validation loop function
    validation_fn: VALIDATION_LOOP_CHOICE = "signal_efficiency"
    loss_mode: SIGNAL_EFFICIENCY_LOSS_MODE = "no_unc" # only relevant if signal_efficiency loss is chosen, determines which formula is used for the asimov calculation
    loss_uncertainty: float = 0.0 # # only relevant if signal_efficiency loss is chosen, determines the background uncertainty used in the asimov calculation

    # Sampler Settings
    sample_ratio: Dict[str, float] = field(default_factory=lambda:{"dy": 1 / 3, "tt": 1 / 3, "hh": 1 / 3})
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
        choice_check(self.loss_mode, SIGNAL_EFFICIENCY_LOSS_MODE)
        choice_check(self.loss_fn, LOSS_CHOICE)
        choice_check(self.model_choice, MODEL_CHOICE)



@dataclass
class SchedulerConfig:
    # TODO if wanna have multiple scheduler types create nested dataclass configs
    patience: int = 4 # wait x number of checks - Marcel: 10
    min_delta: float = 0.0 # minimum improvement before increase patience - Marcel: 0
    threshold_mode: str = "abs" # type of min_delta - Marcel: abs
    factor: float = 0.5 # LR reduce by factor

@dataclass
class OptimizerConfig:
    optimizer_choice: OPTIMIZER_CHOICE = "adamw"

    @dataclass
    class ADAMWConfig():
        decay_factor: float = 500 # factor of L2 - Marcel: 500
        normalize: bool = True # normalize weight decay factor to number of parameters
        lr: float = 1e-3 # start learning rate

    @dataclass
    class SAMConfig():
        TODO = None

    def __post_init__(self):
        # move optimizer attribute to top level
        if self.optimizer_choice == "adamw":
            choice_config = self.ADAMWConfig()
        elif self.optimizer_choice == "sam":
            choice_config = self.SAMConfig()

        for _key, _attr in choice_config.__dict__.items():
            setattr(self, _key, _attr)

        choice_check(self.optimizer_choice, OPTIMIZER_CHOICE)


@dataclass
class DebugConfig:
    get_batch_statistic_return_dummy: bool = False # do not calculate preprocessing, instead use mean = 0, std = 1
    load_marcel_stats: bool = False # load Marcels mean and std from preprocessing into STD Layer
    load_marcel_weights: bool = False # preload Marcels Pytorch Model weights, HINT: will break now, since architecture changes


# Combination of all Configs so only 1 import is necessary
@dataclass
class FullConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_building_config: ModelBuildingConfig = field(default_factory=ModelBuildingConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    binning_config: BinningConfig = field(default_factory=BinningConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    debug_config: DebugConfig = field(default_factory=DebugConfig)

    def __post_init__(self):
        # when using optimizer sam specific training routine needs to be used
        if (self.training_config.training_fn != "sam") & (self.optimizer_config.optimizer_choice=="sam"):
            raise ValueError(f"When using Optimizer SAM, training_fn needs to be set to 'sam', is currently:{self.training_config.training_fn}).")


full_config = FullConfig()
