from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple

from utils.utils import EMPTY_FLOAT, choice_check

LAST_ACTIVATION_CHOICE = Literal["Softmax", "Sigmoid", None]

@dataclass
class RotationLayerConfig:
    enable_rotation: bool = False # turn off rotation, currently data is rotated in preprocessing by Marcel
    ref_phi_columns: Tuple[str, str] = ("vis_tau1", "vis_tau2") # reference column to calculate rotation angle
    rotate_columns: Tuple[str, ...] = (
        "bjet1",
        "bjet2",
        "fatjet",
        "vis_tau1",
        "vis_tau2",
    ) # which columns should be rotated

@dataclass
class PaddingConfig:
    enable_categorical_padding: bool = False
    categorical_target_value: Optional[float] = None # which categorical value is targeted by the padding
    categorical_masking_value: Optional[float] = -1 # value that is used to mask the categorical target value

    enable_continuous_padding: bool = False
    continuous_target_value: Optional[float] = None # which continuous value is targeted by the padding, if None no value is masked
    continuous_masking_value: Optional[float] = EMPTY_FLOAT # value that is used to mask the continuous target value

@dataclass
class EmbeddingConfig:
    tokenizer_add_unknown_category: Optional[None] = None # value added to categories to symbolize unknown category. If None, no extra category is added
    embedding_dim: int = 10 # dimension of embedding layer - Marcel 10

@dataclass
class DenseNetworkConfig:
    nodes: int = 128 # base number of nodes of the dense blocks, due to DenseNet connection, increases with more layers
    activation_functions: str = "elu" # string of the activation function of DenseBlocks - Marcel DNN: elu
    skip_connection_init: float = 1 # init value of the skip connection, 1 = exact copy
    freeze_skip_connection: bool = True # True = non-learnable skip connection value
    batch_norm_eps: float = 0.001 # epsilon denominator of batch norm - increase stability Marcel: 0.001
    last_activation_fn: LAST_ACTIVATION_CHOICE = "Softmax" # add activation function after last layer
    use_last_activation: bool = True # whether to use the last activation function, can be deactivated if not wanted - for example when using a loss function that already includes an activation like cross entropy, Marcel: False

@dataclass
class StandardizationConfig:
    mean: Optional[Any] = None # these values are determined by your data
    std: Optional[Any] = None

@dataclass
class LorentzBoostNetworkConfig:
    number_of_particles: int = 10 # number of particles of the lbn network Marcel: 10

@dataclass
class ModelConfig:
    enable_binning: bool = True # turn off binning layer, for example when using a model that does not support binning
    eps_batchnorm: float = 0.001 # epsilon of batch norm layers
    normalize_linear: bool = False # activate weight normalization of linear layer, TODO currently BUGGED, leave at False

    rotation_layer: RotationLayerConfig = field(default_factory=RotationLayerConfig)

    padding_layer: PaddingConfig = field(default_factory=PaddingConfig)

    embedding_layer: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    std_layer: StandardizationConfig = field(default_factory=StandardizationConfig)

    dense_network: DenseNetworkConfig = field(default_factory=DenseNetworkConfig)
    lbn_network: LorentzBoostNetworkConfig = field(default_factory=LorentzBoostNetworkConfig)


    def __post_init__(self):

        choice_check(self.dense_network.last_activation_fn, LAST_ACTIVATION_CHOICE)

        # currently parametrization and l2 maybe buggy
        if self.normalize_linear is True:
            raise ValueError("Norm of Linear Layer is currently buggy and is therefore disabled for now")
