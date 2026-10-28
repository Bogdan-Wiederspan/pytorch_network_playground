from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple


@dataclass
class RotationLayerConfig:
    ref_phi_columns: Tuple[str, str] = ("vis_tau1", "vis_tau2")  # reference column to calculate rotation angle
    rotate_columns: Tuple[str, ...] = (
        "bjet1",
        "bjet2",
        "fatjet",
        "vis_tau1",
        "vis_tau2",
    )  # which columns should be rotated



# replacement of a missing continuous value: ("raw", v) is v in units of the raw feature,
# ("std", k) is mean + k * std of that feature, resolved when the model is built
REPLACEMENT_TYPE_CHOICE = Tuple[Literal["raw", "std"], float]

@dataclass
class PaddingConfig:
    categorical_target_value: Optional[float] = None  # which categorical value is targeted by the padding
    categorical_masking_value: Optional[float] = -1  # value that is used to mask the categorical target value

    # the value that marks a missing continuous feature comes from DataConfig.dummy_values
    continuous_replacement: dict[str, REPLACEMENT_TYPE_CHOICE] = field(default_factory=dict)  # noqa per-feature replacement of the marker with 2 modes: e.g. {"bjet1_e": ("raw", 0.0), "bjet1_tag_b": ("std", -1.0)}
    continuous_replacement_default: REPLACEMENT_TYPE_CHOICE = ("raw", 0.0)  # replacement for every continuous feature not listed above


@dataclass
class EmbeddingConfig:
    tokenizer_add_unknown_category: Optional[None] = None  # noqa value added to categories to symbolize unknown category. If None, no extra category is added
    embedding_dim: int = 10  # dimension of embedding layer - Marcel 10


@dataclass
class DenseNetworkConfig:
    nodes: int = 128  # base number of nodes of the dense blocks, due to DenseNet connection, increases with more layers
    activation_functions: str = "elu"  # string of the activation function of DenseBlocks - Marcel DNN: elu
    skip_connection_init: float = 1  # init value of the skip connection, 1 = exact copy
    freeze_skip_connection: bool = True  # True = non-learnable skip connection value
    batch_norm_eps: float = 0.001  # epsilon denominator of batch norm - increase stability Marcel: 0.001

    normalize_linear: bool = False  # noqa activate weight normalization of linear layer, TODO currently BUGGED, leave at False


@dataclass
class StandardizationConfig:
    mean: Optional[Any] = None  # these values are determined by your data
    std: Optional[Any] = None


@dataclass
class LorentzBoostNetworkConfig:
    number_of_particles: int = 10  # number of particles of the lbn network Marcel: 10
    weight_init_scale = 1.0
    clip_weights = False
    eps = 1.0e-5
    safe_sqrt_eps = 1.0e-5


@dataclass
class ModelConfig:
    enable_categorical_padding: bool = False # replace PaddingConfig.categorical_target_value in categorical inputs Marcel: False (by default categories are always filled)
    enable_continuous_padding: bool = True  # replace DataConfig dummy markers in continuous inputs, see PaddingConfig

    enable_rotation: bool = False  # turn off rotation, currently data is rotated in preprocessing by Marcel
    enable_binning: bool = True  # turn off binning layer, for example when using a model that does not support binning

    # needs to be false for CrossEntropy and True for SignalEfficiency
    use_last_activation: bool = False  # use last activation defined inside model. If false logits are returned Marcel: False

    rotation: RotationLayerConfig = field(default_factory=RotationLayerConfig)

    padding: PaddingConfig = field(default_factory=PaddingConfig)

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    standardization: StandardizationConfig = field(default_factory=StandardizationConfig)

    dense_network: DenseNetworkConfig = field(default_factory=DenseNetworkConfig)
    lbn_network: LorentzBoostNetworkConfig = field(default_factory=LorentzBoostNetworkConfig)

    def __post_init__(self):
        # currently parametrization and l2 maybe buggy
        if self.dense_network.normalize_linear is True:
            raise ValueError("Norm of Linear Layer is currently buggy and is therefore disabled for now")
