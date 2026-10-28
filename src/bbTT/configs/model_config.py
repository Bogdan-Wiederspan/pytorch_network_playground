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


REPLACEMENT_TYPE_CHOICE = Tuple[Literal["raw", "std"], float]


@dataclass
class PaddingConfig:
    categorical_target_value: Optional[float] = None  # which categorical value is targeted by the padding
    categorical_masking_value: Optional[float] = -1  # value that is used to mask the categorical target value

    # the value that marks a missing continuous feature comes from DataConfig.dummy_values
    # 2 Mods exist:
    # raw: "htt_e": ("raw", 0.0) --> missing values are set to 0.0 GeV, after standardization value is not 0.0
    # std: "htt_e": ("std", -1.0) --> missing value are set to -1std, thus after standardization value is -1.0

    # my choice:
    # - physical: kinematics 0 is same as non-existing
    # - tags: 0 means its not, so something outside of range
    # - cov: cov00,11 are variance >=0, cov01,10 are mixture 0 is neutral
    continuous_replacement: dict[str, REPLACEMENT_TYPE_CHOICE] = field(
        default_factory=lambda: {
            "met_px": ("raw", 0.0), "met_py": ("raw", 0.0),
            "met_cov00": ("raw", 0.0), "met_cov01": ("raw", 0.0), "met_cov11": ("raw", 0.0),
            "vis_tau1_px": ("raw", 0.0),"vis_tau1_py": ("raw", 0.0),"vis_tau1_pz": ("raw", 0.0),"vis_tau1_e": ("raw", 0.0),
            "vis_tau2_px": ("raw", 0.0), "vis_tau2_py": ("raw", 0.0), "vis_tau2_pz": ("raw", 0.0), "vis_tau2_e": ("raw", 0.0),
            "bjet1_px": ("raw", 0.0), "bjet1_py": ("raw", 0.0), "bjet1_pz": ("raw", 0.0), "bjet1_e": ("raw", 0.0),
            "bjet1_tag_b": ("raw", -1.0), "bjet1_tag_cvsb": ("raw", -1.0), "bjet1_tag_cvsl": ("raw", -1.0), "bjet1_hhbtag": ("raw", -1.0),
            "bjet2_px": ("raw", 0.0), "bjet2_py": ("raw", 0.0), "bjet2_pz": ("raw", 0.0), "bjet2_e": ("raw", 0.0),
            "bjet2_tag_b": ("raw", -1.0), "bjet2_tag_cvsb": ("raw", -1.0), "bjet2_tag_cvsl": ("raw", -1.0), "bjet2_hhbtag": ("raw", -1.0),
            "fatjet_px": ("raw", 0.0), "fatjet_py": ("raw", 0.0), "fatjet_pz": ("raw", 0.0), "fatjet_e": ("raw", 0.0),
            "htt_e": ("raw", 0.0), "htt_px": ("raw", 0.0), "htt_py": ("raw", 0.0), "htt_pz": ("raw", 0.0),
            "hbb_e": ("raw", 0.0), "hbb_px": ("raw", 0.0), "hbb_py": ("raw", 0.0), "hbb_pz": ("raw", 0.0),
            "htthbb_e": ("raw", 0.0), "htthbb_px": ("raw", 0.0), "htthbb_py": ("raw", 0.0), "htthbb_pz": ("raw", 0.0),
            "httfatjet_e": ("raw", 0.0), "httfatjet_px": ("raw", 0.0), "httfatjet_py": ("raw", 0.0), "httfatjet_pz": ("raw", 0.0),
            "nu1_px": ("raw", 0.0), "nu1_py": ("raw", 0.0), "nu1_pz": ("raw", 0.0),
            "nu2_px": ("raw", 0.0), "nu2_py": ("raw", 0.0), "nu2_pz": ("raw", 0.0),
        }
    )  # noqa per-feature replacement of the marker with 2 modes: e.g. {"bjet1_e": ("raw", 0.0), "bjet1_tag_b": ("std", -1.0)}
    continuous_replacement_default: REPLACEMENT_TYPE_CHOICE = (
        "raw",
        0.0,
    )  # replacement for every continuous feature not listed above


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
    enable_categorical_padding: bool = False  # replace PaddingConfig.categorical_target_value in categorical inputs Marcel: False (by default categories are always filled)
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
