import dataclasses
import os
import pathlib
from typing import Literal

import torch

from bbTT.configs.full_config import FullConfig
from bbTT.evaluation.rebuild_dataclasses import from_dict
from bbTT.models.register import MODEL_REGISTRY
from bbTT.utils.utils import CPU_DEVICE


def rebuild_dataclass_from_dict(full_cfg: dict[dict]) -> dataclasses.dataclass:
    """
    Pickle of dataclasses relies on the dataclass being available. To prevent conversion to dict is used.
    This comes with a lost of information (like post_inits being not serialized).
    To at least reuse code that relies on accessing dataclasses, this converts the dictionary back to a dataclass.
    When *full_cfg* is already a dataclass nothing will be done and it will be returned.

    Args:
        full_cfg (dict[dict]): Dictionary of dictionary, where first level describes a dataclass, while second one are the normal values.

    Returns:
        dataclasses.dataclass: A dataclass with dataclasses as attributes
    """
    if dataclasses.is_dataclass(full_cfg):
        return full_cfg

    # create sub-configs
    sub_dataclasses = {}
    for cls_cfg, sub_cfg in full_cfg.items():
        sub_dataclasses[cls_cfg] = (dataclasses.make_dataclass(cls_cfg, [(k, type(v)) for k,v in sub_cfg.items()])(**sub_cfg))

    # create full config that hosts all sub-configs
    full_config = dataclasses.make_dataclass("full_config", [(dataclass, type(dataclass)) for dataclass in sub_dataclasses])(**sub_dataclasses)
    return full_config


def resolve_checkpoint_path(model, suffix=None):
    p = pathlib.Path(os.environ["MODELS_DIR"])
    p = (p / model)
    if suffix is not None:
        p = p.with_suffix(suffix)
    return p

def load_checkpoint(path):
    return torch.load(path, map_location=CPU_DEVICE, weights_only=False)


def rebuild_checkpoint_information(path) -> tuple[torch.nn.Module, Literal["DataClass"]]:
    """
    Load checkpoint instance and rebuild model and full config instance from this checkpoint.

    Args:
        path (str): Path to checkpoint

    Returns:
        tuple(torch.nn.Module, Literal[&quot;DataClass&quot;]): Tuple of model instance and an dataclass instance
    """
    checkpoint_path = resolve_checkpoint_path(path)
    checkpoint_inst = load_checkpoint(checkpoint_path)

    reconstructed_full_config = from_dict(FullConfig, checkpoint_inst["full_config"])
    model_inst = rebuild_model_from_checkpoint(checkpoint_inst, cfg_dataclass=reconstructed_full_config)
    return model_inst, reconstructed_full_config


def rebuild_model_from_checkpoint(checkpoint: dict, cfg_dataclass: FullConfig) -> torch.nn.Module:
    """
    Load checkpoint file from *path* and reconstruct correct model instance and load weight into model

    Args:
        path (pathlib.Path): Path to checkpoint file, typically with .pt suffix

    Returns:
        _type_: _description_
    """
    # when instance is saved load this
    # otherwise rebuild model from module and class name and load state dict
    model_choice = cfg_dataclass.training_config.model_choice
    model_cls = MODEL_REGISTRY[model_choice] # pick correct cls from registered models
    model_inst = model_cls(cfg_dataclass) # create new instance with config
    model_inst.load_state_dict(checkpoint["model_state_dict"])
    model_inst.eval()
    return model_inst
