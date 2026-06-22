import dataclasses
import os
import pathlib
from typing import Literal

import torch

from models.register import MODEL_REGISTRY

from .utils import CPU_DEVICE


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
    full_config = rebuild_dataclass_from_dict(checkpoint_inst["full_config"])
    model_inst = rebuild_model_from_checkpoint(checkpoint_path)
    return model_inst, full_config


def rebuild_model_from_checkpoint(path: pathlib.Path) -> torch.nn.Module:
    """
    Load checkpoint file from *path* and reconstruct correct model instance and load weight into model

    Args:
        path (pathlib.Path): Path to checkpoint file, typically with .pt suffix

    Returns:
        _type_: _description_
    """
    # load always on cpu
    checkpoint = torch.load(str(path), weights_only=False, map_location="cpu")

    # when instance is saved load this
    # otherwise rebuild model from module and class name and load state dict
    if "model_inst" in checkpoint:
        model_inst = checkpoint["model_inst"]
    else:
        full_cfg = checkpoint["full_config"]

        model_choice = full_cfg.training_config.model_choice
        model_cls = MODEL_REGISTRY[model_choice] # pick correct cls from registered models
        model_inst = model_cls(full_cfg) # create new instance with config
        model_inst.load_state_dict(checkpoint["model_state_dict"])
    model_inst.eval()
    return model_inst
