from __future__ import annotations

import pathlib
from typing import Any

import torch

from bbTT.checkpoint.paths import models_dir
from bbTT.configs.full_config import FullConfig
from bbTT.evaluation.rebuild_dataclasses import from_dict
from bbTT.models.register import MODEL_REGISTRY
from bbTT.utils.utils import CPU_DEVICE


def resolve_checkpoint_path(model: str | pathlib.Path, suffix: str | None = None) -> pathlib.Path:
    """
    Args:
        model (str | pathlib.Path): Checkpoint name or path, relative names are looked up in ``MODELS_DIR``.
        suffix (str | None, optional): Suffix to enforce, e.g. ".pt". Defaults to None.

    Returns:
        pathlib.Path: Path to the checkpoint file.
    """
    p = models_dir() / model
    if suffix is not None:
        p = p.with_suffix(suffix)
    return p


def load_checkpoint(path: pathlib.Path) -> dict[str, Any]:
    """
    Load a checkpoint written by ``CheckPoint.create_checkpoint``. Since the checkpoint holds only data,
    ``weights_only=True`` is enforced: a checkpoint that needs code objects to unpickle is rejected.

    Args:
        path (pathlib.Path): Path to the checkpoint file.

    Returns:
        dict[str, Any]: The checkpoint dictionary, tensors mapped to CPU.

    Raises:
        FileNotFoundError: If *path* does not exist, the message lists the available checkpoints next to it.
    """
    if not path.exists():
        available_checkpoints = "\n".join(list(sorted([p.stem for p in path.parent.glob("*.pt")])))
        msg = f"Checkpoint file {path} does not exist.\n Available checkpoints: \n{available_checkpoints}"
        raise FileNotFoundError(msg)
    return torch.load(path, map_location=CPU_DEVICE, weights_only=True)


def rebuild_checkpoint_information(path: str | pathlib.Path) -> tuple[torch.nn.Module, FullConfig]:
    """
    Load a checkpoint and rebuild model and full config from it.

    Args:
        path (str | pathlib.Path): Checkpoint name or path, see ``resolve_checkpoint_path``.

    Returns:
        tuple[torch.nn.Module, FullConfig]: Model in eval mode with the checkpoint weights, and the reconstructed config.
    """
    checkpoint_path = resolve_checkpoint_path(path)
    checkpoint_inst = load_checkpoint(checkpoint_path)

    # checkpoints from before physics parameters (pNN) existed have no such field, these networks were
    # trained without physics parameters. Not setting it leaves the attribute missing, since from_dict skips it
    checkpoint_inst["full_config"]["dataset_config"].setdefault("physics_parameters", {})

    reconstructed_full_config = from_dict(FullConfig, checkpoint_inst["full_config"])
    model_inst = rebuild_model_from_checkpoint(checkpoint_inst, cfg_dataclass=reconstructed_full_config)
    return model_inst, reconstructed_full_config


def rebuild_model_from_checkpoint(checkpoint: dict[str, Any], cfg_dataclass: FullConfig) -> torch.nn.Module:
    """
    Build the model from the registry entry named in the config and load the checkpoint weights.

    Args:
        checkpoint (dict[str, Any]): Checkpoint dictionary as returned by ``load_checkpoint``.
        cfg_dataclass (FullConfig): Config the model is built from, ``training_config.model_choice`` picks the architecture.

    Returns:
        torch.nn.Module: Model instance in eval mode.
    """
    model_choice = cfg_dataclass.training_config.model_choice
    model_cls = MODEL_REGISTRY[model_choice]  # pick correct cls from registered models
    model_inst = model_cls(cfg_dataclass)  # create new instance with config
    model_inst.load_state_dict(checkpoint["model_state_dict"])
    model_inst.eval()
    return model_inst
