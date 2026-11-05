from __future__ import annotations

import os
import pathlib

CHECKPOINT_SUFFIX = ".pt"


def models_dir() -> pathlib.Path:
    """
    Returns:
        pathlib.Path: Directory where checkpoints live, from the ``MODELS_DIR`` environment variable.
    """
    return pathlib.Path(os.environ["MODELS_DIR"])


def checkpoint_path(name: str, fold: int | str) -> pathlib.Path:
    """
    The one place that defines where a checkpoint of a training is stored.
    Multiple places in the codebase consume a checkpoint path. This function centralizes the logic of how to construct a checkpoint path.

    Args:
        name (str): Name of the training, e.g. ``training_config.save_model_name``.
        fold (int | str): Fold the checkpoint belongs to.

    Returns:
        pathlib.Path: ``<MODELS_DIR>/<name>_fold<fold>.pt``
    """
    return (models_dir() / f"{name}_fold{fold}").with_suffix(CHECKPOINT_SUFFIX)
