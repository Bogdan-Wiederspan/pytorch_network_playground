from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils import parametrize

from bbTT.models.register import MODEL_REGISTRY
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)


class WeightNormalizedLinear(torch.nn.Linear):  # noqa: F811
    def __init__(self, *args, normalize=False, **kwargs):
        """
        If normalize is set to True, Linear layer is replaced by weight normalized layer as described in https://arxiv.org/abs/1602.07868.
        If false, the layer is a normal linear layer.

        WeightNormalizedLayer decouple the length of the weight vector from its direction.
        This is done by convert weights from Linear Layer to weight_original0 and 1.
        0 stands for the magnitude parameter g, while 1 is for the direction v.

        Args:
            normalize (bool, optional): True to replace Linear Layer with WeightNormalizedLayer. Defaults to True.
        """
        super().__init__(*args, **kwargs)
        if normalize:
            self = torch.nn.utils.parametrizations.weight_norm(self, name="weight", dim=0)


def linear_state(layer: torch.nn.Linear) -> dict[str, Any]:
    """
    Monitoring snapshot of a linear layer, torch layers have no state method of their own.

    For a weight normalized layer (torch parametrization) the norm of the direction parameter `v` is added;
    the magnitude `g` is skipped since its norm equals the weight norm.

    Args:
        layer (torch.nn.Linear): Linear layer, also subclasses and weight normalized layers.

    Returns:
        dict[str, Any]: "weight_norm" (float), "bias_norm" (float, only with bias), "direction_norm" (float,
            only when weight normalized); all Frobenius norms.
    """

    # layer.weight is the effective weight, also under a parametrization
    weight = layer.weight.detach()

    state: dict[str, Any] = {
        "weight_norm": torch.linalg.vector_norm(weight).item(),
        "weights": weight.flatten().cpu(),
    }
    if layer.bias is not None:
        state["bias_norm"] = torch.linalg.vector_norm(layer.bias.detach()).item()
    if parametrize.is_parametrized(layer, "weight"):
        # weight_norm stores g as original0 and v as original1
        direction = layer.parametrizations.weight.original1
        state["direction_norm"] = torch.linalg.vector_norm(direction.detach()).item()
    return state


def batch_norm_state(layer: torch.nn.BatchNorm1d) -> dict[str, Any]:
    """
    Monitoring snapshot of a batch norm layer, torch layers have no state method of their own.

    Args:
        layer (torch.nn.BatchNorm1d): Batch norm layer with running statistics.

    Returns:
        dict[str, Any]: "running_mean" and "running_var" as detached cpu tensors of shape (num_features,).
    """
    return {
        "running_mean": layer.running_mean.detach().cpu(),
        "running_var": layer.running_var.detach().cpu(),
    }

def embedding_state(layer: torch.nn.Embedding) -> dict[str, Any]:
    """
    Args:
        layer (torch.nn.Embedding): Embedding table of shape (num_embeddings, embedding_dim).

    Returns:
        dict[str, Any]: "weight_norm" (float) of the whole table and "token_norms" (torch.Tensor of shape
            (num_embeddings,)), the L2 norm of each token's embedding vector.
    """
    weight = layer.weight.detach()
    return {
        "weight_norm": torch.linalg.vector_norm(weight).item(),
        "token_norms": torch.linalg.vector_norm(weight, dim=1).cpu(),
    }

def optional_layer(layer: torch.nn.Module | None) -> torch.nn.Module:
    if layer is None:
        return torch.nn.Identity()
    return layer


def init_model(full_config):
    """
    Helper function to init model. Take a full config object and pick correct
    model from register and init it. Returns model instance.

    Args:
        full_config (_type_): FullConfig Instance, combining all config instances.

    Returns:
        _type_: _description_
    """
    model_choice = full_config.training_config.model_choice

    logger_inst.info(f"Initialize Model {model_choice}")

    model_cls = MODEL_REGISTRY[model_choice]
    model_inst = model_cls(full_config)

    # extra settings for specific models
    if model_choice in ("binned_lbn_dense",):
        model_inst.set_learning_mode("model_only")
    return model_inst
