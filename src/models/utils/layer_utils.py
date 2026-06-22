from __future__ import annotations

import torch

from ..register import MODEL_REGISTRY
from ..preprocessing import EmptyLayer

class WeightNormalizedLinear(torch.nn.Linear):  # noqa: F811
    def __init__(self ,*args, normalize=False, **kwargs):
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
            self = torch.nn.utils.parametrizations.weight_norm(self, name='weight', dim=0)

def optional_layer(layer: torch.nn.Module | None) -> torch.nn.Module:
    if layer is None:
        return torch.nn.Identity()
    return layer

def dummy_empty(condition, layer: torch.nn.Module | None) -> torch.nn.Module:
    if condition:
        return EmptyLayer()
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

    model_cls = MODEL_REGISTRY[model_choice]
    model_inst = model_cls(full_config)

    # extra settings for specific models
    if model_choice in ("binned_lbn_dense",):
        model_inst.set_learning_mode("model_only")
    return model_inst
