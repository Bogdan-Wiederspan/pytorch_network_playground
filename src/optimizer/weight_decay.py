from __future__ import annotations

import torch

from utils import logger

logger_inst = logger.get_logger(__name__)

def normalized_weight_decay(
    model: torch.nn.Module,
    decay_factor: float = 1e-1,
    normalize_decay_factor: bool = True,
) -> tuple[dict, dict]:
    """
    Weight decay should only be applied to the linear layers or convolutional layers.
    All other layers should not have weight decay applied.
    Pytorch Optimizers will apply weight decay to all parameters that have `requires_grad=True`.
    This function can be overwritten by specify the parameters that should have weight decay applied.

    Args:
        model (torch.nn.Module): The model to apply weight decay to.
        decay_factor (float): The weight decay factor to apply to the parameters.
        normalize (bool): If True, the decay factor is normalized by the number of parameters.
            This ensures that the end L2 loss is about the same size for big and small models.

    Returns:

        tuple: A tup    le containing two dictionaries where:
            - The first dictionary contains parameters that should not have weight decay applied.
            - The second dictionary contains parameters that should have weight decay applied.
    """
    def get_l2_layer_parameters(model):
        ### PROBLEM: https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/2
        with_weight_decay = []
        for name, module in model.named_modules():
            # first filter the layers correctly:
            # only apply on linear layer, but not last layer
            is_linear_layer = isinstance(module, torch.nn.Linear)
            is_not_last_layer = (name != "last_linear") # last linear layer is the classification node

            # add only specific parameters
            # special case for parametrization:
            if (is_linear_layer & is_not_last_layer):
                for parameter_name, parameter in module.named_parameters():
                    if parameter_name == "bias":
                        continue
                    if torch.nn.utils.parametrize.is_parametrized(module):
                    # parametrization of linear layers have 2 parameters - magnitude and direction
                    # since both coming from an operation they are now non-leaf tensors (which is not allowed for parameter)
                    # TODO: currently only using magnitude, maybe wrong?
                        # magnitude
                        if parameter_name == "parametrizations.weight.original0":
                            with_weight_decay.append(parameter)
                        # directions
                        elif parameter_name == "parametrizations.weight.original1":
                            continue
                    else:
                        # the normal linear layer case with
                        with_weight_decay.append(parameter)
        return with_weight_decay

    weight_decay_parameters = get_l2_layer_parameters(model=model)

    # decouple lambda choice from number of parameters, by normalizing the decay factor
    num_weight_decay_params = sum([len(weight.flatten()) for weight in weight_decay_parameters])
    if normalize_decay_factor:
        decay_factor = decay_factor / num_weight_decay_params
        logger_inst.debug(f"\tNormalize weight decay factor by number of parameters to: {decay_factor}")

    return {
        "weight_decay_params":{"params": weight_decay_parameters, "weight_decay": decay_factor}
            }


# def init_optimizer(optimizer, optimizer_config) -> None:
#     no_weight_decay_param, weight_decay_param = prepare_weight_decay(optimizer_config)
#     optimizer = optimizer(
#         (no_weight_decay_param, weight_decay_param),
#         lr=optimizer_config.learning_rate,
#     )
