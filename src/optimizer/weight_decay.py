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
    ### PROBLEM: https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/2
    # weight filtering after modules is bad

    with_weight_decay = []
    no_weight_decay = []

    # from IPython import embed; embed(header="MESSAGE Line 38 | File: weight_decay.py")
    # filter linear layers out to separate task into code chunks
    for module_name, module in model.named_modules():
        # recurse false is necessary with out the same parameters are iterated multiple times
        for parameter_name, parameter in module.named_parameters(recurse=False):
            # when linear and not last linear layer add weight decay
            is_linear = isinstance(module, torch.nn.Linear)
            is_not_last_linear = "last_linear" not in module_name
            is_weight = "weight" in parameter_name
            not_parametrized = torch.nn.utils.parametrize.is_parametrized(module)

            if is_linear and is_not_last_linear and is_weight:
                with_weight_decay.append((parameter_name, parameter))
                logger_inst.debug(f"add weight decay - parameter: {parameter_name} of layer: {module_name}\n")
            else:
                no_weight_decay.append((parameter_name, parameter))
    # decouple lambda choice from number of parameters, by normalizing the decay factor
    num_weight_decay_params = sum([len(weight.flatten()) for name, weight in with_weight_decay])
    if normalize_decay_factor:
        decay_factor = decay_factor / num_weight_decay_params
        logger_inst.debug(f"\tNormalize weight decay by number of parameters: {decay_factor}")

    return {
        "no_weight_decay_params" :{"params": no_weight_decay, "weight_decay": 0.0},
        "weight_decay_params":{"params": with_weight_decay, "weight_decay": decay_factor}
            }


# def init_optimizer(optimizer, optimizer_config) -> None:
#     no_weight_decay_param, weight_decay_param = prepare_weight_decay(optimizer_config)
#     optimizer = optimizer(
#         (no_weight_decay_param, weight_decay_param),
#         lr=optimizer_config.learning_rate,
#     )
