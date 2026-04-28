from __future__ import annotations

import torch
from utils import logger

logger_inst = logger.get_logger(__name__)

def normalized_weight_decay(
    model: torch.nn.Module,
    decay_factor: float = 1e-1,
    normalize: bool = True,
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

        tuple: A tuple containing two dictionaries where:
            - The first dictionary contains parameters that should not have weight decay applied.
            - The second dictionary contains parameters that should have weight decay applied.
    """
    ### PROBLEM: https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/2
    # weight filtering after modules is bad

    with_weight_decay = []
    no_weight_decay = []
    linear_layers = []
    non_linear_layers = []

    # filter linear layers out to separate task into code chunks
    for module_tuple in model.named_modules():
        module_name, module = module_tuple
        if (isinstance(module, torch.nn.Linear)) and ("last_linear" not in module_name):
            linear_layers.append(module_tuple)
        else:
            non_linear_layers.append(module_tuple)

    # for linear layer original weight and bias are always accessible with weight and bias
    # even when parametrization created other parameters (see weight_norm as example)

    for module_name, module in linear_layers:
        # linear layer always have weight and bias
        # parametrization renaming is avoided and original weight is taken
        # from IPython import embed; embed(header="MESSAGE Line 50 | File: optimizer.py")
        for parameter_name, parameter in module.named_parameters():
            with_weight_decay.append(parameter)
        # with_weight_decay.append(module.bias)
        # no_weight_decay.append(module.bias)
        logger_inst.debug(f"add weight - name{module_name} -> module:{module}\n")

    # for non linear layers any parameters are collected
    for module_name, module in non_linear_layers:
        for parameter_name, parameter in module.named_parameters():
            no_weight_decay.append(parameter)

    a = {"no_weight_decay_params" : {"params": no_weight_decay, "weight_decay": 0.0}, "weight_decay_params" : {"params": with_weight_decay, "weight_decay": decay_factor}}

    optimizer_inst = torch.optim.AdamW(list(a.values()), lr=1e-5)
    from IPython import embed; embed(header="MESSAGE Line 60 | File: optimizer.py")
    # decouple lambda choice from number of parameters, by normalizing the decay factor
    num_weight_decay_params = sum([len(weight.flatten()) for weight in with_weight_decay])
    if normalize:
        decay_factor = decay_factor / num_weight_decay_params
        logger_inst.debug(f"\tNormalize weight decay by number of parameters: {decay_factor}")
    return {"params": no_weight_decay, "weight_decay": 0.0}, {"params": with_weight_decay, "weight_decay": decay_factor}

def prepare_weight_decay(model, optimizer_config) -> None:
    # define which layers should contribute to the weight decay
    no_weight_decay_parameters, weight_decay_parameters = normalized_weight_decay(
        model,
        decay_factor=optimizer_config.decay_factor,
        normalize=optimizer_config.normalize,
    )
    return {"no_weight_decay_params": no_weight_decay_parameters, "weight_decay_params": weight_decay_parameters}

def init_optimizer(optimizer, optimizer_config) -> None:
    no_weight_decay_param, weight_decay_param = prepare_weight_decay(optimizer_config)
    optimizer = optimizer(
        (no_weight_decay_param, weight_decay_param),
        lr=optimizer_config.learning_rate,
    )
