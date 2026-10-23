from __future__ import annotations

import torch

from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)


def get_weight_decay_parameter_names(model: torch.nn.Module) -> set[str]:
    """
    Collect the full names of all parameters that should receive weight decay.

    Only weights of linear layers are decayed, with two exceptions:
    the classification layer ``last_linear`` is excluded and for weight-normalized
    linear layers only the magnitude is decayed, not the direction. Decaying the
    direction as well would shrink the whole weight instead of only its norm.
    Biases are never decayed.

    Args:
        model (torch.nn.Module): The model whose parameters are inspected.

    Returns:
        set[str]: Full parameter names as produced by ``model.named_parameters()``,
            e.g. ``"dense_block_1.linear.weight"``.
    """
    # see https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/2
    decay_names: set[str] = set()
    for module_name, module in model.named_modules():
        is_linear_layer = isinstance(module, torch.nn.Linear)
        is_not_last_layer = module_name != "last_linear"  # last linear layer is the classification node
        if not (is_linear_layer and is_not_last_layer):
            continue

        # recurse=True is required: a weight-normalized linear layer keeps its parameters
        # in the child module `parametrizations.weight`, not in the linear layer itself
        for parameter_name, _ in module.named_parameters():
            if parameter_name == "bias":
                continue
            if torch.nn.utils.parametrize.is_parametrized(module):
                # weight-normalized linear layers have 2 parameters: magnitude (original0) and
                # direction (original1); only the magnitude is decayed
                if parameter_name != "parametrizations.weight.original0":
                    continue
            # the root module has an empty name, named_parameters() then yields the bare parameter name
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            decay_names.add(full_name)
    return decay_names


def normalized_weight_decay(
    model: torch.nn.Module,
    decay_factor: float = 1e-1,
    normalize_decay_factor: bool = True,
) -> tuple[dict, dict]:
    """
    Split the trainable parameters of ``model`` into two optimizer parameter groups:
    one without weight decay (biases, normalization layers, embeddings, custom ``nn.Parameter``,
    the classification layer and the direction of weight-normalized layers) and one with
    weight decay (weights of all other linear layers, see ``get_weight_decay_parameter_names``).

    Pytorch Optimizers only update parameters that are passed to them, so both groups together
    must cover every trainable parameter of the model. This is asserted here.

    Args:
        model (torch.nn.Module): The model whose parameters are grouped.
        decay_factor (float): The weight decay factor applied to the decay group.
        normalize_decay_factor (bool): If True, the decay factor is divided by the number of
            parameters in the decay group. This keeps the total L2 penalty of comparable size
            for big and small models.

    Returns:
        tuple[dict, dict]: Parameter groups in the format expected by ``torch.optim.Optimizer``:
            - no weight decay: ``{"params": [...], "weight_decay": 0.0}``
            - weight decay: ``{"params": [...], "weight_decay": decay_factor}``
    """
    decay_names = get_weight_decay_parameter_names(model)

    # single pass over all trainable parameters, split by name membership
    decay_parameters: list[torch.nn.Parameter] = []
    no_decay_parameters: list[torch.nn.Parameter] = []
    seen_decay_names: set[str] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in decay_names:
            decay_parameters.append(parameter)
            seen_decay_names.add(name)
        else:
            no_decay_parameters.append(parameter)

    # sanity checks:
    # both checks ensure that something is not counted doubled or is missing
    # 1) every collected decay name must exist
    missing_names = decay_names - seen_decay_names
    if missing_names:
        raise RuntimeError(f"Weight decay parameter names not found in model.named_parameters(): {missing_names}")
    # 2)  both groups must cover the models parameter
    num_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    num_grouped = sum(parameter.numel() for parameter in decay_parameters + no_decay_parameters)
    if num_grouped != num_trainable:
        raise RuntimeError(
            f"Optimizer parameter groups cover {num_grouped} parameters, "
            f"but the model has {num_trainable} trainable parameters."
        )

    # decouple lambda choice from number of parameters, by normalizing the decay factor
    num_weight_decay_params = sum(parameter.numel() for parameter in decay_parameters)
    if normalize_decay_factor:
        decay_factor = decay_factor / num_weight_decay_params
        logger_inst.debug(f"\tNormalize weight decay factor by number of parameters to: {decay_factor}")

    def _msg(names):
        return "\n".join(sorted(names))
    logger_inst.debug(f"Parameters with weight decay: \n{_msg(seen_decay_names)}\n")
    logger_inst.debug(
        f"Parameters without weight decay: \n"
        f"{_msg(sorted(name for name, p in model.named_parameters() if p.requires_grad and name not in decay_names))}"
    )

    no_decay_group = {"params": no_decay_parameters, "weight_decay": 0.0}
    decay_group = {"params": decay_parameters, "weight_decay": decay_factor}
    return no_decay_group, decay_group
