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


def weight_decay_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
    lr: float,
    decouple_weight_decay_from_lr: bool = True,
) -> tuple[dict, dict]:
    """
    Split the trainable parameters of ``model`` into two optimizer parameter groups:
    one with weight decay, covering weights of all other linear layers and without weight day (everything else).
    The layers that are altered by weight decay are determined by ``get_weight_decay_parameter_names``.

    Pytorch Optimizers only update parameters that are passed to them, so both groups together
    must cover every trainable parameter of the model. Due to an issue before this is now asserted here.

    ``weight_decay`` is the lambda of the AdamW paper (Loshchilov & Hutter, Algorithm 2).
    The weights are shrunk by ``1 - scheduler_scale * weight_decay`` per step, i.e. the decay
    follows the LR scheduler but is independent of the base learning rate.

    The PyTorch only stores the product ``lr = base_lr * scheduler_scale`` and shrinks therefore by ``1 - lr * weight_decay``,
    which couples lambda to the base learning rate.
    Dividing by ``lr`` here undoes that coupling, so learning rate and weight decay can be scanned independently.

    Args:
        model (torch.nn.Module): The model whose parameters are grouped.
        weight_decay (float): Weight decay lambda in the sense of the AdamW paper, applied to the decay group.
        lr (float): Base learning rate passed to the optimizer, used to undo PyTorch's coupling.


    Returns:
        tuple[dict, dict]: Parameter groups in the format expected by ``torch.optim.Optimizer``:
            - no weight decay: ``{"params": [...], "weight_decay": 0.0}``
            - weight decay: ``{"params": [...], "weight_decay": weight_decay / lr}``

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

    # undo PyTorch's coupling of weight decay to the base learning rate (see docstring)
    pytorch_weight_decay = weight_decay
    if decouple_weight_decay_from_lr:
        pytorch_weight_decay = weight_decay / lr
    logger_inst.debug(f"\tWeight decay lambda {weight_decay} passed to PyTorch as {pytorch_weight_decay}")


    def _msg(names):
        return "\n".join(sorted(names))
    logger_inst.debug(f"Parameters with weight decay: \n{_msg(seen_decay_names)}\n")
    logger_inst.debug(
        f"Parameters without weight decay: \n"
        f"{_msg(sorted(name for name, p in model.named_parameters() if p.requires_grad and name not in decay_names))}"
    )

    no_decay_group = {"params": no_decay_parameters, "weight_decay": 0.0}
    decay_group = {"params": decay_parameters, "weight_decay": pytorch_weight_decay}
    return no_decay_group, decay_group
