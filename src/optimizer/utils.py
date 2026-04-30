import torch
from optimizer import weight_decay
from optimizer.SAM import SAM

def init_optimizer(full_config, model_inst):
    # only linear layers contribute to weight decay, prepare config that separates them for the optimizer

    weight_decay_parameters = weight_decay.normalized_weight_decay(
        model_inst,
        full_config.optimizer_config.decay_factor,
        full_config.optimizer_config.normalize,
        )
    # is a dictionary, expect list of tuples
    weight_decay_parameters = list(weight_decay_parameters.values())
    name = full_config.optimizer_config.optimizer_choice

    if name == "adamw":
        optimizer_inst = torch.optim.AdamW(
            weight_decay_parameters,
            lr=full_config.optimizer_config.lr
            )
    elif name == "sam":
        optimizer_inst = SAM(
            weight_decay_parameters,
            torch.optim.AdamW,
            lr=full_config.optimizer_config.lr,
            rho = 2.0,
            adaptive=True
        )
    else:
        raise ValueError(f"Chosen Optimizer {name} does not exist")
    return optimizer_inst
