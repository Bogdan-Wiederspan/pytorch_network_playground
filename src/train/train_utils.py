from __future__ import annotations

from collections.abc import Iterable

import torch

from train import plotting
from utils import logger

logger_inst = logger.get_logger(__name__)

def do_scheduler_step(
    loss: torch.tensor,
    logger_inst: logger.logging.Logger,
    scheduler_inst: torch.optim.lr_scheduler,
    model_inst: torch.nn.Module,
    optimizer_inst: torch.optim.Optimizer,
    checkpoint_inst: dict[torch.tensor]) -> None:
    """
    Perform with given *scheduler_inst* a scheduler step, under the conditions of the given *scheduler_inst*.
    If a step is performed be vebose using *logger_inst* and reload the last best state for *model_inst* and *optimizer_inst* from
    *checkpoint_inst*.

    Args:
        loss (torch.tensor): The current validation loss.
        logger_inst (logger.logging.Logger): Instance of the used Logger
        scheduler_inst (torch.optim.lr_scheduler): Instance of a Learning Rate Scheduler
        model_inst (torch.nn.Module): Instance of the Model
        optimizer_inst (torch.optim.Optimizer): Instance of an Optimizer
        checkpoint_inst (dict[torch.tensor]): Dictionary containing the learning rate, model state dict and optimizer state dict.
    """
    previous_lr =  optimizer_inst.param_groups[0]["lr"]
    scheduler_inst.step(loss)
    current_lr = optimizer_inst.param_groups[0]["lr"]
    if previous_lr != current_lr:
        logger_inst.info(
            f"{previous_lr} -> {current_lr}\n"
            f"Reload model and optimizer from iteration "
            )
        model_inst.load_state_dict(checkpoint_inst.last_checkpoint["model_state_dict"])
        optimizer_inst.load_state_dict(checkpoint_inst.last_checkpoint["optimizer_state_dict"])

def log_metrics(
    tensorboard_inst,
    iteration_step: int,
    sampler_output: tuple[torch.tensor],
    target_map: dict,
    mode: str ="train",
    exist_check: bool = False,
    **data: dict[torch.tensor]
    ) -> None:
    """
    Helper function to store all logging efforts in given *tensorboard_inst*.
    Each log is saved under a given *iteration_step*, with a given *mode* prefix.
    The *sampler_output* is a tuple the inputs and output of the model.
    The *target_map* is a dict defining the mapping of a node to an index.
    Data is an dictionary with arbitrary tensors to log, what data is used for is defined within the function.
    A plot in data is only created when respective entry exist. If one does want to be strict in checking turn on
    "exist_check", then a check if fields exist in data is perform every time.

    Args:
        tensorboard_inst (_type_): Active tensorboard instance
        iteration_step (int): Current iteration step
        sampler_output (tuple[torch.tensor]): tuple of tensors, defining prediction, target and weights.
        target_map (dict): Mapping of output node to index.
        mode (str, optional): Suffix for name, typically "train" oder "validation". Defaults to "train".
    """
    def _optional(*keys, log_name=None) -> bool:
        """
        Helper function to enable optional logs only when all *keys* exist.
        Typical case: A log describe a model specific information like binnings of learnable edges.
        Only specific models have this information, thus this should not be logged if it is not prevalent.

        Returns:
            bool: True if all keys exist, else False
        """
        missing_keys = [k for k in keys if k not in data.keys()]
        if any(missing_keys):
            if exist_check:
                # name = keys if log_name is None else log_name # TODO what is this for?
                logger_inst.warning(f"Can't produce log of - {log_name} - missing: {missing_keys}")
            return False
        return True

    logger_inst.info(f"Start {mode} Logs at iteration {iteration_step}")
    # logs and plots that are ALWAYS plotted

    # outputs of network
    pred, tar, weights = sampler_output

    # log crossentropy as metric
    # weights are EVENT WEIGHTS, but cross entropy want to have cls weights
    # for this reason reduce is set to false and manual average is calculated
    cce_metric = torch.nn.functional.cross_entropy(pred, tar, weight=None, reduction="none")
    weights = weights.reshape(cce_metric.shape)
    cce_metric = torch.mean(cce_metric * weights)
    tensorboard_inst.log_scalar(values={mode: cce_metric}, step=iteration_step, name=f"{mode} CrossEntropy")

    # network prediction of all nodes
    pred_fig, pred_ax = plotting.network_predictions(
        tar,
        pred,
        target_map,
        normalize=True,
    )
    tensorboard_inst.log_figure(f"{mode} node output", pred_fig, step=iteration_step)

    # confusion matrix plot
    c_mat_fig, c_mat_ax, c_mat = plotting.confusion_matrix(
        tar,
        pred,
        target_map,
        sample_weight=weights,
        normalized="true"
    )
    tensorboard_inst.log_figure(f"{mode} confusion matrix", c_mat_fig, step=iteration_step)

    roc_fig, roc_ax = plotting.roc_curve(
        tar,
        pred,
        sample_weight=weights,
        labels=list(target_map.keys())
    )
    tensorboard_inst.log_figure(f"{mode} roc curve one vs rest", roc_fig, step=iteration_step)

    if _optional("dataset_id", log_name="ProcessID Loss"):
        uids = data["dataset_id"]

        _ids = {
        "hh": (21101,),
        "tt":(1100,1200,1300), # groups are possible, and mixes are allowed
        "dy":(51667, 51683, 51664, 51680, 51720, 51723, 51726,
        51729, 51732, 51735, 51674, 51690, 51665, 51681,
        51661, 51677, 51670, 51671, 51672, 51673, 51675,
        51686, 51687, 51688, 51689, 51691, 51666, 51682,
        51699, 51702, 51705, 51708, 51711, 51714, 51693,
        51668, 51684, 51663, 51679,),
        }

        to_filter = []
        for _id in _ids.values():
            to_filter.extend(_id)

        for _id in to_filter:
            mask = (uids == _id).flatten()
            masked_pred = pred[mask]
            masked_tar = tar[mask]
            masked_weight = weights[mask]

            cce_pid_metric = torch.nn.functional.cross_entropy(
                masked_pred,
                masked_tar,
                weight=None,
                reduction="none"
                )

            cce_pid_weights = masked_weight.reshape(cce_pid_metric.shape)
            cce_pid_weighted_metric = torch.mean(cce_pid_metric * cce_pid_weights)
            tensorboard_inst.log_scalar(
                values={str(_id): cce_pid_weighted_metric},
                step=iteration_step,
                name=f"{mode} - PID CrossEntropy")


    if _optional("loss", log_name="Loss"):
        tensorboard_inst.log_loss({mode: data["loss"]}, step=iteration_step)

    if _optional("lr", log_name="Learning Rate"):
        tensorboard_inst.log_lr(data["lr"], step=iteration_step)

    if _optional("binning_edges", log_name="HH Node Prediction"):
    # binned network prediction is only available by models with binned activation layer
        pred_fig, pred_ax = plotting.network_predictions_hh(
            tar,
            pred,
            target_map,
            normalize=True,
            binning_edges=data["binning_edges"],
            current_iteration=iteration_step,
        )
        tensorboard_inst.log_figure(f"{mode} HH node output", pred_fig, step=iteration_step)

    if _optional("kernels", log_name="Lernable Bin Edges"):
        kernel_fig, kernel_ax = plotting.visualize_bins(data["kernels"])
        tensorboard_inst.log_figure("bin edges", kernel_fig, step=iteration_step)

def clip_gradients(parameters: Iterable[torch.nn.Parameter], clip_value: float = 1.0):
    for p in parameters:
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
