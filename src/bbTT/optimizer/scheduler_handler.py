from __future__ import annotations

import logging

import torch

from bbTT.checkpoint.checkpoint import CheckPoint


class SchedulerHandler:
    def __init__(
        self,
        scheduler_inst: torch.optim.lr_scheduler.LRScheduler | None,
        checkpoint_inst: CheckPoint,
        logger_inst: logging.Logger,
    ):
        """
        Couples the scheduler to the checkpoint: a metric scheduler (ReduceLROnPlateau) is fed the confirmed best
        loss of the checkpoint instead of the raw validation loss, so checkpoint and scheduler share one definition
        of "improvement" and a single validation spike cannot trigger a learning rate drop. After a drop the model
        and optimizer are rolled back to the last confirmed checkpoint.

        Args:
            scheduler_inst (torch.optim.lr_scheduler.LRScheduler | None): Scheduler to drive, None disables stepping.
            checkpoint_inst (CheckPoint): Checkpoint handler, source of the confirmed best loss and the rollback state.
            logger_inst (logging.Logger): Logger for reload messages.
        """
        self.scheduler_inst = scheduler_inst
        self.checkpoint_inst = checkpoint_inst
        self.logger_inst = logger_inst
        self._needs_metric = isinstance(scheduler_inst, (torch.optim.lr_scheduler.ReduceLROnPlateau,))

    def _confirmed_metric(self, metric: float) -> float:
        """
        Args:
            metric (float): Raw validation loss of this evaluation.

        Returns:
            float: The checkpoint's confirmed best loss, or *metric* while no checkpoint has been confirmed yet.
        """
        best_loss = self.checkpoint_inst.best_loss
        return metric if best_loss is None else best_loss


    def step(
        self,
        model_inst: torch.nn.Module,
        optimizer_inst: torch.optim.Optimizer,
        metric: float = None,
    ) -> bool:
        """
        Call once per evaluation.
        There are two place where a step happens: Once per batch OR when a metric is evaluated.
        The latter is at the end of the training, while the former is direct after the optimizer step.
        The correct one is chosen depending on _needs_metric and chosen scheduler_inst.

        A scheduler step with a metric reloads the model when the learning rate dropped.
        The metric handed to the scheduler is the confirmed best loss, see ``_confirmed_metric``.

        Args:
            model_inst (torch.nn.Module): Passed ML model instance.
            optimizer_inst (torch.optim.Optimizer): Passed Optimizer instance.
            metric (float | None, optional): Raw validation loss, only for metric schedulers. Defaults to None.

        Returns (bool): If the handler did a step or not
        """

        if self.scheduler_inst is None:
            return False

        # check mutual exclusivity:
        # when a metric is given but scheduler does not need it
        # OR when a metric that needs a metric has none
        # do nothing in these cases
        metric_given = metric is not None
        if metric_given != self._needs_metric:
            return False

        # step for plateau scheduler is different
        previous_lr = optimizer_inst.param_groups[0]["lr"]
        if self._needs_metric:
            self.scheduler_inst.step(self._confirmed_metric(metric))
        else:
            self.scheduler_inst.step()
        current_lr = optimizer_inst.param_groups[0]["lr"]

        if previous_lr == current_lr:
            return False

        if metric_given:
            return self._reload_after_lr_drop(
                model_inst=model_inst,
                optimizer_inst=optimizer_inst,
                previous_lr=previous_lr,
                current_lr=current_lr,
            )
        return True

    def _reload_after_lr_drop(
        self,
        model_inst: torch.nn.Module,
        optimizer_inst: torch.optim.Optimizer,
        previous_lr: torch.Tensor,
        current_lr: torch.Tensor,
    ) -> bool:
        checkpoint = self.checkpoint_inst.last_checkpoint

        if checkpoint is None:
            return False

        self.logger_inst.info(
            f"{previous_lr} -> {current_lr}" + f"\nReload model and optimizer from iteration {checkpoint['iteration']}"
        )

        model_inst.load_state_dict(checkpoint["model_state_dict"])
        optimizer_inst.load_state_dict(checkpoint["optimizer_state_dict"])
        # since scheduler and optimizer are coupled
        # overwrite old lr in checkpoint with lr after scheduler step
        for group in optimizer_inst.param_groups:
            group["lr"] = current_lr

        return True
