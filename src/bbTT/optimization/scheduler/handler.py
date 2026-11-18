from __future__ import annotations

import logging

import torch

from bbTT.checkpoint.checkpoint import CheckPoint
from bbTT.optimization.scheduler.warmup import WarmupScheduler


class SchedulerHandler:
    def __init__(
        self,
        scheduler_inst: WarmupScheduler | None,
        checkpoint_inst: CheckPoint,
        logger_inst: logging.Logger,
    ):
        """
        Couples the scheduler to the checkpoint: a metric-driven main scheduler (ReduceLROnPlateau) is fed the
        confirmed best loss of the checkpoint instead of the raw validation loss, so checkpoint and scheduler share
        one definition of "improvement" and a single validation spike cannot trigger a learning rate drop. After a
        drop the model and optimizer are rolled back to the last confirmed checkpoint.

        Which phase (warm-up / main) and which kind of step applies is decided by the WarmupScheduler itself, the
        handler only provides the two entry points ``step_batch`` and ``step_metric``.

        Args:
            scheduler_inst (WarmupScheduler | None): Scheduler to drive, None disables stepping.
            checkpoint_inst (CheckPoint): Checkpoint handler, source of the confirmed best loss and the rollback state.
            logger_inst (logging.Logger): Logger for reload messages.
        """
        self.scheduler_inst = scheduler_inst
        self.checkpoint_inst = checkpoint_inst
        self.logger_inst = logger_inst

    def _confirmed_metric(self, metric: float) -> float:
        """
        Args:
            metric (float): Raw validation loss of this evaluation.

        Returns:
            float: The checkpoint's confirmed best loss, or *metric* while no checkpoint has been confirmed yet.
        """
        best_loss = self.checkpoint_inst.best_loss
        return metric if best_loss is None else best_loss

    def step_batch(self) -> None:
        """
        Call once after every optimizer step. Learning rate changes here are the normal course of the schedule,
        no rollback happens.
        """
        if self.scheduler_inst is None:
            return
        self.scheduler_inst.step_batch()

    def step_metric(
        self,
        model_inst: torch.nn.Module,
        optimizer_inst: torch.optim.Optimizer,
        metric: float,
    ) -> bool:
        """
        Call once per validation. The metric handed to the scheduler is the confirmed best loss, see
        ``_confirmed_metric``. When the step lowered the learning rate the model is reloaded.

        Args:
            model_inst (torch.nn.Module): Passed ML model instance.
            optimizer_inst (torch.optim.Optimizer): Passed Optimizer instance.
            metric (float): Raw validation loss.

        Returns:
            bool: Whether the model was rolled back after a learning rate drop.

        """

        if self.scheduler_inst is None:
            return False

        previous_lr = optimizer_inst.param_groups[0]["lr"]
        stepped = self.scheduler_inst.step_metric(self._confirmed_metric(metric))
        if not stepped:
            return False
        current_lr = optimizer_inst.param_groups[0]["lr"]

        if previous_lr == current_lr:
            return False

        return self._reload_after_lr_drop(
            model_inst=model_inst,
            optimizer_inst=optimizer_inst,
            previous_lr=previous_lr,
            current_lr=current_lr,
        )

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
