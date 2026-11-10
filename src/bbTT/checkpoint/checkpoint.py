from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import torch

from bbTT.checkpoint.paths import checkpoint_path
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)


class CheckPoint:
    def __init__(
        self,
        checkpoint_name: str,
        checkpoint_fold: int | str,
        patience: int = 0,
        delta: float = 0.0,
        confirm: bool = True,
    ):
        """
        Writes the training state to disk whenever the validation loss improves and keeps the last written
        state in memory, so the scheduler can roll back to it after a learning rate drop.

        A checkpoint holds only data (state dicts and the config as dict), never model, optimizer or scheduler
        instances, so it loads with ``weights_only=True`` and survives renames in the code base. The model is
        rebuilt from the config, see ``bbTT.checkpoint.loading``.

        With *confirm* a new best loss is not accepted at once: the state is kept as candidate in memory, and only
        written if the following validation also beats the previous best. A single downward spike of the validation
        loss can therefore neither block later checkpoints nor become the state the scheduler rolls back to.
        On confirmation the better of candidate and confirming measurement is written.

        Args:
            checkpoint_name (str): Name of the training, defines the file name together with the fold.
            checkpoint_fold (int | str): Fold of the training.
            patience (int, optional): Number of criteria checks to wait before an improvement is accepted again. Defaults to 0.
            delta (float, optional): Minimum improvement of the loss to count as improvement. Defaults to 0.0.
            confirm (bool, optional): Require the next validation to confirm a new best before writing. Defaults to True.
        """
        self.fold = checkpoint_fold
        self.name = checkpoint_name

        self.patience = patience  # number of iteration before starting to look again
        self.delta = delta  # minimum threshold that needs to be overcome
        self.confirm = confirm

        self.best_loss: float | None = None
        self.last_checkpoint: dict[str, Any] | None = None  # confirmed state, identical to the file on disk
        self.candidate: dict[str, Any] | None = None  # unconfirmed state, waiting for the next validation

        self.no_improvement_count = 0

    @property
    def save_path(self) -> str:
        """
        Returns:
            str: Destination of the checkpoint file, see ``checkpoint_path``.
        """
        return checkpoint_path(self.name, self.fold)

    def _improves(self, loss: float) -> bool:
        """
        Args:
            loss (float): Current validation loss.

        Returns:
            bool: True if *loss* beats the confirmed best loss by more than *delta*, or no best exists yet.
        """
        return (self.best_loss is None) or (self.best_loss > (loss - self.delta))


    def update(
        self,
        loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        current_iteration: int,
        full_config: Any,
    ) -> bool:
        """
        Feed one validation result. Decides whether the current state becomes a candidate, confirms or discards a
        pending candidate, and writes the checkpoint when a best is confirmed.

        Args:
            loss (float): Current validation loss.
            model (torch.nn.Module): Model in its current state.
            optimizer (torch.optim.Optimizer): Optimizer in its current state.
            scheduler (torch.optim.lr_scheduler.LRScheduler): Scheduler in its current state.
            current_iteration (int): Training iteration the validation belongs to.
            full_config (Any): Full config dataclass, stored as dict.

        Returns:
            bool: True if a checkpoint was written in this call.
        """
        # when best loss is higher than current overwrite it
        if self.no_improvement_count < self.patience:
            self.no_improvement_count += 1
            return False

        improves = self._improves(loss)


        if not improves:
            # next loss is worse than the best, discard candidate if any
            if self.candidate is not None:
                logger_inst.info(
                    f"Checkpoint candidate from iteration {self.candidate['iteration']} "
                    f"({self.candidate['loss']:6E}) not confirmed by {loss:6E}, discarded"
                )
                self.candidate = None
            else:
                # no candidate, just log the lack of improvement
                logger_inst.info(
                    f"Checkpoint criteria not met 'best loss - delta': {self.best_loss}-{self.delta} - "
                    f"count/Patience: {self.no_improvement_count}/{self.patience}"
                )
            self.no_improvement_count += 1
            return False


        snapshot = self._snapshot(model, optimizer, scheduler, current_iteration, full_config)

        if self.confirm and self.candidate is None:
            # first measurement below the best: hold as candidate, decide with the next validation
            logger_inst.info(f"Checkpoint candidate at {loss:6E} (iteration {current_iteration}), awaiting confirmation")
            self.candidate = {"loss": loss, "iteration": current_iteration, "checkpoint": snapshot}
            return False

        # confirmed (or confirmation disabled): take the better of candidate and current state
        chosen_loss, chosen_iteration, chosen_checkpoint = loss, current_iteration, snapshot
        if self.candidate is not None and self.candidate["loss"] < loss:
            chosen_loss = self.candidate["loss"]
            chosen_iteration = self.candidate["iteration"]
            chosen_checkpoint = self.candidate["checkpoint"]
        self.candidate = None

        logger_inst.info(
            f"Checkpoint criteria trigger at {chosen_loss:6E} (iteration {chosen_iteration}) "
            f"- after {self.no_improvement_count} waiting epochs"
        )
        self.no_improvement_count = 0
        self.best_loss = chosen_loss
        self._write(chosen_checkpoint)
        return True


    def _snapshot(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        current_iteration: int,
        full_config: Any,
    ) -> dict[str, Any]:
        """
        Copy the training state. The state dicts are deep-copied, since a held snapshot must not alias the live parameters.

        Args:
            model (torch.nn.Module): Model to snapshot.
            optimizer (torch.optim.Optimizer): Optimizer to snapshot.
            scheduler (torch.optim.lr_scheduler.LRScheduler): Scheduler to snapshot.
            current_iteration (int): Training iteration the snapshot belongs to.
            full_config (Any): Full config dataclass, stored as dict.

        Returns:
            dict[str, Any]: Checkpoint dictionary, only data.
        """
        return {
            "iteration": current_iteration,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
            "full_config": asdict(full_config),
        }


    def _write(self, checkpoint: dict[str, Any]) -> None:
        """
        Keep *checkpoint* as ``last_checkpoint`` and write it to ``save_path``.

        Args:
            checkpoint (dict[str, Any]): Checkpoint dictionary as returned by ``_snapshot``.
        """
        self.last_checkpoint = checkpoint
        torch.save(self.last_checkpoint, self.save_path)
