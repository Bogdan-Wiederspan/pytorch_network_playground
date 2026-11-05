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
        verbose: bool = True,
    ):
        """
        Writes the training state to disk whenever the validation loss improves and keeps the last written
        state in memory, so the scheduler can roll back to it after a learning rate drop.

        A checkpoint holds only data (state dicts and the config as dict), never model, optimizer or scheduler
        instances, so it loads with ``weights_only=True`` and survives renames in the code base. The model is
        rebuilt from the config, see ``bbTT.checkpoint.loading``.

        Args:
            checkpoint_name (str): Name of the training, defines the file name together with the fold.
            checkpoint_fold (int | str): Fold of the training.
            patience (int, optional): Number of criteria checks to wait before an improvement is accepted again. Defaults to 0.
            delta (float, optional): Minimum improvement of the loss to count as improvement. Defaults to 0.0.
            verbose (bool, optional): Unused, kept for compatibility. Defaults to True.
        """
        self.fold = checkpoint_fold
        self.name = checkpoint_name

        self.patience = patience  # number of iteration before starting to look again
        self.delta = delta  # minimum threshold that needs to be overcome
        self.verbose = verbose

        self.best_loss: float | None = None
        self.last_checkpoint: dict[str, Any] | None = None
        self.no_improvement_count = 0

    @property
    def save_path(self) -> str:
        """
        Returns:
            str: Destination of the checkpoint file, see ``checkpoint_path``.
        """
        return str(checkpoint_path(self.name, self.fold))

    def check_criteria(self, loss: float) -> bool:
        """
        Decide whether the current *loss* is a new best and a checkpoint should be written.

        Args:
            loss (float): Current validation loss.

        Returns:
            bool: True if *loss* improves on the best loss by more than *delta* and the patience is over.
        """
        # when best loss is higher than current overwrite it
        if self.no_improvement_count >= self.patience:
            if (self.best_loss is None) or (self.best_loss > (loss - self.delta)):
                logger_inst.info(
                    f"Checkpoint criteria trigger started at {loss:6E} - after {self.no_improvement_count} waiting epochs"
                )
                self.no_improvement_count = 0
                self.best_loss = loss
                return True
            logger_inst.info(
                f"Checkpoint criteria not met 'best loss - delta': {self.best_loss}-{self.delta} - "
                f"count/Patience: {self.no_improvement_count}/{self.patience}"
            )

        self.no_improvement_count += 1
        return False

    def create_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        current_iteration: int,
        full_config: Any,
    ) -> None:
        """
        Snapshot the training state, keep it as ``last_checkpoint`` and write it to ``save_path``.
        The state dicts are deep-copied, since ``last_checkpoint`` must not alias the live parameters.

        Args:
            model (torch.nn.Module): Model to snapshot.
            optimizer (torch.optim.Optimizer): Optimizer to snapshot.
            scheduler (torch.optim.lr_scheduler.LRScheduler): Scheduler to snapshot.
            current_iteration (int): Training iteration the snapshot belongs to.
            full_config (Any): Full config dataclass, stored as dict.
        """
        checkpoint = {
            "iteration": current_iteration,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
            "full_config": asdict(full_config),
        }

        self.last_checkpoint = checkpoint
        torch.save(self.last_checkpoint, self.save_path)
