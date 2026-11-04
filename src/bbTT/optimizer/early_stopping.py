from __future__ import annotations

import copy
import os
import pathlib
from dataclasses import asdict

import torch

from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)

class CheckPoint:
    def __init__(self, checkpoint_name, checkpoint_fold, patience=0, delta=0, verbose=True):
        self.fold = checkpoint_fold
        self.name = checkpoint_name

        self.patience = patience  # number of iteration before starting to look again
        self.delta = delta  # minimum threshold that needs to be overcome
        self.verbose = verbose

        self.best_loss = None
        self.last_checkpoint = None
        self.no_improvement_count = 0

    @property
    def save_path(self):
        base_dir = os.environ["MODELS_DIR"]
        dst = (pathlib.Path(base_dir) / f"{self.name}_fold{self.fold}").with_suffix(".pt")
        return str(dst)

    def check_criteria(self, loss):
        # when best loss is higher than current overwrite it
        if self.no_improvement_count >= self.patience:
            if (self.best_loss is None) or (self.best_loss > (loss - self.delta)):
                logger_inst.info(
                    f"Checkpoint criteria trigger started at {loss:6E} - after {self.no_improvement_count} waiting epochs"
                )
                self.no_improvement_count = 0
                self.best_loss = loss
                return True
        self.no_improvement_count += 1
        return False

    def create_checkpoint(self, model, optimizer, scheduler, current_iteration, full_config):
        checkpoint = {
            "iteration": current_iteration,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
            "full_config": asdict(full_config),
        }

        self.last_checkpoint = checkpoint
        torch.save(self.last_checkpoint, self.save_path)
