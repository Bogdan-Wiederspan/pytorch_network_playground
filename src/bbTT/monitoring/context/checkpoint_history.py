
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CheckpointRecord:
    """
    One checkpoint that was written to disk.

    Args:
        name (str): File stem of the checkpoint, e.g. ``my_run_fold0``, see ``CheckPoint.save_path``.
        iteration (int): Training iteration at which the checkpoint was written.
        loss (float): Validation loss that triggered the checkpoint.
    """

    name: str
    iteration: int
    loss: float


@dataclass
class CheckpointHistory:
    """
    History of all checkpoints written during a training, across folds.

    Accumulated in the training loop and rendered as a markdown table for the TensorBoard
    text dashboard. Exists to make visible at which iterations a checkpoint was taken and
    which of them still exist on disk: checkpoints with the same name are written to the same
    file, so every record except the most recent one per name has been overwritten.

    Args:
        records (list[CheckpointRecord]): One record per written checkpoint, in chronological order.
    """

    records: list[CheckpointRecord] = field(default_factory=list)

    def record(self, name: str, iteration: int, loss: float) -> None:
        """
        Append the checkpoint that was just written.

        Args:
            name (str): File stem of the checkpoint, e.g. ``my_run_fold0``, see ``CheckPoint.save_path``.
            iteration (int): Training iteration at which the checkpoint was written.
            loss (float): Validation loss that triggered the checkpoint.
        """
        self.records.append(CheckpointRecord(name=name, iteration=iteration, loss=loss))

    def to_markdown(self) -> str:
        """
        Render the history as a markdown table.

        Returns:
            str: Table with the columns name, iteration, loss and whether the checkpoint was overwritten.
        """
        # the last record per name is the one that currently exists on disk
        current_iteration_per_name: dict[str, int] = {record.name: record.iteration for record in self.records}

        rows = ["| Name | Iteration | Loss | Overwritten |", "| --- | --- | --- | --- |"]
        for record in self.records:
            overwritten = "yes" if record.iteration != current_iteration_per_name[record.name] else "no"
            rows.append(f"| {record.name} | {record.iteration} | {record.loss:.4E} | {overwritten} |")
        return "\n".join(rows)
