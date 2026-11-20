from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure
    import torch
    from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    """
    Tensorboard output of one training, split into one run per data mode.

    Tensorboard draws one panel per *tag* and one line per *run* that wrote this tag. To get training and
    validation of the same quantity into one panel, the two are written as sub-runs ``<path>/training`` and
    ``<path>/validation`` under the same tag, e.g. ``Loss``. Colors are assigned per run, so one training keeps
    its colors in every panel and the run regex (``/validation$``) selects one kind of data over all trainings.

    Everything without a mode (text, monitored model blocks, ...) goes to the main run at ``<path>``.
    """

    def __init__(self, name: str | None = None, path: str | None = None):
        """
        Args:
            name (str, optional): Name of the training, suffix of the timestamped default directory.
            path (str, optional): Directory relative to env TENSORBOARD_DIR; replaces the default directory.
        """
        self.name = name
        self.log_dir = pathlib.Path(os.environ["TENSORBOARD_DIR"])
        self.path = self.logger_path() if path is None else self.log_dir / path
        self._writers: dict[str | None, SummaryWriter] = {}

    @property
    def writer(self) -> SummaryWriter:
        """SummaryWriter of the main run, for everything that belongs to no mode."""
        return self.writer_for(None)

    def writer_for(self, mode: str | None) -> SummaryWriter:
        """
        Get the writer of a mode, creating the sub-run ``<path>/<mode>`` on first use.

        Args:
            mode (str | None): Data mode ("batch", "training", "validation", ...) or None for the main run.

        Returns:
            SummaryWriter: Writer of this mode.
        """
        if mode not in self._writers:
            log_dir = self.path if mode is None else self.path / mode
            self._writers[mode] = self.create_tensorboard_writer(log_dir=log_dir)
        return self._writers[mode]

    def log_scalar(self, tag: str, value: float | dict[str, float], step: int, mode: str | None = None) -> None:
        """
        Log one scalar, or one scalar per key of a dict under ``<tag>/<key>``.

        Args:
            tag (str): Tensorboard tag, "/" separates collapsible groups in the dashboard.
            value (float | dict[str, float]): Value, or named values that become the tags ``<tag>/<key>``.
            step (int): Global step.
            mode (str, optional): Run the scalar belongs to, see writer_for. Defaults to the main run.
        """
        writer = self.writer_for(mode)
        if isinstance(value, dict):
            for key, key_value in value.items():
                writer.add_scalar(f"{tag}/{key}", key_value, global_step=step)
        else:
            writer.add_scalar(tag, value, global_step=step)

    def log_scalars_overlay(self, tag: str, values: dict[str, float], step: int) -> None:
        """
        Log several named values as lines of *one* panel of the main run.

        Tensorboard has no per-line toggle inside a panel, so ``add_scalars`` fakes it by writing every key
        as its own sub-run ``<path>/<tag>_<key>``. This pollutes the run list with one entry per key and is
        only worth it where many lines of one training belong in one picture, e.g. one line per model block.

        Args:
            tag (str): Tensorboard tag of the panel.
            values (dict[str, float]): Line name -> value.
            step (int): Global step.
        """
        self.writer.add_scalars(tag, values, global_step=step)

    def log_lr(self, optimizer: torch.optim.Optimizer, step: int, mode: str | None = None) -> None:
        """
        Log the learning rate of the first parameter group.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to read the learning rate from.
            step (int): Global step.
            mode (str, optional): Run the scalar belongs to. Defaults to the main run.
        """
        value = optimizer.param_groups[0]["lr"]
        self.writer_for(mode).add_scalar("Learning_Rate", value, global_step=step)

    def log_precision(self, values, step, mode="train"):
        acc = {k: v["precision"] for k, v in values.items()}
        self.writer.add_scalars(f"{mode} Precision", acc, step)

    def log_f1(self, values, step, mode="train"):
        acc = {k: v["f1"] for k, v in values.items()}
        self.writer.add_scalars(f"{mode} F1-Score", acc, step)

    def log_sensitivity(self, values, step, mode="train"):
        acc = {k: v["sensitivity"] for k, v in values.items()}
        self.writer.add_scalars(f"{mode} Recall", acc, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int, mode: str | None = None) -> None:
        """
        Log a histogram of a tensor.

        Args:
            tag (str): Tensorboard tag.
            values (torch.Tensor): Values to histogram.
            step (int): Global step.
            mode (str, optional): Run the histogram belongs to. Defaults to the main run.
        """
        self.writer_for(mode).add_histogram(tag, values, global_step=step)

    def log_figure(self, tag: str, figure: matplotlib.figure.Figure, step: int, mode: str | None = None) -> None:
        """
        Log a matplotlib figure.

        Args:
            tag (str): Tensorboard tag.
            figure (matplotlib.figure.Figure): Figure to render.
            step (int): Global step.
            mode (str, optional): Run the figure belongs to. Defaults to the main run.
        """
        self.writer_for(mode).add_figure(tag, figure, global_step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Write a markdown text entry to the TensorBoard text dashboard.

        TensorBoard keeps only a small reservoir sample of text entries per tag, so callers that
        log a growing history should re-log the complete text on every call instead of one entry per item.

        Args:
            tag (str): TensorBoard tag of the entry.
            text (str): Markdown content.
            step (int): Global step the entry belongs to.
        """
        self.writer.add_text(tag, text, global_step=step)

    def close(self) -> None:
        """Flush and close all writers."""
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def logger_path(self) -> pathlib.Path:
        """
        Default directory ``<TENSORBOARD_DIR>/<timestamp>-<name>``.

        Returns:
            pathlib.Path: Directory of the main run.
        """
        from time import localtime, strftime

        t = strftime("%Y_%m_%d-%H_%M_%S", localtime())
        new_stem = f"{t}-{self.name}"
        logger_path = self.log_dir / new_stem
        return logger_path

    def create_tensorboard_writer(self, log_dir: pathlib.Path) -> SummaryWriter:
        """
        Create a SummaryWriter; torch is imported lazily to keep the module importable without it.

        Args:
            log_dir (pathlib.Path): Directory of the run.

        Returns:
            SummaryWriter: The writer.
        """
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
