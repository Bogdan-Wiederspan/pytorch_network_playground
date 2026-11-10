from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from bbTT.monitoring.context.eval_context import EvalContext

if TYPE_CHECKING:
    from bbTT.configs.record_config import RecordConfig
    from bbTT.monitoring.context.batch_composition_context import BatchCompositionHistory
    from bbTT.monitoring.context.checkpoint_history import CheckpointHistory
    from bbTT.monitoring.logger.tensorboard_logger import TensorboardLogger
    from bbTT.monitoring.runner import EvaluationRunner
    from bbTT.monitoring.training_monitor import TrainingMonitor

# keys under which the TrainingMonitor captures tensors and gradients, passed as `kind_of_data` to the loops
MONITOR_MODE_BATCH = "training_batch"
MONITOR_MODE_TRAINING = "evaluation_training"
MONITOR_MODE_VALIDATION = "evaluation_validation"


@dataclass(frozen=True)
class MonitoringMode:
    """
    Everything that distinguishes one kind of monitored data from another.

    Args:
        context_mode (str): Mode of the EvalContext, becomes the tensorboard tag prefix ("batch", "training", ...).
        monitor_mode (str): Key under which the TrainingMonitor captured tensors for this data.
        plots (tuple[str, ...]): Plot names from PLOT_REGISTRY to run.
        scalars (dict[str, str]): Tensorboard "tag/name" -> context artifact to log, see EvaluationRunner.run_scalars.
        with_gradients (bool): Also attach the captured gradients, only the training batch has them.
    """

    context_mode: str
    monitor_mode: str
    plots: tuple[str, ...]
    scalars: dict[str, str]
    with_gradients: bool = False


class EvaluationReporter:
    """
    Turns the result of a loop into tensorboard output and owns the monitoring state that grows over a training.

    One instance per fold. It knows for every kind of data which monitor bucket, plots and scalars belong to
    it, so the training loop only states *which* data it hands over, not *how* it is logged.
    """

    def __init__(
        self,
        evaluation_runner_inst: EvaluationRunner,
        training_monitor_inst: TrainingMonitor,
        tensorboard_writer: TensorboardLogger,
        record_config: RecordConfig,
        target_map: dict[str, int],
        default_n_bins: int,
        checkpoint_history_inst: CheckpointHistory,
        batch_composition_history_inst: BatchCompositionHistory | None = None,
    ) -> None:
        """
        Args:
            evaluation_runner_inst (EvaluationRunner): Runs plots and scalars on a context and logs them.
            training_monitor_inst (TrainingMonitor): Holds the tensors and gradients captured during the loops.
            tensorboard_writer (TensorboardLogger): Writer for everything that bypasses the runner (text).
            record_config (RecordConfig): Plot and scalar lists per kind of data.
            target_map (dict[str, int]): Mapping of target names to output nodes, passed into every context.
            default_n_bins (int): Default binning for histograms, passed into every context.
            checkpoint_history_inst (CheckpointHistory): History of written checkpoints, rendered as text.
            batch_composition_history_inst (BatchCompositionHistory | None): Batch composition per iteration,
                only present when batch metrics are requested; without it the related plot is skipped.
        """
        self.evaluation_runner_inst = evaluation_runner_inst
        self.training_monitor_inst = training_monitor_inst
        self.tensorboard_writer = tensorboard_writer
        self.target_map = target_map
        self.default_n_bins = default_n_bins
        self.checkpoint_history_inst = checkpoint_history_inst
        self.batch_composition_history_inst = batch_composition_history_inst

        # the batch has no scalars: its loss is logged separately every verbose_interval
        self.modes: dict[str, MonitoringMode] = {
            "batch": MonitoringMode(
                context_mode="batch",
                monitor_mode=MONITOR_MODE_BATCH,
                plots=tuple(record_config.batch_plots),
                scalars={},
                with_gradients=True,
            ),
            "training": MonitoringMode(
                context_mode="training",
                monitor_mode=MONITOR_MODE_TRAINING,
                plots=tuple(record_config.training_plots),
                scalars=dict(record_config.training_scalars),
            ),
            "validation": MonitoringMode(
                context_mode="validation",
                monitor_mode=MONITOR_MODE_VALIDATION,
                plots=tuple(record_config.validation_plots),
                scalars=dict(record_config.validation_scalars),
            ),
        }

    def _shared_meta(self, global_step: int, model_evaluation_state: dict[str, Any]) -> dict[str, Any]:
        """
        Context metadata that is identical for every kind of data at one evaluation point.

        Args:
            global_step (int): Current training iteration.
            model_evaluation_state (dict[str, Any]): Snapshot from Model.evaluation_state().

        Returns:
            dict[str, Any]: Keyword arguments for EvalContext.
        """
        meta: dict[str, Any] = {
            "model_evaluation_state": model_evaluation_state,
            "target_map": self.target_map,
            "global_step": global_step,
            "default_n_bins": self.default_n_bins,
        }
        if self.batch_composition_history_inst is not None:
            meta["batch_composition_history"] = self.batch_composition_history_inst
        return meta

    def report(
        self,
        mode: str,
        result: dict[str, torch.Tensor],
        global_step: int,
        model_evaluation_state: dict[str, Any],
        loss: float | None = None,
    ) -> EvalContext:
        """
        Build the context for one kind of data and log its plots and scalars.

        Args:
            mode (str): One of the keys in `self.modes`: "batch", "training" or "validation".
            result (dict[str, torch.Tensor]): Loop result with "predictions", "targets" and "event_weights".
            global_step (int): Current training iteration.
            model_evaluation_state (dict[str, Any]): Snapshot from Model.evaluation_state().
            loss (float | None): Loss of this data, added as feature "loss" when given.

        Returns:
            EvalContext: The populated context, in case the caller wants to inspect it.
        """
        monitoring_mode = self.modes[mode]

        ctx = EvalContext(
            mode=monitoring_mode.context_mode,
            predictions=result["predictions"],
            targets=result["targets"],
            event_weights=result["event_weights"],
            **self._shared_meta(global_step=global_step, model_evaluation_state=model_evaluation_state),
        )
        ctx.add_features(*self.training_monitor_inst.get_plot_tensors(monitoring_mode.monitor_mode))
        if monitoring_mode.with_gradients:
            ctx.add_features(*self.training_monitor_inst.get_plot_gradients(monitoring_mode.monitor_mode))
        if loss is not None:
            ctx.add_feature("loss", loss)

        self.evaluation_runner_inst.run_plots(ctx, plots=monitoring_mode.plots)
        self.evaluation_runner_inst.run_scalars(ctx=ctx, artifact_names=monitoring_mode.scalars)
        return ctx

    def record_batch_composition(self, step: int, cursors: dict[str, Any]) -> None:
        """
        Record the composition of the current training batch, if batch metrics are tracked.

        Args:
            step (int): Current training iteration.
            cursors (dict[str, Any]): Sampler cursors after the batch was drawn, see ProcessSampler.cursors.
        """
        if self.batch_composition_history_inst is not None:
            self.batch_composition_history_inst.record(step=step, cursors=cursors)

    def record_checkpoint(self, name: str, iteration: int, loss: float, step: int) -> None:
        """
        Record a written checkpoint and re-log the complete history as tensorboard text.

        Args:
            name (str): File stem of the checkpoint, see CheckPoint.save_path.
            iteration (int): Iteration whose state was written; can differ from `step` with confirmation enabled.
            loss (float): Validation loss of the written state.
            step (int): Current training iteration, used as tensorboard step.
        """
        self.checkpoint_history_inst.record(name=name, iteration=iteration, loss=loss)
        # tensorboard only keeps a sample of text entries per tag, so the whole table is written every time
        self.tensorboard_writer.log_text(
            tag="Checkpoints",
            text=self.checkpoint_history_inst.to_markdown(),
            step=step,
        )
