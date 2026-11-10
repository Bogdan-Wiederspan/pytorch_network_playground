from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import torch

from bbTT.checkpoint.resume import format_overrides, load_resume_state, restore_training_state
from bbTT.configs.full_config import FullConfig
from bbTT.data_handling.utils import hash_dictionary
from bbTT.monitoring import (
    MONITOR_MODE_BATCH,
    MONITOR_MODE_TRAINING,
    MONITOR_MODE_VALIDATION,
    EvaluationReporter,
    EvaluationRunner,
    load_registers,
)
from bbTT.monitoring.context.batch_composition_context import BatchCompositionHistory
from bbTT.monitoring.context.checkpoint_history import CheckpointHistory
from bbTT.monitoring.logger.logger import get_logger
from bbTT.monitoring.logger.tensorboard_logger import TensorboardLogger
from bbTT.train.setup import FoldData, TrainingComponents, build_training_components, prepare_fold_data
from bbTT.utils.utils import DEVICE

logger_inst = get_logger(__name__)


def run_training_loop(
    full_config: FullConfig,
    training_components: TrainingComponents,
    fold_data: FoldData,
    reporter_inst: EvaluationReporter,
    tensorboard_writer: TensorboardLogger,
    start_iteration: int = 0,
) -> None:
    """
    Run training on a given fold.
    Inside *training_components* one find all instances that are trained.
    *fold_data* describes all data of the training folds.


    Args:
        full_config (FullConfig): Complete configuration of the run.
        training_components (TrainingComponents): Model, optimizer, losses, scheduler, checkpoint and monitor of the fold.
        fold_data (FoldData): Samplers of the training and validation split.
        reporter_inst (EvaluationReporter): Turns loop results into tensorboard output.
        tensorboard_writer (TensorboardLogger): Writer for the per-batch scalars (loss, learning rate).
        start_iteration (int): First iteration to run, > 0 when resuming from a checkpoint.
    """
    sample_columns = full_config.sampler_config.sample_attributes
    record_config = full_config.record_config

    logger_inst.info("Start training loop")
    for current_iteration in range(start_iteration, full_config.training_config.max_train_iteration):
        batch_result = training_components.training_loop(
            model_inst=training_components.model_inst,
            monitor=training_components.training_monitor_inst,
            kind_of_data=MONITOR_MODE_BATCH,
            loss_fn=training_components.training_loss_inst,
            sampler=fold_data.training_sampler,
            device=DEVICE,
            sample_columns=sample_columns,
            scheduler_handler_inst=training_components.scheduler_handler_inst,
            optimizer_inst=training_components.optimizer_inst,
        )
        reporter_inst.record_batch_composition(step=current_iteration, cursors=fold_data.training_sampler.cursors)

        # --- batch scalars ---
        if current_iteration % record_config.verbose_interval == 0:
            tensorboard_writer.log_lr(training_components.optimizer_inst, current_iteration)
            batch_loss = batch_result["loss"].item()
            tensorboard_writer.log_loss({"batch_loss": batch_loss}, step=current_iteration)
            current_lr = training_components.optimizer_inst.param_groups[0]["lr"]
            logger_inst.training(f"T-It: {current_iteration} - LR: {current_lr} - batch loss: {batch_loss:.2E}")

        # ----
        #### Evaluation of training and validation data, logging and checkpointing
        # ----
        evaluation_condition = current_iteration % record_config.validation_interval == 0
        if not evaluation_condition:
            continue

        # --- evaluation training ---
        # optional - the full pass is pure monitoring, so it only runs every n-th validation, None when skipped
        validation_count = current_iteration // record_config.validation_interval
        training_evaluation_condition = validation_count % record_config.training_evaluation_interval == 0
        evaluation_training_result = None
        if training_evaluation_condition:
            logger_inst.info(f"Iteration {current_iteration}. Start evaluation of training data.")
            evaluation_training_result = training_components.validation_loop(
                model_inst=training_components.model_inst,
                monitor=training_components.training_monitor_inst,
                kind_of_data=MONITOR_MODE_TRAINING,
                loss_fn_inst=training_components.validation_loss_inst,
                sampler_inst=fold_data.training_sampler,
                sample_columns=sample_columns,
                device=DEVICE,
            )

        # --- evaluation validation ---
        logger_inst.info(f"Iteration {current_iteration}. Start evaluation of validation data.")
        evaluation_validation_result = training_components.validation_loop(
            model_inst=training_components.model_inst,
            monitor=training_components.training_monitor_inst,
            kind_of_data=MONITOR_MODE_VALIDATION,
            loss_fn_inst=training_components.validation_loss_inst,
            sampler_inst=fold_data.validation_sampler,
            sample_columns=sample_columns,
            device=DEVICE,
        )

        eval_t_loss = evaluation_training_result["loss"].item() if evaluation_training_result is not None else None
        eval_v_loss = evaluation_validation_result["loss"].item()
        eval_t_loss_text = f"{eval_t_loss:.2E}" if eval_t_loss is not None else "skipped"
        logger_inst.training(f"Iteration: {current_iteration} - TLoss: {eval_t_loss_text} VLoss: {eval_v_loss:.2E}")

        # --- monitoring ---
        if record_config.log_metrics:
            model_evaluation_state = training_components.model_inst.evaluation_state()
            reporter_inst.report(
                "batch", batch_result, global_step=current_iteration, model_evaluation_state=model_evaluation_state
            )
            if evaluation_training_result is not None:
                reporter_inst.report(
                    "training", evaluation_training_result, global_step=current_iteration,
                    model_evaluation_state=model_evaluation_state, loss=eval_t_loss,
                )
            reporter_inst.report(
                "validation", evaluation_validation_result, global_step=current_iteration,
                model_evaluation_state=model_evaluation_state, loss=eval_v_loss,
            )

        # --- checkpoint criteria checks and saving ---
        is_checkpoint_written = training_components.checkpoint_inst.update(
            loss=eval_v_loss,
            model=training_components.model_inst,
            optimizer=training_components.optimizer_inst,
            scheduler=training_components.scheduler_inst,
            current_iteration=current_iteration,
            full_config=full_config,
        )
        if is_checkpoint_written:
            # with confirmation enabled the written state may be the candidate from the previous
            # validation, so take iteration and loss from what was actually written, not from this step
            reporter_inst.record_checkpoint(
                name=training_components.checkpoint_inst.save_path.stem,
                iteration=training_components.checkpoint_inst.last_checkpoint["iteration"],
                loss=training_components.checkpoint_inst.best_loss,
                step=current_iteration,
            )

        training_components.scheduler_handler_inst.step(training_components.model_inst, training_components.optimizer_inst, metric=eval_v_loss)


def main(
    tensorboard_name: str,
    rebuild_cache: bool,
    save_cache: bool,
    resume: str | None = None,
    overrides: Sequence[str] = (),
    ) -> None:
    """
    Train the configured folds one after another, sharing logger, tensorboard writer and runner.

    With *resume* the config comes from the checkpoint (plus *overrides* from the code config), only the
    checkpoint's fold is trained, and checkpoints and tensorboard run get a "-resume-<tag>" suffix so the
    original run stays untouched.

    Args:
        tensorboard_name (str): Sub directory of the tensorboard logs for this run.
        rebuild_cache (bool): Rebuild the event cache instead of reading it.
        save_cache (bool): Write a rebuilt event cache to disk.
        resuem (str | None): Checkpoint to continue from, see checkpoint.resume.load_resume_state.
        overrides (Sequence[str]): Dotted config paths to take from the code config on a resume.
    """
    full_config = FullConfig()

    # handle checkpoint resume loading
    # full config will be partly overwritten (merged)
    resume_state = None
    if resume is not None:
        resume_state = load_resume_state(resume, current_config=full_config, overrides=overrides)
        full_config = resume_state.full_config
    elif overrides:
        raise ValueError("--override only has an effect together with --resume")

    torch.manual_seed(full_config.training_config.seed)
    np.random.seed(full_config.training_config.seed)

    logger_inst.info(f"DEVICE: {DEVICE}")

    # create tensorboard instance to monitor plots for the training
    # when doing a continue a NEW instance of tensorboard is created, but the name of the previous training is used as suffix
    run_name = hash_dictionary(dataclasses.asdict(full_config.training_config))
    if resume_state is not None:
        run_name = resume_state.run_name(run_name)
        tensorboard_name = resume_state.run_name(tensorboard_name) if tensorboard_name is not None else None
    tensorboard_writer = TensorboardLogger(name=run_name, path=tensorboard_name)


    evaluation_runner_inst = EvaluationRunner(tensorboard_writer)
    load_registers()
    # TODO add LOGGER FILEPATH in same directory of tensorboard
    logger_inst.i_info(f"Tensorboard logs: {tensorboard_writer.path}")

    # a resume trains only the fold of its checkpoint
    # mixing folds would mix their data splits (so a complete mess)
    folds = full_config.training_config.train_folds if resume_state is None else (resume_state.fold,)
    for current_fold in folds:
        logger_inst.info(f"Trainings fold: {current_fold}/{full_config.training_config.k_fold - 1}")

        # --- data ---
        fold_data = prepare_fold_data(
            full_config=full_config,
            current_fold=current_fold,
            rebuild_cache=rebuild_cache,
            save_cache=save_cache,
        )
        # the standardization follows the training split of this fold, so the model config is set per fold
        full_config.model_building_config.standardization.mean = fold_data.feature_statistic_cache.mean
        full_config.model_building_config.standardization.std = fold_data.feature_statistic_cache.std

        # --- model, optimizer, loss, scheduler, checkpoint, monitor hooks ---
        components = build_training_components(
            full_config=full_config,
            current_fold=current_fold,
            training_sampler=fold_data.training_sampler,
        )

        # --- monitoring state of this fold ---
        # histories grow with the training and are rendered by the reporter. The batch composition history
        # grows with every iteration, so it is only created when requested; without it its plot is skipped.
        batch_composition_history_inst = BatchCompositionHistory() if full_config.record_config.batch_metrics else None
        reporter_inst = EvaluationReporter(
            evaluation_runner_inst=evaluation_runner_inst,
            training_monitor_inst=components.training_monitor_inst,
            tensorboard_writer=tensorboard_writer,
            record_config=full_config.record_config,
            target_map=full_config.dataset_config.target_map,
            default_n_bins=full_config.binning_config.num_bins,
            checkpoint_history_inst=CheckpointHistory(),
            batch_composition_history_inst=batch_composition_history_inst,
        )

        # --- resume: load the checkpoint into the fresh components ---
        start_iteration = 0
        if resume_state is not None:
            start_iteration = restore_training_state(components, resume_state)

        # --- training ---
        run_training_loop(
            full_config=full_config,
            training_components=components,
            fold_data=fold_data,
            reporter_inst=reporter_inst,
            tensorboard_writer=tensorboard_writer,
            start_iteration=start_iteration,
        )


if __name__ == "__main__":
    from bbTT.utils.parser import ParserBuilder

    parser = ParserBuilder(
        "tensorboard",
        "cache",
        "resume",
        description="Train the HH -> bbtautau DNN with the configuration in bbTT/configs.",
        epilog=(
            "examples:\n"
            "  python src/bbTT/train/train.py -tn my_first_run -rc\n"
            "  python src/bbTT/train/train.py -tn my_first_run -r my_model_fold0 -o training_config.max_train_iteration"
            "\nto see all overwrite add --list-overrides"
            ),
        )

    # print every possible overwrite option
    if parser.args.list_overrides:
        print(format_overrides(FullConfig()))
        raise SystemExit(0)

    main(
        rebuild_cache=parser.args.rebuild_cache,
        save_cache=parser.args.save_cache,
        tensorboard_name=parser.args.tensorboard_name,
        resume=parser.args.resume,
        overrides=parser.args.overrides,
    )
