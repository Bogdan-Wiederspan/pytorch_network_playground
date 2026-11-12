from __future__ import annotations

# standard imports
import dataclasses

# package imports
import numpy as np
import torch

import bbTT.data_handling.sampling.sampler as sampler
from bbTT.checkpoint.checkpoint import CheckPoint
from bbTT.configs.full_config import FullConfig

# personal imports
from bbTT.data_handling.io import get_data
from bbTT.data_handling.preprocessing.k_fold import FoldAndSplitCoordinator
from bbTT.data_handling.preprocessing.standardization import FeatureStatisticCache
from bbTT.data_handling.sampling.weight import WeightAggregator
from bbTT.data_handling.utils import hash_dictionary
from bbTT.loss import init_loss
from bbTT.models.utils import init_model

# from .train_utils import log_metrics
from bbTT.monitoring import (
    MONITOR_MODE_BATCH,
    MONITOR_MODE_TRAINING,
    MONITOR_MODE_VALIDATION,
    EvaluationReporter,
    EvaluationRunner,
    TrainingMonitor,
    load_registers,
    setup_monitoring,
)
from bbTT.monitoring.context.batch_composition_context import BatchCompositionHistory
from bbTT.monitoring.context.checkpoint_history import CheckpointHistory
from bbTT.monitoring.logger.logger import get_logger
from bbTT.monitoring.logger.tensorboard_logger import TensorboardLogger
from bbTT.optimizer.scheduler_handler import SchedulerHandler
from bbTT.optimizer.utils import init_optimizer, init_scheduler
from bbTT.train.loops import TrainingLoop, ValidationLoop
from bbTT.utils.utils import DEVICE


def main(**kwargs):

    full_config = FullConfig()
    torch.manual_seed(full_config.training_config.seed)
    np.random.seed(full_config.training_config.seed)

    # prepare logger
    logger_inst = get_logger(__name__)
    logger_inst.info(f"DEVICE: {DEVICE}")
    tensorboard_writer = TensorboardLogger(
        name=hash_dictionary(dataclasses.asdict(full_config.training_config)),
        path=kwargs["tensorboard_name"],
    )
    evaluation_runner_inst = EvaluationRunner(tensorboard_writer)

    # load all registered plots and metrics
    load_registers()
    # TODO add LOGGER FILEPATH in same directory of tensorboard
    logger_inst.i_info(f"Tensorboard logs: {tensorboard_writer.path}")

    # load data
    for current_fold in full_config.training_config.train_folds:
        logger_inst.info(f"Trainings fold: {current_fold}/{full_config.training_config.k_fold - 1}")
        # -----
        ### data loading and preprocessing
        # -----
        # events is of form : {uid : {"COLUMN_NAME": torch.Tensor}}
        events = get_data(
            full_config.dataset_config,
            rebuild_cache=kwargs["rebuild_cache"],
            save_cache=kwargs["save_cache"],
        )

        # split data into training, validation or test set - according to fold
        # HINT: order matters, due to memory constraints, events is changed inplace.
        #  To get Weight statistics all events need to be present. Thus this needs to run before apply splitting

        # Calculate all Indices for Splitting
        fold_split_coordinator = FoldAndSplitCoordinator(
            events=events,
            c_fold=current_fold,
            k_fold=full_config.training_config.k_fold,
            seed=full_config.training_config.seed,
            training_percentage=full_config.training_config.train_ratio,
            randomize=full_config.training_config.randomize,
        )

        columns_to_split = (
            "continuous",
            "categorical",
            "event_id",
            "normalization_weights",
            "product_of_weights",
            "evaluation_mask",
        )
        # calculate weight statistics, NEEDS TO RUN BEFORE APPLY INDICES
        weight_aggregator = WeightAggregator(events, fold_split_coordinator.indices)
        # actually apply indices, but release on the fly to reduce peak memory
        split_events = fold_split_coordinator.split_and_free(
            events, kinds=("training", "validation"), columns=columns_to_split
        )
        train_events, validation_events = split_events["training"], split_events["validation"]
        del events

        # create new sampler, splitted dict are released
        logger_inst.info("Create Sampler")
        training_sampler = sampler.create_sampler(
            train_events,
            train=True,
            weight_aggregator_inst=weight_aggregator,
            full_config=full_config,
        )
        del train_events

        validation_sampler = sampler.create_sampler(
            validation_events,
            train=False,
            weight_aggregator_inst=weight_aggregator,
            full_config=full_config,
        )
        del validation_events

        training_sampler.print_epoch_statistics()

        # load calculated sampler from training sampler to validation sampler
        validation_sampler.load_process_weights_from(training_sampler)
        # calculate or load statistic of DATA used to for training, this information is necessary for standardization.
        feature_statistic_cache = FeatureStatisticCache(
            dataset_config=full_config.dataset_config,
            sampler=training_sampler,
            split_fingerprint=fold_split_coordinator.fingerprint(),
            return_dummy=full_config.debug_config.get_batch_statistic_return_dummy,
            verbose=True,
        )
        full_config.model_building_config.standardization.mean = feature_statistic_cache.mean
        full_config.model_building_config.standardization.std = feature_statistic_cache.std

        # ----
        ### model build and configuration, including optimizer, scheduler, early stopping and loss function
        # ----
        model_inst = init_model(full_config=full_config)
        model_inst = model_inst.to(DEVICE).train()
        training_loop, validation_loop = (
            TrainingLoop(full_config),
            ValidationLoop(full_config),
        )

        optimizer_inst = init_optimizer(full_config=full_config, model_inst=model_inst)
        training_loss_inst, validation_loss_inst = init_loss(
            full_config=full_config, device=DEVICE, training_sampler=training_sampler
        )
        scheduler_inst = init_scheduler(full_config=full_config, optimizer_inst=optimizer_inst)
        checkpoint_inst = CheckPoint(
            checkpoint_name=full_config.training_config.save_model_name,
            checkpoint_fold=current_fold,
            patience=full_config.record_config.checkpoint_patience,
            delta=full_config.record_config.checkpoint_delta,
            confirm=full_config.record_config.checkpoint_confirm,
        )

        training_monitor_inst = TrainingMonitor(to_cpu=True, non_blocking=True)
        scheduler_handler_inst = SchedulerHandler(
            scheduler_inst=scheduler_inst,
            checkpoint_inst=checkpoint_inst,
            logger_inst=logger_inst,
        )

        # monitoring state of this fold: histories grow with the training and are rendered by the reporter.
        # The batch composition history grows with every iteration, so it is only created when requested;
        # without it the optional plot is skipped by the runner.
        checkpoint_history_inst = CheckpointHistory()
        batch_composition_history_inst = BatchCompositionHistory() if full_config.record_config.batch_metrics else None
        reporter_inst = EvaluationReporter(
            evaluation_runner_inst=evaluation_runner_inst,
            training_monitor_inst=training_monitor_inst,
            tensorboard_writer=tensorboard_writer,
            record_config=full_config.record_config,
            target_map=full_config.dataset_config.target_map,
            default_n_bins=full_config.binning_config.num_bins,
            checkpoint_history_inst=checkpoint_history_inst,
            batch_composition_history_inst=batch_composition_history_inst,
        )

        setup_monitoring(
            training_monitor_inst,
            model_inst,
            # model_inst.binning_layer,
            training_loss_inst,
            validation_loss_inst,
        )

        # ----
        ### training loop
        # ----
        logger_inst.info("Start training loop")
        for current_iteration in range(full_config.training_config.max_train_iteration):
            batch_result = training_loop(
                model_inst=model_inst,
                monitor=training_monitor_inst,
                kind_of_data=MONITOR_MODE_BATCH,
                loss_fn=training_loss_inst,
                sampler=training_sampler,
                device=DEVICE,
                sample_columns=full_config.sampler_config.sample_attributes,
                scheduler_handler_inst=scheduler_handler_inst,
                optimizer_inst=optimizer_inst,
            )

            reporter_inst.record_batch_composition(step=current_iteration, cursors=training_sampler.cursors)

            # --- Batch - training metrics
            if current_iteration % full_config.record_config.verbose_interval == 0:
                tensorboard_writer.log_lr(optimizer_inst, current_iteration)
                batch_loss = batch_result["loss"].item()
                tensorboard_writer.log_loss({"batch_loss": batch_loss}, step=current_iteration)
                current_lr = optimizer_inst.param_groups[0]["lr"]
                logger_inst.training(f"T-It: {current_iteration} - LR: {current_lr} - batch loss: {batch_loss:.2E}")

            # ----
            #### Evaluation of training and validation data, logging and checkpointing
            # ----
            evaluation_condition = current_iteration % full_config.record_config.validation_interval == 0

            if evaluation_condition:
                # evaluation of training data

                # --- Evaluation training ---
                # optional - every n-th validation do training evaluation, None when skipped
                validation_count = current_iteration // full_config.record_config.validation_interval
                training_evaluation_condition = (
                    validation_count % full_config.record_config.training_evaluation_interval == 0
                )
                evaluation_training_result = None
                if training_evaluation_condition:
                    logger_inst.info(f"Iteration {current_iteration}. Start evaluation of training data.")
                    evaluation_training_result = validation_loop(
                        model_inst=model_inst,
                        monitor=training_monitor_inst,
                        kind_of_data=MONITOR_MODE_TRAINING,
                        loss_fn_inst=validation_loss_inst,
                        sampler_inst=training_sampler,
                        sample_columns=full_config.sampler_config.sample_attributes,
                        device=DEVICE,
                    )


                logger_inst.info(f"Iteration {current_iteration}. Start evaluation of validation data.")
                # --- Evaluation validation ---
                evaluation_validation_result = validation_loop(
                    model_inst=model_inst,
                    monitor=training_monitor_inst,
                    kind_of_data=MONITOR_MODE_VALIDATION,
                    loss_fn_inst=validation_loss_inst,
                    sampler_inst=validation_sampler,
                    sample_columns=full_config.sampler_config.sample_attributes,
                    device=DEVICE,
                )

                eval_t_loss = (
                    evaluation_training_result["loss"].item() if evaluation_training_result is not None else None
                )
                eval_v_loss = evaluation_validation_result["loss"].item()
                eval_t_loss_text = f"{eval_t_loss:.2E}" if eval_t_loss is not None else "skipped"

                logger_inst.training(
                    f"Iteration: {current_iteration} - TLoss: {eval_t_loss_text} VLoss: {eval_v_loss:.2E}"
                )

                # --- Monitoring ---
                if full_config.record_config.log_metrics:
                    model_evaluation_state = model_inst.evaluation_state()
                    reporter_inst.report_model_state(model_evaluation_state=model_evaluation_state, global_step=current_iteration) # noqa

                    reporter_inst.report(mode="batch", result=batch_result, global_step=current_iteration, model_evaluation_state=model_evaluation_state) # noqa
                    reporter_inst.report(mode="batch", result=batch_result, global_step=current_iteration, model_evaluation_state=model_evaluation_state) # noqa
                    # optional - since training can also be monitored by batch monitoring
                    if evaluation_training_result is not None:
                        reporter_inst.report(mode="training", result=evaluation_training_result, global_step=current_iteration, model_evaluation_state=model_evaluation_state, loss=eval_t_loss) # noqa
                    reporter_inst.report(mode="validation", result=evaluation_validation_result, global_step=current_iteration, model_evaluation_state=model_evaluation_state, loss=eval_v_loss) # noqa

                ### --- checkpoint criteria checks and saving ---
                is_checkpoint_written = checkpoint_inst.update(
                    loss=eval_v_loss,
                    model=model_inst,
                    optimizer=optimizer_inst,
                    scheduler=scheduler_inst,
                    current_iteration=current_iteration,
                    full_config=full_config,
                )

                if is_checkpoint_written:
                    # with confirmation enabled the written state may be the candidate from the previous
                    # validation, so take iteration and loss from what was actually written, not from this step
                    reporter_inst.record_checkpoint(
                        name=checkpoint_inst.save_path.stem,
                        iteration=checkpoint_inst.last_checkpoint["iteration"],
                        loss=checkpoint_inst.best_loss,
                        step=current_iteration,
                    )

                scheduler_handler_inst.step(model_inst, optimizer_inst, metric=eval_v_loss)

        from IPython import embed

        embed(header="Training ends: Check if everything is as you thought it would be")


if __name__ == "__main__":
    from bbTT.utils.parser import ParserBuilder

    parser = ParserBuilder(
        "tensorboard",
        "cache",
        description="Train the HH -> bbtautau DNN with the configuration in bbTT/configs.",
        epilog="example:\n  python src/bbTT/train/train.py -tn my_first_run -rc",
    )

    main(
        rebuild_cache=parser.args.rebuild_cache,
        save_cache=parser.args.save_cache,
        tensorboard_name=parser.args.tensorboard_name,
    )
