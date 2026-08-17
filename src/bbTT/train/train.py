from __future__ import annotations

# standard imports
import dataclasses

# package imports
import numpy as np
import torch

from bbTT.configs.full_config import FullConfig

# personal imports
from bbTT.data_handling import io, preprocessing, sampler
from bbTT.data_handling.utils import hash_dictionary
from bbTT.loss import init_loss
from bbTT.models.utils import init_model

# from .train_utils import log_metrics
from bbTT.monitoring import EvalContext, EvaluationRunner, TrainingMonitor, load_registers, setup_monitoring
from bbTT.optimizer.early_stopping import CheckPoint
from bbTT.optimizer.scheduler_handler import SchedulerHandler
from bbTT.optimizer.utils import init_optimizer, init_scheduler
from bbTT.train.loops import TrainingLoop, ValidationLoop
from bbTT.utils import logger

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
full_config = FullConfig()
torch.manual_seed(full_config.training_config.seed)
np.random.seed(full_config.training_config.seed)


def main(**kwargs):
    # prepare logger
    logger_inst = logger.get_logger(__name__)
    logger_inst.info(f"DEVICE: {DEVICE}")
    tensorboard_writer = logger.TensorboardLogger(
        name=hash_dictionary(dataclasses.asdict(full_config.training_config)),
        path=kwargs["tensorboard_name"],
        )
    evaluation_runner_inst = EvaluationRunner(tensorboard_writer)
    # load all registered plots and metrics
    load_registers()
    # TODO add LOGGER FILEPATH in same directory of tensorboard
    logger_inst.i_info(f"Tensorboard logs: {tensorboard_writer.path}")

    # load data
    for current_fold in (full_config.training_config.train_folds):
        logger_inst.info(f"Trainings fold: {current_fold}/{full_config.training_config.k_fold - 1}")
        #-----
        ### data loading and preprocessing
        #-----
        # HINT: order matters, due to memory constraints views are moved in and out of dictionaries
        # load data from cache is necessary or from root files
        # events is of form : {uid : {"continuous","categorical", "weight": torch tensor}}
        events = io.get_data(full_config.dataset_config, ignore_cache=kwargs["ignore_cache"], save_cache=kwargs["save_cache"])
        # split data into training and validation according to fold and get collect all weight statistics
        fold_split_coordinator = preprocessing.FoldAndSplitCoordinator(
            events=events,
            c_fold=current_fold,
            k_fold=full_config.training_config.k_fold,
            seed=full_config.training_config.seed,
            training_percentage=0.75,
            randomize=True,
        )

        columns_to_split = ("continuous", "categorical", "event_id", "normalization_weights", "product_of_weights", "evaluation_mask")
        train_events, validation_events = fold_split_coordinator(events, which="training", columns=columns_to_split), fold_split_coordinator(events, which="validation", columns=columns_to_split) #noqa
        weight_aggregator = preprocessing.WeightAggregator(events, fold_split_coordinator.indices)

        # release initial fields
        for key in list(events.keys()):
            del events[key]

        logger_inst.info("Start creation of Sampler")

        _sampler_config = {
            "weight_aggregator_inst" : weight_aggregator,
            "target_map" : full_config.dataset_config.target_map,
            "min_size" : full_config.training_config.min_events_in_batch,
            "batch_size" : full_config.training_config.t_batch_size,
            "sample_ratio" : full_config.training_config.sample_ratio,
            "sub_sample_ratio" : full_config.training_config.sub_process_ratios,
        }

        training_sampler = sampler.create_sampler(
            train_events,
            train=True,
            **_sampler_config,
        )
        validation_sampler = sampler.create_sampler(
            validation_events,
            train=False,
            **_sampler_config,
        )
        # share relative weight from training batch statistic to validation sampler
        training_sampler.share_weights_between_sampler(validation_sampler)
        # get weighted mean and std of expected batch composition
        logger_inst.info("Start model building and configuration")

        full_config.model_building_config.mean, full_config.model_building_config.std = preprocessing.get_batch_statistics_from_sampler(
            training_sampler,
            padding_values=full_config.dataset_config.dummy_values,
            features=full_config.dataset_config.continuous_features,
            return_dummy=full_config.debug_config.get_batch_statistic_return_dummy,
        )
        #----
        ### model build and configuration, including optimizer, scheduler, early stopping and loss function
        #----
        model_inst = init_model(full_config=full_config)
        model_inst = model_inst.to(DEVICE).train()

        training_loop, validation_loop = TrainingLoop(full_config), ValidationLoop(full_config)

        optimizer_inst = init_optimizer(full_config=full_config, model_inst=model_inst)
        training_loss_inst, validation_loss_inst = init_loss(full_config=full_config, device=DEVICE, training_sampler=training_sampler)
        scheduler_inst = init_scheduler(full_config=full_config, optimizer_inst=optimizer_inst)
        checkpoint_inst = CheckPoint(checkpoint_name=full_config.training_config.save_model_name, checkpoint_fold=current_fold)
        training_monitor_inst = TrainingMonitor(to_cpu=True, non_blocking=True)
        scheduler_handler_inst = SchedulerHandler(scheduler_inst=scheduler_inst, checkpoint_inst=checkpoint_inst, logger_inst=logger)
        mode_batch, mode_eval_training, mode_eval_validation = "training_batch", "evaluation_training", "evaluation_validation"
        setup_monitoring(
            training_monitor_inst,
            model_inst,
            # model_inst.binning_layer,
            training_loss_inst,
            validation_loss_inst
            )
        #----
        ### training loop
        #----
        logger_inst.info("Start training loop")
        for current_iteration in range(1_000_000):
            batch_result = training_loop(
                model_inst=model_inst,
                monitor = training_monitor_inst,
                kind_of_data= mode_batch,
                loss_fn=training_loss_inst,
                optimizer=optimizer_inst,
                sampler=training_sampler,
                device=DEVICE,
                sample_columns=full_config.training_config.sample_attributes,
                scheduler_handler_inst=scheduler_handler_inst,
            )
            # ----
            # Verbose and Metrics that are triggered often
            # ----
            if current_iteration % full_config.training_config.verbose_interval == 0:
                tensorboard_writer.log_lr(optimizer_inst, current_iteration)
                batch_loss = batch_result["loss"].item()
                tensorboard_writer.log_loss({"batch_loss": batch_loss}, step=current_iteration)
                current_lr = optimizer_inst.param_groups[0]["lr"]
                logger_inst.training(f"T-It: {current_iteration} - LR: {current_lr} - batch loss: {batch_loss:.2E}")

            #----
            #### Evaluation of training and validation data, logging and checkpointing
            #----
            evaluation_condition = (current_iteration % full_config.training_config.validation_interval == 0) & (current_iteration >= 0)
            if evaluation_condition:
                # evaluation of training data
                logger_inst.info(f"Iteration {current_iteration}. Start evaluation of training data.")

                evaluation_training_result = validation_loop(
                    model_inst=model_inst,
                    monitor = training_monitor_inst,
                    kind_of_data= mode_eval_training,
                    loss_fn_inst=validation_loss_inst,
                    sampler_inst=training_sampler,
                    sample_columns=full_config.training_config.sample_attributes,
                    device=DEVICE,
                    )
                # evaluation of validation
                logger_inst.info(f"Iteration {current_iteration}. Start evaluation of validation data.")

                evaluation_validation_result = validation_loop(
                    model_inst=model_inst,
                    monitor = training_monitor_inst,
                    kind_of_data= mode_eval_validation,
                    loss_fn_inst=validation_loss_inst,
                    sampler_inst=validation_sampler,
                    sample_columns=full_config.training_config.sample_attributes,
                    device=DEVICE,
                    )

                eval_t_loss = evaluation_training_result["loss"].item()
                eval_v_loss = evaluation_validation_result["loss"].item()
                logger_inst.training(f"Iteration: {current_iteration} - TLoss: {eval_t_loss:.2E} VLoss: {eval_v_loss:.2E}")

                # TODO when edges should be tracked add this in a way that is universal and does not break for models without binning layer, e.g. add property to model that returns None if no binning layer is present and add check in log_metrics
                if full_config.training_config.log_metrics:
                    # --- plots on evaluation, on training data ---
                    model_evaluation_state = model_inst.evaluation_state()

                    shared_eval_context_meta_data = {
                        "model_evaluation_state": model_evaluation_state,
                        "target_map": full_config.dataset_config.target_map,
                        "global_step": current_iteration,
                    }

                    ctx_batch = EvalContext(
                        mode="batch",
                        predictions=batch_result["predictions"],
                        targets=batch_result["targets"],
                        event_weights=batch_result["event_weights"],
                        **shared_eval_context_meta_data,
                    )

                    ctx_train = EvalContext(
                        mode="training",
                        predictions=evaluation_training_result["predictions"],
                        targets=evaluation_training_result["targets"],
                        event_weights=evaluation_training_result["event_weights"],
                        **shared_eval_context_meta_data,
                    )

                    ctx_validation = EvalContext(
                        mode="validation",
                        predictions=evaluation_validation_result["predictions"],
                        targets=evaluation_validation_result["targets"],
                        event_weights=evaluation_validation_result["event_weights"],
                        **shared_eval_context_meta_data,
                    )
                    # ctx_batch.add_feature("kernels", model_inst.kernels)
                    # ctx_batch.add_feature("binning_fn", model_inst.binning_fn)

                    ctx_batch.add_features(
                        *training_monitor_inst.get_plot_gradients(mode_batch),
                        *training_monitor_inst.get_plot_tensors(mode_batch),
                    )

                    ctx_train.add_features(
                        *training_monitor_inst.get_plot_tensors(mode_eval_training),
                        ("loss", eval_t_loss),
                    )

                    ctx_validation.add_features(
                        *training_monitor_inst.get_plot_tensors(mode_eval_validation),
                        ("loss", eval_v_loss),
                    )

                    # --- Plotting
                    evaluation_runner_inst.run_plots(
                        ctx_batch,
                        plots=[
                            # "active_kernels",
                            "active_kernels_advance"
                        ]
                    )
                    # run metrics and store them
                    evaluation_runner_inst.run_plots(
                        ctx_train,
                        plots=[
                            "confusion_matrix",
                            "roc",
                            # "asimov_small_signal",
                            "output_score_hh_node",
                            "output_score_hh_node_untransformed",
                            "active_kernels_advance"
                        ],
                    )

                    evaluation_runner_inst.run_plots(
                        ctx_validation,
                        plots=[
                            "confusion_matrix",
                            "roc",
                            # "asimov_small_signal",
                            "output_score_hh_node",
                            "output_score_hh_node_untransformed",
                            "active_kernels_advance"
                        ],
                    )

                    evaluation_runner_inst.run_scalars(
                        ctx=ctx_train,
                        artifact_names={
                            "CrossEntropy/Evaluation Training" : "cross_entropy",
                            "Loss/Evaluation Training Loss" :"loss",
                            },
                    )

                    evaluation_runner_inst.run_scalars(
                        ctx=ctx_validation,
                        artifact_names={
                            "CrossEntropy/Evaluation Validation": "cross_entropy",
                            "Loss/Validation VLoss": "loss"
                            },

                    )

                ### checkpoint criteria checks and saving
                if checkpoint_inst.check_criteria(eval_v_loss):
                    checkpoint_inst.create_checkpoint(
                        model=model_inst,
                        optimizer=optimizer_inst,
                        scheduler=scheduler_inst,
                        current_iteration=current_iteration,
                        full_config=full_config,
                    )

                scheduler_handler_inst.step(model_inst, optimizer_inst, metric=eval_v_loss)


        from IPython import embed
        embed(header="Training ends: Check if everything is as you thought it would be")

if __name__ == "__main__":
    from bbTT.utils.parser import ParserBuilder
    parser = ParserBuilder("tensorboard", "cache")

    main(
        ignore_cache=parser.args.ignore_cache,
        save_cache=parser.args.save_cache,
        tensorboard_name=parser.args.tensorboard_name
        )
