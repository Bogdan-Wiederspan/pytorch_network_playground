# standard imports
import torch

# package imports
import optimizer

from models import create_model
from data import load_data, preprocessing, sampler, cache
from utils import logger

from train_config import (
    config, dataset_config, model_building_config, optimizer_config, target_map, scheduler_config, binning_config
)
from train_utils import training_fn, validation_fn, log_metrics
from early_stopping import EarlyStopOnPlateau, CheckPoint
from export import torch_save, torch_export_v2
import marcel_weight_translation as mwt

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(**kwargs):
    # prepare logger
    logger_inst = logger.get_logger(__name__)
    logger_inst.info(f"Running DEVICE is {DEVICE}")
    tensorboard_writer = logger.TensorboardLogger(
        name=cache.hash_config(config),
        path=kwargs["tensorboard_name"]
        )
    logger_inst.info(f"Tensorboard logs are stored in {tensorboard_writer.path}")

    # load data
    for current_fold in (config["train_folds"]):
        logger_inst.info(f"Start Training of fold {current_fold} from {config["k_fold"] - 1}")
        #-----
        ### data loading and preprocessing
        #-----
        # HINT: order matters, due to memory constraints views are moved in and out of dictionaries
        # load data from cache is necessary or from root files
        # events is of form : {uid : {"continuous","categorical", "weight": torch tensor}}
        events = load_data.get_data(dataset_config, ignore_cache=kwargs["ignore_cache"], _save_cache=kwargs["save_cache"])

        fold_split_coordinator = preprocessing.FoldAndSplitCoordinator(
            events=events,
            c_fold=current_fold,
            k_fold=config["k_fold"],
            seed=config["seed"],
            training_percentage=0.75,
            randomize=True,
        )

        train_events, validation_events = fold_split_coordinator(events, which="training"), fold_split_coordinator(events, which="validation")

        weight_aggregator = preprocessing.WeightAggregator(events, fold_split_coordinator.indices)

        # release fie
        for key in list(events.keys()):
            del events[key]

        training_sampler = sampler.create_sampler(
            train_events,
            weight_aggregator_inst = weight_aggregator,
            target_map = target_map,
            min_size=config["min_events_in_batch"],
            batch_size=config["t_batch_size"],
            train=True,
            sample_ratio=config["sample_ratio"],
        )
        validation_sampler = sampler.create_sampler(
            validation_events,
            weight_aggregator_inst = weight_aggregator,
            target_map = target_map,
            min_size=config["min_events_in_batch"],
            batch_size=config["v_batch_size"],
            train=False,
            sample_ratio=config["sample_ratio"],
        )
        # share relative weight from training batch statistic to validation sampler
        training_sampler.share_weights_between_sampler(validation_sampler)

        # get weighted mean and std of expected batch composition
        model_building_config["mean"], model_building_config["std"] = preprocessing.get_batch_statistics_from_sampler(
            training_sampler,
            padding_values=-99999,
            features=dataset_config["continuous_features"],
            return_dummy=config["get_batch_statistic_return_dummy"],
        )
        #----
        ### model build and configuration, including optimizer, scheduler, early stopping and loss function
        #----
        if config["model_choice"] == "binned_lbn":
            model_inst = create_model.BinnedLBNDenseNet(
                dataset_config["continuous_features"],
                dataset_config["categorical_features"],
                config=model_building_config,
                binning_config=binning_config,
                )
            model_inst.set_learning_mode("default")
        elif config["model_choice"] == "bnet_lbn":
            model_inst = create_model.BNetLBNDenseNet(dataset_config["continuous_features"], dataset_config["categorical_features"], config=model_building_config)
        else:
            raise ValueError(f"Model choice {config['model_choice']} not recognized, but must be set")

        # # adding activaton fn
        # model = create_model.AddActFnToModel(model=model, act_fn="softmax")

        # kernel_cfg = {"general": {"smooth_width": torch.tensor(0.05) , "smooth_zone_value":torch.tensor(0.5) , "abs_mode":False}}
        # model = create_model.AddBinning(
        #     binning_edges=torch.linspace(0,1,15),
        #     model=model,
        #     kernel_cls=kernel.GaussianKernel,
        #     kernel_cfg=kernel_cfg,
        #     signal_cls=target_map["hh"],
        # )

        model_inst = model_inst.to(DEVICE).train()


        # TODO load mean from marcel if activated, probably break for new model
        model_inst = mwt.load_marcels_weights(model_inst, continuous_features=dataset_config["continuous_features"], with_std=config["load_marcel_stats"], with_weights=config["load_marcel_weights"])

        # only linear layers contribute to weight decay, prepare config that separates them for the optimizer
        weight_decay_parameters = optimizer.prepare_weight_decay(model_inst, optimizer_config)

        if config["training_fn"] == "default":
            optimizer_inst = torch.optim.AdamW(list(weight_decay_parameters.values()), lr=optimizer_config["lr"])
        elif config["training_fn"] == "sam":
            optimizer_inst = optimizer.SAM(list(weight_decay_parameters.values()), torch.optim.AdamW, lr=optimizer_config["lr"], rho = 2.0, adaptive=True)

        scheduler_inst = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer_inst,
            mode='min',
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            threshold=scheduler_config["min_delta"],
            threshold_mode=scheduler_config["threshold_mode"],
            cooldown=0,
            min_lr=0,
            eps=1e-08
        )

        early_stopper_inst = EarlyStopOnPlateau()
        checkpoint_inst = CheckPoint(checkpoint_name=config["save_model_name"], checkpoint_fold=current_fold)

        if config["loss_fn"] == "default":
            train_loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None,label_smoothing=config["label_smoothing"])

            validation_loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None,label_smoothing=config["label_smoothing"])
        elif config["loss_fn"] == "signal_efficiency":
            from loss import SignalEfficiency, BinningAwareSignificance
            import binning
            train_loss_fn = SignalEfficiency(sampler=training_sampler ,device=DEVICE, train=True, mode=config["loss_mode"], uncertainty=config["loss_uncertainty"])
            validation_loss_fn = SignalEfficiency(sampler=training_sampler, device=DEVICE, train=False, mode=config["loss_mode"], uncertainty=config["loss_uncertainty"])
            # train_loss_fn_new = BinningAwareSignificance(
            #     num_bins=15, sampler=training_sampler ,device=DEVICE, train=True, mode=config["loss_mode"], uncertainty=config["loss_uncertainty"], binning_fn=binning.EqualDistant(),
            #     )
            # validation_loss_fn_new = BinningAwareSignificance(
            #     num_bins=15, sampler=training_sampler, device=DEVICE, train=False, mode=config["loss_mode"], uncertainty=config["loss_uncertainty"], binning_fn=binning.EqualDistant(),
            # )

        #----
        ### training loop
        #----
        for current_iteration in range(1_000_000):
            t_loss, (t_pred, t_targets) = training_fn(
                model = model_inst,
                loss_fn = train_loss_fn,
                optimizer = optimizer_inst,
                sampler = training_sampler,
                device=DEVICE,
                sample_from = config["sample_attributes"],
            )

            if current_iteration % config["verbose_interval"] == 0:
                tensorboard_writer.log_loss({"batch_loss": t_loss.item()}, step=current_iteration)
                logger_inst.green(f"Training: {current_iteration} - batch loss: {t_loss.item():.2E}")

            #----
            #### Evaluation of training and validation data, logging and checkpointing
            #----
            if (current_iteration % config["validation_interval"] == 0) & (current_iteration >= 0):
                # evaluation of training data
                print(f"Running evaluation of training data at iteration {current_iteration}...")
                eval_t_loss, (eval_t_pred, eval_t_tar, eval_t_weights) = validation_fn(
                    model_inst,
                    validation_loss_fn,
                    training_sampler,
                    sample_from=config["sample_attributes"],
                    device=DEVICE
                    )
                # TODO when edges should be tracked add this in a way that is universal and does not break for models without binning layer, e.g. add property to model that returns None if no binning layer is present and add check in log_metrics
                # log_metrics(
                #     tensorboard_inst = tensorboard_writer,
                #     iteration_step = current_iteration,
                #     sampler_output = (eval_t_pred, eval_t_tar, eval_t_weights),
                #     target_map = target_map,
                #     mode = "train",
                #     loss = eval_t_loss.item(),
                #     lr = optimizer_inst.param_groups[0]["lr"],
                #     binning_edges = model_inst.binning_layer.edges.detach().cpu(),
                #     current_iteration = current_iteration,
                #     kernels = model_inst.binning_layer.kernels,
                # )
                print(f"Running evaluation of validation data at iteration {current_iteration}...")
                # evaluation of validation
                eval_v_loss, (eval_v_pred, eval_v_tar, eval_v_weights) = validation_fn(
                    model_inst,
                    validation_loss_fn,
                    validation_sampler,
                    sample_from=config["sample_attributes"],
                    device=DEVICE
                    )
                # TODO when edges should be tracked add this in a way that is universal and does not break for models without binning layer, e.g. add property to model that returns None if no binning layer is present and add check in log_metrics
                # log_metrics(
                #     tensorboard_inst = tensorboard_writer,
                #     iteration_step = current_iteration,
                #     sampler_output = (eval_v_pred, eval_v_tar, eval_v_weights),
                #     target_map = target_map,
                #     mode = "validation",
                #     loss = eval_v_loss.item(),
                #     binning_edges = model_inst.binning_layer.edges.detach().cpu(),
                #     current_iteration = current_iteration,
                #     kernels=model_inst.binning_layer.kernels,
                # )
                print(f"Evaluation: it: {current_iteration} - TLoss: {eval_t_loss:.2E} VLoss: {eval_v_loss:.2E}")

                # if (current_iteration % 1000 == 0) and (current_iteration > 0):

                ### checkpoint criteria checks and saving

                if checkpoint_inst.check_criteria(eval_v_loss):
                    checkpoint_inst.create_checkpoint(
                        model=model_inst,
                        optimizer=optimizer_inst,
                        scheduler=scheduler_inst,
                        current_iteration=current_iteration,
                    )

                # if metric does not improve x-times reduce
                # load previous checkpoint with reduces learning rate
                previous_lr =  optimizer_inst.param_groups[0]["lr"]
                scheduler_inst.step(eval_v_loss)
                current_lr = optimizer_inst.param_groups[0]["lr"]
                if previous_lr != current_lr:
                    logger_inst.info(f"{previous_lr} -> {current_lr}")
                    logger_inst.info(f"Reload model and optimizer from iteration ")
                    model_inst.load_state_dict(checkpoint_inst.last_checkpoint["model_state_dict"])
                    optimizer_inst.load_state_dict(checkpoint_inst.last_checkpoint["optimizer_state_dict"])
                    # since scheduler and optimizer are coupled
                    # overwrite old lr in checkpoint with lr after scheduler step
                    optimizer_inst.param_groups[0]["lr"] = current_lr

                # TODO release DATA from previous RUN
                if (current_iteration % config["max_train_iteration"] == 0) & (current_iteration > 0):
                    from IPython import embed; embed(
                        header=f"Current break at {current_iteration} if you wanna continue press y else save")

        from IPython import embed; embed(header="END - 89 in train.py ")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model runner")
    parser.add_argument("--ignore_cache", "-ic", action="store_true", default=False, help="Ignore cache when running the program (default: False)")
    parser.add_argument("--save-cache", "-s", dest="save_cache", action="store_true", default=False, help="Save cache (default: False)")
    parser.add_argument("--tensorboard_name", "-tn", dest="tensorboard_name", action="store", default=None, help="Name of the tensorboard, if given, turns off generated name (default: None)")

    args = parser.parse_args()

    # call main with parsed args
    main(
        ignore_cache=args.ignore_cache,
        save_cache=args.save_cache,
        tensorboard_name=args.tensorboard_name

        )
