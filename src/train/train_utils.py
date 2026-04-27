import torch

import plotting
from train_config import full_config
from utils import logger

functions = {}
logger_inst = logger.get_logger(__name__)

def do_scheduler_step(
    loss: torch.tensor,
    logger_inst: logger.logging.Logger,
    scheduler_inst: torch.optim.lr_scheduler,
    model_inst: torch.nn.Module,
    optimizer_inst: torch.optim.Optimizer,
    checkpoint_inst: dict[torch.tensor]) -> None:
    """
    Perform with given *scheduler_inst* a scheduler step, under the conditions of the given *scheduler_inst*.
    If a step is performed be vebose using *logger_inst* and reload the last best state for *model_inst* and *optimizer_inst* from
    *checkpoint_inst*.

    Args:
        loss (torch.tensor): The current validation loss.
        logger_inst (logger.logging.Logger): Instance of the used Logger
        scheduler_inst (torch.optim.lr_scheduler): Instance of a Learning Rate Scheduler
        model_inst (torch.nn.Module): Instance of the Model
        optimizer_inst (torch.optim.Optimizer): Instance of an Optimizer
        checkpoint_inst (dict[torch.tensor]): Dictionary containing the learning rate, model state dict and optimizer state dict.
    """
    previous_lr =  optimizer_inst.param_groups[0]["lr"]
    scheduler_inst.step(loss)
    current_lr = optimizer_inst.param_groups[0]["lr"]
    if previous_lr != current_lr:
        logger_inst.info(
            f"{previous_lr} -> {current_lr}\n"
            f"Reload model and optimizer from iteration "
            )
        model_inst.load_state_dict(checkpoint_inst.last_checkpoint["model_state_dict"])
        optimizer_inst.load_state_dict(checkpoint_inst.last_checkpoint["optimizer_state_dict"])

def register(fn):
    # helper to register functions in pool of functions
    functions[fn.__name__] = fn
    return fn

class BaseLoop():
    registered_loops = {"training": {}, "validation": {}}

    def __init__(self, mode, which_fn):
        self.mode = mode
        self.which_fn = which_fn
        # register all Loop Functions in registered_loops for checking purpose
        self._register_loop_methods()

    @staticmethod
    def register(fn):
        fn._is_loop_method = True
        return staticmethod(fn)

    def _register_loop_methods(self):
        for name, fn in type(self).__dict__.items():
            # only add methods if is part of self
            if getattr(fn, "_is_loop_method", False):
                self.registered_loops[self.mode][name] = fn

    def __call__(self, *args, **kwargs):
        fn = getattr(self, self.which_fn)
        return fn(*args, **kwargs)


class TrainingLoop(BaseLoop):
    def __init__(self, which_fn, *args, **kwargs):
        super().__init__(mode="training", which_fn=which_fn, *args, **kwargs)

    @BaseLoop.register
    def sam_optimizer(model, loss_fn, optimizer, sampler, device):
        optimizer.zero_grad()

        cont, cat, targets = sampler.sample_batch(device=device)
        pred = model(categorical_inputs=cat, continuous_inputs=cont)

        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # second forward step with disabled bachnorm running stats in second forward step
        optimizer.disable_running_stats(model)
        pred_2 = model(categorical_inputs=cat, continuous_inputs=cont)
        loss_fn(pred_2, targets).backward()

        optimizer.second_step(zero_grad=True)

        optimizer.enable_running_stats(model)  # <- this is the important line
        return loss, (pred, targets)

    @BaseLoop.register
    def default(model, loss_fn, optimizer, sampler, sample_from, device):
        optimizer.zero_grad()

        events = sampler.sample_batch(sample_from=sample_from, device=device)
        pred = model(categorical_inputs=events.pop("categorical"), continuous_inputs=events.pop("continuous"))

        targets = events.pop("targets")
        loss = loss_fn(pred, targets.reshape(-1,3), events)
        loss.backward()
        optimizer.step()
        return loss, (pred, targets)

class ValidationLoop(BaseLoop):
    def __init__(self, which_fn, *args, **kwargs):
        super().__init__(mode="validation", which_fn=which_fn, *args, **kwargs)

    @BaseLoop.register
    def default(model, loss_fn, sampler, sample_from, device):
        with torch.no_grad():
            val_loss = []
            model.eval()
            predictions = []
            truth = []
            weights = []

            for uid, validation_batch_generator in sampler.create_sample_generator(
                batch_size=sampler.batch_size, # do validation over whole dataset at once
                sample_from=sample_from,
                device=device).items():
                dataset_losses = []

                for events in validation_batch_generator:
                    val_pred = model(categorical_inputs=events.pop("categorical"), continuous_inputs=events.pop("continuous"))
                    targets = events.pop("targets")
                    loss = loss_fn(val_pred, targets.reshape(-1,3), events, debug=False)
                    # collect data for metric plots
                    dataset_losses.append(loss)
                    predictions.append(torch.softmax(val_pred, dim=1).cpu())
                    truth.append(targets.cpu())
                    weights.append(torch.full(size=(val_pred.shape[0], 1), fill_value=sampler[uid].relative_weight).cpu())

                process_contribution = sampler[uid].relative_weight
                average_val = sum(dataset_losses) / len(dataset_losses) * process_contribution
                val_loss.append(average_val)

            final_validation_loss = sum(val_loss).cpu()#/ len(val_loss)
            model.train()

            truth = torch.concatenate(truth, dim=0)
            predictions = torch.concatenate(predictions, dim=0)
            weights = torch.flatten(torch.concatenate(weights, dim=0))
            return final_validation_loss, (predictions, truth, weights)


    @BaseLoop.register
    def signal_efficiency(model, loss_fn, sampler, sample_from, device):
        with torch.no_grad():
            val_loss = []
            model.eval()
            predictions = []
            truth = []
            weights = []

            # get constant process factors and filter signal or background out
            p, t = [], []
            e = {"product_of_weights" : [], "evaluation_space_mask" : []}
            for uid, validation_batch_generator in sampler.create_sample_generator(
                batch_size=sampler.batch_size, # do validation over whole dataset at once
                sample_from=sample_from,
                device=device).items():
                dataset_losses = []



                for events in validation_batch_generator:
                    val_pred = model(categorical_inputs=events.pop("categorical"), continuous_inputs=events.pop("continuous"))
                    p.append(val_pred)
                    targets = events.pop("targets")
                    t.append(targets)
                    e["product_of_weights"].append(events.pop("product_of_weights"))
                    e["evaluation_space_mask"].append(events.pop("evaluation_space_mask"))

                    # collect data for metric plots

                    predictions.append(torch.softmax(val_pred, dim=1).cpu())
                    truth.append(targets.cpu())
                    weights.append(torch.full(size=(val_pred.shape[0], 1), fill_value=sampler[uid].relative_weight).cpu())

            p = torch.concatenate(p)
            t = torch.concatenate(t)
            e["product_of_weights"] = torch.concatenate(e["product_of_weights"])
            e["evaluation_space_mask"] = torch.concatenate(e["evaluation_space_mask"])
            loss = loss_fn(p, t.reshape(-1,3), e, debug=False)
            # loss = loss_fn(val_pred, targets.reshape(-1,3), events, debug=False)
            dataset_losses.append(loss)

            # from IPython import embed; embed(header = "VALID LOSS line: 182 in train_utils.py")
            # process_contribution = sampler[uid].relative_weight
            # average_val = sum(dataset_losses) / len(dataset_losses) * process_contribution
            # val_loss.append(average_val)

            # final_validation_loss = sum(val_loss).cpu()#/ len(val_loss)
            final_validation_loss = loss.cpu()#/ len(val_loss)
            model.train()

            truth = torch.concatenate(truth, dim=0)
            predictions = torch.concatenate(predictions, dim=0)
            weights = torch.flatten(torch.concatenate(weights, dim=0))
            return final_validation_loss, (predictions, truth, weights)


@register
def trainings_loop_sam(model, loss_fn, optimizer, sampler, device):
    optimizer.zero_grad()

    cont, cat, targets = sampler.sample_batch(device=device)
    pred = model(categorical_inputs=cat, continuous_inputs=cont)

    loss = loss_fn(pred, targets)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # second forward step with disabled bachnorm running stats in second forward step
    optimizer.disable_running_stats(model)
    pred_2 = model(categorical_inputs=cat, continuous_inputs=cont)
    loss_fn(pred_2, targets).backward()

    optimizer.second_step(zero_grad=True)

    optimizer.enable_running_stats(model)  # <- this is the important line
    return loss, (pred, targets)

@register
def trainings_loop_default(model, loss_fn, optimizer, sampler, sample_from, device):
    optimizer.zero_grad()

    events = sampler.sample_batch(sample_from=sample_from, device=device)
    pred = model(categorical_inputs=events.pop("categorical"), continuous_inputs=events.pop("continuous"))

    # loss = loss_fn(pred, targets.reshape(-1,3), weight, idx)
    targets = events.pop("targets")
    loss = loss_fn(pred, targets.reshape(-1,3), events)
    loss.backward()
    optimizer.step()
    return loss, (pred, targets)

# @register
# def validation_default(model, loss_fn, sampler, device):
#     with torch.no_grad():
#         # run validation every x steps
#         val_loss = []
#         model.eval()
#         predictions = []
#         truth = []
#         weights = []

#         for uid, validation_batch_generator in sampler.get_dataset_batch_generators(batch_size=sampler.batch_size, device=device).items():
#             dataset_losses = []
#             for cont, cat, targets in validation_batch_generator:
#                 val_pred = model(categorical_inputs=cat, continuous_inputs=cont)
#                 # if torch.any(torch.isnan(val_pred)):
#                 #     from IPython import embed; embed(header="string - 63 in train_utils.py ")
#                 loss = loss_fn(val_pred, targets.reshape(-1, 3))

#                 dataset_losses.append(loss)

#                 predictions.append(torch.softmax(val_pred, dim=1).cpu())
#                 truth.append(targets.cpu())
#                 weights.append(torch.full(size=(val_pred.shape[0], 1), fill_value=sampler[uid].relative_weight).cpu())
#             # create event based weight tensor for dataset

#             average_val = sum(dataset_losses) / len(dataset_losses) * sampler[uid].relative_weight
#             val_loss.append(average_val)

#         final_validation_loss = sum(val_loss).cpu()#/ len(val_loss)
#         model.train()

#         truth = torch.concatenate(truth, dim=0)
#         predictions = torch.concatenate(predictions, dim=0)
#         weights = torch.flatten(torch.concatenate(weights, dim=0))
#         return final_validation_loss, (predictions, truth, weights)


@register
def validation_default(model, loss_fn, sampler, sample_from, device):
    with torch.no_grad():
        val_loss = []
        model.eval()
        predictions = []
        truth = []
        weights = []

        for uid, validation_batch_generator in sampler.create_sample_generator(
            batch_size=sampler.batch_size, # do validation over whole dataset at once
            sample_from=sample_from,
            device=device).items():
            dataset_losses = []

            for events in validation_batch_generator:
                val_pred = model(categorical_inputs=events.pop("categorical"), continuous_inputs=events.pop("continuous"))
                targets = events.pop("targets")
                loss = loss_fn(val_pred, targets.reshape(-1,3), events, debug=False)
                # collect data for metric plots
                dataset_losses.append(loss)
                predictions.append(torch.softmax(val_pred, dim=1).cpu())
                truth.append(targets.cpu())
                weights.append(torch.full(size=(val_pred.shape[0], 1), fill_value=sampler[uid].relative_weight).cpu())

            process_contribution = sampler[uid].relative_weight
            average_val = sum(dataset_losses) / len(dataset_losses) * process_contribution
            val_loss.append(average_val)

        final_validation_loss = sum(val_loss).cpu()#/ len(val_loss)
        model.train()

        truth = torch.concatenate(truth, dim=0)
        predictions = torch.concatenate(predictions, dim=0)
        weights = torch.flatten(torch.concatenate(weights, dim=0))
        return final_validation_loss, (predictions, truth, weights)


@register
def validation_signal_efficiency(model, loss_fn, sampler, sample_from, device):
    with torch.no_grad():
        val_loss = []
        model.eval()
        predictions = []
        truth = []
        weights = []

        # get constant process factors and filter signal or background out
        p, t = [], []
        e = {"product_of_weights" : [], "evaluation_space_mask" : []}
        for uid, validation_batch_generator in sampler.create_sample_generator(
            batch_size=sampler.batch_size, # do validation over whole dataset at once
            sample_from=sample_from,
            device=device).items():
            dataset_losses = []



            for events in validation_batch_generator:
                val_pred = model(categorical_inputs=events.pop("categorical"), continuous_inputs=events.pop("continuous"))
                p.append(val_pred)
                targets = events.pop("targets")
                t.append(targets)
                e["product_of_weights"].append(events.pop("product_of_weights"))
                e["evaluation_space_mask"].append(events.pop("evaluation_space_mask"))

                # collect data for metric plots

                predictions.append(torch.softmax(val_pred, dim=1).cpu())
                truth.append(targets.cpu())
                weights.append(torch.full(size=(val_pred.shape[0], 1), fill_value=sampler[uid].relative_weight).cpu())

        p = torch.concatenate(p)
        t = torch.concatenate(t)
        e["product_of_weights"] = torch.concatenate(e["product_of_weights"])
        e["evaluation_space_mask"] = torch.concatenate(e["evaluation_space_mask"])
        loss = loss_fn(p, t.reshape(-1,3), e, debug=False)
        # loss = loss_fn(val_pred, targets.reshape(-1,3), events, debug=False)
        dataset_losses.append(loss)

        # from IPython import embed; embed(header = "VALID LOSS line: 182 in train_utils.py")
        # process_contribution = sampler[uid].relative_weight
        # average_val = sum(dataset_losses) / len(dataset_losses) * process_contribution
        # val_loss.append(average_val)

        # final_validation_loss = sum(val_loss).cpu()#/ len(val_loss)
        final_validation_loss = loss.cpu()#/ len(val_loss)
        model.train()

        truth = torch.concatenate(truth, dim=0)
        predictions = torch.concatenate(predictions, dim=0)
        weights = torch.flatten(torch.concatenate(weights, dim=0))
        return final_validation_loss, (predictions, truth, weights)


def log_metrics(
    tensorboard_inst,
    iteration_step: int,
    sampler_output: tuple[torch.tensor],
    target_map: dict,
    mode: str ="train",
    exist_check: bool = False,
    **data: dict[torch.tensor]
    ) -> None:
    """
    Helper function to store all logging efforts in given *tensorboard_inst*.
    Each log is saved under a given *iteration_step*, with a given *mode* prefix.
    The *sampler_output* is a tuple the inputs and output of the model.
    The *target_map* is a dict defining the mapping of a node to an index.
    Data is an dictionary with arbitrary tensors to log, what data is used for is defined within the function.
    A plot in data is only created when respective entry exist. If one does want to be strict in checking turn on
    "exist_check", then a check if fields exist in data is perform every time.

    Args:
        tensorboard_inst (_type_): Active tensorboard instance
        iteration_step (int): Current iteration step
        sampler_output (tuple[torch.tensor]): tuple of tensors, defining prediction, target and weights.
        target_map (dict): Mapping of output node to index.
        mode (str, optional): Suffix for name, typically "train" oder "validation". Defaults to "train".
    """
    def _optional(*keys, log_name=None) -> bool:
        """
        Helper function to enable optional logs only when all *keys* exist.
        Typical case: A log describe a model specific information like binnings of learnable edges.
        Only specific models have this information, thus this should not be logged if it is not prevalent.

        Returns:
            bool: True if all keys exist, else False
        """
        missing_keys = [k for k in keys if k not in data.keys()]
        if any(missing_keys):
            if exist_check:
                name = keys if log_name is None else log_name
                logger_inst.warning(f"Can't produce log of - {log_name} - missing: {missing_keys}")
            return False
        return True


    logger_inst.info(f"Start {mode} Logs at iteration {iteration_step}")
    # logs and plots that are ALWAYS plotted

    # outputs of network
    pred, tar, weights = sampler_output

    # log crossentropy as metric
    # weights are EVENT WEIGHTS, but cross entropy want to have cls weights
    # for this reason reduce is set to false and manual average is calculated
    cce_metric = torch.nn.functional.cross_entropy(pred, tar, weight=None, reduction="none")
    cce_metric = torch.mean(cce_metric * weights)
    tensorboard_inst.log_scalar(values={mode: cce_metric}, step=iteration_step, name=f"{mode} CrossEntropy")

    # network prediction of all nodes
    pred_fig, pred_ax = plotting.network_predictions(
        tar,
        pred,
        target_map,
        normalize=True,
    )
    tensorboard_inst.log_figure(f"{mode} node output", pred_fig, step=iteration_step)

    # confusion matrix plot
    c_mat_fig, c_mat_ax, c_mat = plotting.confusion_matrix(
        tar,
        pred,
        target_map,
        sample_weight=weights,
        normalized="true"
    )
    tensorboard_inst.log_figure(f"{mode} confusion matrix", c_mat_fig, step=iteration_step)

    roc_fig, roc_ax = plotting.roc_curve(
        tar,
        pred,
        sample_weight=weights,
        labels=list(target_map.keys())
    )
    tensorboard_inst.log_figure(f"{mode} roc curve one vs rest", roc_fig, step=iteration_step)


    if _optional("loss", log_name="Loss"):
        tensorboard_inst.log_loss({mode: data["loss"]}, step=iteration_step)

    if _optional("lr", log_name="Learning Rate"):
        tensorboard_inst.log_lr(data["lr"], step=iteration_step)

    if _optional("binning_edges", log_name="HH Node Prediction"):
    # binned network prediction is only available by models with binned activation layer
        pred_fig, pred_ax = plotting.network_predictions_hh(
            tar,
            pred,
            target_map,
            normalize=True,
            binning_edges=data["binning_edges"],
            current_iteration=iteration_step,
        )
        tensorboard_inst.log_figure(f"{mode} HH node output", pred_fig, step=iteration_step)

    if _optional("kernels", log_name="Lernable Bin Edges"):
        kernel_fig, kernel_ax = plotting.visualize_bins(data["kernels"])
        tensorboard_inst.log_figure("bin edges", kernel_fig, step=iteration_step)



training_fn = functions.get(f"trainings_loop_{full_config.training_config.training_fn}")
validation_fn = functions.get(f"validation_{full_config.training_config.validation_fn}")
