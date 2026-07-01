from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from utils import logger

if TYPE_CHECKING:
    from data.sampler import ProcessSampler

logger_inst = logger.get_logger(__name__)

def register_loop(name):
    # add markers to function for later registration in loop registry
    def decorator(fn):
        fn._is_loop_method = True
        fn._loop_name = name
        return fn
    return decorator

class BaseLoop():
    """
    Base Class that implements are registry where registered loops are categorized
    as training and validation loops.
    The registry is used to check if a given loop function exists and to call the correct function in the training script.
    """
    REGISTERED_LOOPS = {}
    MODE = None # is overwitten by child class

    def __init__(self, full_config, necessary_columns: set[str] | None = None, *args, **kwargs):
        self.which_fn = (
            full_config.training_config.training_fn
            if self.MODE == "training"
            else full_config.training_config.validation_fn
            )
        # these column are always necessary for the loops
        self.necessary_columns = {"continuous", "categorical","targets"} if necessary_columns is None else necessary_columns
        self.target_shape = (-1, len(full_config.dataset_config.target_map))

    def __init_subclass__(cls):
        for _name, fn in cls.__dict__.items():
            # only add methods if is part of self
            if getattr(fn, "_is_loop_method", False):
                cls.REGISTERED_LOOPS.setdefault(cls.MODE, {}) # make sure that dict for mode exists
                cls.REGISTERED_LOOPS[cls.MODE][fn._loop_name] = fn

    def separate_prediction(
        self,
        pred: torch.tensor | tuple[torch.tensor],
        ) -> tuple[torch.tensor, torch.tensor]:
        """
        Depending on the network architecture the output can either be a tuple of prediction and binned prediction
        or just the prediction. To have a uniform interface the singular output uses a dummy value.

        Args:
            pred (torch.tensor | tuple[torch.tensor]): _description_

        Returns:
            tuple[torch.tensor, torch.tensor]: Tuple of predictions where first value is used for logging and second value is used for optimization.
            In case of non binned output both values are the same.
        """
        # TODO maybe replace with dictionary?
        is_binned = isinstance(pred, tuple)
        if is_binned:
            logging_pred, optimization_pred = pred
        else:
            logging_pred, optimization_pred = pred, pred

        return logging_pred, optimization_pred


    def __call__(self, *args, **kwargs):
        # registered loop is just an unbound function
        # bound by passing self as first argument
        fn = self.REGISTERED_LOOPS[self.MODE][self.which_fn]
        return fn(self, *args, **kwargs)


class TrainingLoop(BaseLoop):
    MODE = "training"

    def __init__(self, full_config, *args, **kwargs):
        super().__init__(*args, **kwargs, full_config=full_config)

    @register_loop(name="sam")
    def sam_optimizer(
        self,
        model,
        loss_fn,
        optimizer,
        sampler,
        device,
        *args,
        **kwargs,
        ):
        optimizer.zero_grad()

        cont, cat, targets = sampler.sample_batch(device=device)
        pred = model(categorical_inputs=cat, continuous_inputs=cont)
        logging_pred, optimization_pred = self.separate_prediction(pred)

        loss = loss_fn(optimization_pred, targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # second forward step with disabled bachnorm running stats in second forward step
        optimizer.disable_running_stats(model)
        pred_2 = model(categorical_inputs=cat, continuous_inputs=cont)
        loss_fn(pred_2, targets).backward()

        optimizer.second_step(zero_grad=True)

        optimizer.enable_running_stats(model)  # <- this is the important line
        return loss, (logging_pred, targets)

    @register_loop(name="cross_entropy")
    def cross_entropy_loss(
        self,
        model,
        loss_fn,
        optimizer,
        sampler,
        device,
        scheduler_inst=None,
        *args,
        **kwargs,
        ):
        # this loss never uses a binning network thus, a single prediction is expected

        optimizer.zero_grad()

        events = sampler.sample_batch(sample_from=self.necessary_columns, device=device)
        targets = events["targets"].reshape(self.target_shape)
        predictions = model(
            categorical_inputs=events.get("categorical"),
            continuous_inputs=events.get("continuous")
            )

        # training does not need event weights. The oversampling algorithm takes care of this
        loss = loss_fn(predictions, targets, event_weights=None)
        loss.backward()
        optimizer.step()
        if scheduler_inst is not None:
            scheduler_inst.step()

        return loss

    @register_loop(name="signal_efficiency")
    def signal_efficiency_loop(
        self,
        model,
        loss_fn,
        optimizer,
        sampler,
        sample_columns,
        device,
        scheduler_inst=None,
        *args,
        **kwargs,

        ):
        optimizer.zero_grad()

        events = sampler.sample_batch(sample_from=sample_columns, device=device)
        pred = model(
            categorical_inputs=events.pop("categorical"),
            continuous_inputs=events.pop("continuous"),
            )

        _, optimization_pred = self.separate_prediction(pred)

        targets = events.pop("targets")
        loss = loss_fn(
            prediction=optimization_pred,
            truth=targets.reshape(self.target_shape),
            product_of_weights=events["product_of_weights"],
            evaluation_mask=events["evaluation_space_mask"],
            )
        loss.backward()
        optimizer.step()
        if scheduler_inst is not None:
            scheduler_inst.step()
        return loss


class ValidationLoop(BaseLoop):
    MODE = "validation"
    def __init__(self, full_config, *args, **kwargs):
        super().__init__(*args, **kwargs, full_config=full_config)

    def collect_from_generators(
        self,
        model_inst: torch.nn.Module,
        sampler_inst: ProcessSampler,
        sample_columns: list[str],
        skip_concatenate_columns: tuple[str] = ("",),
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
        ) -> dict[torch.tensors]:
        """
        During Validation no sampling is used, instead we loop over all unsampeled Datasets.
        The *model_inst* is the DNN you want to use, while *sampler_inst* should be a ValidationSampler.
        Which column is aggregated over all datasets is defined in *sample_columns*.
        In the end a concatenation is done over the event axis. If this step should be skipped for
        certain columns (e.g. to maintain the shape), add the name in *skip_concatenate_columns*.

        Args:
            model_inst (torch.nn.Module): Your neural network model.
            sampler_inst (ProcessSampler): Instance of a ProcessSampler, with registered Processes.
            sample_columns (list[str]): List of columns to sample from.
            skip_concatenate_columns (tuple[str], optional): List of columns to skip concatenation. Defaults to ("",).
            device (torch.device, optional): Device everything should run on. Defaults to torch.device("cpu").

        Returns:
            dict[torch.tensors]: Dictonary with all aggregated tensors.
        """
        device = torch.device(device) if isinstance(device, str) else device

        model_inst.eval()
        collected_data = {
            "class_predictions" : [],
            "targets" : [],
            "sample_weights" : [],
            "loss_predictions" : [],
            "relative_weights" : [],

            # "dataset_id" : [], # TODO add ID to enable filtering by dataset
            }

        dynamic_data = {dynamic_d : [] for dynamic_d in tuple(sample_columns - self.necessary_columns)}

        with torch.no_grad():
            # sample over all dataset generator and run network
            # store output in lists which are then concatenated in the end
            for uid, validation_batch_generator in sampler_inst.create_sample_generator(
                batch_size=sampler_inst.batch_size,
                sample_from=sample_columns,
                device=device
                ).items():
                # hold data for current dataset, saved in loop to avoid memory issues
                for events in validation_batch_generator:
                    pred = model_inst(
                        categorical_inputs=events.pop("categorical"),
                        continuous_inputs=events.pop("continuous")
                        )
                    # network can output either binned or non binned predictions
                    # depending on model architecture, separate_prediction gives us a uniform interface to handle both cases
                    class_pred, loss_pred = self.separate_prediction(pred)

                    # -- collect data that is always done --

                    collected_data["targets"].append(events.pop("targets"))
                    collected_data["sample_weights"].append(events.pop("sample_weights"))
                    collected_data["relative_weights"].append(
                        torch.full(size=(class_pred.shape[0], 1), fill_value=sampler_inst[uid].relative_weight)
                        )

                    if not model_inst.use_last_activation:
                        # TODO if sigmoid is necessary add switch case
                        class_pred = torch.softmax(class_pred, dim=1)
                    collected_data["class_predictions"].append(class_pred)

                    # for loss calculation, can be the same as class_prediction e.g. crossentropy
                    collected_data["loss_predictions"].append(loss_pred)

                    # -- collect dynamic data --
                    for k in dynamic_data.keys():
                        dynamic_data[k].append(events.pop(k))

                    # it is necessary to ensure the same shape
                    # when mapping of activation function is on (only having softmax) do not apply them, else do apply

            # combine datas
            collected_data.update(dynamic_data)

            for k in tuple(collected_data.keys()):
                value = collected_data.pop(k)
                # event dimension is always the second last axis
                event_dim = len(value[0].shape) - 2
                # for whatever reason maybe want to skip concatenate?
                if k not in skip_concatenate_columns:
                    value = torch.concatenate(value, dim = event_dim)
                collected_data[k] = value
        return collected_data

    @register_loop(name="cross_entropy")
    def cross_entropy_loop(
        self,
        model_inst,
        loss_fn_inst,
        sampler_inst,
        sample_columns,
        device,
        *args,
        **kwargs,
        ):
        """
        Used for normal loss functions like Crossentropy.
        """
        to_sample_columns = self.necessary_columns.union(sample_columns)
        tensors = self.collect_from_generators(
            model_inst=model_inst,
            sampler_inst=sampler_inst,
            sample_columns=to_sample_columns,
            skip_concatenate_columns=("",),
            device=device,
            )

        # validation needs to be reweightes by sample weights
        loss = loss_fn_inst(
            tensors["loss_predictions"],
            tensors["targets"],
            tensors["sample_weights"],
        )
        return loss, (tensors["class_predictions"], tensors["targets"], tensors["sample_weights"])

    @register_loop(name="signal_efficiency")
    def signal_efficiency_loop(
        self,
        model_inst,
        loss_fn_inst,
        sampler_inst,
        sample_columns,
        device
        ):
        loss_columns = {"product_of_weights", "evaluation_space_mask"}
        to_sample_columns = self.necessary_columns.union(loss_columns).union(sample_columns)


        tensors = self.collect_from_generators(
            model_inst=model_inst,
            sampler_inst=sampler_inst,
            sample_columns=to_sample_columns,
            skip_concatenate_columns=("",),
            device=device,
            )

        loss = loss_fn_inst(
            tensors["loss_predictions"],
            tensors["targets"].reshape(self.target_shape),
            tensors["product_of_weights"],
            tensors["evaluation_space_mask"]
            # TODO  sampler weight is not used, but maybe should ?
            )

        return loss, (tensors["class_predictions"], tensors["targets"], tensors["sample_weights"])
