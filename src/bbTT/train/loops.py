from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from bbTT.monitoring.logger.logger import get_logger

if TYPE_CHECKING:
    from data_handling.sampler import ProcessSampler

logger_inst = get_logger(__name__)



def register_loop(loss: str):
    """
    Marks a loop method as the loop that belongs to the loss *loss* (a LossConfig.loss_fn value).
    A loop exists only because its loss needs certain columns and a certain call signature, so the loss
    name is the key of the registry: the loop is picked by LossConfig.loss_fn, there is no separate loop choice.

    Args:
        loss (str): Name of the loss this loop belongs to, must be a LOSS_CHOICE value.
    """
    def decorator(fn):
        fn._is_loop_method = True
        fn._loss_name = loss
        return fn

    return decorator


class BaseLoop:
    """
    Base Class that implements are registry where registered loops are categorized
    as training and validation loops.
    The registry is used to check if a given loop function exists and to call the correct function in the training script.

    How to add a new loop (use the *cross_entropy* loops as reference, since they are the most basic implementation):

        1. Add a method to TrainingLoop (or ValidationLoop) and decorate it with @register_loop(loss="my_loss").
            The name must match the LossConfig.loss_fn value of the loss the loop is written for.

        2. Use the argument names of the mode you are in. They differ on purpose-historically:
            training  -> loss_fn, sampler
            validation -> loss_fn_inst, sampler_inst
            Everything else (monitor, model_inst, optimizer_inst, ...) is passed by __call__ as keyword.

        3. The model returns logits via its forward pass. Apply the head your loss needs via model_inst.head(name, <what you want>),
            see 'your_model_inst.HEADS' to see get an idea what is available.

        4. Return a dict with the keys: "loss", "predictions", "targets", "event_weights" , "composition_weights".
            Training loops add "grad_norm".
            loss = Value of the chosen Loss Function
            targets = Vector of your target
            event_weights = Product of all Weights on event base
            composition_weights = Transfer weight to convert an unsampled to an sampled version
            grad_norm = L2 norm of the whole gradient before clipping, see TrainingLoop.clip_gradient_norm

        5. Add a rule to FullConfig._compatibility_rules if your loss only works with certain models (heads).

    Remarks for each loop type:
        Training loops: sample one batch, then run: zero_grad -> forward -> loss -> backward -> clip -> step.
        Event weights are not part of the training loss, the oversampling takes care of that.

        Validation loops: run over all data via collect_from_generators (eval mode, no gradients) and weight the loss.
    """

    REGISTERED_LOOPS = {}
    MODE = None  # is overwritten by child class

    def __init__(
        self,
        full_config,
        necessary_columns: set[str] | None = None,
        *args,
        **kwargs):
        self.full_config = full_config


        self.loss_name = self.full_config.loss_config.loss_fn
        # resolve once at construction, so a loss without a loop fails in setup and not in the first iteration
        self.loop_fn = self.registered_loop(self.loss_name)

        # these column are always necessary for the loops
        self.necessary_columns = (
            {"continuous", "categorical", "targets"} if necessary_columns is None else necessary_columns
        )
        self.target_shape = (-1, len(self.full_config.dataset_config.target_map))
        self.target_map = self.full_config.dataset_config.target_map

    @property
    def batch_size(self):
        if self.MODE == "training":
            return self.full_config.training_config.training_batch_size
        else:
            return self.full_config.training_config.validation_batch_size

    def __init_subclass__(cls):
        for _name, fn in cls.__dict__.items():
            # only add methods if is part of self
            if getattr(fn, "_is_loop_method", False):
                cls.REGISTERED_LOOPS.setdefault(cls.MODE, {})  # make sure that dict for mode exists
                cls.REGISTERED_LOOPS[cls.MODE][fn._loss_name] = fn

    @classmethod
    def registered_loop(cls, loss_name: str):
        """
        Looks up the loop of this mode (training or validation) that belongs to *loss_name*.

        Args:
            loss_name (str): LossConfig.loss_fn value.

        Returns:
            Callable: The unbound loop method.

        Raises:
            KeyError: If no loop of this mode is registered for the loss.
        """
        loops = cls.REGISTERED_LOOPS.get(cls.MODE, {})
        if loss_name not in loops:
            raise KeyError(
                f"No {cls.MODE} loop registered for loss {loss_name!r}, available: {tuple(loops)}. "
                f"Decorate a {cls.__name__} method with @register_loop(loss={loss_name!r})."
            )
        return loops[loss_name]

    @staticmethod
    def yield_head_name(model_inst: torch.nn.Module) -> str:
        """
        Name of the head a yield based loss (Asimov) works on: binned when the model offers it, otherwise
        the plain probabilities. SignalEfficiency accepts both shapes.

        Args:
            model_inst (torch.nn.Module): Model with a ``HEADS`` declaration.

        Returns:
            str:  "binned" or "probability"
        """
        return "binned" if "binned" in model_inst.HEADS else "probabilities"

    def setup(self, monitor=None, model_inst=None, kind_of_data="batch"):
        if monitor is not None:
            monitor.set_mode(kind_of_data)
            monitor.start_step(kind_of_data)

    def cleanup(self, monitor, model_inst, **kwargs):
        pass

    def __call__(
        self,
        *args,
        monitor=None,
        model_inst=None,
        kind_of_data=None,
        scheduler_handler_inst=None,
        optimizer_inst=None,
        **kwargs,
    ):
        # registered loop is just an unbound function
        # bound by passing self as first argument
        self.setup(monitor=monitor, model_inst=model_inst, kind_of_data=kind_of_data)
        try:
            result = self.loop_fn(
                self, *args, monitor=monitor, model_inst=model_inst, optimizer_inst=optimizer_inst, **kwargs
            )

        finally:
            self.cleanup(
                monitor=monitor,
                model_inst=model_inst,
                scheduler_handler_inst=scheduler_handler_inst,
                optimizer_inst=optimizer_inst,
            )
        return result


class TrainingLoop(BaseLoop):
    MODE = "training"

    def __init__(self, full_config, *args, **kwargs):
        super().__init__(*args, **kwargs, full_config=full_config)

    def setup(self, monitor=None, model_inst=None, kind_of_data="batch"):
        super().setup(monitor, model_inst, kind_of_data)
        model_inst.train() # ensure that model is in training mode
        model_inst.enable_gradient_hooks()

    def cleanup(self, monitor=None, model_inst=None, optimizer_inst=None, scheduler_handler_inst=None, **kwargs):
        super().cleanup(monitor=monitor, model_inst=model_inst)
        if scheduler_handler_inst is not None:
            scheduler_handler_inst.step_batch()

        model_inst.disable_gradient_hooks()
        if monitor is not None:
            monitor.check_gradient_correctness(expected_names=model_inst.monitored_gradient_names())

    def clip_gradient_norm(self, model_inst: torch.nn.Module) -> torch.Tensor:
        """
        Clips the gradient of *model_inst* in place, call it between ``loss.backward()`` and ``optimizer.step()``.

        The gradients of all parameters are treated as one vector $g$. If its L2 norm exceeds
        ``optimizer_config.max_grad_norm`` ($c$), the whole vector is scaled by $c / \\|g\\|$, so its length
        becomes $c$ and its direction is kept. With ``max_grad_norm = None`` nothing is clipped, the norm
        is still computed and returned.

        Args:
            model_inst (torch.nn.Module): Model whose parameter gradients are clipped.

        Returns:
            torch.Tensor: L2 norm of the whole gradient BEFORE clipping, scalar tensor.
        """
        max_grad_norm = self.full_config.optimizer_config.max_grad_norm
        # an infinite bound never clips, but still gives the norm for the monitoring
        max_norm = float("inf") if max_grad_norm is None else max_grad_norm
        return torch.nn.utils.clip_grad_norm_(model_inst.parameters(), max_norm=max_norm)


    @register_loop(loss="cross_entropy")
    def cross_entropy_loss(
        self,
        model_inst,
        loss_fn,
        optimizer_inst,
        sampler,
        device,
        sample_columns=None,
        monitor=None,
        **kwargs,
    ):
        """
        Reference training loop, copy this when writing a new one.
        See BaseLoop for further instructions.
        """
        optimizer_inst.zero_grad()

        events = sampler.sample_batch(sample_from=sample_columns, device=device)
        categorical, continuous = events.get("categorical"), events.get("continuous")
        targets = events["targets"].reshape(self.target_shape)

        logits = model_inst(categorical_inputs=categorical, continuous_inputs=continuous)

        # training does not need event weights. The oversampling algorithm takes care of this
        # CrossEntropy loss has build in softmax
        loss = loss_fn(logits, targets, event_weights=None)

        loss.backward()

        grad_norm = self.clip_gradient_norm(model_inst)

        optimizer_inst.step()
        return {
            "loss": loss,
            "grad_norm": grad_norm,
            "predictions": model_inst.probabilities(logits),
            "targets": targets,
            "event_weights": events["product_of_weights"],
            "composition_weights": None, # training loops do not need composition weight, since their are already in sampled space
        }

    @register_loop(loss="signal_efficiency")
    def signal_efficiency_loop(
        self,
        model_inst,
        loss_fn,
        optimizer_inst,
        sampler,
        sample_columns,
        device,
        **kwargs,
    ):
        optimizer_inst.zero_grad()

        events = sampler.sample_batch(sample_from=sample_columns, device=device)
        categorical, continuous = events.get("categorical"), events.get("continuous")
        targets = events["targets"].reshape(self.target_shape)

        logits = model_inst(categorical_inputs=categorical, continuous_inputs=continuous)

        # loss expects binned probabilities, monitor expect unbinned probabilities
        loss = loss_fn(
            prediction=model_inst.head(self.yield_head_name(model_inst), logits),
            truth=targets,
            product_of_weights=events["product_of_weights"],
            evaluation_mask=events["evaluation_space_mask"],
        )

        loss.backward()

        grad_norm = self.clip_gradient_norm(model_inst)

        optimizer_inst.step()
        return {
            "loss": loss,
            "grad_norm": grad_norm,
            "predictions": model_inst.probabilities(logits),
            "targets": targets,
            "event_weights": events["product_of_weights"],
            "composition_weights": None, # training loops do not need composition weight, since their are already in sampled space
        }


class ValidationLoop(BaseLoop):
    MODE = "validation"

    def __init__(self, full_config, *args, **kwargs):
        super().__init__(*args, **kwargs, full_config=full_config)

    def setup(self, monitor=None, model_inst=None, kind_of_data="batch"):
        super().setup(monitor, model_inst, kind_of_data)
        # evaluation never look into gradients
        model_inst.eval() # ensure that model is in validation mode
        model_inst.disable_gradient_hooks()

    def cleanup(self, monitor, model_inst, **kwargs):
        super().cleanup(monitor=monitor, model_inst=model_inst)

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
        During Validation no sampling is used, instead we loop over all unsampled Datasets.
        The *model_inst* is the DNN you want to use, while *sampler_inst* should be a ValidationSampler.
        Which column is aggregated over all datasets is defined in *sample_columns*.
        In the end a concatenation is done over the event axis. If this step should be skipped for
        certain columns (e.g. to maintain the shape), add the name in *skip_concatenate_columns*.
        Only the logits are collected under "logits". The heads (probabilities, binned) act per event,
        so the caller applies them once on the concatenated logits instead of once per batch.

        Args:
            model_inst (torch.nn.Module): Your neural network model.
            sampler_inst (ProcessSampler): Instance of a ProcessSampler, with registered Processes.
            sample_columns (list[str]): List of columns to sample from.
            skip_concatenate_columns (tuple[str], optional): List of columns to skip concatenation. Defaults to ("",).
            device (torch.device, optional): Device everything should run on. Defaults to torch.device("cpu").

        Returns:
            dict[str, torch.tensors]: Dictonary with all aggregated tensors.
        """
        device = torch.device(device) if isinstance(device, str) else device

        collected_data = {
            "targets": [],
            "sample_weights": [],
            "post_relative_weights": [],
            "process_id": [],
            "logits": []
        }

        dynamic_data = {dynamic_d: [] for dynamic_d in tuple(sample_columns - self.necessary_columns)}

        with torch.no_grad():
            # sample over all dataset generator and run network
            # store output in lists which are then concatenated in the end
            for uid, events in sampler_inst.full_pass(
                batch_size=self.batch_size, sample_from=sample_columns, device=device
            ):
                # hold data for current dataset, saved in loop to avoid memory issues
                current_process = sampler_inst.registry[uid]
                logits = model_inst(categorical_inputs=events.pop("categorical"), continuous_inputs=events.pop("continuous"))

                num_events = logits.shape[0]

                # -- collect data that is always done --

                collected_data["targets"].append(events.pop("targets"))
                collected_data["sample_weights"].append(events.pop("sample_weights"))

                # batch share divided by process size: summed over the full pass each process
                # weighs exactly as much as in a training batch, see Process.per_event_composition_weight
                collected_data["post_relative_weights"].append(
                    current_process.per_event_composition_weight(num_events, denominator=len(current_process)
                    ).to(device)
                )

                collected_data["process_id"].append(
                    torch.full((num_events, 1), uid[1], device=device)
                    )

                collected_data["logits"].append(logits)

                # -- collect dynamic data --
                for k in dynamic_data.keys():
                    dynamic_data[k].append(events.pop(k))

            # combine datas
            collected_data.update(dynamic_data)

            for k in tuple(collected_data.keys()):
                value = collected_data.pop(k)
                # event dimension is always the second last axis
                event_dim = len(value[0].shape) - 2
                # for whatever reason maybe want to skip concatenate?
                if k not in skip_concatenate_columns:
                    value = torch.concatenate(value, dim=event_dim)
                collected_data[k] = value
        return collected_data

    @register_loop(loss="cross_entropy")
    def cross_entropy_loop(
        self,
        model_inst,
        loss_fn_inst,
        sampler_inst,
        sample_columns,
        device,
        monitor=None,
        *args,
        **kwargs,
    ):
        """
        Reference training loop, copy this when writing a new one.
        See BaseLoop for further instructions.
        """
        to_sample_columns = self.necessary_columns.union(sample_columns)
        tensors = self.collect_from_generators(
            model_inst=model_inst,
            sampler_inst=sampler_inst,
            sample_columns=to_sample_columns,
            skip_concatenate_columns=("",),
            device=device,
        )
        # validation runs on unsampled data: weighted by batch share / process size so the
        # loss reproduces the training batch composition and is comparable to the batch loss
        loss = loss_fn_inst(
            tensors["logits"],
            tensors["targets"],
            tensors["post_relative_weights"],
        )
        return {
            "loss": loss,
            "predictions": model_inst.probabilities(tensors["logits"]),
            "targets": tensors["targets"],
            "event_weights": tensors["product_of_weights"],
            "composition_weights": tensors["post_relative_weights"],
        }

    @register_loop(loss="signal_efficiency")
    def signal_efficiency_loop(
        self,
        model_inst,
        loss_fn_inst,
        sampler_inst,
        sample_columns,
        device,
        monitor=None,
        *args,
        **kwargs,
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

        # heads are applied once on the full pass, logits were collected without gradients
        loss = loss_fn_inst(
            model_inst.head(self.yield_head_name(model_inst), tensors["logits"]),
            tensors["targets"].reshape(self.target_shape),
            tensors["product_of_weights"],
            tensors["evaluation_space_mask"],
            # TODO  sampler weight is not used, but maybe should ?
        )
        return {
            "loss": loss,
            "predictions": model_inst.probabilities(tensors["logits"]),
            "targets": tensors["targets"],
            "event_weights": tensors["sample_weights"],
            "composition_weights": tensors["post_relative_weights"],
        }
