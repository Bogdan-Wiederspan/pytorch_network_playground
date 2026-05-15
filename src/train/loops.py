from __future__ import annotations

import torch

from utils import logger

functions = {}
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
    Base Class that implements are registry where registered loops are cateogrized
    as training and validation loops.
    The registry is used to check if a given loop function exists and to call the correct function in the training script.
    """
    REGISTERED_LOOPS = {}
    MODE = None # is overwitten by child class

    def __init__(self, full_config):
        self.which_fn = (
            full_config.training_config.training_fn
            if self.MODE == "training"
            else full_config.training_config.validation_fn
            )

    def __init_subclass__(cls):
        for name, fn in cls.__dict__.items():
            # only add methods if is part of self
            if getattr(fn, "_is_loop_method", False):
                cls.REGISTERED_LOOPS.setdefault(cls.MODE, {}) # make sure that dict for mode exists
                cls.REGISTERED_LOOPS[cls.MODE][fn._loop_name] = fn

    def prepare_binned_output(pred: torch.tensor | tuple[torch.tensor]) -> tuple[torch.tensor, torch.tensor]:
        """
        Depending on the network architecture the output can either be a tuple of prediction and binned prediction
        or just the prediction. To have a uniform interface the singular output uses a dummy value.

        Args:
            pred (torch.tensor | tuple[torch.tensor]): _description_

        Returns:
            tuple[torch.tensor, torch.tensor]: Tuple of predictions where first value is used for logging and second value is used for optimization.
            In case of non binned output both values are the same.
        """
        is_binned = isinstance(pred, tuple)
        if is_binned:
            logging_pred, optimization_pred = pred
        else:
            logging_pred, optimization_pred = pred, pred
        return logging_pred, optimization_pred


    def __call__(self, *args, **kwargs):

        fn = getattr(self, self.which_fn)
        return fn(*args, **kwargs)


class TrainingLoop(BaseLoop):
    MODE = "training"

    def __init__(self, full_config, *args, **kwargs):
        super().__init__(full_config=full_config, *args, **kwargs)

    @register_loop(name="sam")
    def sam_optimizer(self, model, loss_fn, optimizer, sampler, device):
        optimizer.zero_grad()

        cont, cat, targets = sampler.sample_batch(device=device)
        pred = model(categorical_inputs=cat, continuous_inputs=cont)
        logging_pred, optimization_pred = self.prepare_binned_output(pred)

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

    @register_loop(name="default")
    def default(self, model, loss_fn, optimizer, sampler, sample_from, device, scheduler_inst=None):
        optimizer.zero_grad()

        events = sampler.sample_batch(sample_from=sample_from, device=device)
        pred = model(categorical_inputs=events.pop("categorical"), continuous_inputs=events.pop("continuous"))

        logging_pred, optimization_pred = self.prepare_binned_output(pred)

        targets = events.pop("targets")
        loss = loss_fn(optimization_pred, targets.reshape(-1,3), events)
        loss.backward()
        optimizer.step()
        if scheduler_inst is not None:
            scheduler_inst.step()

        return loss, (logging_pred, targets)

class ValidationLoop(BaseLoop):
    MODE = "validation"
    def __init__(self, full_config, *args, **kwargs):
        super().__init__(full_config=full_config, *args, **kwargs)

    @register_loop(name="default")
    def default(self, model, loss_fn, sampler, sample_from, device):
        """
        Used for normal loss functions like Crossentropy.
        """
        # TODO: MAYBE BUGGED
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
                    pred = model(categorical_inputs=events.pop("categorical"), continuous_inputs=events.pop("continuous"))
                    logging_pred, optimization_pred = self.prepare_binned_output(pred)

                    targets = events.pop("targets")
                    loss = loss_fn(optimization_pred, targets.reshape(-1,3), events, debug=False)
                    # collect data for metric plots
                    dataset_losses.append(loss)
                    # predictions.append(torch.softmax(val_pred, dim=1).cpu())
                    predictions.append(logging_pred).cpu()
                    truth.append(targets.cpu())
                    weights.append(torch.full(size=(logging_pred.shape[0], 1), fill_value=sampler[uid].relative_weight).cpu())

                process_contribution = sampler[uid].relative_weight
                average_val = sum(dataset_losses) / len(dataset_losses) * process_contribution
                val_loss.append(average_val)

            final_validation_loss = sum(val_loss).cpu()#/ len(val_loss)
            model.train()

            truth = torch.concatenate(truth, dim=0)
            predictions = torch.concatenate(predictions, dim=0)
            weights = torch.flatten(torch.concatenate(weights, dim=0))
            return final_validation_loss, (predictions, truth, weights)


    def gather_from_(self, model, loss_fn, sampler, sample_from, collect ,device):
        model.eval()

        collected_data = {"predictions": [], "targets": [], "weights": []}
        collected_data.update({key: [] for key in collect.keys()})

        with torch.no_grad():
            # get constant process factors and filter signal or background out
            for uid, validation_batch_generator in sampler.create_sample_generator(
                batch_size=sampler.batch_size, # do validation over whole dataset at once
                sample_from=sample_from,
                device=device).items():
                dataset_losses = []

                for events in validation_batch_generator:
                    pred = model(categorical_inputs=events.pop("categorical"), continuous_inputs=events.pop("continuous"))
                    logging_pred, optimization_pred = self.prepare_binned_output(pred)
                    targets = events.pop("targets")
                    t.append(targets)
                    e["product_of_weights"].append(events.pop("product_of_weights"))
                    e["evaluation_space_mask"].append(events.pop("evaluation_space_mask"))

                    # collect data for metric plots
                    # TODO: outsource softmax for all things
                    predictions.append(torch.softmax(logging_pred, dim=1))
                    truth.append(targets)
                    weights.append(torch.full(size=(logging_pred.shape[0], 1), fill_value=sampler[uid].relative_weight))


    @register_loop(name="signal_efficiency")
    def signal_efficiency(self, model, loss_fn, sampler, sample_from, device):
        with torch.no_grad():
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
                    pred = model(categorical_inputs=events.pop("categorical"), continuous_inputs=events.pop("continuous"))
                    logging_pred, optimization_pred = self.prepare_binned_output(pred)

                    p.append(optimization_pred)
                    targets = events.pop("targets")
                    t.append(targets)
                    e["product_of_weights"].append(events.pop("product_of_weights"))
                    e["evaluation_space_mask"].append(events.pop("evaluation_space_mask"))

                    # collect data for metric plots
                    # TODO: outsource softmax for all things
                    predictions.append(torch.softmax(logging_pred, dim=1))
                    truth.append(targets)
                    weights.append(torch.full(size=(logging_pred.shape[0], 1), fill_value=sampler[uid].relative_weight))

            _dim = len(p[0].shape)
            if _dim == 3: #then it means that we have a binned output, concat happens over the 1 axis
                event_dim = 1
            elif _dim == 2: # then we have a normal output, concat happens over the 0 axis
                event_dim = 0
            p = torch.concatenate(p, dim=event_dim)

            t = torch.concatenate(t, dim=0)
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
            return final_validation_loss, (predictions.cpu(), truth.cpu(), weights.cpu())
