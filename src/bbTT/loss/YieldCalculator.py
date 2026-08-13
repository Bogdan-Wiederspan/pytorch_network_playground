import torch


class YieldCalculator:
    def __init__(self, sampler_inst, training=True):
        self.training = training
        self.total_product_of_weights = sampler_inst.weights_aggregator_inst("product_of_weights", "whole_sum")
        self.eval_weights, self.total_process_weights = self._get_eval_weights(sampler_inst=sampler_inst)

    def _get_eval_weights(self, sampler_inst):
        if self.training:
            return None, None
        # validation mode goes over whole validation phase space and not only over a batch
        # this information is stored in the sampler instance.
        eval_weights = sampler_inst.weights_aggregator_inst("product_of_weights", "evaluation_sum")
        total_process_weights = sampler_inst.weights_aggregator_inst("product_of_weights", "validation_sum")

        return eval_weights, total_process_weights

    def _compute_transfer_factor(
        self,
        event_weight: torch.Tensor,
        evaluation_mask: dict[torch.Tensor],
        is_signal: bool
        ) -> torch.Tensor:
        """
        Calculate the Transfer Factor which moves result from batch space to total evaluation space.
        Computation changes in training and evaluation mode.
        During Training calculation is done on batch base.
        During Validation no batch exist, but whole data phase space is used instead

        Args:
            event_weight (torch.Tensor): Filtered weight for Signal or Background
            evaluation_mask (dict[torch.Tensor]): Mask to describe evaluation events
            is_signal (bool): Flag to indicate if event belongs to signal node or not.

        Returns:
            torch.Tensor: Transfer weight factor to describe movement from batch to evaluation space.

        """
        if self.training:
            # when in trainings mode, calculate factors on batch base
            eval_yield = torch.sum(event_weight * evaluation_mask) # term 3
            batch_yield = torch.sum(event_weight) # term 2
        else:
            # for validation a batch cover whole validation space, not just a batch
            # these reduces the factors to constants, where s is hh and b is dy + tt factor
            if is_signal:
                eval_yield = self.eval_weights["hh"]
                batch_yield = self.total_process_weights["hh"]
            else:
                eval_yield = self.eval_weights["dy"] + self.eval_weights["tt"]
                batch_yield = self.total_process_weights["dy"] + self.total_process_weights["tt"]

        return eval_yield  / batch_yield

    def compute_yield(
        self,
        prediction: torch.Tensor,
        truth: torch.Tensor,
        event_weight: torch.Tensor,
        evaluation_mask: torch.Tensor,
        is_signal: bool,
        ) -> torch.Tensor:
        """
        Computes s or b yield depending on if *is_signal* is set true or false.

        Args:
            prediction (torch.Tensor): _description_
            truth (torch.Tensor): _description_
            event_weight (torch.Tensor): _description_
            evaluation_mask (torch.Tensor): _description_
            is_signal (bool): _description_

        Returns:
            _type_: _description_
        """

        if is_signal:
            selector =  truth
        else:
            selector = 1 - truth

        selected_weight = selector * event_weight
        weighted_prediction = prediction * selected_weight

        transfer_factor = self._compute_transfer_factor(
            event_weight = selected_weight,
            evaluation_mask = evaluation_mask,
            is_signal = is_signal
        )
        reduced_prediction = weighted_prediction.sum(dim=-1)
        return transfer_factor * reduced_prediction

    def signal_yield(
        self,
        prediction: torch.Tensor,
        truth: torch.Tensor,
        event_weight: torch.Tensor,
        evaluation_mask: torch.Tensor,
        ) -> torch.Tensor:
        return self.compute_yield(
            prediction=prediction,
            truth=truth,
            event_weight=event_weight,
            evaluation_mask=evaluation_mask,
            is_signal=True,
        )

    def background_yield(
        self,
        prediction: torch.Tensor,
        truth: torch.Tensor,
        event_weight: torch.Tensor,
        evaluation_mask: torch.Tensor,
        ) -> torch.Tensor:
        return self.compute_yield(
            prediction=prediction,
            truth=truth,
            event_weight=event_weight,
            evaluation_mask=evaluation_mask,
            is_signal=False,
        )

    def __call__(self, prediction, truth, event_weight, evaluation_mask):
        s = self.signal_yield(
            prediction=prediction,
            truth=truth,
            event_weight=event_weight,
            evaluation_mask=evaluation_mask,
        )
        b = self.background_yield(
            prediction=prediction,
            truth=truth,
            event_weight=event_weight,
            evaluation_mask=evaluation_mask,
        )
        return s, b
