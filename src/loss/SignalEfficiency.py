
from statistics.asimov import asimov, asimov_no_background, asimov_small_signal_and_no_background

import torch

asimov_functions = {
    "full" : asimov,
    "no_unc" : asimov_no_background,
    "approximation" : asimov_small_signal_and_no_background,
    }


class SignalEfficiency(torch.nn.Module):
    def __init__(
        self,
        sampler_inst,
        device,
        train=True,
        asimov_cfg=None,
        *args,
        **kwargs,
        ) -> torch.tensor:
        """
        This function creates a loss using the Asimov Significance as introduced in https://arxiv.org/abs/1806.00322.

        The loss comes in three *modes*: "full", "no_unc", "approximation".
        The loss is calculated differently in train and validation mode, which is set by *train* flag.
        All tensors are placed on *device*.
        Information about phase space and weights is contained in the *sampler_inst*, which is defined Sampler.py.
        *uncertainty* sets the relative uncertainty of the background yield, everything above 1 is a constant uncertainty.

        Args:
            sampler_inst: Sampler instance that provides necessary information about phase space and weights.
            device: Device on which the tensors are placed.
            train: Bool to signal if the loss is used in training or validation mode, which changes the way the loss is calculated.
            mode: String to signal which implementation of the Asimov Significance is used, choose between "full", "no_unc", "approximation".
            uncertainty: Float to signal the level of uncertainty, if between 0 and 1 a relative uncertainty of

        Returns: (torch.tensor): Torch Tensor Scalar defining the loss.

        """
        super().__init__()

        self.train = train
        self.total_product_of_weights = sampler_inst.weights_aggregator_inst("product_of_weights", "whole_sum")
        self.target_map = sampler_inst.target_map

        self.asimov_name = asimov_cfg.asimov_mode
        self.asimov_fn = asimov_functions[self.asimov_name]
        self.uncertainty = asimov_cfg.background_uncertainty
        self.asimov_epsilon = {
            "eps_log" : asimov_cfg.epsilon_log,
            "eps_sqrt" : asimov_cfg.epsilon_sqrt,
            "eps" : asimov_cfg.epsilon_small_signal,
        }

        self.s_cls = self.target_map["hh"] # definition of signal class
        self.device = device

        self.eval_weights = None
        self.total_process_weights = None
        if not train:
            # validation mode goes over whole validation phase space and not only over a batch
            # this information is stored in the sampler instance.
            self.eval_weights = sampler_inst.weights_aggregator_inst("product_of_weights", "evaluation_sum")
            self.total_process_weights = sampler_inst.weights_aggregator_inst("product_of_weights", "validation_sum")

    def _uncertainty(self, b):
        if self.uncertainty >=1:
            return self.uncertainty
        return self.uncertainty * b + 1

    def build_selected_weight_and_prediction(
        self,
        truth: torch.Tensor,
        weights: dict[torch.Tensor],
        is_signal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Filter *weights* after signal or background events.
        The selection depends on the *is_signal* flag.

        Args:
            prediction (torch.Tensor): SoftMaxed model output.
            truth (torch.Tensor): True class labels, one-hot encoded.
            weights (dict[torch.Tensor]): Dictionary holding all weights for different processes
            is_signal (bool): bool to signal if calculation for s or b is done

        Returns:
            tuple[torch.Tensor]: Filtered weights, every other entry is 0.
        """
        selector = truth if is_signal else (1 - truth)
        selected_weight = selector * weights
        return selected_weight

    def compute_transfer_factor(
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
        if self.train:
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


    def reduce_yield(self, event):
        # sum over all predictions in bins
        return torch.sum(event, dim = -1)

    def stabilize(self, x, eps):
        return torch.clamp(x, min=eps)

    def _loss(self, x):
        return 1 / x.sum()

    def loss(self, *args, **kwargs):
        return self._loss(*args, **kwargs)

    def forward(self, prediction, truth, product_of_weights, evaluation_mask):
        # prediction can be of shape [bin, event, node] or [event, node]
        signal_node_prediction = prediction[..., self.s_cls] # can be 2D or 1D
        signal_node_truth = truth[..., self.s_cls] # is 1D

        # filter weights after signal and background only events
        s_weights = self.build_selected_weight_and_prediction(
            truth=signal_node_truth,
            weights=product_of_weights,
            is_signal=True
        )

        b_weights = self.build_selected_weight_and_prediction(
            truth=signal_node_truth,
            weights=product_of_weights,
            is_signal=False
        )
        # calculate weighted predictions for signal and background
        s_weighted_prediction = s_weights * signal_node_prediction
        b_weighted_prediction = b_weights * signal_node_prediction

        # compute transfer factor
        tf_s = self.compute_transfer_factor(
            event_weight=s_weights,
            evaluation_mask=evaluation_mask,
            is_signal=True
        )

        tf_b = self.compute_transfer_factor(
            event_weight=b_weights,
            evaluation_mask=evaluation_mask,
            is_signal=False
        )

        # reduction
        s = tf_s * self.reduce_yield(s_weighted_prediction)
        b = tf_b * self.reduce_yield(b_weighted_prediction)

        asimov = self.asimov_fn(
            s=s,
            b=b,
            unc_b=self._uncertainty(b),
            **self.asimov_epsilon
            )
        loss = self.loss(asimov)
        return loss
