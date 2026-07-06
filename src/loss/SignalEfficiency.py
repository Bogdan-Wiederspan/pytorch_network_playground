
from statistics.asimov import asimov, asimov_no_background, asimov_small_signal_and_no_background

import torch

from models.binning import GaussianKernelV2

# mapping of asimov functions to a name
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



class BinningAwareSignificance(SignalEfficiency):
    def __init__(
        self,
        model_inst: torch.nn.Module=None,
        bins: torch.tensor=None,
        binning_config: dict=None,
        *args,
        **kwargs
        ):
        """
        Extension of SignalEfficiency Loss where weighted predictions are scaled by a Gaussian binning kernel.
        The kernel is configured using a *binning_cfg*, the actual bin edges are defined in *bins*.

        Args:
            bins (torch.tensor): _description_
            binning_cfg (dict, optional): _description_. Defaults to None.
        """

        super().__init__(*args, **kwargs)
        if model_inst is not None:
            pass
        self.edges = bins
        if self.edges is not None:
            self.build_kernels()

        self.binning_config = self.binning_config

    def reduce_yield():
        raise NotImplementedError

    def build_kernels(self):
        kernels = []
        for bin_num, (low, upper) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            if bin_num == 0:
                bin_type = "underflow"
            elif bin_num == len(self.edges) - 2:
                bin_type = "overflow"
            else:
                bin_type = "normal"
            k = GaussianKernelV2(
                initial_edge=(low, upper),
                bin_type=bin_type,
                **self.binning_config,
            )
            kernels.append(k)
        self.kernels = kernels


    def digitize_masks(self, x, bin_edges, include_left_edge = True):
        # TODO maybe Delete
        # create masks for each bin and save them in a mask dictionary, where the key is the bin number
        # torch implemented right as "right border is open" and "left is closed"
        indices = torch.bucketize(x, torch.tensor(bin_edges).to(x.device), right=include_left_edge)

        underflow_bin_number = 0
        overflow_bin_number = len(bin_edges)

        masks = {}
        for bin_number in range(underflow_bin_number, overflow_bin_number + 1):
            masks[bin_number] = (indices == bin_number)
        return masks

    def approximation_sb(
        self,
        prediction: torch.tensor,
        truth: torch.tensor,
        product_of_weights: dict[torch.tensor],
        evaluation_phase_space_mask: dict[torch.tensor],
        is_signal: bool,
        epsilon=None
        ):
        """
        Approximation of (s)ignal and (b)ackground yield as defined in 4.1 and 4.2 in https://arxiv.org/abs/1806.00322
        The approximation is calculated batchwise and only defined for binary classification, which is calculated is defined by *is_signal*.
        The physics weight information is handed over by *product_of_weights* and *evaluation_phase_space_mask*.
        The actual network output is handed over by *prediction* and *truth*.
        Args:
            prediction (torch.tensor): torch tensor coming from model output, containing the predicted probabilities for each class.
            truth (torch.tensor): torch tensor containing the true class labels, one-hot encoded.
            product_of_weights (dict[torch.tensor]): Dictionary of torch tensors containing the product of all weights for different processes
            evaluation_phase_space_mask (dict[torch.tensor]): Dictionary of torch tensors containing a mask for the evaluation phase space, which is 1 for events in the evaluation phase space and 0 otherwise
            is_signal (bool): bool to signal if calculation for s or b is done

        Returns:
            torch.tensor: torch tensor to be either s or b, depending on *is_signal*
        """
        # get parts of approximation of s and b and then scale the prediction part
        transfer_factor, weighted_predictions = self.approximation_sb_parts(
        prediction=prediction,
        truth=truth,
        product_of_weights=product_of_weights,
        evaluation_phase_space_mask=evaluation_phase_space_mask,
        is_signal=is_signal,
        )

        # apply binning scaling via kernel multiplication
        # for each kernel a scaling is calculated
        # in the end a sum over all kernel result is done
        binned_yield = []
        for _kernel in self.kernels:
            scale = _kernel(prediction) # tensor of shape len(events)
            weighted_yield = torch.sum(weighted_predictions * scale) # term 1
            binned_yield.append(weighted_yield * transfer_factor)
        binned_yield = torch.stack(binned_yield, dim=0)
        if epsilon is not None:
            self.stabilize(binned_yield, epsilon)
        return binned_yield

    def stabilize(y, eps):
        return torch.clamp(y, min=eps)

    def forward(self, prediction, truth, product_of_weights, evaluation_mask):
        signal_node_truth = truth[:, self.s_cls]
        signal_node_prediction = prediction[:, self.s_cls]
        # signal can be 0, since this will yield 0
        s = self.approximation_sb(
            signal_node_prediction,
            signal_node_truth,
            product_of_weights,
            evaluation_mask,
            is_signal=True,
            epsilon=1e-4
            )
        # background cant be 0 -> inf
        b = self.approximation_sb(
            signal_node_prediction,
            signal_node_truth,
            product_of_weights,
            evaluation_mask,
            is_signal=False,
            epsilon=1e-4
            )
        loss = self.asimov_fn(s=s, b=b, unc_b=0, epsilon=1e-2)
        loss = loss.sum()
        return 1 / loss
