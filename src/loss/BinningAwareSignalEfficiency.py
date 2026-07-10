
from statistics.asimov import asimov, asimov_no_background, asimov_small_signal_and_no_background

import torch

from models.binning import GaussianKernel  # mapping of asimov functions to a name

asimov_functions = {
    "full" : asimov,
    "no_unc" : asimov_no_background,
    "approximation" : asimov_small_signal_and_no_background,
    }

from .SignalEfficiency import SignalEfficiency


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
            k = GaussianKernel(
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
