from __future__ import annotations

import torch
import kernel
import binning
from utils import logger

logger_inst = logger.get_logger(__name__)


class SignalEfficiency(torch.nn.Module):
    def __init__(
        self,
        sampler_inst,
        device,
        train=True,
        mode="full",
        uncertainty=0.1,
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
        self.uncertainty = uncertainty

        self.mode = mode
        self.loss = self._loss()

        self.s_cls = self.target_map["hh"] # definition of signal class
        self.device = device

        self.eval_weights = None
        self.total_process_weights = None
        if not train:
            # validation mode goes over whole validation phase space and not only over a batch
            # this information is stored in the sampler instance.
            self.eval_weights = sampler_inst.weights_aggregator_inst("product_of_weights", "evaluation_sum")
            self.total_process_weights = sampler_inst.weights_aggregator_inst("product_of_weights", "validation_sum")

    def _loss(self):
        """
        Function that returns the correct loss function according to the set mode of the instance.

        Raises:
            ValueError: If mode is not one of "full", "no_unc", "approximation"

        Returns:
            function: The loss function corresponding to the set mode.
        """
        losses = {
            "full" : self.asimov_full,
            "no_unc" : self.asimov_no_uncertainty,
            "approximation" : self.asimov_small_signal_approximation,
            }

        if self.mode in losses:
            return losses[self.mode]
        raise ValueError(f"Unknown mode {self.mode} for SignalEfficiency loss, choose between {tuple(losses.keys())}")

    def _uncertainty(self, b):
        if self.uncertainty >=1:
            return self.uncertainty
        return self.uncertainty * b + 1

    def asimov_small_signal_approximation(self, s, b, *args, **kwargs):
        # approximation for small signal yield  3.3
        return s / torch.sqrt(s + b)

    def asimov_no_uncertainty(self, s, b, epsilon=0, *args, **kwargs):
        # approximation coming from asimov for no background uncertainty: https://arxiv.org/abs/1806.00322 3.2
        return torch.sqrt( (2 * ((s + b) * torch.log(1 + s / b) - s)))

    def asimov_full(self, s, b, unc_b, *args, **kwargs):
        # full asimov formula with background uncertainty: https://arxiv.org/abs/1806.00322 3.1
        ln_nominator_1 = ( s + b ) * ( b + unc_b**2 )
        ln_denominator_1 = ( b**2 + ( s + b ) )
        part_1 = ( s + b ) * torch.log( ln_nominator_1 / ln_denominator_1 )

        ln_nominator_2 = 1 + (unc_b**2 * s)
        ln_denominator_2 = b * ( b + unc_b**2 )
        part_2 = b**2 / unc_b**2 * torch.log( ln_nominator_2 / ln_denominator_2)

        _asimov = torch.sqrt(
        2 * ( part_1 - part_2 )
        )
        return _asimov

    def approximation_sb_parts(
        self,
        prediction: torch.tensor,
        truth: torch.tensor,
        product_of_weights: dict[torch.tensor],
        evaluation_phase_space_mask: dict[torch.tensor],
        is_signal: bool
        ) -> tuple[torch.tensor]:
        """
        Calculate the different factors of the Signal and Background Approximation.
        The resulting tuple contains a weight transfer factor and the weighted predictions, that still needs to be summed up.

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
            tuple(torch.tensor): tuple of torch tensor describing: (transfer factor from batch to evaluation, weighted predictions)
        """
        # create filter to filter all results after signal or background
        # since filtered batch is 0 for either signal or background this reduces the sum
        # to be specifically only over signal OR background events
        signal_background_filter = truth if is_signal else (1 - truth)
        filtered_signal_background_batch_weight = signal_background_filter * product_of_weights

        # term 1
        weighted_predictions = prediction * filtered_signal_background_batch_weight

        # calculate sum of weights
        if self.train:
            # when in trainings mode, calculate factors on batch base
            # term 3
            evaluation_phase_space_yield = torch.sum(filtered_signal_background_batch_weight * evaluation_phase_space_mask)
            # term 2
            batch_yield = torch.sum(filtered_signal_background_batch_weight)
        else:
            # for validation batches cover whole validation space, not just a batch
            # these reduces the factors to constants, where s is hh and b is dy + tt factor
            # for background or signal different
            if is_signal:
                evaluation_phase_space_yield = self.eval_weights["hh"]
                batch_yield = self.total_process_weights["hh"]
            else:
                evaluation_phase_space_yield = self.eval_weights["dy"] + self.eval_weights["tt"]
                batch_yield = self.total_process_weights["dy"] + self.total_process_weights["tt"]
        weight_transfer_factor_from_batch_to_evaluation = evaluation_phase_space_yield  / batch_yield

        return weight_transfer_factor_from_batch_to_evaluation, weighted_predictions


    def approximation_sb(
        self,
        prediction: torch.tensor,
        truth: torch.tensor,
        product_of_weights: dict[torch.tensor],
        evaluation_phase_space_mask: dict[torch.tensor],
        is_signal: bool
        ):
        """
        Function to assemble together the result of the approximation parts.

        Args:
            prediction (torch.tensor): torch tensor coming from model output, containing the predicted probabilities for each class.
            truth (torch.tensor): torch tensor containing the true class labels, one-hot encoded.
            product_of_weights (dict[torch.tensor]): Dictionary of torch tensors containing the product of all weights for different processes
            evaluation_phase_space_mask (dict[torch.tensor]): Dictionary of torch tensors containing a mask for the evaluation phase space, which is 1 for events in the evaluation phase space and 0 otherwise
            is_signal (bool): bool to signal if calculation for s or b is done

        Returns:
            torch.tensor: torch tensor to be either s or b, depending on *is_signal*
        """
        transfer_factor, weighted_predictions = self.approximation_sb_parts(
            prediction=prediction,
            truth=truth,
            product_of_weights=product_of_weights,
            evaluation_phase_space_mask=evaluation_phase_space_mask,
            is_signal=is_signal,
        )
        weighted_yield = torch.sum(weighted_predictions)
        return transfer_factor * weighted_yield


    # TODO CHECK again

    def main(self, prediction, truth, weight, *args, **kwargs):
        signal_node_truth = truth[:, self.s_cls]
        prediction = torch.softmax(prediction, dim = 1) # network does not contain softmax
        signal_node_prediction = prediction[:, self.s_cls]

        s = self.approximation_sb(signal_node_prediction, signal_node_truth, weight["product_of_weights"], weight["evaluation_space_mask"], is_signal=True)
        b = self.approximation_sb(signal_node_prediction, signal_node_truth, weight["product_of_weights"], weight["evaluation_space_mask"], is_signal=False)

        if (s <= 0 or b <= 0):
            from IPython import embed; embed(header = "Either s or b is below 0, which is not physical, starting debugger in Forward of SignalEfficiency loss")
        uncertainty = self._uncertainty(b)

        loss = self.loss(s= s, b=b, unc_b=uncertainty)
        if loss == 0:
            return 0
        return 1 / loss
    # TODO REPLACE

    def forward(self, prediction, truth, weight, *args, **kwargs):
        signal_node_truth = truth[:, self.s_cls]
        prediction = torch.softmax(prediction, dim = 1) # network does not contain softmax
        signal_node_prediction = prediction[:, self.s_cls]

        s = self.approximation_sb(signal_node_prediction, signal_node_truth, weight["product_of_weights"], weight["evaluation_space_mask"], is_signal=True)
        b = self.approximation_sb(signal_node_prediction, signal_node_truth, weight["product_of_weights"], weight["evaluation_space_mask"], is_signal=False)

        if (s <= 0 or b <= 0):
            from IPython import embed; embed(header = "Either s or b is below 0, which is not physical, starting debugger in Forward of SignalEfficiency loss")
        uncertainty = self._uncertainty(b)
        loss = self.loss(s= s, b=b, unc_b=uncertainty)
        if loss == 0:
            return 0
        if torch.isnan(1 / loss):
            from IPython import embed; embed(header = "IsNan line: 176 in loss.py")

        return 1 / loss

class BinningAwareSignificance(SignalEfficiency):
    def __init__(
        self,
        bins: torch.tensor,
        binning_cfg: dict=None,
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

        self.edges = bins
        kernels = []
        for bin_num, (low, upper) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            if bin_num == 0:
                bin_type = "underflow"
            elif bin_num == len(self.edges) - 2:
                bin_type = "overflow"
            else:
                bin_type = "normal"
            k = kernel.GaussianKernelV2(
                initial_edge=(low, upper),
                bin_type=bin_type,
                **binning_cfg
            )
            kernels.append(k)

        self.kernels = kernels

    def digitize_masks(self, x, bin_edges, include_left_edge = True):
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
        for kernel in self.kernels:
            scale = kernel(prediction) # tensor of shape len(events)
            weighted_yield = torch.sum(weighted_predictions * scale) # term 1
            binned_yield.append(weighted_yield * transfer_factor)
        binned_yield = torch.stack(binned_yield, axis=0)
        if epsilon is not None:
            binned_yield = torch.clamp(binned_yield, min=epsilon)
        return binned_yield

    def forward(self, prediction, truth, weight, *args, **kwargs):
        signal_node_truth = truth[:, self.s_cls]
        prediction = torch.softmax(prediction, dim = 1) # network does not contain softmax
        signal_node_prediction = prediction[:, self.s_cls]
        # signal can be 0, since this will yield 0
        s = self.approximation_sb(
            signal_node_prediction,
            signal_node_truth,
            weight["product_of_weights"],
            weight["evaluation_space_mask"],
            is_signal=True,
            epsilon=1e-3
            )
        # background cant be 0 -> inf
        b = self.approximation_sb(
            signal_node_prediction,
            signal_node_truth,
            weight["product_of_weights"],
            weight["evaluation_space_mask"],
            is_signal=False,
            epsilon=1e-3
            )
        loss = self.loss(s=s, b=b, unc_b=0, epsilon=1e-2)
        loss = loss.sum()
        return 1 / loss


class WeightedCrossEntropy(torch.nn.CrossEntropyLoss):
    def forward(self, input, target, weight: torch.Tensor | None = None):
        # save original reduction mode
        reduction = self.reduction
        if weight is not None:
            self.reduction = "none"
            loss = super().forward(input, target)
            self.reduction = reduction

            # dot product is only defined for flat tensors, so flatten
            loss = torch.flatten(loss)
            weight = torch.flatten(weight)
            loss = torch.dot(loss, weight)
            if self.reduction == "mean":
                loss = loss / torch.sum(weight)
        else:
            loss = super().forward(input, target)
        return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Taken from https://github.com/itakurah/Focal-loss-PyTorch/blob/main/focal_loss.py

        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                    Shape:
                        - binary/multi-label: (batch_size, num_classes)
                        - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                        - binary: (batch_size,)
                        - multi-label: (batch_size, num_classes)
                        - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = torch.nn.functional.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
