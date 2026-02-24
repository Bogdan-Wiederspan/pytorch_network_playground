from __future__ import annotations

import torch

class SignalEfficiency(torch.nn.Module):
    def __init__(self, sampler, device, train=True, mode="full", *args, **kwargs):
        """
        This function creates a loss using the Asimov Significance as introduced in https://arxiv.org/abs/1806.00322.
        The loss is calculated as 1 / significance, thus maximizing the significance is equivalent to minimizing the loss.

        The loss comes in three implementations: "full", "no_unc", "approximation", which is used in the forward pass is determined by the *mode* argument.
        The loss is calculated differently in *train* and validation mode.
        All tensors are placed on *device*.
        The necessary information about phase space and weights is handed over by *sampler*.

        Args:
            sampler (_type_): _description_
            device (_type_): _description_
            train (bool, optional): _description_. Defaults to True.
        """
        super().__init__()

        self.train = train
        self.total_product_of_weights = sampler.weights_aggregator_inst("product_of_weights", "whole_sum")
        self.target_map = sampler.target_map

        self.mode = mode
        self.loss = self.chosen_loss()

        self.s_cls = self.target_map["hh"] # definition of signal class
        self.device = device

        if not train:
            # validation mode goes over whole validation phase space and not only over a batch
            # thus this information is handed over by sampler
            self.eval_weights = sampler.weights_aggregator_inst("product_of_weights", "evaluation_sum")
            self.total_process_weights = sampler.weights_aggregator_inst("product_of_weights", "validation_sum")

    def chosen_loss(self):
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

    def approximation_sb(
        self,
        prediction: torch.tensor,
        truth: torch.tensor,
        product_of_weights: dict[torch.tensor],
        evaluation_phase_space_mask: dict[torch.tensor],
        is_signal: bool
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
        # determine if signal (true positive) or false identified background events (false negative) are calculated
        if is_signal:
            bool_filter = truth
        else:
            bool_filter = (1 - truth)
        filtered_batch_weight = bool_filter * product_of_weights

        if self.train:
            # when training, calculate factors on batch base
            evaluation_yield = torch.sum(filtered_batch_weight * evaluation_phase_space_mask) # term 3
            batch_yield = torch.sum(filtered_batch_weight) # term 2
        else:
            # for validation batches cover whole validation space, not just a batch
            # these reduces the factors to constants, where s is hh and b is dy + tt factor
            # for background or signal different
            if is_signal:
                evaluation_yield = self.eval_weights["hh"]
                batch_yield = self.total_process_weights["hh"]
            else:
                evaluation_yield = self.eval_weights["dy"] + self.eval_weights["tt"]
                batch_yield = self.total_process_weights["dy"] + self.total_process_weights["tt"]

        # since filtered batch is 0 for either signal or background this reduces the sum
        # to be specifically only over signal OR background events
        weighted_yield = torch.sum(prediction * filtered_batch_weight) # term 1
        return weighted_yield * evaluation_yield / batch_yield

    def asimov_small_signal_approximation(self, s, b, *args, **kwargs):
        # approximation for small signal yield  3.3
        return s / torch.sqrt(s + b)

    def asimov_no_uncertainty(self, s, b, *args, **kwargs):
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

    def forward(self, prediction, truth, weight, uncertainty=10, *args, **kwargs):
        signal_node_truth = truth[:, self.s_cls]
        prediction = torch.softmax(prediction, dim = 1) # network does not contain softmax
        signal_node_prediction = prediction[:, self.s_cls]

        s = self.approximation_sb(signal_node_prediction, signal_node_truth, weight["product_of_weights"], weight["evaluation_space_mask"], is_signal=True)
        b = self.approximation_sb(signal_node_prediction, signal_node_truth, weight["product_of_weights"], weight["evaluation_space_mask"], is_signal=False)

        if (s <= 0 or b <= 0):
            from IPython import embed; embed(header = "Either s or b is below 0, which is not physical, starting debugger in Forward of SignalEfficiency loss")

        loss = self.loss(s= s, b=b, unc_b=uncertainty)
        if loss == 0:
            return 0
        return 1 / loss

class ZScore(torch.nn.Module):
    """
    Calculates a surrogate loss by calculating signal efficiency for given *signal_cls_idx*.
    This discrete value is used as an coefficient to amplify a given *base_loss_inst*, extending
    this discrete values to be differentiable.

    Idea come from https://arxiv.org/pdf/2412.09500

    Args:
        signal_cls_idx (int): Index of the signal class in the prediction/target tensors.
        base_loss_inst (torch.nn.Module): Base loss function to be amplified.
    """

    def __init__(self, signal_cls_idx, base_loss_inst):
        super().__init__()
        self.signal_cls_idx = signal_cls_idx
        self.eps = 1
        self.base_loss_inst = base_loss_inst

    def split_signal_and_background(self, pred, truth):
        signal_events_idx = torch.min(torch.argwhere(truth[:,self.signal_cls_idx]))
        signal_truth = truth[signal_events_idx:]
        background_truth = truth[:signal_events_idx]
        signal_pred = pred[signal_events_idx:]
        background_pred = pred[:signal_events_idx]
        return signal_pred, signal_truth, background_pred, background_truth

    def get_slices(self, idx):
        start = torch.tensor([0] + idx[:-1]).cumsum(0).tolist()
        end = torch.tensor(idx).cumsum(0).tolist()
        return start, end

    def signal_score(self, signal_prediction, signal_truth, weight, idx):
        start_idx, end_idx = self.get_slices(idx["signal"])

        signal_scores = []
        for current_signal_idx, (l, r) in enumerate(zip(start_idx, end_idx)):
            # select sub batches
            sub_targets = signal_truth[l:r]
            sub_predictions = signal_prediction[l:r]

            pred_cls = torch.argmax(sub_predictions, dim=1)
            truth_cls = torch.argmax(sub_targets, dim=1)

            # false negative is any bg classified as signal
            # TODO: differitiazion between multiple signal will not work with this code
            fn_signal = torch.sum(pred_cls != truth_cls)

            # calculate weighted fraction of correctly classified signal events
            num_signal = len(sub_targets)
            fraction = (num_signal - fn_signal)/num_signal
            score = fraction * weight["signal"][current_signal_idx]
            signal_scores.append(score)
        return torch.tensor(signal_scores)

    def background_scores(self, background_pred, background_truth, weight, idx):
        """ calculates for each background sample the sqrt of the score based on misclassification into signal"""
        start_idx, end_idx = self.get_slices(idx["background"])

        bg_scores = []
        # HINT: this score does not differentiate between different background types
        for current_bg_idx, (l, r) in enumerate(zip(start_idx, end_idx)):
            sub_targets = background_truth[l:r]
            sub_predictions = background_pred[l:r]

            pred_cls = torch.argmax(sub_predictions, dim=1)

            # false negative in the case of background is signal classified as background, thus being signal_cls
            # TODO maybe include misclassification into other background classes as well?
            fn_signal_as_bg = torch.sum(pred_cls == self.signal_cls_idx)

            num_bg = len(sub_targets)
            rate_of_miscl_into_signal  = (fn_signal_as_bg) / num_bg
            score = torch.sqrt(
                rate_of_miscl_into_signal * weight["background"][current_bg_idx] + self.eps
                )
            bg_scores.append(score)
        return torch.tensor(bg_scores)

    def forward(self, pred, truth, weight, idx,debug=False):
        if debug:
            from IPython import embed; embed(header = "FORWARD LOSS line: 129 in loss.py")
        signal_predictions, signal_truth, background_prediction, background_truth = self.split_signal_and_background(pred, truth)
        # separate signal and background

        signal_fraction_score = torch.sum(self.signal_score(signal_predictions, signal_truth, weight, idx))
        background_fraction_score = torch.sum(self.background_scores(background_prediction, background_truth, weight, idx))

        max_score = torch.sum(torch.tensor(weight["signal"])) / self.eps**0.5
        # print(signal_fraction_score, background_fraction_score, max_score)
        coefficient = (max_score - signal_fraction_score) / background_fraction_score
        loss = self.base_loss_inst(pred, truth)
        if torch.isnan(coefficient) or torch.isnan(loss):
            from IPython import embed; embed(header="Either coefficient or loss is NaN")
        return loss * coefficient


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
