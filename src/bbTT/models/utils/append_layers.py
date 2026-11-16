from __future__ import annotations

from typing import Callable, Literal

import torch

HEAD_CHOICE = Literal["logits", "probabilities", "binned"]


class SelectHead(torch.nn.Module):
    """
    Wraps a model so its forward returns one named head instead of the logits.
    Used for export and evaluation, where a single tensor output is wanted (e.g. torch.export).
    """

    def __init__(self, model: torch.nn.Module, head: HEAD_CHOICE, *args, **kwargs):
        """
        Args:
            model (torch.nn.Module): Model whose forward returns logits and that declares ``HEADS``.
            head (HEAD_CHOICE): Name of the head to return, must be in ``model.HEADS``.

        Raises:
            ValueError: If *model* does not offer *head*.
        """
        super().__init__(*args, **kwargs)
        if head not in model.HEADS:
            raise ValueError(f"Model {type(model).__name__} has no head {head!r}, available heads: {model.HEADS}")
        self.model = model
        self.head = head
        self.categorical_features = model.categorical_features
        self.continuous_features = model.continuous_features

    def forward(self, categorical_inputs: torch.Tensor, continuous_inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(categorical_inputs, continuous_inputs)
        return self.model.head(self.head, logits)



class AddBinningLayer(torch.nn.Module):
    def __init__(self, model, binning_cls, kernel_cls, kernel_cfg, binning_edges, signal_cls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unbinned_model = model
        self.binning_layer = binning_cls(
            init_edges=binning_edges,
            kernel_cls=kernel_cls,
            kernel_cfg=kernel_cfg,
        )
        self.signal_cls = ... if signal_cls is None else signal_cls  # used for class slicing

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.unbinned_model(categorical_inputs, continuous_inputs)[
            :, self.signal_cls
        ]  # normal prediction as probabilities
        x = self.binning_layer(x)
        return x


ACTIVATION_CHOICE = Literal["sigmoid", "softmax", "identity"]

class AddActFnToModel(torch.nn.Module):
    def __init__(self, model, act_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

        self.act_func = self.activation_layer(act_fn)
        self.categorical_features = model.categorical_features
        self.continuous_features = model.continuous_features

    @staticmethod
    def activation_layer(name: ACTIVATION_CHOICE) -> torch.nn.Module:
        """
        Build the activation layer that belongs to *name*.

        Args:
            name (ACTIVATION_CHOICE): One of "sigmoid", "softmax" or "identity".

        Returns:
            torch.nn.Module: Fresh instance of the activation layer, softmax acts on the last dimension.

        Raises:
            KeyError: If *name* is not a known activation.
        """
        choices: dict[str, Callable[[], torch.nn.Module]] = {
            "sigmoid": torch.nn.Sigmoid,
            "softmax": lambda: torch.nn.Softmax(dim=-1),
            "identity": torch.nn.Identity,
        }
        return choices[name]()


    def forward(self, categorical_inputs: torch.Tensor, continuous_inputs: torch.Tensor) -> torch.Tensor:
        x = self.model(categorical_inputs, continuous_inputs)
        x = self.act_func(x)
        return x
