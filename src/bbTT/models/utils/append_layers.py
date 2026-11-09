from __future__ import annotations

from typing import Callable, Literal

import torch


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

    def forward(self, categorical_inputs, continuous_inputs, physics_parameters=None):
        x = self.unbinned_model(categorical_inputs, continuous_inputs, physics_parameters)[
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
        self.num_physics_parameters = model.num_physics_parameters

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


    def forward(
        self,
        categorical_inputs: torch.Tensor,
        continuous_inputs: torch.Tensor,
        physics_parameters: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.model(categorical_inputs, continuous_inputs, physics_parameters)
        x = self.act_func(x)
        return x
