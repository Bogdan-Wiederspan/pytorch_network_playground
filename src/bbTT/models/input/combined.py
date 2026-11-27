from __future__ import annotations

from typing import Any

import torch


class InputLayer(torch.nn.Module):
    def __init__(
        self,
        continuous_layer_inst: torch.nn.Module | None = None,
        categorical_layer_inst: torch.nn.Module | None = None,
    ):
        """
        Combine the preprocessing branches for continuous and categorical features into one input tensor.
        If one branch is not needed (marked by None), its inputs are ignored in the forward pass.
        At least one branch is required.

        Args:
            continuous_layer_inst (torch.nn.Module | None, optional): Branch for continuous features, needs an ``ndim``
                attribute with its output dimension. Defaults to None.
            categorical_layer_inst (torch.nn.Module | None, optional): Branch for categorical features, needs an ``ndim``
                attribute with its output dimension. Defaults to None.

        Raises:
            ValueError: If both branches are None.
        """
        super().__init__()
        if continuous_layer_inst is None and categorical_layer_inst is None:
            raise ValueError("InputLayer needs at least one of continuous_layer or categorical_layer")

        self.continuous_layer = continuous_layer_inst
        self.categorical_layer = categorical_layer_inst
        self.ndim = sum(layer.ndim for layer in self.branches)

    @property
    def branches(self) -> tuple[torch.nn.Module, ...]:
        """
        Returns:
            tuple[torch.nn.Module, ...]: The branches that exist, in concatenation order (continuous, categorical).
        """
        return tuple(layer for layer in (self.continuous_layer, self.categorical_layer) if layer is not None)

    def create_evaluation_state(self) -> dict[str, Any]:
        """
        Snapshot of the categorical branch; the continuous branch (standardization, rotation, padding) holds no
        learnable state.

        Returns:
            dict[str, Any]: See CategoricalInputLayer.create_evaluation_state, empty without a categorical branch.
        """
        if self.categorical_layer is None:
            return {}
        return self.categorical_layer.create_evaluation_state()


    def forward(
        self,
        *,
        categorical_inputs: torch.Tensor | None,
        continuous_inputs: torch.Tensor | None
        ) -> torch.Tensor:
        """
        Args:
            categorical_inputs (torch.Tensor | None): Categorical features, ignored when no categorical branch exists.
            continuous_inputs (torch.Tensor | None): Continuous features, ignored when no continuous branch exists.

        Returns:
            torch.Tensor: Concatenation of the branch outputs along the feature dimension.
        """
        outputs = []
        if self.continuous_layer is not None:
            outputs.append(self.continuous_layer(continuous_inputs))
        if self.categorical_layer is not None:
            outputs.append(self.categorical_layer(categorical_inputs))
        return torch.cat(outputs, dim=1)
[]
