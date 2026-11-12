from __future__ import annotations

from typing import Any

import torch

from bbTT.models.utils import WeightNormalizedLinear, linear_state


class DenseNetBlock(torch.nn.Module):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        skip_connection_init=1.0,
        freeze_skip_connection=False,
        activation_functions="PReLu",
        eps=1e-5,
        normalize=False,
        *args,
        **kwargs,
    ):
        # TODO Docstring
        super().__init__(*args, **kwargs)
        self.input_dim = input_nodes
        self.output_dim = output_nodes + input_nodes
        self.dense_block = DenseBlock(
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            activation_functions=activation_functions,
            normalize=normalize,
            eps=eps,
        )
        self.skip_connection_amplifier = torch.nn.Parameter(torch.ones(1) * skip_connection_init)
        if freeze_skip_connection:
            self.skip_connection_amplifier.requires_grad = False

    def forward(self, x):
        _input = x * self.skip_connection_amplifier
        x = self.dense_block(x)
        x = torch.concatenate((x, _input), dim=1)
        return x

    def create_evaluation_state(self) -> dict[str, Any]:
        """
        A DenseNetBlock is literally a DenseBlock with some change to be using a DenseNet structure.
        This Snapshot just shows the internal DenseBlock, see DenseBlock.create_evaluation_state.

        Returns:
            dict[str, Any]: Same content as the inner DenseBlock.
        """
        return self.dense_block.create_evaluation_state()


class DenseBlock(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        input_nodes: float,
        output_nodes: float,
        activation_functions: str = "LeakyReLu",
        eps: float = 1e-5,
        normalize: bool = True,
        *args,
        **kwargs,
    ):
        """
        DenseBlock is a dense block that consists of a linear layer, batch normalization, and an activation function.

        Args:
            nodes (int): Number of nodes in the block.
            activation_functions (str, optional): Name of the pytorch activation function, case insenstive.
                Defaults to "LeakyReLu".
        """
        super().__init__()
        self.input_dim = input_nodes
        self.output_dim = output_nodes

        self.linear = WeightNormalizedLinear(self.input_dim, self.output_dim, bias=True, normalize=normalize)
        # self.bn = torch.nn.BatchNorm1d(self.output_dim, eps=eps)
        self.act_fn = self._get_attr(torch.nn.modules.activation, activation_functions)()

    def _get_attr(self, obj, attr):
        for o in dir(obj):
            if o.lower() == attr.lower():
                return getattr(obj, o)
        else:
            raise AttributeError(f"Object has no attribute '{attr}'")

    def forward(self, x):
        x = self.linear(x)
        # x = self.bn(x)
        x = self.act_fn(x)
        return x

    def create_evaluation_state(self) -> dict[str, Any]:
        """
        Detached cpu snapshot of the block state for monitoring.

        Returns:
            dict[str, Any]: Linear norms (see linear_state) and batch norm running statistics
                (see batch_norm_state).
        """
        return {
            **linear_state(self.linear),
            # **batch_norm_state(self.bn)
            }
