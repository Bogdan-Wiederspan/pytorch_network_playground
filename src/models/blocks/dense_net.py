from __future__ import annotations

import torch

from ..utils import WeightNormalizedLinear


class DenseNetBlock(torch.nn.Module):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        skip_connection_init=1.0,
        freeze_skip_connection=False,
        activation_functions = "PReLu",
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
            eps=eps
        )
        self.skip_connection_amplifier = torch.nn.Parameter(torch.ones(1) * skip_connection_init)
        if freeze_skip_connection:
            self.skip_connection_amplifier.requires_grad = False

    def forward(self, x):
        _input = x * self.skip_connection_amplifier
        x = self.dense_block(x)
        x = torch.concatenate((x, _input), dim = 1)
        return x


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
        self.bn = torch.nn.BatchNorm1d(self.output_dim, eps=eps)
        self.act_fn = self._get_attr(torch.nn.modules.activation, activation_functions)()

    def _get_attr(self, obj, attr):
        for o in dir(obj):
            if o.lower() == attr.lower():
                return getattr(obj, o)
        else:
            raise AttributeError(f"Object has no attribute '{attr}'")

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.act_fn(x)
        return x
