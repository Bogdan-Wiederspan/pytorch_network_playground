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
        normalize=False,
        num_physics_parameters=0,
        *args,
        **kwargs,
    ):
        # TODO Docstring
        # num_physics_parameters: physics parameters (e.g. kappa lambda) that are concatenated to the
        # input of the dense layer, but not to the skip connection
        super().__init__(*args, **kwargs)
        self.input_dim = input_nodes
        self.output_dim = output_nodes + input_nodes
        self.num_physics_parameters = num_physics_parameters
        self.dense_block = DenseBlock(
            input_nodes=input_nodes + num_physics_parameters,
            output_nodes=output_nodes,
            activation_functions=activation_functions,
            normalize=normalize,
        )
        self.skip_connection_amplifier = torch.nn.Parameter(torch.ones(1) * skip_connection_init)
        if freeze_skip_connection:
            self.skip_connection_amplifier.requires_grad = False

    def forward(self, x, physics_parameters=None):
        _input = x * self.skip_connection_amplifier
        if self.num_physics_parameters:
            x = torch.concatenate((x, physics_parameters), dim=1)
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
        normalize: bool = True,
        *args,
        **kwargs,
    ):
        """
        DenseBlock is a dense block that consists of a linear layer and an activation function.

        Args:
            nodes (int): Number of nodes in the block.
            activation_functions (str, optional): Name of the pytorch activation function, case insenstive.
                Defaults to "LeakyReLu".
        """
        super().__init__()
        self.input_dim = input_nodes
        self.output_dim = output_nodes

        self.linear = WeightNormalizedLinear(self.input_dim, self.output_dim, bias=True, normalize=normalize)
        self.act_fn = self._get_attr(torch.nn.modules.activation, activation_functions)()

        # the block initializes itself, the model only has to build it
        self.init_parameters()

    def init_parameters(self) -> None:
        """
        Initializes the learnable parameters of the block, called once at the end of ``__init__``.
        The base block keeps the torch default of the linear layer. Subclasses override this
        method to define their own initialization, see KaimingDenseBlock.
        """
        return None

    def _get_attr(self, obj, attr):
        for o in dir(obj):
            if o.lower() == attr.lower():
                return getattr(obj, o)
        else:
            raise AttributeError(f"Object has no attribute '{attr}'")

    def forward(self, x):
        x = self.linear(x)
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
            }


class KaimingDenseBlock(DenseBlock):
    def __init__(
        self,
        input_nodes: int,
        output_nodes: int,
        activation_functions: str = "LeakyReLu",
        normalize: bool = False,
        nonlinearity: str = "relu",
    ) -> None:
        """
        DenseBlock whose linear layer is initialized with Kaiming normal weights and a zero bias,
        instead of the torch default (uniform weights and bias).

        Args:
            input_nodes (int): Number of input nodes of the linear layer (fan in).
            output_nodes (int): Number of output nodes of the linear layer.
            activation_functions (str, optional): Name of the pytorch activation function, case insensitive.
                Defaults to "LeakyReLu".
            normalize (bool, optional): Weight normalization of the linear layer, not supported by this block.
                Defaults to False.
            nonlinearity (str, optional): Name of the nonlinearity the gain is calculated for, see
                ``torch.nn.init.calculate_gain``. Defaults to "relu".

        Raises:
            ValueError: If *normalize* is True. A weight normalized layer stores magnitude and direction
                instead of the weight, an initialization of the weight itself is not defined for it here.
        """
        if normalize:
            raise ValueError("KaimingDenseBlock does not support weight normalized linear layers")
        # needs to be set before super().__init__, since the parent calls init_parameters at its end
        self.nonlinearity = nonlinearity
        super().__init__(
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            activation_functions=activation_functions,
            normalize=normalize,
        )

    def init_parameters(self) -> None:
        """
        Kaiming normal initialization (fan in) of the linear weight, the bias is set to zero.
        """
        torch.nn.init.kaiming_normal_(
            self.linear.weight,
            mode="fan_in",
            nonlinearity=self.nonlinearity
            )
        torch.nn.init.zeros_(self.linear.bias)
