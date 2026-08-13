from __future__ import annotations

import torch

from ..utils import WeightNormalizedLinear


class ResNetBlock(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        nodes: int,
        activation_functions: str = "LeakyReLu",
        skip_connection_init: float = 1,
        freeze_skip_connection: float = False,
        eps: float = 1e-5,
        normalize = True,
        *args,
        **kwargs,
    ):
        """
        ResNetBlock consisting of a linear layer, batch normalization, and an activation function.
        A adjustable skip connection connects input and output of the block.
        The adjustable skip connection has a learnable parameter, *skip_connection_amplifier*.
        The dimension of the input and output of the block are defined by *nodes*.
        If skip_connection_init is set to 0, the skip connection is disabled.
        This also make if possible to use different in_nodes and out_nodes must can be different.
        To freeze the skip connection parameter, set *freeze_skip_connection* to True.

        Args:
            nodes (int): Number of nodes in the block.
            activation_functions (str, optional): Name of the pytorch activation function, case insenstive.
                Defaults to "LeakyReLu".
            skip_connection_init (int, optional): Start value of the skipconnection. Defaults to 1.
            freeze_skip_connection (bool, optional): Freeze learnable skipconnection parameter. Defaults to False.
        """
        super().__init__(*args, **kwargs)

        self.nodes = nodes
        self.act_func = self._get_attr(torch.nn.modules.activation, activation_functions)()
        self.skip_connection_amplifier = torch.nn.Parameter(torch.ones(1) * skip_connection_init)
        if freeze_skip_connection:
            self.skip_connection_amplifier.requires_grad = False

        self.linear = WeightNormalizedLinear(self.nodes, self.nodes, bias=True, normalize=normalize)
        self.bn = torch.nn.BatchNorm1d(self.nodes, eps=eps)
        self.act_fn = self.act_func


    def _get_attr(self, obj, attr):
        for o in dir(obj):
            if o.lower() == attr.lower():
                return getattr(obj, o)
        else:
            raise AttributeError(f"Object has no attribute '{attr}'")

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        skip_connection = self.skip_connection_amplifier * x
        x = self.linear(x)
        x = self.bn(x)
        x = self.act_fn(x)
        x = x + skip_connection
        return x

class ResNetPreactivationBlock(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        nodes: int,
        activation_functions: str = "PReLu",
        skip_connection_init: float = 1,
        freeze_skip_connection: bool = False,
        eps: float = 1e-5,
        normalize: bool = True,
        *args,
        **kwargs,
    ):
        """
        Residual block that consists of a linear layer, batch normalization, and an activation function.
        A adjustable skip connection connects input and output of the block.
        The adjustable skip connection has a learnable parameter, *skip_connection_amplifier*.
        The dimension of the input and output of the block are defined by *nodes*.
        If skip_connection_init is set to 0, the skip connection is disabled.
        This also make if possible to use different in_nodes and out_nodes must can be different.
        To freeze the skip connection parameter, set *freeze_skip_connection* to True.

        More information can be found in the original paper: https://arxiv.org/abs/1603.05027

        Args:
            nodes (int): Number of nodes in the block.
            activation_functions (str, optional): Name of the pytorch activation function, case insenstive.
                Defaults to "LeakyReLu".
            skip_connection_init (int, optional): Start value of the skipconnection. Defaults to 1.
            freeze_skip_connection (bool, optional): Freeze skipconnection parameter. Defaults to False.
        """
        super().__init__()
        self.nodes = nodes
        self.act_func = self._get_attr(torch.nn.modules.activation, activation_functions)()
        self.skip_connection_amplifier = torch.nn.Parameter(torch.ones(1) * skip_connection_init)
        if freeze_skip_connection:
            self.skip_connection_amplifier.requires_grad = False

        self.linear_1 = WeightNormalizedLinear(self.nodes, self.nodes, bias=True, normalize=normalize)
        self.bn_1 = torch.nn.BatchNorm1d(self.nodes, eps=eps)
        self.act_fn_1 = self.act_func
        self.linear_2 = WeightNormalizedLinear(self.nodes, self.nodes, bias=True, normalize=normalize)
        self.bn_2 = torch.nn.BatchNorm1d(self.nodes, eps=eps)
        self.act_fn2 = self.act_func

    def _get_attr(self, obj, attr):
        for o in dir(obj):
            if o.lower() == attr.lower():
                return getattr(obj, o)
        else:
            raise AttributeError(f"Object has no attribute '{attr}'")

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        skip_connection = self.skip_connection_amplifier * x
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = self.act_fn_1(x)
        x = self.linear_2(x)
        x = self.bn_2(x)
        x = x + skip_connection
        return self.act_fn2(x)
