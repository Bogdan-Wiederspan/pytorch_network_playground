from __future__ import annotations

import torch

from bbTT.models.architectures.base_model import BaseModel
from bbTT.models.blocks import KaimingDenseBlock
from bbTT.models.register import register_model


@register_model("fc")
class FullyConnectedNet(BaseModel):
    """
    Plain fully connected network:
        - input layer
        - six hidden layers of the same width and a linear output layer.
    """

    monitored_blocks: tuple[str, ...] = (
        "input_layer",
        "dense_1",
        "dense_2",
        "dense_3",
        "dense_4",
        "dense_5",
        "dense_6",
        "last_linear",
    )

    def __init__(self, full_config, *args, **kwargs) -> None:
        """
        Args:
            full_config (FullConfig): Merged config, ``model_building_config.dense_network`` defines the hidden layers.
        """
        super().__init__(full_config, *args, **kwargs)
        self.init_layers()

    def init_layers(self) -> None:
        """
        Defines all layers of the model. The hidden layers are written out one by one,
        so the depth is visible here and each layer can be listed in ``monitored_blocks``.
        """
        dense_cfg = self.model_config.dense_network
        nodes = int(dense_cfg.nodes)
        _dense_config = {
            "output_nodes": nodes,
            "activation_functions": dense_cfg.activation_functions,
            "normalize": dense_cfg.normalize_linear,
            "nonlinearity": dense_cfg.weight_init_nonlinearity,

        }

        self.input_layer = self.init_input_layer()

        self.dense_1 = KaimingDenseBlock(input_nodes=self.input_layer.ndim, **_dense_config)
        self.dense_2 = KaimingDenseBlock(input_nodes=nodes, **_dense_config)
        self.dense_3 = KaimingDenseBlock(input_nodes=nodes, **_dense_config)
        self.dense_4 = KaimingDenseBlock(input_nodes=nodes, **_dense_config)
        self.dense_5 = KaimingDenseBlock(input_nodes=nodes, **_dense_config)
        self.dense_6 = KaimingDenseBlock(input_nodes=nodes, **_dense_config)
        self.last_linear = torch.nn.Linear(nodes, self.num_targets)

        # can only be sigmoid or softmax, uses only dim as configuration
        self.last_activation_fn = self.init_last_activation_layer()

    def forward(self, categorical_inputs: torch.Tensor, continuous_inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            categorical_inputs (torch.Tensor): Categorical features, shape ``[event, num_categorical_features]``.
            continuous_inputs (torch.Tensor): Continuous features, shape ``[event, num_continuous_features]``.

        Returns:
            torch.Tensor: Logits, shape ``[event, node]``.
        """
        x = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        x = self.dense_6(x)
        x = self.last_linear(x)
        return x
