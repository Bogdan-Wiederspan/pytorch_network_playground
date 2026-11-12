from __future__ import annotations

from typing import Any

import torch

from bbTT.models.architectures.base_model import BaseModel
from bbTT.models.blocks import DenseBlock, DenseNetBlock
from bbTT.models.register import register_model
from bbTT.models.utils import linear_state


@register_model("dense")
class DenseNet(BaseModel):
    monitored_blocks: tuple[str, ...] = (
        "input_layer",
        "transition_dense_1",
        "dense_block_1",
        "dense_block_2",
        "dense_block_3",
        "dense_block_4",
        "dense_block_5",
        "last_linear",
    )


    def __init__(self, full_config, *args, **kwargs):
        super().__init__(full_config, *args, **kwargs)
        self.init_layers()

    def init_layers(self):
        dense_cfg = self.model_config.dense_network
        _dense_config = {
            "skip_connection_init": dense_cfg.skip_connection_init,
            "freeze_skip_connection": dense_cfg.freeze_skip_connection,
            "activation_functions": dense_cfg.activation_functions,
            "eps": dense_cfg.batch_norm_eps,  # increasing eps helps to stabilize training, to counter batch norm and L2 reg counter play when used together
            "normalize": dense_cfg.normalize_linear,  # activate weight normalization on linear layer weights
            "output_nodes": int(dense_cfg.nodes),
        }

        self.input_layer = self.init_input_layer()

        self.transition_dense_1 = DenseBlock(
            input_nodes=self.input_layer.ndim,
            output_nodes=_dense_config["output_nodes"],
            activation_functions=_dense_config["activation_functions"],
            eps=_dense_config["eps"],
            normalize=_dense_config["normalize"],
        )
        self.dense_block_1 = DenseNetBlock(input_nodes=self.transition_dense_1.output_dim, **_dense_config)
        self.dense_block_2 = DenseNetBlock(input_nodes=self.dense_block_1.output_dim, **_dense_config)
        self.dense_block_3 = DenseNetBlock(input_nodes=self.dense_block_2.output_dim, **_dense_config)
        self.dense_block_4 = DenseNetBlock(input_nodes=self.dense_block_3.output_dim, **_dense_config)
        self.dense_block_5 = DenseNetBlock(input_nodes=self.dense_block_4.output_dim, **_dense_config)
        self.last_linear = torch.nn.Linear(self.dense_block_5.output_dim, self.num_targets)

        # can only be sigmoid or softmax, uses only dim as configuration
        self.last_activation_fn = self.init_last_activation_layer()

    def evaluation_state(self) -> dict[str, Any]:
        """
        Monitored blocks of the parent plus the output layer, which is a plain torch layer owned by the model
        and therefore has no state method of its own.

        Returns:
            dict[str, Any]: {"monitored_blocks": {..., "last_linear": linear norms}}.
        """
        state = super().evaluation_state()
        state["monitored_blocks"]["last_linear"] = linear_state(self.last_linear)
        return state


    def forward(self, categorical_inputs, continuous_inputs):
        x = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        x = self.transition_dense_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        x = self.dense_block_3(x)
        x = self.dense_block_4(x)
        x = self.dense_block_5(x)
        x = self.last_linear(x)
        if self.use_last_activation:
            x = self.last_activation_fn(x)
        return x
