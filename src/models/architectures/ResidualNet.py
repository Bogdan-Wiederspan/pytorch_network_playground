from __future__ import annotations

import torch

from ..blocks import DenseBlock, ResNetPreactivationBlock
from ..register import register_model
from .base_model import BaseModel


@register_model("residual")
class ResidualNet(BaseModel):
    def __init__(self, full_config, *args, **kwargs):
        super().__init__(full_config, *args, **kwargs)
        self.init_layers()

    def init_layers(self):
        # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
        # eps = 0.5e-5

        # helper where all layers are defined
        # std layers are filled when statistics are known
        self.input_layer = self.init_optional_input_layer()

        dense_cfg = self.model_config.dense_network
        cfg = {
            "nodes" : dense_cfg.nodes,
            "activation_functions" : dense_cfg.activation_functions,
            "skip_connection_init" : dense_cfg.skip_connection_init,
            "freeze_skip_connection" :dense_cfg.freeze_skip_connection,
            "eps" : dense_cfg.batch_norm_eps,
            "normalize" : False # activate weight normalization on linear layer weights
        }

        self.transition_dense_1 = DenseBlock(
            input_nodes = self.input_layer.ndim,
            output_nodes = dense_cfg.nodes,
            **cfg,
            )
        self.resnet_block_1 = ResNetPreactivationBlock(**cfg)
        self.resnet_block_2 = ResNetPreactivationBlock(**cfg)
        self.resnet_block_3 = ResNetPreactivationBlock(**cfg)
        self.last_linear = torch.nn.Linear(dense_cfg.nodes, self.num_targets)

        # can only be sigmoid or softmax, uses only dim as configuration
        self.last_activation_fn = self.init_last_activation_layer()


    def forward(self, categorical_inputs, continuous_inputs):
        x = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        x = self.transition_dense_1(x)
        x = self.resnet_block_1(x)
        x = self.resnet_block_2(x)
        x = self.resnet_block_3(x)
        x = self.last_linear(x)
        if self.use_last_activation:
            x = self.last_activation_fn(x)
        return x
