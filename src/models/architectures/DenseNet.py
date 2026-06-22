from __future__ import annotations

import torch

from ..blocks import DenseBlock, DenseNetBlock
from ..register import register_model
from .base_model import BaseModel


@register_model("dense")
class DenseNet(BaseModel):
    def __init__(self, full_config, *args, **kwargs):
        super().__init__(full_config, *args, **kwargs)

        # init layers
        self.dense_config = {
            "skip_connection_init" : self.model_building_config.skip_connection_init,
            "freeze_skip_connection" : self.model_building_config.freeze_skip_connection,
            "activation_functions" : self.model_building_config.activation_functions,
            "eps": self.model_building_config.eps_batchnorm, # increasing eps helps to stabilize training, to counter batch norm and L2 reg counter play when used together
            "normalize" : self.model_building_config.normalize_linear, # activate weight normalization on linear layer weights
        }
        self.init_layers()


    def init_layers(self):
        m_cfg = self.model_building_config

        # self.input_layer = self.init_input_layer()
        self.input_layer = self.init_optional_input_layer()

        self.transition_dense_1 = DenseBlock(
            input_nodes=self.input_layer.ndim,
            output_nodes=int(m_cfg.nodes),
            activation_functions=m_cfg.activation_functions,
            eps=self.dense_config["eps"],
            normalize=self.dense_config["normalize"],
            )
        self.dense_block_1 = DenseNetBlock(input_nodes=self.transition_dense_1.output_dim, output_nodes=int(m_cfg.nodes), **self.dense_config)
        self.dense_block_2 = DenseNetBlock(input_nodes=self.dense_block_1.output_dim, output_nodes=int(m_cfg.nodes), **self.dense_config)
        self.dense_block_3 = DenseNetBlock(input_nodes=self.dense_block_2.output_dim, output_nodes=int(m_cfg.nodes), **self.dense_config)
        self.dense_block_4 = DenseNetBlock(input_nodes=self.dense_block_3.output_dim, output_nodes=int(m_cfg.nodes), **self.dense_config)
        self.dense_block_5 = DenseNetBlock(input_nodes=self.dense_block_4.output_dim, output_nodes=int(m_cfg.nodes), **self.dense_config)
        self.last_linear = torch.nn.Linear(self.dense_block_5.output_dim, len(self.dataset_config.target_map.keys()))

        # can only be sigmoid or softmax, uses only dim as configuration
        self.last_activation_fn = self.init_last_activation_layer()

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
