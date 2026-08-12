from __future__ import annotations

import torch

from ..blocks import DenseBlock
from ..physics import LBNPipeline
from ..register import register_model
from .DenseNet import DenseNet


@register_model("lbn_dense")
class LBNDenseNet(DenseNet):
    def __init__(
        self,
        full_config,
        *args,
        **kwargs
        ):
        # has same init as DenseNet
        super().__init__(full_config, *args, **kwargs)

    def init_layers(self):
        # create normal DenseNet as defined in DenseNet.py, including on top an LBN
        super().init_layers()

        # dense net uses output of preprocessing layers and lbn, thus concat both features
        lbn_config = self.model_config.lbn_network
        self.lbn = LBNPipeline(
            self.continuous_features, # number of input parameters
            M =lbn_config.number_of_particles,
            weight_init_scale=lbn_config.weight_init_scale,
            clip_weights=lbn_config.clip_weights,
            eps=lbn_config.eps,
        )
        dense_cfg = self.model_config.dense_network
        self.transition_dense_1 = DenseBlock(
            input_nodes=(self.input_layer.ndim + self.lbn.ndim),
            output_nodes=dense_cfg.nodes,
            activation_functions=dense_cfg.activation_functions,
            eps=dense_cfg.batch_norm_eps,
            normalize=dense_cfg.normalize_linear
            )

    def forward(self, categorical_inputs, continuous_inputs):
        # preprocessing with lbn
        x = torch.concatenate(
            (self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs),
            self.lbn(continuous_inputs)),
            axis=1
        )
        # dnn
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
