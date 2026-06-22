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
        self.lbn = LBNPipeline(
            self.dataset_config.continuous_features, # number of input parameters
            M =self.model_building_config.LBN_M,
            weight_init_scale=1.0,
            clip_weights=False,
            eps=1.0e-5,
        )
        self.transition_dense_1 = DenseBlock(
            input_nodes=(self.input_layer.ndim + self.lbn.ndim),
            output_nodes=self.model_building_config.nodes,
            activation_functions=self.model_building_config.activation_functions,
            eps=self.model_building_config.eps_batchnorm,
            normalize=self.model_building_config.normalize_linear
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
