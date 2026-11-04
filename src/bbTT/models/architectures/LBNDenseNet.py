from __future__ import annotations

import torch

from bbTT.models.architectures.DenseNet import DenseNet
from bbTT.models.blocks import DenseBlock
from bbTT.models.physics import LBNPipeline
from bbTT.models.register import register_model


@register_model("lbn_dense")
class LBNDenseNet(DenseNet):
    def __init__(self, full_config, *args, **kwargs):
        # has same init as DenseNet
        super().__init__(full_config, *args, **kwargs)

    def init_layers(self):
        # create normal DenseNet as defined in DenseNet.py, including on top an LBN
        super().init_layers()
        # dense net uses output of preprocessing layers and lbn, thus concat both features
        lbn_config = self.model_config.lbn_network

        # the lbn manages its own preprocessing (padding and rotation, see ``sanitize`` in the pipeline),
        # own layer instances built from the same config as the dense branch
        self.lbn = LBNPipeline(
            self.continuous_features,  # number of input parameters
            padding_layer_inst=self.init_continuous_padding_layer(),
            rotation_layer_inst=self.init_rotation_layer(),
            M=lbn_config.number_of_particles,
            weight_init_scale=lbn_config.weight_init_scale,
            clip_weights=lbn_config.clip_weights,
            eps=lbn_config.eps,
            safe_sqrt_eps=lbn_config.safe_sqrt_eps,
        )
        dense_cfg = self.model_config.dense_network
        self.transition_dense_1 = DenseBlock(
            input_nodes=(self.input_layer.ndim + self.lbn.ndim),
            output_nodes=dense_cfg.nodes,
            activation_functions=dense_cfg.activation_functions,
            eps=dense_cfg.batch_norm_eps,
            normalize=dense_cfg.normalize_linear,
        )

    def forward(self, categorical_inputs: torch.Tensor, continuous_inputs: torch.Tensor) -> torch.Tensor:
        normal_inputs = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        # lbn needs padding sanitized values, but not standardized. Reason for this is that physics has scale dependency
        lbn_outputs = self.lbn(continuous_inputs)

        x = torch.concatenate(
            (normal_inputs, lbn_outputs),
            axis=1,
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
