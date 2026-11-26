from __future__ import annotations

import torch

from bbTT.models.architectures.LBNDenseNet import LBNDenseNet
from bbTT.models.blocks import KaimingDenseBlock, KaimingDenseNetBlock
from bbTT.models.physics import LBNPipeline
from bbTT.models.register import register_model


@register_model("lbn_kaiming_dense")
class LBNKaimingDenseNet(LBNDenseNet):
    """
    LBNDenseNet whose linear layers are initialized with Kaiming normal weights and a zero bias.

    Same structure as LBNDenseNet (one transition block, five DenseNet blocks that concatenate their
    input to their output, a linear output layer, the LBN next to the input layer) and the same
    forward pass; only the blocks differ. With ``dense_network.nodes = n`` the width grows by n per
    block: n, 2n, ..., 6n in front of the output layer. The output layer keeps the torch default.
    """

    def init_layers(self) -> None:
        """
        Defines all layers of the model. Written out instead of calling the parent, since every
        block of the parent is replaced by its Kaiming variant.
        """
        dense_cfg = self.model_config.dense_network
        lbn_config = self.model_config.lbn_network
        nodes = int(dense_cfg.nodes)
        _block_config = {
            "output_nodes": nodes,
            "skip_connection_init": dense_cfg.skip_connection_init,
            "freeze_skip_connection": dense_cfg.freeze_skip_connection,
            "activation_functions": dense_cfg.activation_functions,
            "normalize": dense_cfg.normalize_linear,
            "nonlinearity": dense_cfg.weight_init_nonlinearity,
        }

        self.input_layer = self.init_input_layer()

        # the lbn manages its own preprocessing (padding and rotation, see ``sanitize`` in the pipeline),
        # own layer instances built from the same config as the dense branch
        self.lbn = LBNPipeline(
            self.continuous_features,
            padding_layer_inst=self.init_continuous_padding_layer(),
            rotation_layer_inst=self.init_rotation_layer(),
            M=lbn_config.number_of_particles,
            weight_init_scale=lbn_config.weight_init_scale,
            clip_weights=lbn_config.clip_weights,
            eps=lbn_config.eps,
            safe_sqrt_eps=lbn_config.safe_sqrt_eps,
        )

        self.transition_dense_1 = KaimingDenseBlock(
            input_nodes=self.input_layer.ndim + self.lbn.ndim,
            output_nodes=nodes,
            activation_functions=dense_cfg.activation_functions,
            normalize=dense_cfg.normalize_linear,
            nonlinearity=dense_cfg.weight_init_nonlinearity,
        )
        self.dense_block_1 = KaimingDenseNetBlock(input_nodes=self.transition_dense_1.output_dim, **_block_config)
        self.dense_block_2 = KaimingDenseNetBlock(input_nodes=self.dense_block_1.output_dim, **_block_config)
        self.dense_block_3 = KaimingDenseNetBlock(input_nodes=self.dense_block_2.output_dim, **_block_config)
        self.dense_block_4 = KaimingDenseNetBlock(input_nodes=self.dense_block_3.output_dim, **_block_config)
        self.dense_block_5 = KaimingDenseNetBlock(input_nodes=self.dense_block_4.output_dim, **_block_config)
        self.last_linear = torch.nn.Linear(self.dense_block_5.output_dim, self.num_targets)

        # can only be sigmoid or softmax, uses only dim as configuration
        self.last_activation_fn = self.init_last_activation_layer()
