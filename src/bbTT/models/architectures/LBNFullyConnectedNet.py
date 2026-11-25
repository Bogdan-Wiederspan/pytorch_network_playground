from __future__ import annotations

import torch

from bbTT.models.architectures.FullyConnectedNet import FullyConnectedNet
from bbTT.models.blocks import KaimingDenseBlock
from bbTT.models.physics import LBNPipeline
from bbTT.models.register import register_model


@register_model("lbn_fc")
class LBNFullyConnectedNet(FullyConnectedNet):
    """
    Fully connected network with a Lorentz Boost Network next to the input layer.

    The dense branch and the LBN branch read the same continuous input tensor but use different columns:
    the dense branch takes ``DataConfig.dense_features`` (see ContinuousInputLayer), the LBN takes the
    four-vector components of its particles (see LBNFeatureExtractor). Both outputs are concatenated
    and fed into the first hidden layer. The batch norm inside the LBN branch is the only one of the model.
    """

    monitored_blocks: tuple[str, ...] = FullyConnectedNet.monitored_blocks + ("lbn",)

    def init_layers(self) -> None:
        """
        Builds the fully connected network, adds the LBN branch and rebuilds the first hidden layer,
        whose input now also holds the LBN features.
        """
        super().init_layers()
        lbn_config = self.model_config.lbn_network
        dense_cfg = self.model_config.dense_network

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
        self.dense_1 = KaimingDenseBlock(
            input_nodes=self.input_layer.ndim + self.lbn.ndim,
            output_nodes=int(dense_cfg.nodes),
            activation_functions=dense_cfg.activation_functions,
            normalize=dense_cfg.normalize_linear,
            nonlinearity=dense_cfg.weight_init_nonlinearity,
        )

    def forward(self, categorical_inputs: torch.Tensor, continuous_inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            categorical_inputs (torch.Tensor): Categorical features, shape ``[event, num_categorical_features]``.
            continuous_inputs (torch.Tensor): Continuous features, shape ``[event, num_continuous_features]``.

        Returns:
            torch.Tensor: Logits, shape ``[event, node]``.
        """
        dense_inputs = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        # the lbn needs padded but unstandardized four-vectors, physics is scale dependent
        lbn_outputs = self.lbn(continuous_inputs)

        x = torch.concatenate((dense_inputs, lbn_outputs), dim=1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        x = self.dense_6(x)
        x = self.last_linear(x)
        return x
