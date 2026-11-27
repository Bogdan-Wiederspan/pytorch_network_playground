from __future__ import annotations

import torch

from bbTT.models.architectures.base_model import BaseModel
from bbTT.models.blocks import DenseBlock, DenseNetBlock
from bbTT.models.register import register_model


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
            "normalize": dense_cfg.normalize_linear,  # activate weight normalization on linear layer weights
            "output_nodes": int(dense_cfg.nodes),
            # physics parameters (e.g. kappa lambda) are fed into every dense block, 0 if none are configured
            "num_physics_parameters": self.num_physics_parameters,
        }

        self.input_layer = self.init_input_layer()
        self.physics_parameter_layer = self.init_physics_parameter_layer()

        self.transition_dense_1 = DenseBlock(
            input_nodes=self.input_layer.ndim + self.num_physics_parameters,
            output_nodes=_dense_config["output_nodes"],
            activation_functions=_dense_config["activation_functions"],
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


    def forward(
        self,
        categorical_inputs: torch.Tensor,
        continuous_inputs: torch.Tensor,
        physics_parameters: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # scaled physics parameters are concatenated to the input of the dense part and fed into every dense block
        physics_parameters = self.scale_physics_parameters(physics_parameters)
        x = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        if physics_parameters is not None:
            x = torch.concatenate((x, physics_parameters), dim=1)
        x = self.transition_dense_1(x)
        x = self.dense_block_1(x, physics_parameters)
        x = self.dense_block_2(x, physics_parameters)
        x = self.dense_block_3(x, physics_parameters)
        x = self.dense_block_4(x, physics_parameters)
        x = self.dense_block_5(x, physics_parameters)
        x = self.last_linear(x)
        return x
