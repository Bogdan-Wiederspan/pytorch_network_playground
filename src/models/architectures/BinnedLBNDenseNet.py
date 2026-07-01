from __future__ import annotations

from dataclasses import asdict

import torch

from ..binning import BinningLayer
from ..binning import gaussian_kernel as kernel
from ..register import register_model
from .LBNDenseNet import LBNDenseNet


@register_model("binned_lbn_dense")
class BinnedLBNDenseNet(LBNDenseNet):
    """
    Reuse LBNDenseNet Architecture, but add a learnable BinningLayer at the end of the network.
    The output is now two-headed: One binned and one non-binned network.
    The network is only trained on the binned output.
    """
    def __init__(
        self,
        full_config,
        *args,
        **kwargs,
        ):
        super().__init__(full_config, *args, **kwargs)
        self.is_binned = True


    def init_layers(self):
        # create normal LBN DenseNet
        super().init_layers()
        # init binning layer
        # add binning layer if necessary
        if self.binning_config is None:
            self.binning_layer = torch.nn.Identity()
            self.is_binned = False
        else:
            self.binning_layer = BinningLayer(
                num_bins=self.binning_config.num_bins,
                bounds=self.binning_config.bounds,
                binning_fn=self.binning_config.binning_fn,
                binning_cfg=self.binning_config.binning_cfg,
                kernel_cls=getattr(kernel, self.binning_config.kernel_cls),
                kernel_cfg=self.binning_config.kernel_config[self.binning_config.kernel_cls],
                )

    def learning_mode_bin_only(self):
        all_layers = dict(self.named_children())
        all_layers["binning_layer"].requires_grad = True


    def learning_mode_model_only(self):
        all_layers_except_binning = dict(self.named_children())
        all_layers_except_binning.pop("binning_layer")
        for _, layer in all_layers_except_binning.items():
            layer.requires_grad = True

    def forward(self, categorical_inputs, continuous_inputs):
        normal_network_output = super().forward(categorical_inputs, continuous_inputs)
        binned_output = self.binning_layer(normal_network_output) # increases dimension at axis 0
        return normal_network_output, binned_output
