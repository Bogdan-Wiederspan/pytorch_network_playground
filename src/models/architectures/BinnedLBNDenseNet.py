from __future__ import annotations

import torch

from ..binning import KERNEL_MAP, BinningLayer
from ..register import register_model
from .LBNDenseNet import LBNDenseNet


@register_model("binned_lbn_dense")
class BinnedLBNDenseNet(LBNDenseNet):
    is_binned = True

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

    def init_layers(self):
        # create normal LBN DenseNet
        super().init_layers()
        # init binning layer, its possible to deactivate the binning layer.
        if not self.model_building_config.enable_binning:
            self.binning_layer = torch.nn.Identity()
            self.is_binned = False
        else:
            self.binning_layer = BinningLayer(
                num_bins=self.binning_config.num_bins,
                bounds=self.binning_config.bounds,
                binning_fn=self.binning_config.binning_fn,
                binning_cfg=self.binning_config.binning_cfg,
                kernel_map=KERNEL_MAP[self.binning_config.kernel_cls],
                kernel_cfg=self.binning_config.kernel_config,
                )
            self.freeze_edges()

    def learning_mode_bin_only(self):
        all_layers = dict(self.named_children())
        all_layers["binning_layer"].requires_grad = True


    def learning_mode_model_only(self):
        all_layers_except_binning = dict(self.named_children())
        all_layers_except_binning.pop("binning_layer")
        for _, layer in all_layers_except_binning.items():
            layer.requires_grad = True

    @property
    def bin_edges_active(self):
        return self.binning_layer.bin_edges

    @property
    def bin_edges_original(self):
        return self.binning_layer.bin_edges_original

    @property
    def kernels(self):
        return self.binning_layer.get_kernels()

    def freeze_binning(self):
        self.binning_layer.freeze_edges()

    def unfreeze_binning(self):
        self.binning_layer.unfreeze_edges()

    def forward(self, categorical_inputs, continuous_inputs):
        normal_network_output = super().forward(categorical_inputs, continuous_inputs)
        binned_output = self.binning_layer(normal_network_output) # increases dimension at axis 0
        return normal_network_output, binned_output
