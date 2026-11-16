from __future__ import annotations

from typing import Any

import torch

from bbTT.models.architectures.LBNDenseNet import LBNDenseNet
from bbTT.models.binning import KERNEL_MAP, BinningLayer
from bbTT.models.register import register_model


@register_model("binned_lbn_dense")
class BinnedLBNDenseNet(LBNDenseNet):
    """
    Reuse LBNDenseNet Architecture, but add a learnable BinningLayer as additional head.

    The forward is unchanged and returns logits. The ``binned`` head maps logits to kernel weighted
    probabilities, so cross entropy (logits), monitoring (probabilities) and Asimov (binned) can be
    served from one forward pass.
    """

    HEADS = LBNDenseNet.HEADS + ("binned",)

    def __init__(
        self,
        full_config,
        *args,
        **kwargs,
    ):
        super().__init__(full_config, *args, **kwargs)
        self.freeze_edges()  # TODO for now, change when not working with fixed binning

    def init_layers(self):
        # create normal LBN DenseNet
        super().init_layers()
        self.binning_layer = BinningLayer(
            num_bins=self.binning_config.num_bins,
            bounds=self.binning_config.bounds,
            binning_fn=self.binning_config.binning_fn,
            binning_cfg=self.binning_config.binning_cfg,
            kernel_map=KERNEL_MAP[self.binning_config.kernel_cls],
            kernel_cfg=self.binning_config.kernel_config,
        )

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
        return self.binning_layer.kernels

    def freeze_edges(self):
        self.binning_layer.freeze_edges()

    def unfreeze_edges(self):
        self.binning_layer.unfreeze_edges()

    @property
    def binning_fn(self):
        return self.binning_layer.binning_fn

    @property
    def num_bins(self):
        return self.binning_layer.num_bins

    def evaluation_state(self) -> dict[str, Any]:
        """
        Returns a collection of safe detached cpu snapshots, ready to evaluate.
        Each layer manages how they create this snapshot and no blind search happens.
        So user must define what is returned in the evaluation snapshot.

        Returns:
            dict[Any]: Dictionary with all snapshots exposed from the internal components.
        """
        return {"binning": self.binning_layer.create_evaluation_state()}

    def binned(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Head that bins the class probabilities with the learnable kernels.

        Args:
            logits (torch.Tensor): Output of forward, shape ``[event, node]``.

        Returns:
            torch.Tensor: Kernel weighted probabilities, shape ``[bin, event, node]``.
        """
        # binning acts on probabilities only, binning logits has no physical meaning
        return self.binning_layer(self.probabilities(logits))  # increases dimension at axis 0
