from __future__ import annotations

from typing import Any

import torch

from bbTT.models.utils.layer_utils import batch_norm_state, optional_layer

from . import LBN, LBNFeatureExtractor


class LBNPipeline(torch.nn.Module):
    def __init__(
        self,
        continuous_features: list[str],
        padding_layer_inst: torch.nn.Module,
        rotation_layer_inst: torch.nn.Module | None = None,
        M: int = 10,
        # no N necessary for LBN, getting information from Feature extractor
        weight_init_scale: float | int = 1.0,
        clip_weights: bool = False,
        eps: float = 1.0e-5,
        safe_sqrt_eps: float = 1.0e-5,
    ):
        """
        Self-contained LBN branch: padding → rotation → feature extraction → LBN → batch norm.
        The branch takes the raw continuous features and manages its own preprocessing, since the LBN needs
        sanitized but unstandardized four-vectors (physics is scale dependent).

        Args:
            continuous_features (list[str]): Names of the continuous features, used to locate the four-vector components.
            padding_layer_inst (torch.nn.Module): Replaces dummy markers before the four-vectors are built. Mandatory,
                the LBN would otherwise boost dummy values.
            rotation_layer_inst (torch.nn.Module | None, optional): Phi rotation applied after padding. Defaults to None.
            M (int, optional): Number of particle and rest frame combinations. Defaults to 10.
            weight_init_scale (float | int, optional): Scale of the LBN weight initialization. Defaults to 1.0.
            clip_weights (bool, optional): Clip LBN combination weights to be non-negative. Defaults to False.
            eps (float, optional): Numerical stability constant. Defaults to 1.0e-5.
            safe_sqrt_eps (float, optional): Lower bound inside square roots. Defaults to 1.0e-5.

        Raises:
            ValueError: If *padding_layer_inst* is None.
        """
        super().__init__()
        if padding_layer_inst is None:
            raise ValueError("LBNPipeline needs a padding layer, the LBN must not see dummy values")
        self.padding_layer = padding_layer_inst
        self.rotation_layer = optional_layer(rotation_layer_inst)

        self.lbn_feature_extractor = LBNFeatureExtractor(continuous_features=continuous_features)

        self.lbn = LBN(
            N=self.lbn_feature_extractor.num_particles,
            M=M,
            clip_weights=clip_weights,
            eps=eps,
            weight_init_scale=weight_init_scale,
            safe_sqrt_eps=safe_sqrt_eps,
        )
        self.lbn_batch_norm = torch.nn.BatchNorm1d(self.lbn.ndim)

    @property
    def ndim(self) -> int:
        return self.lbn.ndim

    def sanitize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Padding and rotation on raw values.
        The replacement is re-applied after the rotation so missing particles stay on their replacement value whatever it is.
        The LBN needs sanitized but unstandardized four-vectors (physics is scale dependent).

        Args:
            x (torch.Tensor): Raw continuous features of shape (batch, n_features).

        Returns:
            torch.Tensor: Padded and rotated features.
        """
        mask = self.padding_layer.mask(x)
        x = self.padding_layer.replace(x, mask)
        x = self.rotation_layer(x)
        x = self.padding_layer.replace(x, mask)
        return x

    def create_evaluation_state(self) -> dict[str, Any]:
        """
        Snapshot of the LBN combination weights and the running statistics of the batch norm on its features;
        padding, rotation and feature extraction hold no learnable state.

        Returns:
            dict[str, Any]: See LBN.create_evaluation_state and batch_norm_state.
        """
        return {**self.lbn.create_evaluation_state(), **batch_norm_state(self.lbn_batch_norm)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Raw continuous features of shape (batch, n_features).

        Returns:
            torch.Tensor: Batch-normalized LBN features of shape (batch, ndim).
        """
        x = self.sanitize(x)
        x = self.lbn_feature_extractor(x)
        x = self.lbn(x)
        x = self.lbn_batch_norm(x)
        return x
