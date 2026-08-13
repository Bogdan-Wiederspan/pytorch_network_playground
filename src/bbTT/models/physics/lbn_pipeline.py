from __future__ import annotations

import torch

from . import LBN, LBNFeatureExtractor


class LBNPipeline(torch.nn.Module):
    def __init__(
        self,
        continuous_features: list[str],
        M: int = 10,
        # no N necessary for LBN, getting information from Feature extractor
        weight_init_scale: float | int = 1.0,
        clip_weights: bool = False,
        eps: float = 1.0e-5,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.lbn_feature_extractor = LBNFeatureExtractor(continuous_features=continuous_features)

        self.lbn = LBN(
            N=self.lbn_feature_extractor.num_particles,
            M=M, clip_weights=clip_weights,
            eps=eps,
            weight_init_scale=weight_init_scale
            )
        self.lbn_batch_norm = torch.nn.BatchNorm1d(self.lbn.ndim)


    @property
    def ndim(self):
        return self.lbn.ndim

    def forward(self, x):
        x = self.lbn_feature_extractor(x)
        x = self.lbn(x)
        x = self.lbn_batch_norm(x)
        return x
