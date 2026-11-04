from __future__ import annotations

import torch

from ..utils.layer_utils import optional_layer


class CategoricalInputLayer(torch.nn.Module):
    def __init__(
        self,
        embedding_layer: torch.nn.Module,
        padding_categorical_layer: torch.nn.Module | None = None,
        *args,
        **kwargs,
    ):
        """
        Preprocessing branch for categorical features:
            padding → embedding.

        The embedding layer tokenizes the categories and maps them to a combined embedding vectors space.
        Its ``ndim`` defines the output dimension of this branch.

        A padding layer passed as None is replaced by an identity.

        Args:
            embedding_layer (torch.nn.Module): Tokenizer plus embedding, needs an ``ndim`` attribute.
            padding_categorical_layer (torch.nn.Module | None, optional): Padding layer for dummy values. Defaults to None.
         """

        super().__init__()
        self.embedding_layer = embedding_layer
        self.ndim = self.embedding_layer.ndim
        self.padding_categorical_layer = optional_layer(padding_categorical_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Categorical features of shape (batch, n_categories), integer-valued.

        Returns:
            torch.Tensor: Concatenated embedding vectors of shape (batch, ndim).
        """
        x = self.padding_categorical_layer(x)
        x = self.embedding_layer(x)
        return x
