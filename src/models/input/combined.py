from __future__ import annotations

import torch

from models.preprocessing.embedding import CatEmbeddingLayer
from models.utils.layer_utils import optional_layer


class OptionalInputLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        continuous_layer_inst: torch.nn.Module | None = None,
        categorical_layer_inst: torch.nn.Module | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.continuous_layer = continuous_layer_inst
        self.categorical_layer = categorical_layer_inst
        self.ndim = self.continuous_layer.ndim + self.categorical_layer.ndim

    def forward(self, *, categorical_inputs, continuous_inputs):
        x = torch.cat(
            [
                self.continuous_layer(continuous_inputs),
                self.categorical_layer(categorical_inputs),
            ],
            dim=1,
        )
        return x

class InputLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        continuous_inputs: tuple[str],
        embedding_dim: int,
        categorical_inputs: tuple[str] | None = None,
        category_dims: int | None = None,
        expected_categorical_inputs: dict[str, list[int]] | None = None,
        empty: int = 15,
        std_layer: torch.nn.Module | None = None,
        rotation_layer: torch.nn.Module | None = None,
        padding_continuous_layer: torch.nn.Module | None = None,
        padding_categorical_layer: torch.nn.Module | None = None,
        *args,
        **kwargs,
    ):
        """
        Enables the use of categorical and continuous features in a single model.
        A tokenizer and embedding layer are created  is created using and an embedding layer.
        The continuous features are passed through a linear layer and then concatenated with the
        categorical features.
        """
        super().__init__()
        self.empty = empty
        self.ndim = len(continuous_inputs)
        self.embedding_layer = None
        # when categories exist
        if categorical_inputs is not None:
            # and categories has clear ranges
            if expected_categorical_inputs is not None:
                self.embedding_layer = CatEmbeddingLayer(
                    embedding_dim=embedding_dim,
                    categories=categorical_inputs,
                    expected_categorical_inputs=expected_categorical_inputs,
                    empty=empty)
            # otherwise ?
            elif category_dims:
                self.embedding_layer = CatEmbeddingLayer(
                    embedding_dim=embedding_dim,
                    category_dims=category_dims,
                    categories=categorical_inputs,
                    empty=empty,
                )

        if self.embedding_layer:
            self.ndim += self.embedding_layer.ndim

        self.rotation_layer = optional_layer(rotation_layer)
        self.std_layer = optional_layer(std_layer)
        self.padding_continuous_layer = optional_layer(padding_continuous_layer)
        self.padding_categorical_layer = optional_layer(padding_categorical_layer)

    def categorical_preprocessing_pipeline(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.padding_categorical_layer(x)
        return self.embedding_layer(x)

    def continuous_preprocessing_pipeline(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # preprocessing
        x = self.padding_continuous_layer(x)
        x = self.rotation_layer(x)
        x = self.std_layer(x)
        return x

    def forward(self, categorical_inputs, continuous_inputs):
        # HINT: When comparing this layer with other compare order of inputs
        x = torch.cat(
            [
                self.continuous_preprocessing_pipeline(continuous_inputs),
                self.categorical_preprocessing_pipeline(categorical_inputs),
            ],
            dim=1,
        )
        return x
