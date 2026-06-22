from __future__ import annotations

import torch

from models.preprocessing.tokenizer import CategoricalTokenizer
from utils import logger

logger_inst = logger.get_logger(__name__)


class CatEmbeddingLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        embedding_dim: int,
        categories: tuple[str],
        expected_categorical_inputs: dict[str, list[int]] | None = None,
        category_dims: int | None = None,
        empty: int = 15,
    ):
        """
        Initializes the categorical feature interface with a tokenizer and an embedding layer with
        given *embedding_dim*.

        The tokenizer maps given *categories* to values defined in *expected_categorical_inputs*.
        Missing values are given a *empty* value, which
        The mapping is defined in .
        The embedding layer then maps this combined feature space into a dense representation.

            embedding_dim (int): Number of dimensions for the embedding layer.
            categories (tuple[str]): Names of the categories as strings.
            expected_categorical_inputs (dict[list[int]]): Dictionary where keys are category
                names and values are lists of integers representing the expected values for
                each category.
            empty (int, optional): Value used to represent missing values in the input tensor.
        """
        super().__init__()
        self.tokenizer = None
        self.category_dims = category_dims
        if not self.category_dims and all(x is not None for x in (categories, expected_categorical_inputs)):
            self.tokenizer = CategoricalTokenizer(
                categories=categories,
                expected_categorical_inputs=expected_categorical_inputs,
                empty=empty)
            self.category_dims = self.tokenizer.num_dim
        self.embeddings = torch.nn.Embedding(
            self.category_dims,
            embedding_dim,
        )

        self.ndim = embedding_dim * len(categories)

    @property
    def look_up_table(self) -> torch.FloatTensor | None:
        return self.tokenizer.map if self.tokenizer else None

    def normalize_embeddings(self):
        # normalize the embedding layer to have unit length
        with torch.no_grad():
            norm = torch.sqrt(torch.sum(self.embeddings.weight**2, dim=-1)).reshape(-1, 1)
            self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight / norm)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.tokenizer:
            x = self.tokenizer(x)

        x = self.embeddings(x)
        return x.flatten(start_dim=1)
