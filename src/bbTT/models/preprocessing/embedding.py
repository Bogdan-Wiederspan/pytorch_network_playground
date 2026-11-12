from __future__ import annotations

import torch

from bbTT.models.preprocessing.tokenizer import CategoricalTokenizer
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)

from typing import Any

from bbTT.models.utils.layer_utils import embedding_state


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
                categories=categories, expected_categorical_inputs=expected_categorical_inputs, empty=empty
            )
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

    def create_evaluation_state(self) -> dict[str, Any]:
        """
        Detached cpu snapshot of the embedding table for monitoring, the tokenizer holds no learnable state.

        With a tokenizer the geometry of the token vectors within each category is exposed. Per category the norm
        of the mean vector (the offset shared by all values, absorbed by the following bias). Per pair of values:
        the distance between the two vectors (what separates the values after that offset), the cosine of the raw
        vectors (direction including the offset) and, for three or more values, the cosine after subtracting the
        mean (direction of the differences only; for two values this is trivially -1 and skipped).

        Returns:
            dict[str, Any]: See embedding_state, plus "offset" {category: float} and per category
                "distance_<category>", "cosine_raw_<category>" and (k >= 3) "cosine_centered_<category>", each
                {"<value a>_vs_<value b>": float}.        """
        state = embedding_state(self.embeddings)
        # there will always be a tokenizer ( expect when one did it them self )
        if self.tokenizer is None:
            return state

        # calculate cosine similarity
        weight = self.embeddings.weight.detach()
        for category, value_to_token in self.tokenizer.tokens_per_category().items():
            # extract corresponding embedding vectors by token
            values = list(value_to_token)
            pairs = [(i, j) for i in range(len(values)) for j in range(i + 1, len(values))]
            offsets = {}

            vectors = weight[list(value_to_token.values())]  # (k, embedding_dim)
            # vectors can have SAME OFFSET, cosine sim. is sensitive to that
            # this is problematic IF offset is much bigger than directional vector
            mean = vectors.mean(dim=0, keepdim=True)
            centered = vectors - mean
            offsets[category] = torch.linalg.vector_norm(mean).item()
            # cos sim. makes only sense for PAIRS of the category
            distance = torch.cdist(vectors, vectors)  # (k, k), invariant under centering
            state[f"distance_{category}"] = {f"{values[i]}_vs_{values[j]}": distance[i, j].item() for i, j in pairs}

            cosine_of = [("raw", vectors)] + ([("centered", centered)] if len(values) >= 3 else [])
            for name, vecs in cosine_of:

                cosine = torch.nn.functional.cosine_similarity(
                    vecs[:, None, :], vecs[None, :, :], dim=-1
                    )  # (k, k)

                state[f"cosine_{name}_{category}"] = {f"{values[i]}_vs_{values[j]}": cosine[i, j].item() for i, j in pairs}
        state["offset"] = offsets
        return state


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.tokenizer:
            x = self.tokenizer(x)

        x = self.embeddings(x)
        return x.flatten(start_dim=1)
