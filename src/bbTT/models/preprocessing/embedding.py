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

        Every value of a category (e.g. pair_type = 0, 1, 2) owns one vector in the table. The network can only
        tell two values apart by how different their vectors are, so the quantities below compare the vectors of
        each category with each other. Its important that interpretation needs an ANCHOR. Thus random init values are given in the following explanation.

        How to read the plots:

        - "distance_<category>": one line per pair of values.
            They describe how far apart the two vectors are. Larger means the network can separate these two values better.
            As comparison, a random initialization gives about sqrt(2 * embedding_dim).
            A pair far above that was pushed apart by training, a pair far below it was pulled together.

        - "angle_raw_<category>": one line per pair.
            Its describing the angle between the two vectors in degrees: 0 = same direction, 90 = unrelated, 180 = opposite.
            Random initialization gives about 90 +- 20.
            Two vectors can look similar here only because they share a common part, see offset description.

        - "angle_centered_<category>": only for three or more values.
            The same as angle after removing the common part/offset of the category.
            This shows which values group together (small angle) and which are opposites
            (large angle).
            The random values depend on number of categories: k values the random expectation is acos(-1 / (k - 1)), e.g. 120 for k = 3.

        - "offset", one line per category.
            Describes the length of the common part (common = mean of all vectors in that category).
            This part is tricky to interpreted, since its the same for every event and carries no information about which value is active.
            For example the bias of the next layer could also do the same job.
            A large offset compared to the distances means most of the token
            norm is shared ballast rather than separation. This affects also <angle_raw_*> since a very offset value will be more narrow by definition.

        Returns:
            dict[str, Any]: See embedding_state, plus "offset" {category: float} and per category
                "distance_<category>", "angle_raw_<category>" and (k >= 3) "angle_centered_<category>", each
                {"<value a>_vs_<value b>": float}.
        """
        state = embedding_state(self.embeddings)
        # there will always be a tokenizer in den Pipeline
        # this is just there for the case one uses pre-tokenized data)
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


            angle_of = [("raw", vectors)] + ([("centered", centered)] if len(values) >= 3 else [])
            for name, vecs in angle_of:
                cosine = torch.nn.functional.cosine_similarity(vecs[:, None, :], vecs[None, :, :], dim=-1)  # (k, k)
                # clamp against float error outside [-1, 1], which would give nan in acos
                angle = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
                state[f"angle_{name}_{category}"] = {f"{values[i]}_vs_{values[j]}": angle[i, j].item() for i, j in pairs}

        state["offset"] = offsets
        return state


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.tokenizer:
            x = self.tokenizer(x)

        x = self.embeddings(x)
        return x.flatten(start_dim=1)
