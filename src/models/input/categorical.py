import torch

from models.utils.layer_utils import optional_layer


class CategoricalInputLayer(torch.nn.Module):
    def __init__(
        self,
        embedding_layer: torch.nn.Module,
        padding_categorical_layer: torch.nn.Module | None = None,

        *args,
        **kwargs,
    ):
        """
        Input Layer for categorical features.
        Convert the categorical feature into tokens, via Tokenizer and result are Embedding vectors

        embedding_dim (int): Number of dimensions for the embedding layer.
        categories (tuple[str]): Names of the categories as strings.
        expected_categorical_inputs (dict[list[int]]): Dictionary where keys are category
            names and values are lists of integers representing the expected values for
            each category.
        empty (int, optional): Value used to represent missing values in the input tensor.
        """

        super().__init__()
        self.embedding_layer = embedding_layer
        self.ndim = self.embedding_layer.ndim
        self.padding_categorical_layer = optional_layer(padding_categorical_layer)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.padding_categorical_layer(x)
        x = self.embedding_layer(x)
        return x
