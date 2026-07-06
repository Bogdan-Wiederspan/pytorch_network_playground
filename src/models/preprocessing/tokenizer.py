from __future__ import annotations

import copy

import torch

from utils import logger, utils

logger_inst = logger.get_logger(__name__)

class CategoricalTokenizer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        categories: tuple[str ],
        expected_categorical_inputs: dict[str, list[int]],
        empty: int = None,
    ):
        """
        Initializes tokenizer for given *expected_categorical_inputs*.
        The tokenizer creates a mapping array in the order of given columns defined in *categories*.
        Empty values are represented as *empty*.
        All categories will be mapped into a common categorical space and ready to be used by a embedding layer.

        Args:
            categories (tuple[str]): Names of the categories as strings.
            Sorting of the entries must correspond to the order of columns in input tensor!
            expected_categorical_inputs (dict[list[int]], optional): Dictionary where keys are category
                names and values are lists of integers representing the expected values for
                each category.
            empty (int, optional): Value used to represent missing values in the input tensor.
                The empty value must be positive and not already used in the categories.
                If not given, no handling of missing values will be done.
                Defaults to None.
        """
        super().__init__()
        self._expected_inputs, self._empty = self.setup(categories, expected_categorical_inputs, empty)

        # setup look up table, returns dummy if None
        _map, _min = self.LookUpTable(self.pad_to_longest())
        _indices = None if _min is None else torch.arange(len(_min))

        # register buffer
        self.map = torch.nn.Buffer(_map, persistent=True)
        self.min = torch.nn.Buffer(_min, persistent=True)
        self.indices = torch.nn.Buffer(_indices, persistent=True)

    def load_state_dict(self, state_dict: dict, strict: bool, assign: bool):
        # overload load_state_dict to set buffer sizes to same of state dict
        for name in self.state_dict().keys():
            self.__setattr__(name, torch.zeros_like(state_dict[name]))
        super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

    def setup(
        self,
        categories: list[str],
        expected_inputs: list[str],
        empty: int,
    ) -> tuple[dict[str, list[int]], int | None]:
        # do all the preparation steps like value checking and adding of empty categories
        # also remove double categories and only take the categories used by the network
        def _empty(expected_inputs, empty):
            if empty is None:
                return None
            if empty < 0:
                raise ValueError("Empty value must be positive")
            if empty in set([item for sublist in expected_inputs.values() for item in sublist]):
                raise ValueError(f"Empty value {empty} is already used in on the categories")
            return empty

        # check if cateogries are part of expected_inputs at least one
        if not set(categories) & set(expected_inputs.keys()):
            sep = "\n"
            raise ValueError(
                f"Categories must not be part of Expected categories:\n"
                f"categories:\n{sep.join(categories)}\nexpected categories:\n{sep.join(expected_inputs.keys())}"
            )

        if expected_inputs is None:
            return {}, None

        # check empty for faulty values
        # add empty category with value of empty to each value
        expected_inputs = copy.deepcopy(expected_inputs)
        empty = _empty(expected_inputs, empty)
        _expected_inputs = {}
        for category in map(str, categories):
            data = expected_inputs[category]
            if empty is not None:
                # when empty value is given append it to category
                data.append(empty)
            _expected_inputs[category] = data
        return _expected_inputs, empty

    @property
    def num_dim(self) -> torch.IntTensor:
        return torch.max(self.map) + 1

    def __repr__(self):
        # create dummy input from expected_categorical_inputs
        padded_array = self.pad_to_longest()
        if padded_array is None:
            return "Not initialized Tokenizer"

        expected_pad = padded_array.transpose(0, 1).to(device=self.map.device)
        shifted = expected_pad - self.min
        output_per_feature = self.map[self.indices, shifted].transpose(0, 1)
        _str = []
        _str.append("Translation (input : output):")
        for ind, (category, expected_value) in enumerate(self._expected_inputs.items()):
            _str.append(f"{category}: {expected_value} -> {output_per_feature[ind][:len(expected_value)].tolist()}")
        return "\n".join(_str)

    def check_for_values_outside_range(self, input_tensor: torch.FloatTensor):
        """
        Helper function checks *input_tensor* for values the tokenizer does not expect but found.

        Args:
            input_tensor (torch.tensor): Input tensor of categorical features.
        """
        # reshape to have features in the first dimension
        input_tensor = input_tensor.transpose(0, 1)
        for i, (category, expected_value) in enumerate(self._expected_inputs.items()):
            uniques = set(torch.unique(input_tensor[i]).to(torch.int32).tolist())
            expected = set(expected_value)
            if uniques != expected:
                difference = uniques - expected
                logger_inst.critical(
                    f"{category} has values outside the expected range: {difference}.\n"
                    "The tokenizer will return wrong values for these inputs."
                    )

    def pad_to_longest(self) -> torch.FloatTensor:
        if not self._expected_inputs:
            return None
        # helper function to pad the input tensor to the longest category
        # first value of the category is used as padding value
        local_max = max([
            len(input_for_category)
            for input_for_category in self._expected_inputs.values()
        ])
        # pad with first value of the category, so we guarantee to not introduce new values
        array = torch.stack(
            [
                torch.nn.functional.pad(
                    torch.tensor(input_for_category),
                    (0, local_max - len(input_for_category)),
                    mode="constant",
                    value=input_for_category[0],
                )
                for input_for_category in self._expected_inputs.values()
            ],
        )
        return array

    def LookUpTable(
        self,
        array: torch.FloatTensor,
        padding_value: int = utils.EMPTY_INT,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor] | None:
        """
        Maps multiple categories given in *array* into a sparse vectoriced lookuptable.
        The given *padding_value* represents no matching categories.

        Args:
            array (torch.tensor): 2D array of categories.

        Returns:
            tuple([torch.tensor]), None: Returns minimum and LookUpTable
        """
        if array is None:
            return None, None
        # append empty to the array representing the empty category if empty is set
        # if self._empty is not None:
        #     array = torch.cat([array, torch.ones(array.shape[0], dtype=torch.int32).reshape(-1, 1) * self.empty], axis=-1)

        # shift input by minimum, pushing the categories to the valid indice space
        minimum = array.min(axis=-1).values
        # shift the input array by their respective minimum
        indices_array = array - minimum.reshape(-1, 1)
        # biggest shifted value + 1
        upper_bound = torch.max(indices_array) + 1

        # warn for big categories
        if upper_bound > 100:
            logger_inst.warning("Be aware that a large number of categories will result in a large sparse lookup array")

        # create mapping empty
        mapping_array = torch.full(
            size=(len(minimum), upper_bound),
            fill_value=padding_value,
            dtype=torch.int32,
        )

        # fill empty with vocabulary
        stride = 0
        # transpose from event to feature loop
        for feature_idx, feature in enumerate(indices_array):
            unique = torch.unique(feature, dim=None)
            mapping_array[feature_idx, unique] = torch.arange(
                stride, stride + len(unique),
                dtype=torch.int32,
            )
            stride += len(unique)
        return mapping_array, minimum

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # shift input array by their respective minimum and slice translation accordingly
        # map to int to be used as indices
        shifted = (x - self.min).to(torch.int32)
        output = self.map[self.indices, shifted]
        return output
