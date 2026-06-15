from __future__ import annotations

import typing
import dataclasses
from collections import defaultdict
from collections.abc import Iterable

import torch


EMPTY_INT = -99999
EMPTY_FLOAT = -99999.0

CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda") if torch.cuda.is_available() else CPU_DEVICE

def multiply_sub_process_rates(which_sub_process, sub_process_rates):
    """
    Helper function to multiply sub process rates together up to 2 nested level down.


    Args:
        which_sub_process (_type_): Describes the current sub process
        sub_process_rates (_type_):
    """
    def validate_exactly_two_levels(d):
        for _, inner in d.items():
            # only dictionaries on top level are accepted
            if not isinstance(inner, dict):
                raise TypeError("Expected dict at top-level key")

            # no more than another dict is accepted
            has_dict_values = any(isinstance(v, dict) for v in inner.values())
            if has_dict_values:
                raise ValueError("Nesting deeper than 2 levels detected")

    validate_exactly_two_levels(sub_process_rates)
    final_rate = defaultdict(lambda: 1)
    for name in which_sub_process:
        for key, rate in sub_process_rates[name].items():
            # if a whole group should be handled
            if isinstance(key, Iterable) and not isinstance(key, str):
                for sub_process in key:
                    final_rate[sub_process] *= rate
            # no nested case
            else:
                final_rate[key] *= rate

    return dict(final_rate)


def choice_check(selected, choices):
    # helper to verify that a selected value is part of a set of valid choices, can be used for runtime checks of config values
    choices = typing.get_args(choices)
    if selected not in choices:
        raise ValueError(f"Selected ({selected}) is not part of valid choices {choices}")
