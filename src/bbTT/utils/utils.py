from __future__ import annotations

import typing

import torch

EMPTY_INT = -99999
EMPTY_FLOAT = -99999.0

CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda") if torch.cuda.is_available() else CPU_DEVICE


COLOR_CYCLE = [
    #e41a1c,
    #377eb8,
    #4daf4a,
    #984ea3,
    #ff7f00,
    #ffff33,
    #a65628,
    #f781bf,
    #999999,
    #cab2d6,
    #6a3d9a,
    #ffff99,
    #b15928,
    #a6cee3,
    #fb9a99,
    #b3de69,
    #8dd3c7,
]


def choice_check(selected, choices):
    # helper to verify that a selected value is part of a set of valid choices, can be used for runtime checks of config values
    choices = typing.get_args(choices)
    if selected not in choices:
        raise ValueError(f"Selected ({selected}) is not part of valid choices {choices}")


def expand_braces(input: tuple[str]) -> tuple[str]:
    """
    Expand all braces of given strings and return tuple of all extended strings.
    For example, input ("a{b,c}d", "e{f,g}h") will return ("abd", "acd", "efh", "egh").

    Args:
        input (tuple[str]): Iterable of strings with braces to be expanded.

    Returns:
        tuple[str]: Tuple of expanded strings.
    """

    def brace_expand(s):
        # "a{b,c}d" -> ["abd", "acd"]
        if "{" not in s:
            return [s]
        pre, post = s.split("{", 1)
        mid, post = post.split("}", 1)
        parts = mid.split(",")
        expanded = []
        for part in parts:
            for rest in brace_expand(post):
                expanded.append(pre + part + rest)
        return expanded

    cols = []
    for _str in input:
        cols.extend(brace_expand(_str))
    return tuple(cols)
