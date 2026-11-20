from __future__ import annotations

import torch

from bbTT.utils.constants import EMPTY_FLOAT, EMPTY_INT  # noqa: F401  re-exported, torch-free home

CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else CPU_DEVICE

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
