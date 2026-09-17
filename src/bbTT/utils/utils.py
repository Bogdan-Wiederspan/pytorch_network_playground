from __future__ import annotations

import torch

EMPTY_INT = -99999
EMPTY_FLOAT = -99999.0

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
