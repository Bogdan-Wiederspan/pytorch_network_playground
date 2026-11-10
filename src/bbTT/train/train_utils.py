from __future__ import annotations

from collections.abc import Iterable

import torch

from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)

def clip_gradients(parameters: Iterable[torch.nn.Parameter], clip_value: float = 1.0):
    for p in parameters:
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
