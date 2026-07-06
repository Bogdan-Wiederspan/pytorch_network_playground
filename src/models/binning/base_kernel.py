from __future__ import annotations

import abc

import torch


class BaseKernel(abc.ABC):

    @abc.abstractmethod
    def kernel(self, x: torch.tensor) -> torch.tensor:
        """
        Returns a kernel function that operates on a given torch tensors x
        """
        return x

    @property
    @abc.abstractmethod
    def normalization(self) -> torch.tensor:
        """
        Calculates the normalization of the kernel and needs to be provided by you.
        """
        pass

    def wrap_tensors(self, *x: torch.tensor) -> list[torch.tensor]:
        """
        Helper function to wrap everything in *x* into a torch tensor and return as list to unpack.

        Returns:
            (list[torch.Tensor]): A list of all x converted to torch tensors
        """
        tensors = []
        for _x in x:
            if not isinstance(_x, torch.Tensor):
                _x = torch.tensor(_x)
            tensors.append(_x)
        return tensors

def create_kernels(edges, kernel_cls, kernel_cfg):
    lower, upper = edges[:-1], edges[1:]
    kernels = {}
    max_bins = len(lower)
    for bin_num, edge in enumerate(zip(lower, upper), 1):
        if bin_num == 1:
            bin_type = "underflow"
        elif bin_num == max_bins:
            bin_type = "overflow"
        else:
            bin_type = "normal"
        kernel_cfg["bin_type"] = bin_type
        kernels[bin_num] = kernel_cls(edge, **kernel_cfg)
    return kernels
