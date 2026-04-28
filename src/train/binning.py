from __future__ import annotations

import torch
import abc

__all__ = ["BinningBase", "EqualDistant"]
# TODO understand this function before using better
class BinningBase(abc.ABC):
    def __call__(self, lower_edge, upper_edge, num_bins, forward=True):
        # TODO USe forward mode ?
        t_lower, t_upper = self.transformation_fn(lower_edge), self.transformation_fn(upper_edge)
        t_linspace = torch.linspace(t_lower, t_upper, num_bins)
        return self.back_transformation_fn(t_linspace)

    @abc.abstractmethod
    def transformation_fn(self, x):
        raise NotImplementedError("Implement Transformation Function")

    @abc.abstractmethod
    def back_transformation_fn(self, y):
        raise NotImplementedError("Implement Inverse Transformation Function")

class EqualDistant(BinningBase):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def transformation_fn(self, x):
        return x

    def back_transformation_fn(self, y):
        return y

class Decompress(BinningBase):
    def __init__(self, eps=1e-6, *args, **kwargs):
        super().__init__()
        self.eps = eps

    def transformation_fn(self, x):
        return x / (1 - x + self.eps)

    def back_transformation_fn(self, y):
        return (y + y * self.eps)/ (1 + y)

def transformed_edges(transformation_fn, bins, num_bins=25):
        if len(bins) == 2 and isinstance(num_bins, int):
            lower, upper = bins
            bins = torch.linspace(lower, upper, num_bins)
        else:
            bins = torch.tensor(bins)
        return transformation_fn(bins)
