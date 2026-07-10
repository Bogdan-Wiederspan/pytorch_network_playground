from __future__ import annotations

import torch

from models.binning import BaseKernel, OverflowKernel, UnderflowKernel


class TanhKernel(BaseKernel):
    def __init__(
        self,
        edges,
        bin_height: float = 1,
        left_notch=0,
        right_notch=0,
        absolute_notch=True,
        eps=1e-3,
        full_width=1,

        **kwargs,
        ):
        """
        Kernel object that models a bin with smoothed edges.
        """
        super().__init__(
            edges=edges,
            bin_height=bin_height,
            left_notch=left_notch,
            right_notch=right_notch,
            absolute_notch=absolute_notch,
            **kwargs,
            )
        self.eps = torch.tensor(eps)
        self.full_width_from_eps_to_eps = torch.tensor(full_width) if full_width is not None else self._smoothing_width_for_constant()
        self.tau = self.compute_smoothness(self.full_width_from_eps_to_eps, self.eps, 0)
        self.left_cut = None
        self.right_cut = None
        self.checks()

    def _smoothing_width_for_constant(self):
        # when all notches are constant, one can extract the information from current one.
        return torch.tensor(self.left_notch_size + self.right_notch_size)

    # def compute_width_from_smoothness(self, smoothness, eps):
    #     half_width = (smoothness * torch.arctanh(2 * ( 1 / 2 - eps)))
    #     full_width = 2 * half_width
    #     return full_width

    # def compute_smoothness(self, full_width, eps):
    #     # compute smoothnes to go from 50% to eps within full_width / 2
    #     half_width = full_width / 2 # due to symmetry
    #     smoothness = half_width / torch.arctanh( 2 * ( 0.5 - eps) )
    #     return smoothness

    # def right_transition_fn(self, x):
    #     from IPython import embed; embed(header="RIGHt Line 37 | File: tanh_kernel.py")
    #     anchor = self.right_transition_coordinate
    #     smoothing = self.tau
    #     return 0.5 * (1 - torch.tanh((x - anchor) / smoothing))

    def compute_width_from_smoothness(self, smoothness, eps):
        half_width = (smoothness * torch.arctanh(2 * (1 / 2 - eps)))
        full_width = 2 * half_width
        return full_width

    def compute_smoothness(self, full_width, eps, anchor):
        # compute smoothnes to go from 50% to eps within full_width / 2
        half_width = full_width / 2  # due to symmetry
        smoothness = half_width / (torch.arctanh(2 * (0.5 - eps)) + anchor)
        return smoothness

    def right_transition_fn(self, x):
        anchor = self.right_transition_coordinate + self.full_width_from_eps_to_eps / 2
        smoothing = self.tau
        return 0.5 * (1 - torch.tanh((x - anchor) / smoothing))

    def left_transition_fn(self, x):
        anchor = self.left_transition_coordinate - self.full_width_from_eps_to_eps / 2
        smoothing = self.tau
        return 0.5 * (1 + torch.tanh((x - anchor) / smoothing))

    def _compute_normalization(self):
        # function is by definition normalized to 1
        return 1

    def _apply_cut_kernel(self, x, y):
        outside = torch.zeros_like(x, dtype=torch.bool)
        if self.left_cut is not None:
            outside |= (x < self.left_cut)
        if self.right_cut is not None:
            outside |= (x > self.right_cut)
        return y.masked_fill(outside, 0.0)

    def kernel(self, x):
        # extend this by first run base kernel and then set value to 0 when reaching the transition point of the neighbouring bin to ensure locality
        y = self._base_kernel(x)
        y = self._apply_cut_kernel(x, y)
        return y


class TanhUnderflowKernel(UnderflowKernel, TanhKernel):
    pass


class TanhOverflowKernel(OverflowKernel, TanhKernel):
    pass
