from __future__ import annotations

import torch

from bbTT.models.binning import BaseKernel, OverflowKernel, UnderflowKernel


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
        eps = torch.tensor(eps)

        full_width = (
            torch.tensor(full_width)
            if full_width is not None
            else self._smoothing_width_for_constant()
        )

        tau = self.compute_smoothness(full_width, eps, 0)

        self.register_buffer("eps", eps)
        self.register_buffer("full_width", full_width)
        self.register_buffer("tau", tau)
        self.checks()

    def _smoothing_width_for_constant(self):
        # when all notches are constant, one can extract the information from current one.
        return self.left_notch_size + self.right_notch_size

    def compute_width_from_smoothness(self, smoothness, eps):
        half_width = smoothness * torch.arctanh(2 * (1 / 2 - eps))
        full_width = 2 * half_width
        return full_width

    def compute_smoothness(self, full_width, eps, anchor):
        # smoothness width is defined going from 50% to eps within full_width / 2
        half_width = full_width / 2  # due to symmetry
        smoothness = half_width / (torch.arctanh(2 * (0.5 - eps)) + anchor)
        return smoothness

    def right_transition_fn(self, x):
        anchor = self.right_transition_coordinate + self.full_width / 2
        smoothing = self.tau
        return (0.5 * (1 - torch.tanh((x - anchor) / smoothing))) * self.bin_height

    def left_transition_fn(self, x):
        anchor = self.left_transition_coordinate - self.full_width / 2
        smoothing = self.tau
        return (0.5 * (1 + torch.tanh((x - anchor) / smoothing))) * self.bin_height

    def _compute_normalization(self):
        # function is by definition normalized to 1
        return torch.tensor(1)

    def kernel(self, x):
        # extend this by first run base kernel and then set value to 0 when reaching the transition point of the neighbouring bin to ensure locality
        y = self._base_kernel(x)
        y = self._apply_cut_mask(x, y)
        return y


class TanhUnderflowKernel(UnderflowKernel, TanhKernel):
    pass


class TanhOverflowKernel(OverflowKernel, TanhKernel):
    pass
