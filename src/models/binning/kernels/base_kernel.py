from __future__ import annotations

import abc

import torch


class BaseKernel(torch.nn.Module, abc.ABC):
    bin_identity = "normal"
    has_left_transition = True
    has_right_transition = True

    def __init__(self, edges, bin_height=1.0, left_notch=0, right_notch=0, absolute_notch=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("lower_edge", edges[0])
        self.register_buffer("upper_edge", edges[1])
        self.register_buffer("bin_height", torch.as_tensor(bin_height))
        self.register_buffer("left_notch", torch.as_tensor(left_notch))
        self.register_buffer("right_notch", torch.as_tensor(right_notch))
        self.right_cut = None
        self.left_cut = None
        self.absolute_notch = absolute_notch

    # --- Geometry ---
    def set_edges(self, lower, upper):
        # since these are buffers, simple overwriting would unregister buffers
        # only value is copied over
        self.lower_edge.copy_(lower)
        self.upper_edge.copy_(upper)

    def set_cuts(self, left=None, right=None):
        self.left_cut = left
        self.right_cut = right

    @property
    def bin_width(self) -> torch.Tensor:
        return self.upper_edge - self.lower_edge

    @property
    def left_transition_coordinate(self) -> torch.Tensor:
        return self.lower_edge + self.left_notch_size

    @property
    def right_transition_coordinate(self) -> torch.Tensor:
        return self.upper_edge - self.right_notch_size

    @property
    def left_notch_size(self):
        if self.absolute_notch:
            return self.left_notch
        return self.bin_width * self.left_notch

    @property
    def right_notch_size(self):
        if self.absolute_notch:
            return self.right_notch
        return self.bin_width * self.right_notch

    @property
    def transition_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.left_transition_coordinate,
            self.right_transition_coordinate,
            )

    @abc.abstractmethod
    def _compute_normalization(self) -> torch.Tensor:
        """
        Computes the normalization of the kernel and needs to be provided by you.
        If no normalization is necessary return 1
        """
        pass

    @property
    def normalization(self) -> torch.Tensor:
        """
        Returns the normalization of the kernel, which is computed by the _compute_normalization method.
        """
        return self._compute_normalization()

    # --- Debugging ---
    def control_plot(self, x, x_ticks=(0, 1, 21)):
        """
        Small helper to visualize the implemented kernels.
        """
        import matplotlib.pyplot as plt

        y = self(x).cpu()
        plt.plot(x.cpu(), y)

        plt.xticks(torch.linspace(*x_ticks).numpy(), rotation=45)
        return plt.gcf(), plt.gca()

    def checks(self) -> None:
        """
        Initialization checks for the kernel.
        """
        assert self.lower_edge < self.upper_edge, "Lower edge must be smaller than upper edge"
        assert self.bin_height > 0, "Bin height must be positive"

    # --- core ---
    @abc.abstractmethod
    def left_transition_fn(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def right_transition_fn(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def _rectangular_mask(self, x: torch.Tensor) -> torch.Tensor:
        # defines rectangle area, is different for Under and Overflow Bins
        left, right = self.transition_points
        return (x >= left) & (x <= right)

    def _base_kernel(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.transition_points
        rectangle = self._rectangular_mask(x)

        y = torch.where(rectangle, self.bin_height, 0.0)

        if self.has_left_transition:
            y = torch.where(x < left, self.left_transition_fn(x), y)

        if self.has_right_transition:
            y = torch.where(x > right, self.right_transition_fn(x), y)
        return y

    def _apply_cut_mask(self, x, y):
        outside = torch.zeros_like(x, dtype=torch.bool)
        if self.left_cut is not None:
            outside |= (x < self.left_cut)
        if self.right_cut is not None:
            outside |= (x > self.right_cut)
        return y.masked_fill(outside, 0.0)

    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overwrite this when you want to extend the behavior of kernel
        """
        y = self._base_kernel(x)
        return self._apply_cut_mask(y)

    def forward(self, x, *args, **kwds) -> torch.Tensor:
        return self.kernel(x) * self.normalization


class UnderflowKernel(BaseKernel):
    has_left_transition = False
    bin_identity = "underflow"

    def _rectangular_mask(self, x: torch.Tensor) -> torch.Tensor:
        _, right = self.transition_points
        return (x <= right)

    def _apply_cut_mask(self, x, y):
        outside = torch.zeros_like(x, dtype=torch.bool)
        if self.right_cut is not None:
            outside |= (x > self.right_cut)
        return y.masked_fill(outside, 0.0)


class OverflowKernel(BaseKernel):
    has_right_transition = False
    bin_identity = "overflow"

    def _apply_cut_mask(self, x, y):
        outside = torch.zeros_like(x, dtype=torch.bool)
        if self.left_cut is not None:
            outside |= (x < self.left_cut)
        return y.masked_fill(outside, 0.0)

    def _rectangular_mask(self, x: torch.Tensor) -> torch.Tensor:
        left, _ = self.transition_points
        return (x >= left)
