from __future__ import annotations

import torch

from models.binning import BaseKernel, UnderflowKernel, OverflowKernel


class GaussianKernel(BaseKernel):
    def __init__(
        self,
        edges,
        left_notch: float   = 0,
        right_notch: float  = 0,
        smoothing_width: float = 0.1,
        abs_mode: bool = False,
        bin_height: float = 1,
        *args,
        **kwargs,
        ):
        """
        Kernel object that models a bin with smoothed edges.
        The lower and upper edge is contained in *edge*.
        The *left_notch* and *right_notch* describe how much relative notch into the linear part the gaussian have.
        The smoothing width is the width from where the gaussian function PEAK goes from 100% to 10%. If *abs_mode* is False, the width is a percentage value of the original bin width,
        It is not the same as FW10M, which is the width from 10% to 10%.
        If *abs_mode* is True, the smoothing width is interpreted as absolute value.
        The *bin_type* mark if the bin is a normal, underflow or overflow bin. Depending on the smoothing functions used to the left or right are deactivated or activated.
        If another maximum value is requires, set *bin_height*, which is by default set to 1.

        Args:
            edge (torch.tensor): tuple of bin edges
            smoothing_width (float, optional): Width of the gaussian zone after which a certain value is reached. Defaults to 0.1.
            abs_mode (bool, optional): True if smooth_width is interpreted as absolute value or relative of edge length.
            bin_type (str, optional): Determined if bin is "normal", "underflow" or "overflow" bin.
            bin_height (float, optional): Set Maximum Value of bin approximation. Defaults to 1.
        """
        super().__init__(self, edges, left_notch, right_notch)
        self.initial_lower_edge, self.initial_upper_edge = edges
        self.absolute_width = abs_mode

        self._left_notch, self._right_notch, self._smoothing_width  = self.wrap_tensors(left_notch, right_notch, smoothing_width)

        # calculate std for gaussian function based on gi
        self.std = self.sigma_for_given_width_at_percentage(self.smoothing_width, 0.1)

        self.bin_height = bin_height # TODO USE THIS
        self.checks()

    # --- Geometry ---
    @property
    def FW50M(self):
        # calculate the full width at half maximum for the gaussian from 50% to 50%
        return torch.tensor(2.35482) * self.std

    @property
    def FW10M(self):
        # calculate the full width at tenth of maximum for the gaussian from 10% to 10%
        return torch.tensor(4.29193) * self.std

    @property
    def smoothing_width(self):
        if self.absolute_mode:
            return self._smoothing_width
        return self._smoothing_width * self.bin_width

    def sigma_for_given_width_at_percentage(self, half_width, percentage):
        """
        Calculates Sigma for a Gaussian with full width at given percentage of *length*.

        Args:
            half_width (torch.tensor): Length of the gaussian from 100% to given percentage
            percentage (float): Percentage value between 0 and 1

        Returns:
            torch.tensor: Sigma that correspond to given half_width and percentage
        """
        constants = {
            0.5 : torch.tensor(2.35482), # 2 * torch.sqrt(2 * torch.log(2)) exact calculation
            0.1 : torch.tensor(4.29193), # 2 * torch.sqrt(2 * torch.log(10)) exact calculation
        }
        return (2 * half_width) / constants[percentage]

    # --- core ---

    @property
    def normalization(self):
        # depending on type of bin different integrals are necessary
        # linear part is just width * height
        start, end = self.coordinates
        linear_bin_width = end - start
        linear_integral = linear_bin_width * self.bin_height

        # gaussian integral is simpler since integral to inf is happen
        # and lower or upper edge is exactly in the middle
        if self.bin_type == "normal":
            gaussian_integral = 1
        else:
            gaussian_integral = 0.5
        full_integral = linear_integral + gaussian_integral
        return full_integral

    def gaussian(self, x: torch.tensor, center: torch.tensor) -> torch.tensor:
        """
        Gaussian Kernel implementation, where *x* is the input.
        *center* is used as shift.

        Args:
            x (torch.tensor): Input tensor of the gaussian.
            center (torch.tensor): Shift along x-axis.

        Returns:
            torch.tensor: y-value of the gaussian.
        """
        x, center, smooth_std = self.wrap_tensors(x, center, self.std)
        return torch.exp(-(1/2) * ((x - center) / (smooth_std))**2)


    def left_transition_fn(self, x):
        left = self.left_transition_coordinate
        return self.gaussian(x, left)

    def right_transition_fn(self, x):
        right = self.right_transition_coordinate
        return self.gaussian(x, right)


    def control_plot(self, x, x_ticks=(0,1,21), with_h_lines=False):
        # helper plot to visualize the kernel
        import matplotlib.pyplot as plt
        fig, ax = super(self.control_plot(x, x_ticks))

        if with_h_lines:
            # gaussian v line reaching 10%
            low, up = self.coordinates
            low = low.cpu()
            up = up.cpu()
            smoothing_width = self.smoothing_width.cpu()
            init_low, init_up = self.initial_lower_edge.cpu(), self.initial_upper_edge.cpu()
            ax.vlines([low - smoothing_width, up + smoothing_width], ymin=0, ymax=1, color="green")

            # ORIGINAL BIN
            ax.vlines([init_low, init_up], ymin=0, ymax=1, color="black", linestyles="-")

            # horizontal marking 10%
            ax.hlines(0.1, 0, 1, color = "black", linestyles=":")

        return fig, ax


class GaussianUnderflowKernel(UnderflowKernel, GaussianKernel):
    pass


class GaussianOverflowKernel(OverflowKernel, GaussianKernel):
    pass
