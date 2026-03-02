import torch

def transformed_edges(transformation_fn, bins, num_bins=25):
        if len(bins) == 2 and isinstance(num_bins, int):
            lower, upper = bins
            bins = torch.linspace(lower, upper, num_bins)
        else:
            bins = torch.tensor(bins)
        return transformation_fn(bins)

def wrap_tensors(*x: torch.tensor) -> list[torch.tensor]:
    """
    Helper function to wrap everything in *x* into a torch tensor and return as list.

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

class GaussianKernel():
    def __init__(
        self,
        edge: tuple[torch.tensor],
        smooth_width: torch.tensor = torch.tensor(0.1),
        value_at_bin_edge: torch.tensor = torch.tensor(0.5),
        abs_mode = True,
        bin_type = "normal",
        ):
        """
        Kernel object that models a bin with smoothed edges.
        The lower and upper edge is contained in *edge*. The *smoothing_width* defines the numerical zone where the gaussian function goes from 100% to almost 0%.
        The smoothing width reaches to 50% in the lower and upper bin edges, the other half extends the bins in the outer area.
        The *value_at_bin_edge* is the value of the gaussian at given low and upper, and indirectly defines the steepness of the gaussian.
        *abs_mode* changes the interpretation of *smooth_width*. If set to True *smooth_width* is interpreted as absolute value.
        If false, it is interpreted as percentage of edge length.
        The *bin_type* mark if the bin is a normal, underflow or overflow bin. Depending on the smoothing functions used to the left or right are deactivated or activated.

        Args:
            edge (torch.tensor): tuple of bin edges
            smooth_width (torch.tensor, optional): Width of the gaussian zone. Defaults to 0.1.
            value_at_bin_edge (torch.tensor, optional): Value of the gaussian at low and upper bin edge. Defaults to 0.5
            abs_mode (bool, optional): True if smooth_width is interpreted as absolute value or relative of edge length.
            bin_type (str, optional): Determined if bin is "normal", "underflow" or "overflow" bin.
        """
        self.low, self.upper = edge
        if abs_mode:
            self.smooth_width = smooth_width
        else:
            self.smooth_width = (self.upper - self.low) * smooth_width
        self.smooth_std = self.std_for_given_smooth_width(smooth_width)
        self.value_at_bin_edge = value_at_bin_edge
        self.bin_type = bin_type

        self.checks()

    @property
    def _init_variables(self):
        return {name: getattr(self,name) for name in ("low", "upper", "smooth_width", "value_at_bin_edge")}

    def checks(self):
        # bin typ is within range
        assert self.bin_type in ("normal", "underflow", "overflow"), f"Bin type is {self.bin_type}, which is not supported"

        # everything should be tensors
        is_tensor = {name: torch.is_tensor(attribute) for name, attribute in (self._init_variables).items()}
        assert all(is_tensor.values()), f"All arguments should be tensors but have\n {is_tensor.items()}"

        # low and upper should be in the correct order
        assert self.low < self.upper, f"Lower edge is above upper edge - low: {self.low}, upper: {self.upper}"

        # gaussian should not extend middle part
        bin_width = self.upper - self.low
        smoothing_inside_bin = self.smooth_width / 2
        assert smoothing_inside_bin < (bin_width / 2), f"Smoothing width is above bin width, max smoothing width is: {bin_width / 2}"

    @property
    def gaussian_left(self):
        start = self.low - self.smooth_width / 2
        end = self.low + self.smooth_width / 2
        return start, end

    @property
    def gaussian_right(self):
        start = self.upper - self.smooth_width / 2
        end = self.upper + self.smooth_width / 2
        return start, end

    @property
    def middle(self):
        return self.low + self.smooth_width / 2, self.upper - self.smooth_width / 2

    def shifted_gaussian(self, x: torch.tensor, shift: torch.tensor) -> torch.tensor:
        """
        Gaussian Kernel implementation, where *x* is the input, and *shift* is there to simulate the left or right side of the gaussian.

        Args:
            x (torch.tensor): Input tensor of the gaussian.
            shift (torch.tensor): Shift along x-axis.

        Returns:
            torch.tensor: y-value of the gaussian.
        """
        x, shift, smooth_std = wrap_tensors(x, shift, self.smooth_std)
        return torch.exp(-((x - shift) / (2* smooth_std))**2)

    def arg_for_y(self,y, shift, std):
        # returns x for given y
        y, shift, std = wrap_tensors(y, shift, std)
        return shift + (2 * std) * torch.sqrt(-torch.log(y))

    def std_for_given_smooth_width(self, smooth_width, value_after_passing_smooth_width = 0.5) -> torch.tensor:
        """
        Calculates the standard deviation so that the *value_at_width* is reached after a given *smooth_width* going from the peak of the gaussian.
        This means that a steeper gaussian is used when a smaller smooth width or smaller values reached are used.

        Args:
            smooth_width (float): Width of the smoothing zone.
            value_after_passing_smooth_width (float, optional): The value of the scaling factor after passing the smooth width

        Returns:
            torch.tensor: Std to model the gaussian
        """
        smooth_width, value_after_passing_smooth_width = wrap_tensors(smooth_width, value_after_passing_smooth_width)
        return (smooth_width / (2 * torch.sqrt(-torch.log(value_after_passing_smooth_width))))

    # def overflow_kernel(self, x):

    # def underflow_kernel(self, x):

    # def normal_kernel(self, x):

    def kernel(self, x: torch.tensor) -> torch.tensor:
        """
        Actual kernel implementation, containing 3 parts: left gaussian, horizontal 1 and right gaussian.
        A value *x* is mapped to an y value using these function.

        Args:
            x (torch.tensor): Input tensor of x values that are mapped to y.

        Returns:
            torch.tensor: y value tensor resulting form the function. Is by default between 0 and 1
        """
        # prepare edges and function
        start, end = self.middle
        sm_fn = self.shifted_gaussian
        # set values depending on the bin type
        if self.bin_type == "overflow":
            f = torch.where(x < start, sm_fn(x, shift=start), 1)
        elif self.bin_type == "underflow":
            f = torch.where(x > end, sm_fn(x, shift=start), 1)
        else:
            f = torch.where(x < start, sm_fn(x, shift=start), 1)
            f = torch.where(x > end, sm_fn(x, shift=end), f)
        return f

    def control_plot(self, x, show=True):
        # helper plot to visualize the kernel
        import matplotlib.pyplot as plt
        y = self(x).cpu()
        plt.plot(x.cpu(), y)
        start, end = self.middle
        plt.vlines([start, end], ymin=0, ymax=1, color="red")
        if show:
            plt.show()

    def __call__(self, *args, **kwds):
        return self.kernel(*args, **kwds)


if __name__ == "__main__":
    bin_edges = torch.linspace(0,1,10)
    x = torch.linspace(0,1,100)
    kernel_cfg = {"smooth_width": torch.tensor(0.1), "value_at_bin_edge" : torch.tensor(0.5)}

    a = create_kernels(bin_edges, kernel_cls=GaussianKernel, kernel_cfg=kernel_cfg)
    from IPython import embed; embed(header = " line: 130 in kernel.py")
