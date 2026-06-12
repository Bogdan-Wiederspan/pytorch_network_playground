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

class GaussianKernel(BaseKernel):
    def __init__(
        self,
        edge: tuple[torch.tensor],
        left_notch: float  | None = None,
        right_notch: float | None = None,
        smoothing_width: float = 0.1,
        bin_type = "normal",
        bin_height: float = 1,
        ):
        """
        Kernel object that models a bin with smoothed edges.
        The lower and upper edge is contained in *edge*.
        The *smoothing_width* is the width from where the gaussian function goes from 100% to 10% and is given by percentage value of original bin width.
        The *left_notch* is how much relative notch into the linear part the gaussian have.
        Thus 0.1 would mean 10% of the edge width is used as smoothing width
        The *bin_type* mark if the bin is a normal, underflow or overflow bin. Depending on the smoothing functions used to the left or right are deactivated or activated.
        If another maximum value is requires, set *bin_height*, which is by default set to 1.

        Args:
            edge (torch.tensor): tuple of bin edges
            smoothing_width (float, optional): Width of the gaussian zone after which a certain value is reached. Defaults to 0.1.
            abs_mode (bool, optional): True if smooth_width is interpreted as absolute value or relative of edge length.
            bin_type (str, optional): Determined if bin is "normal", "underflow" or "overflow" bin.
            bin_height (float, optional): Set Maximum Value of bin approximation. Defaults to 1.
        """

        self.low, self.upper = edge

        self._left_notch, self._right_notch, self._smoothing_width = self.wrap_tensors(left_notch, right_notch, smoothing_width)

        # calculate std for gaussian function based on gi
        self.std = self.sigma_for_full_width_at_tenth_of_maximum(self.smoothing_width)

        self.bin_type = bin_type
        self.bin_height = bin_height # TODO USE THIS

        self.checks()

    @property
    def left_notch(self):
        return self._left_notch * self.bin_width

    @property
    def right_notch(self):
        return self._right_notch * self.bin_width

    @property
    def smoothing_width(self):
        return self._smoothing_width * self.bin_width

    @property
    def bin_width(self):
        return self.upper - self.low

    def checks(self):
        # helper function to gather all assert checks
        init_variables = {
            name: getattr(self,name)
            for name in ("low", "upper", "smoothing_width", "left_notch", "right_notch")
            }

        assert self.bin_type in ("normal", "underflow", "overflow"), f"Bin type is {self.bin_type}, which is not supported"

        # everything should be tensors
        is_tensor = {name: torch.is_tensor(attribute) for name, attribute in (init_variables).items()}
        assert all(is_tensor.values()), f"All arguments should be tensors but have\n {is_tensor.items()}"

        # low and upper should be in the correct order
        assert self.low < self.upper, f"Lower edge is above upper edge - low: {self.low}, upper: {self.upper}"

        #TODO NOTCHING MECHANISM
        # # gaussian should not extend middle part
        # bin_width = self.upper - self.low
        # smoothing_inside_bin = self.smooth_width
        # assert smoothing_inside_bin < (bin_width / 2), f"Smoothing width is above bin width, max smoothing width is: {bin_width / 2}"

    @property
    def gaussian_left_coordinates(self):
        # start and end is shifted by x of FHWM length
        if self.left_notch is None:
            shift = self.smoothing_width / 2
        else:
            shift = self.left_notch

        start = self.low - shift
        end = self.low + shift
        return start, end

    @property
    def gaussian_right_coordinates(self):
        if self.right_notch is None:
            shift = self.smoothing_width / 2
        else:
            shift = self.right_notch

        start = self.upper - shift
        end = self.upper + shift
        return start, end

    @property
    def middle_coordinates(self):
        shift = self.smooth_width / 2
        start = self.low + shift
        end = self.upper - shift
        return start, end




    def shifted_gaussian(self, x: torch.tensor, shift: torch.tensor) -> torch.tensor:
        """
        Gaussian Kernel implementation, where *x* is the input, and *shift* is there to simulate the left or right side of the gaussian.

        Args:
            x (torch.tensor): Input tensor of the gaussian.
            shift (torch.tensor): Shift along x-axis.

        Returns:
            torch.tensor: y-value of the gaussian.
        """
        x, shift, smooth_std = self.wrap_tensors(x, shift, self.std)
        return torch.exp(-((x - shift) / (2* smooth_std))**2)

    # def arg_for_y(self,y, shift, std):
    #     # returns x for given y
    #     y, shift, std = wrap_tensors(y, shift, std)
    #     return shift + (2 * std) * torch.sqrt(-torch.log(y))

    # def std_for_given_smooth_width(self, smooth_width, value_after_passing_smooth_width = 0.5) -> torch.tensor:
    #     """
    #     Calculates the standard deviation so that the *value_at_width* is reached after a given *smooth_width* going from the peak of the gaussian.
    #     This means that a steeper gaussian is used when a smaller smooth width or smaller values reached are used.

    #     Args:
    #         smooth_width (float): Width of the smoothing zone.
    #         value_after_passing_smooth_width (float, optional): The value of the scaling factor after passing the smooth width

    #     Returns:
    #         torch.tensor: Std to model the gaussian
    #     """
    #     smooth_width, value_after_passing_smooth_width = wrap_tensors(smooth_width, value_after_passing_smooth_width)
    #     return (smooth_width / (2 * torch.sqrt(-torch.log(value_after_passing_smooth_width))))

    @property
    def FW50M(self):
        return torch.tensor(2.35482) * self.std

    @property
    def FW10M(self):
        return torch.tensor(4.29193) * self.std

    def sigma_for_full_width_at_half_maximum(self, half_width):
        """
        Calculates Sigma for a Gaussian with full width at 50% of *length*.

        Args:
            half_width (torch.tensor): Length of the gaussian from 100% to 10%

        Returns:
            torch.tensor: Sigma that correspond to given half_width
        """
        # length for 0.5 FULL = constant * std
        # 2 * torch.sqrt(2 * torch.log(2)) exact calculation
        constant = torch.tensor(2.35482)
        return half_width / constant

    def sigma_for_full_width_at_tenth_of_maximum(self, half_width):
        """
        Calculates Sigma for a Gaussian with full width at 10% of *length*.

        Args:
            half_width (torch.tensor): Length of the gaussian from 100% to 10%

        Returns:
            torch.tensor: Sigma that correspond to given half_width
        """
        # l = 4.29 * sigma
        constant = torch.tensor(4.29193) # 2 * torch.sqrt(2 * torch.log(10)) exact calculation
        return half_width / constant

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
        # TODO add scaling factor as bin_height
        # prepare edges and function
        start, end = self.middle_coordinates
        sm_fn = self.shifted_gaussian
        # set values depending on the bin type
        if self.bin_type == "overflow":
            f = torch.where(x < start, sm_fn(x, shift=start), 1)
        elif self.bin_type == "underflow":
            f = torch.where(x > end, sm_fn(x, shift=end), 1)
        else:
            f = torch.where(x < start, sm_fn(x, shift=start), 1)
            f = torch.where(x > end, sm_fn(x, shift=end), f)
        return f

    @property
    def normalization(self):
        # depending on type of bin different integrals are necessary
        # linear part is just width * height
        start, end = self.middle_coordinates
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

    def control_plot(self, x, show=True, with_h_lines=True):
        # helper plot to visualize the kernel
        import matplotlib.pyplot as plt
        y = self(x).cpu()
        plt.plot(x.cpu(), y)
        plt.xticks(torch.linspace(0,1,21).numpy(), rotation=45)
        if with_h_lines:
            # start is shifted to right, end to left
            start, end = self.middle_coordinates
            # shorted middle bin
            plt.vlines([start, end], ymin=0, ymax=1, color="red", linestyles=":")

            # gaussian v line reaching 10%
            plt.vlines([self.gaussian_left_coordinates[0], self.gaussian_right_coordinates[1]], ymin=0, ymax=1, color="green")
            # ORIGINAL BIN
            plt.vlines([self.low, self.upper], ymin=0, ymax=1, color="black", linestyles="-")
            # horizontal marking 10%
            plt.hlines(0.1, 0, 1, color = "black", linestyles=":")
        if show:
            plt.show()


    def __call__(self, *args, **kwds):
        return self.kernel(*args, **kwds) / self.normalization



class GaussianKernelV2(BaseKernel):
    def __init__(
        self,
        initial_edge: tuple[torch.tensor],
        left_notch: float   = 0,
        right_notch: float  = 0,
        smoothing_width: float = 0.1,
        abs_mode: bool = False,
        bin_type = "normal",
        bin_height: float = 1,
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

        self.initial_lower_edge, self.initial_upper_edge = initial_edge
        self.abs_mode = abs_mode

        self._left_notch, self._right_notch, self._smoothing_width = self.wrap_tensors(left_notch, right_notch, smoothing_width)

        # calculate std for gaussian function based on gi
        self.std = self.sigma_for_given_width_at_percentage(self.smoothing_width, 0.1)

        self.bin_type = bin_type
        self.bin_height = bin_height # TODO USE THIS
        self.checks()

    @property
    def left_notch(self):
        return self._left_notch * self.bin_width

    @property
    def right_notch(self):
        return self._right_notch * self.bin_width

    @property
    def smoothing_width(self):
        if self.abs_mode:
            return self._smoothing_width
        return self._smoothing_width * self.bin_width

    @property
    def bin_width(self):
        return self.initial_upper_edge - self.initial_lower_edge

    def checks(self):
        # helper function to gather all assert checks
        init_variables = {
            name: getattr(self,name)
            for name in ("initial_lower_edge", "initial_upper_edge", "smoothing_width", "left_notch", "right_notch")
            }

        assert self.bin_type in ("normal", "underflow", "overflow"), f"Bin type is {self.bin_type}, which is not supported"

        # everything should be tensors
        is_tensor = {name: torch.is_tensor(attribute) for name, attribute in (init_variables).items()}
        assert all(is_tensor.values()), f"All arguments should be tensors but have\n {is_tensor.items()}"

        # low and upper should be in the correct order
        assert self.initial_lower_edge < self.initial_upper_edge, f"Lower edge is above upper edge - low: {self.initial_lower_edge}, upper: {self.initial_upper_edge}"

        # gaussian should not over whole bin
        assert self._left_notch + self._right_notch <= 1, f"Notches should not exceed whole bin, but left notch is {self._left_notch}, right notch is {self._right_notch} and their sum is {self._left_notch + self._right_notch}"

    @property
    def coordinates(self):
        if self.bin_type == "overflow":
            transition_point_left = self.initial_lower_edge + self.left_notch
            transition_point_right = self.initial_upper_edge
        elif self.bin_type == "underflow":
            transition_point_left = self.initial_lower_edge
            transition_point_right = self.initial_upper_edge - self.right_notch
        else:
            transition_point_left = self.initial_lower_edge + self.left_notch
            transition_point_right = self.initial_upper_edge - self.right_notch
        return transition_point_left, transition_point_right

    def shifted_gaussian(self, x: torch.tensor, shift: torch.tensor) -> torch.tensor:
        """
        Gaussian Kernel implementation, where *x* is the input, and *shift* is there to simulate the left or right side of the gaussian.

        Args:
            x (torch.tensor): Input tensor of the gaussian.
            shift (torch.tensor): Shift along x-axis.

        Returns:
            torch.tensor: y-value of the gaussian.
        """
        x, shift, smooth_std = self.wrap_tensors(x, shift, self.std)
        return torch.exp(-(1/2) * ((x - shift) / (smooth_std))**2)

    @property
    def FW50M(self):
        # calculate the full width at half maximum for the gaussian from 50% to 50%
        return torch.tensor(2.35482) * self.std

    @property
    def FW10M(self):
        # calculate the full width at tenth of maximum for the gaussian from 10% to 10%
        return torch.tensor(4.29193) * self.std

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

    def kernel(self, x: torch.tensor) -> torch.tensor:
        """
        Actual kernel implementation, containing 3 parts: left gaussian, horizontal 1 and right gaussian.
        A value *x* is mapped to an y value using these function.

        Args:
            x (torch.tensor): Input tensor of x values that are mapped to y.

        Returns:
            torch.tensor: y value tensor resulting form the function. Is by default between 0 and 1
        """
        # TODO add scaling factor as bin_height
        # prepare edges and function
        start, end = self.coordinates
        sm_fn = self.shifted_gaussian
        # set values depending on the bin type
        if self.bin_type == "overflow":
            f = torch.where(x < start, sm_fn(x, shift=start), 1)
        elif self.bin_type == "underflow":
            f = torch.where(x > end, sm_fn(x, shift=end), 1)
        else:
            f = torch.where(x < start, sm_fn(x, shift=start), 1)
            f = torch.where(x > end, sm_fn(x, shift=end), f)
        return f

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

    def control_plot(self, x, show=True, with_h_lines=True):
        # helper plot to visualize the kernel
        import matplotlib.pyplot as plt
        y = self(x).cpu()
        plt.plot(x.cpu(), y)
        plt.xticks(torch.linspace(0,1,21).numpy(), rotation=45)
        if with_h_lines:
            # gaussian v line reaching 10%
            low, up = self.coordinates
            plt.vlines([low - self.smoothing_width, up + self.smoothing_width], ymin=0, ymax=1, color="green")

            # ORIGINAL BIN
            plt.vlines([self.initial_lower_edge, self.initial_upper_edge], ymin=0, ymax=1, color="black", linestyles="-")

            # horizontal marking 10%
            plt.hlines(0.1, 0, 1, color = "black", linestyles=":")
        if show:
            plt.show()


    def __call__(self, *args, **kwds):
        return self.kernel(*args, **kwds) #/ self.normalization

class GaussianKernelV3(BaseKernel):
    def __init__(
        self,
        edges,
        left_notch: float   = 0,
        right_notch: float  = 0,
        smoothing_width: float = 0.1,
        abs_mode: bool = False,
        bin_type = "normal",
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
        self.initial_lower_edge, self.initial_upper_edge = edges
        self.abs_mode = abs_mode

        self._left_notch, self._right_notch, self._smoothing_width  = self.wrap_tensors(left_notch, right_notch, smoothing_width)

        # calculate std for gaussian function based on gi
        self.std = self.sigma_for_given_width_at_percentage(self.smoothing_width, 0.1)

        self.bin_type = bin_type
        self.bin_height = bin_height # TODO USE THIS
        self.checks()

    @property
    def bin_width(self):
        return self.initial_upper_edge - self.initial_lower_edge

    @property
    def left_notch(self):
        return self._left_notch * self.bin_width

    @property
    def right_notch(self):
        return self._right_notch * self.bin_width

    @property
    def smoothing_width(self):
        if self.abs_mode:
            return self._smoothing_width
        return self._smoothing_width * self.bin_width

    def checks(self):
        # helper function to gather all assert checks
        init_variables = {
            name: getattr(self,name)
            for name in ("initial_lower_edge", "initial_upper_edge", "smoothing_width", "left_notch", "right_notch")
            }

        assert self.bin_type in ("normal", "underflow", "overflow"), f"Bin type is {self.bin_type}, which is not supported"

        # everything should be tensors
        is_tensor = {name: torch.is_tensor(attribute) for name, attribute in (init_variables).items()}
        assert all(is_tensor.values()), f"All arguments should be tensors but have\n {is_tensor.items()}"

        # low and upper should be in the correct order
        assert self.initial_lower_edge < self.initial_upper_edge, f"Lower edge is above upper edge - low: {self.initial_lower_edge}, upper: {self.initial_upper_edge}"

        # gaussian should not over whole bin
        assert self._left_notch + self._right_notch <= 1, f"Notches should not exceed whole bin, but left notch is {self._left_notch}, right notch is {self._right_notch} and their sum is {self._left_notch + self._right_notch}"

    @property
    def coordinates(self):
        if self.bin_type == "overflow":
            transition_point_left = self.initial_lower_edge + self.left_notch
            transition_point_right = self.initial_upper_edge
        elif self.bin_type == "underflow":
            transition_point_left = self.initial_lower_edge
            transition_point_right = self.initial_upper_edge - self.right_notch
        else:
            transition_point_left = self.initial_lower_edge + self.left_notch
            transition_point_right = self.initial_upper_edge - self.right_notch
        return transition_point_left, transition_point_right

    def shifted_gaussian(self, x: torch.tensor, shift: torch.tensor) -> torch.tensor:
        """
        Gaussian Kernel implementation, where *x* is the input, and *shift* is there to simulate the left or right side of the gaussian.

        Args:
            x (torch.tensor): Input tensor of the gaussian.
            shift (torch.tensor): Shift along x-axis.

        Returns:
            torch.tensor: y-value of the gaussian.
        """
        x, shift, smooth_std = self.wrap_tensors(x, shift, self.std)
        return torch.exp(-(1/2) * ((x - shift) / (smooth_std))**2)

    @property
    def FW50M(self):
        # calculate the full width at half maximum for the gaussian from 50% to 50%
        return torch.tensor(2.35482) * self.std

    @property
    def FW10M(self):
        # calculate the full width at tenth of maximum for the gaussian from 10% to 10%
        return torch.tensor(4.29193) * self.std

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

    def kernel(self, x: torch.tensor) -> torch.tensor:
        """
        Actual kernel implementation, containing 3 parts: left gaussian, horizontal 1 and right gaussian.
        A value *x* is mapped to an y value using these function.

        Args:
            x (torch.tensor): Input tensor of x values that are mapped to y.

        Returns:
            torch.tensor: y value tensor resulting form the function. Is by default between 0 and 1
        """
        # TODO add scaling factor as bin_height
        # prepare edges and function
        start, end = self.coordinates
        sm_fn = self.shifted_gaussian
        # set values depending on the bin type
        if self.bin_type == "overflow":
            f = torch.where(x < start, sm_fn(x, shift=start), 1)
        elif self.bin_type == "underflow":
            f = torch.where(x > end, sm_fn(x, shift=end), 1)
        else:
            f = torch.where(x < start, sm_fn(x, shift=start), 1)
            f = torch.where(x > end, sm_fn(x, shift=end), f)
        return f

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

    def control_plot(self, x, show=True, with_h_lines=True):
        # helper plot to visualize the kernel
        import matplotlib.pyplot as plt
        y = self(x).cpu()
        plt.plot(x.cpu(), y)
        plt.xticks(torch.linspace(0,1,21).numpy(), rotation=45)
        if with_h_lines:
            # gaussian v line reaching 10%
            low, up = self.coordinates
            plt.vlines([low - self.smoothing_width, up + self.smoothing_width], ymin=0, ymax=1, color="green")

            # ORIGINAL BIN
            plt.vlines([self.initial_lower_edge, self.initial_upper_edge], ymin=0, ymax=1, color="black", linestyles="-")

            # horizontal marking 10%
            plt.hlines(0.1, 0, 1, color = "black", linestyles=":")
        if show:
            plt.show()



class GaussianKernelFinal(BaseKernel):
    def __init__(
        self,
        edges,
        left_notch: float   = 0,
        right_notch: float  = 0,
        smoothing_width: float = 0.1,
        abs_mode: bool = False,
        bin_type = "normal",
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
        self.initial_lower_edge, self.initial_upper_edge = edges
        self.abs_mode = abs_mode

        self._left_notch, self._right_notch, self._smoothing_width  = self.wrap_tensors(left_notch, right_notch, smoothing_width)

        # calculate std for gaussian function based on gi
        self.std = self.sigma_for_given_width_at_percentage(self.smoothing_width, 0.1)

        self.bin_type = bin_type
        self.bin_height = bin_height # TODO USE THIS
        self.checks()

    @property
    def bin_width(self):
        return self.initial_upper_edge - self.initial_lower_edge

    @property
    def left_notch(self):
        return self._left_notch * self.bin_width

    @property
    def right_notch(self):
        return self._right_notch * self.bin_width

    @property
    def smoothing_width(self):
        if self.abs_mode:
            return self._smoothing_width
        return self._smoothing_width * self.bin_width

    def checks(self):

        # helper function to gather all assert checks
        init_variables = {
            name: getattr(self,name)
            for name in ("initial_lower_edge", "initial_upper_edge", "smoothing_width", "left_notch", "right_notch")
            }

        if self.bin_type not in ("normal", "underflow", "overflow"):
            raise ValueError(f"Unsupported bin type: {self.bin_type}")

        # everything should be tensors
        is_tensor = {name: torch.is_tensor(attribute) for name, attribute in (init_variables).items()}

        if not all(is_tensor.values()):
            raise TypeError(f"All arguments should be tensors but got:\n{is_tensor}")

        # TODO: These checks break torchscript
        # # low and upper should be in the correct order
        # torch._assert(
        #     self.initial_lower_edge < self.initial_upper_edge,
        #     "Lower edge above upper edge"
        #     )

        # # gaussian should not over whole bin
        # torch._assert(
        #     self._left_notch + self._right_notch <= 1,
        #     "Notches exceed whole bin"
        # )

    @property
    def coordinates(self):
        if self.bin_type == "overflow":
            transition_point_left = self.initial_lower_edge + self.left_notch
            transition_point_right = self.initial_upper_edge
        elif self.bin_type == "underflow":
            transition_point_left = self.initial_lower_edge
            transition_point_right = self.initial_upper_edge - self.right_notch
        else:
            transition_point_left = self.initial_lower_edge + self.left_notch
            transition_point_right = self.initial_upper_edge - self.right_notch
        return transition_point_left, transition_point_right

    def shifted_gaussian(self, x: torch.tensor, shift: torch.tensor) -> torch.tensor:
        """
        Gaussian Kernel implementation, where *x* is the input, and *shift* is there to simulate the left or right side of the gaussian.

        Args:
            x (torch.tensor): Input tensor of the gaussian.
            shift (torch.tensor): Shift along x-axis.

        Returns:
            torch.tensor: y-value of the gaussian.
        """
        x, shift, smooth_std = self.wrap_tensors(x, shift, self.std)
        return torch.exp(-(1/2) * ((x - shift) / (smooth_std))**2)

    @property
    def FW50M(self):
        # calculate the full width at half maximum for the gaussian from 50% to 50%
        return torch.tensor(2.35482) * self.std

    @property
    def FW10M(self):
        # calculate the full width at tenth of maximum for the gaussian from 10% to 10%
        return torch.tensor(4.29193) * self.std

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

    def kernel(self, x: torch.tensor) -> torch.tensor:
        """
        Actual kernel implementation, containing 3 parts: left gaussian, horizontal 1 and right gaussian.
        A value *x* is mapped to an y value using these function.

        Args:
            x (torch.tensor): Input tensor of x values that are mapped to y.

        Returns:
            torch.tensor: y value tensor resulting form the function. Is by default between 0 and 1
        """
        # TODO add scaling factor as bin_height
        # prepare edges and function
        start, end = self.coordinates
        sm_fn = self.shifted_gaussian
        # set values depending on the bin type
        if self.bin_type == "overflow":
            f = torch.where(x < start, sm_fn(x, shift=start), 1)
        elif self.bin_type == "underflow":
            f = torch.where(x > end, sm_fn(x, shift=end), 1)
        else:
            f = torch.where(x < start, sm_fn(x, shift=start), 1)
            f = torch.where(x > end, sm_fn(x, shift=end), f)
        return f

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

    def control_plot(self, x, show=True, with_h_lines=True):
        # helper plot to visualize the kernel
        import matplotlib.pyplot as plt
        y = self(x).cpu()
        plt.plot(x.cpu(), y)
        plt.xticks(torch.linspace(0,1,21).numpy(), rotation=45)
        if with_h_lines:
            # gaussian v line reaching 10%
            low, up = self.coordinates
            plt.vlines([low - self.smoothing_width, up + self.smoothing_width], ymin=0, ymax=1, color="green")

            # ORIGINAL BIN
            plt.vlines([self.initial_lower_edge, self.initial_upper_edge], ymin=0, ymax=1, color="black", linestyles="-")

            # horizontal marking 10%
            plt.hlines(0.1, 0, 1, color = "black", linestyles=":")
        if show:
            plt.show()



    def __call__(self, *args, **kwds):
        return self.kernel(*args, **kwds) #/ self.normalization


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    bin_edges = torch.linspace(0,1,10)
    x = torch.linspace(0,1,100)
    kernel_cfg = {"smooth_width": torch.tensor(0.1), }
    #GaussianKernel((torch.tensor(0.2), torch.tensor(0.4)), torch.tensor(0.0999), torch.tensor(0.5), abs_mode=True).control_plot(x)
    #plt.savefig("test.png")

    normal = GaussianKernelV2(
        initial_edge = (torch.tensor(0.4), torch.tensor(0.5)),
        left_notch =  0.1,
        right_notch = 0.1,
        smoothing_width = 0.2,
        abs_mode = False,
        bin_type = "normal",
        bin_height = 1,
    )

    underflow = GaussianKernelV2(
        initial_edge = (torch.tensor(0.0), torch.tensor(0.3)),
        left_notch =  0.0,
        right_notch = 0.1,
        smoothing_width = 0.2,
        abs_mode = False,
        bin_type = "underflow",
        bin_height = 1,
    )


    overflow = GaussianKernelV2(
        initial_edge = (torch.tensor(0.7), torch.tensor(1)),
        left_notch =  0.0,
        right_notch = 0.1,
        smoothing_width = 0.2,
        abs_mode = False,
        bin_type = "overflow",
        bin_height = 1,
    )


    normal.control_plot(torch.linspace(0,1,400))
    overflow.control_plot(torch.linspace(0,1,400))
    underflow.control_plot(torch.linspace(0,1,400))
    plt.savefig("test.png")

    # from IPython import embed; embed(header = " line: 130 in kernel.py")
    # a = create_kernels(bin_edges, kernel_cls=GaussianKernel, kernel_cfg=kernel_cfg)
