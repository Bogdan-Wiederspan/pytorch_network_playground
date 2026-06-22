import torch
from torch.nn.utils import parametrize

from train.plotting import plot_edges_number_line


class BinningLayerRight(torch.nn.Module):
    def __init__(
        self,
        num_bins: int,
        bounds: tuple[float],

        binning_fn: callable, # like linspace or logspace to create initial edges
        kernel_cls,
        kernel_cfg,
        *args,
        **kwargs
        ):
        """
        Creates *num_bins* kernel instances of *kernel_cls* with configuration defined in *kernel_cfg*.
        The initial edge are defined by a given binning function *binning_fn*.
        The lower and upper bounds are given as tuple *bounds*.

        For every prediction, add another axis, with num_bins entries.

        Example we have an prediction vector of shape [100, 3] and 20 kernels.
        The resulting Tensors would be [20, 100, 3]



        Args:
            num_bins (int): _description_
            bounds (tuple[float]): _description_
            binning_fn (callable): Function that is applied on the input and the initial edges interval
            kernel_cfg (_type_): _description_
        """
        super().__init__(*args, **kwargs)
        # TODO currently no fusion allowed

        self.num_bins = num_bins
        self.bounds = bounds
        self.binning_fn = binning_fn
        self.init_learnable_edges()

        self.kernel_cls = kernel_cls
        self.kernel_cfg = kernel_cfg
        self.kernel_cache = None

    def visualize(self):
        plot_edges_number_line(self.bin_intervals)

    @property
    def lower_edge(self):
        return self.bounds[0]

    @property
    def upper_edge(self):
        return self.bounds[1]

    @property
    def bin_intervals(self):
        # TODO bin edges ignore transformation currently
        # calculate absolute widht of bins
        # from IPython import embed; embed(header="MESSAGE Line 1414 | File: layers.py")

        weights = list(self.buffers())[0]
        relative_part = weights.detach()
        interval = self.upper_edge - self.lower_edge
        abs_width = interval * relative_part

        # right edge is sum of abs_bins + lowest edge
        # left edge is
        shift = self.lower_edge
        right_edge = shift + torch.cumsum(abs_width, dim=0)
        left_edge = right_edge - abs_width
        return torch.stack((left_edge, right_edge), dim = 1)

    @property
    def bin_edges(self):
        intervals = self.bin_intervals
        edges = [intervals[:, 0].reshape(-1, 1), intervals[-1, 1].reshape(-1, 1)]
        return torch.flatten(torch.concatenate(edges, dim = 0))

    def apply_bin_fn(self, eps=1e-6):
        if not torch.is_tensor(eps):
            eps = torch.tensor(eps)

        if self.binning_fn is None:
            return torch.linspace(self.lower_edge, self.upper_edge, self.num_bins + 1)

        # create intervalls using linspace in transformed space
        transformed_lower_edge = self.binning_fn.forward(self.lower_edge, eps=eps)
        transformed_upper_edge = self.binning_fn.forward(self.upper_edge, eps=eps)

        interval = torch.linspace(
            transformed_lower_edge,
            transformed_upper_edge,
            self.num_bins + 1
            )

        # TODO maybe leave inverse out and perform in forward also transformated binning?
        # return to normal space
        interval = self.binning_fn.inverse(interval, eps=eps)
        interval[0] = self.lower_edge
        interval[-1] = self.upper_edge
        # but lower and upper bound are replaced with init bounds
        return interval

    def init_learnable_edges(self):
        """
        Takes *space_fn* that defines an interval space, like linspace and register this interval.
        Afterwards parametrize the interval to their relative contribution.

        Returns:
            _type_: _description_
        """
        interval = self.apply_bin_fn()
        relative_width = (interval[1:] - interval[:-1]) / (self.upper_edge - self.lower_edge)

        self.relative_bin_width = torch.nn.Buffer(relative_width)
        parametrize.register_parametrization(self, "relative_bin_width", torch.nn.Softmax(dim=0))


    def kernels(self ,load_cache=False):
        """
        Construct the gaussian-kernel consisting out of a left gaussian, horizontal middle and right gaussian.
        If kernels already exist, reuse kernel cache.
        TODO: When using a model with learnable edges reconstruct this!!!

        """
        # kernels should only be rebuild, when there is no cache or load_cache is false
        if load_cache and self.kernel_cache:
            return self.kernel_cache

        kernels = []
        for bin_num, edge in enumerate(self.bin_intervals):
            if bin_num == 0:
                bin_type = "underflow"
            elif bin_num == (self.num_bins - 1):
                bin_type = "overflow"
            else:
                bin_type = "normal"
            kernel = self.kernel_cls(
                edges=edge,
                bin_type=bin_type,
                **self.kernel_cfg,
                )
            kernels.append(kernel)
        # save in cache
        self.kernel_cache = kernels
        return kernels


    def forward(self, x):
        scaled_x = []
        kernels = self.kernels(load_cache=False)
        for kernel in kernels:
            scale = kernel(x)
            scaled_x.append(scale * x)
        return torch.stack(scaled_x, dim=0)
