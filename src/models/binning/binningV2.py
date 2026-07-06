import torch


class BinningLayerV2(torch.nn.Module):
    def __init__(
        self,
        num_bins: int,
        lower_bound: float,
        upper_bound: float,
        space_fn: callable, # like linspace or logspace to create initial edges
        *args,
        **kwargs
        ):
        """
        Args:
            init_edges (list[float]): _description_
        """
        super().__init__(*args, **kwargs)
        # TODO currently no fusion allowed
        # TODO maybe good idea, when going below certain threshold
        self.num_bins = num_bins
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.bin_lengths = torch.nn.Parameter(self.init_learnable_edges(space_fn=space_fn), requires_grad=True)


    def init_learnable_edges(self, space_fn):
        # create initial edges with space fn, then create learnable edges by taking the difference of the initial edges
        space_intervalls = space_fn(self.lower_bound, self.upper_bound, self.num_bins + 1)
        differences = space_intervalls[1:] - space_intervalls[:-1]
        return differences

    def reconstruct_bins(self):
        # reconstruct bins from bin_length
        edges = torch.cumsum(self.bin_length, dim=0)
        bin_center = edges - edges / 2
        return edges

    def kernel(self, x: torch.tensor, bin_type="normal") -> torch.tensor:
        """
        Actual kernel implementation, containing 3 parts: left gaussian, horizontal 1 and right gaussian.
        A value *x* is mapped to an y value using these function.

        Args:
            x (torch.tensor): Input tensor of x values that are mapped to y.

        Returns:
            torch.tensor: y value tensor resulting form the function. Is by default between 0 and 1
        """

        # prepare edges and function
        start, end = 0, 0 # TODO
        gaussian_fn = lambda x, shift, std: torch.exp(-((x - shift) / (2* std))**2)
        # set values depending on the bin type
        if bin_type == "overflow":
            f = torch.where(x < start, gaussian_fn(x, shift=start), 1)
        elif bin_type == "underflow":
            f = torch.where(x > end, gaussian_fn(x, shift=end), 1)
        else:
            f = torch.where(x < start, gaussian_fn(x, shift=start), 1)
            f = torch.where(x > end, gaussian_fn(x, shift=end), f)
        return f



    def forward(self, x):
        scaled_x = []
        for kernel in self.kernels:
            scale = kernel(x)
            scaled_x.append(scale * x)
        binned_x = torch.sum(torch.stack(scaled_x, axis=0), axis=0)
        return binned_x
