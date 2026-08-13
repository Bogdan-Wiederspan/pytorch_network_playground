from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn.utils import parametrize

from bbTT.monitoring.monitoring_hooks.hookable_module import HookableMixin


class BinningLayer(HookableMixin, torch.nn.Module):
    def __init__(
        self,
        num_bins: int,
        bounds: tuple[float],

        binning_fn: callable, # like linspace or logspace to create initial edges
        binning_cfg,
        kernel_map, # dict with mapping to bins factories
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
        # TODO currently no fusion allowed, when for example bins are very small
        # --- Status Flags ---
        self.is_frozen = True

        # --- Geometry ---
        self.num_bins = num_bins
        self.original_bounds = bounds
        self.bounds = bounds # after apply trans_fn
        self.is_transformed = False

        # --- Transformations ---
        self.binning_fn = binning_fn
        self.binning_cfg = binning_cfg
        self.init_learnable_edges() # saves parameter as: relative_bin_width

        # --- Kernels ---
        self.kernel_map = kernel_map
        self.kernel_cfg = kernel_cfg

        self.init_kernels()

    # --- Status Flags ---
    def freeze_edges(self):
        self.parametrizations.relative_bin_width.original.requires_grad = False
        self.is_frozen = True

    def unfreeze_edges(self):
        # self.relative_bin_width.requires_grad = True
        self.parametrizations.relative_bin_width.original.requires_grad = True
        self.is_frozen = False

    # --- Core Kernels
    def init_kernels(self):
        kernels = self.create_kernels()
        self.kernels = torch.nn.ModuleList(kernels)

    def synchronize_kernels(self):
        intervals = self.bin_intervals.detach()
        for kernel, interval in zip(self.kernels, intervals):
            kernel.set_edges(
                interval[0],
                interval[1],
            )
        self.connect_kernels(self.kernels)

    def create_kernels(self):
        # kernel_cls is a dict of kernel pointers
        edges = self.bin_intervals.detach() # kernels should NOT have any gradient behavior since they only act as ENCHANCER
        kernels = []
        n_bins = len(edges)
        # --- creation of kernels
        for bin_num, edge in enumerate(edges):
            if bin_num == 0:
                role = "underflow"
            elif bin_num == n_bins - 1:
                role = "overflow"
            else:
                role = "normal"
            cls = self.kernel_map[role]
            kernels.append(cls(edge, **self.kernel_cfg))

        # --- configuration of kernels belong here
        # TODO smoothing width is also a neightbor hood quantity
        # necessary when training the edges
        kernels = self.connect_kernels(kernels)
        return kernels

    def _connect_smoothness(self, kernels):
        # TODO for each overlapp calculate a smoothess
        # necessary for variable edges?
        raise NotImplementedError("NotImplemented")

    def connect_kernels(self, kernels):
        n_bins = len(kernels)

        # share information about neighbors
        for bin_idx, kernel in enumerate(kernels):
            # set left cut
            if bin_idx > 0:
                kernel.set_cuts(
                left = kernels[bin_idx - 1].right_transition_coordinate
            )
            # set right cut
            if bin_idx < n_bins - 1:
                kernel.set_cuts(
                right = kernels[bin_idx + 1].left_transition_coordinate
                )
        return kernels

    # --- Geometry handling ---
    @property
    def lower_edge(self):
        return self.bounds[0]

    @property
    def upper_edge(self):
        return self.bounds[1]

    def _edges_to_relative_width(self, edges):
        widths = edges[1:] - edges[:-1]
        return widths / (self.upper_edge - self.lower_edge)

    def _relative_width_to_edges(self, relative_width):
        # TODO bin edges ignore transformation currently
        # calculate absolute widht of bins
        # from IPython import embed; embed(header="MESSAGE Line 1414 | File: layers.py")
        interval = self.upper_edge - self.lower_edge
        width = interval * relative_width

        right = self.lower_edge + torch.cumsum(width, dim=0)
        left = right - width
        return torch.stack((left, right), dim = 1)

    @property
    def bin_intervals(self):
        return self._relative_width_to_edges(self.relative_bin_width)

    # # TODO usage unclear
    def bin_intervals_in_space(self, transformed=None):
        intervals = self.bin_edges
        # return currently used interval when no transformed status is given
        if transformed is None:
            return intervals

        # if required already exist return it
        if (self.is_transformed and transformed) or (not self.is_transformed and not transformed):
            return intervals

        # when not transformed and transform is requested, apply transformation
        # when transformed and
        if not self.is_transformed and transformed:
            bin_fn = self.binning_fn.forward
        elif self.is_transformed and not transformed:
            bin_fn = self.binning_fn.inverse

        transformed_intervals = bin_fn(intervals, **self.binning_cfg)
        return transformed_intervals

    @property
    def bin_edges(self):
        intervals = self.bin_intervals
        edges = [intervals[:, 0].reshape(-1, 1), intervals[-1, 1].reshape(-1, 1)]
        return torch.flatten(torch.concatenate(edges, dim = 0))

    @property
    def bin_edges_original(self):
        left, right = self.original_bounds
        return torch.linspace(left, right, self.num_bins + 1)

    def _transform_bounds(self) -> tuple[torch.Tensor,torch.Tensor]:
        """
        What are the current active bounds.

        Returns:
            torch.Tensor: Current active bounds with applied transformation.
        """
        if self.binning_fn is None:
            self.is_transformed = False
            return self.original_bounds

        self.is_transformed = True
        return (
            self.binning_fn.forward(torch.as_tensor(self.original_bounds[0]), **self.binning_cfg),
            self.binning_fn.forward(torch.as_tensor(self.original_bounds[1]), **self.binning_cfg)
        )

    def _create_initial_edges(self) -> torch.Tensor:
        """
        Creates and returns linspace edges in current transformed edge space.
        """
        return torch.linspace(
            self.lower_edge,
            self.upper_edge,
            self.num_bins + 1
            )

    def init_learnable_edges(self):
        """
        Register the learnable bin edges as parameters.
        The bin edges are saves as relative width. To ensure non-negative values AND summation to 1 a softmax constraint it put on top.
        """
        self.bounds = self._transform_bounds()

        edges = self._create_initial_edges()
        relative_width = self._edges_to_relative_width(edges)

        self.relative_bin_width = torch.nn.Parameter(relative_width)

        parametrize.register_parametrization(
            self,
            "relative_bin_width",
            torch.nn.Softmax(dim=0)
            )

    def create_evaluation_state(self) -> dict[str, Any]:
            return {
                "kernels": copy.deepcopy(self.kernels).cpu(),
                "binning_fn": self.binning_fn,
                "active_edges": self.bin_edges.detach().cpu(),
                "original_edges": self.bin_edges_original.detach().cpu(),
            }


    def monitored_gradient_names(self):
        names = ["dnn_score", "binned_tensor"]
        names += [f"weighted_dnn_score_bin_{i}" for i in range(self.num_bins)]
        return names

    def monitored_tensor_names(self):
        return ["kernel_weights"]

    def old_binning(self, y):
        weighted_bins_y = []
        kernel_weights = []
        # kernel weight is determined by transformed y
        y = torch.as_tensor(y)
        transformed_y = (self.binning_fn.forward(y, **self.binning_cfg)).detach()

        for bin_num, kernel in enumerate(self.kernels):
            bin_weight = kernel(transformed_y)
            kernel_weights.append(bin_weight)
            bin_y = bin_weight * y
            self.monitor_gradient(tensor=bin_y,name=f"weighted_dnn_score_bin_{bin_num}")
            weighted_bins_y.append(bin_y)
        output = torch.stack(weighted_bins_y, dim=0)
        self.monitor_gradient(tensor=y, name="dnn_score")
        self.monitor_gradient(tensor=output, name="binned_tensor")
        self.monitor_tensor(tensor=torch.stack(kernel_weights), name="kernel_weights")
        return output

    def normalized_binning(self, y):
        weighted_bins_y = []
        kernel_weights = []
        # kernel weight is determined by transformed y
        transformed_y = (self.binning_fn.forward(y, **self.binning_cfg)).detach()

        # normalize
        lower_edge_ind = torch.bucketize(transformed_y, self.bin_edges, right=True)[:,0]
        _bins = self.binning_fn.inverse(self.bin_edges)
        lower_bin_edge, upper_bin_edge = _bins[lower_edge_ind - 1].reshape(-1,1), _bins[lower_edge_ind].reshape(-1,1)
        normalized_y = (y - lower_bin_edge) / (upper_bin_edge - lower_bin_edge)
        # apply kernel weight
        for bin_num, kernel in enumerate(self.kernels):
            bin_weight = kernel(transformed_y)
            kernel_weights.append(bin_weight)
            bin_y = bin_weight * normalized_y
            weighted_bins_y.append(bin_y)
        output = torch.stack(weighted_bins_y, dim=0)

        self.monitor_gradient(tensor=y, name="dnn_score")
        self.monitor_gradient(tensor=bin_y,name=f"weighted_dnn_score_bin_{bin_num}")
        self.monitor_tensor(tensor=kernel_weights, name="kernel_weights")
        self.monitor_gradient(tensor=output, name="binned_tensor")
        return output


    def forward(self, y):
        # y and not x due to y being an neural network output
        if self.training and not self.is_frozen:
            self.synchronize_kernels()

        return self.old_binning(y)
