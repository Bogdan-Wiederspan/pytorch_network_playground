import torch


class BinningLayerV1(torch.nn.Module):
    def __init__(
        self,
        init_edges: list[float],
        kernel_cls,
        kernel_cfg,
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
        self.init_edges = init_edges
        self.edges = torch.nn.Parameter(init_edges)
        self.kernel_cls = kernel_cls

        # config allows bin-wise configuration
        # if bin number (as int) present overwrite general value
        self.original_kernel_cfg = kernel_cfg
        # self.kernel_configs = self.build_cfg(self.original_kernel_cfg)
        self.kernels = self.build_kernels(kernel_cfg=kernel_cfg)

    def get_edges(self):
        e = self.edges.detach()
        low, up = e[:-1], e[1:]
        return [(l,u) for l,u in zip(low,up)]

    def build_cfg(self, base_cfg):
        config = {}
        for _bin in range(0, self.num_bins):
            if _bin in base_cfg.keys():
                fill_cfg = base_cfg[_bin]
            else:
                fill_cfg = base_cfg["general"].copy()
            fill_cfg["bin_type"] = "normal"
            config[_bin] = fill_cfg

        config[0]["bin_type"] = "underflow"
        config[self.num_bins - 1]["bin_type"] = "overflow"
        return config

    def build_kernels(self, kernel_cfg):
        edges = self.get_edges()
        kernels = []

        for bin_index in range(0, self.num_bins):
            bin_config = kernel_cfg.copy()
            edge = edges[bin_index]

            if bin_index == 0:
                bin_config["bin_type"] = "underflow"
            elif bin_index == (self.num_bins - 1):
                bin_config["bin_type"] = "overflow"
            else:
                bin_config["bin_type"] = "normal"

            kernels.append(self.kernel_cls(edge, **bin_config))
        return kernels

    # def build_kernels(self,):
    #     edges = self.get_edges()
    #     kernels = []
    #     range(0, self.num_bins):
    #     for bin_index, cfg in self.kernel_configs.items():
    #         edge = edges[bin_index]
    #         kernels.append(self.kernel_cls(edge=edge, **cfg))
    #     return kernels
    def __repr__(self):
        edges = self.get_edges()
        edges = [(float(l), float(u)) for l,u in edges]

        msg = "Bin Index | Edge Value\n"
        msg+="\n".join([f"{idx}:{edge}" for idx, edge in enumerate(edges)])
        return msg

    @property
    def num_bins(self):
        return len(self.edges) - 1

    def check(self):
        # config has to have "general" key
        assert "general" in self.kernel_cfg.keys()

    def forward(self, x):
        scaled_x = []
        for kernel in self.kernels:
            scale = kernel(x)
            scaled_x.append(scale * x)
        binned_x = torch.sum(torch.stack(scaled_x, axis=0), axis=0)
        return binned_x
