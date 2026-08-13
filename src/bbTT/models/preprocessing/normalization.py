from __future__ import annotations

import torch


class StandardizeLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        mean: float = 0.,
        std: float = 1.,
        active: bool = True,
    ):
        """
        Standardizes the input tensor with given *mean* and *std* tensor.
        If no value is provided, mean and std are set to 0 and 1, resulting in no scaling.

        Args:
            mean (torch.tensor, optional): Mean tensor. Defaults to torch.tensor(0.).
            std (torch.tensor, optional): Standard tensor. Defaults to torch.tensor(1.).
            active (bool, optional): Status Flag to show if identity is passed or layer is used.
        """
        super().__init__()
        self.mean = torch.nn.Buffer(mean.detach().clone(), persistent=True)
        self.std = torch.nn.Buffer(std.detach().clone(), persistent=True)
        self.active = active

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if not self.active:
            return x
        x = (x - self.mean) / self.std
        return x.to(torch.float32)

    def _type_check(self, mean: torch.FloatTensor, std: torch.FloatTensor):
        if not all([isinstance(value, torch.Tensor) for value in [mean, std]]):
            raise TypeError(f"given mean or std needs to be tensor, but is {type(mean)}{type(std)}")

    def update_buffer(self, mean: torch.
                        tensor, std: torch.tensor):
        """
        Update the mean and std parameter.

        Args:
            mean (torch.tensor): Mean value.
            std (torch.tensor): Standard deviation value.
        """
        self._type_check(mean=mean, std=std)
        self.mean = mean.type_as(self.mean)
        self.std = std.type_as(self.std)
