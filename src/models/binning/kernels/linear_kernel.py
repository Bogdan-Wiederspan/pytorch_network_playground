from __future__ import annotations

import torch

from models.binning import (
    BaseKernel,
    UnderflowKernel,
    OverflowKernel,
)


class LinearKernel(BaseKernel):
    def __init__(
        self,
        edges,
        *,
        left_notch: float = 0.0,
        right_notch: float = 0.0,
        absolute_notch: bool = False,

        # --- kernel-specific parameters ---
        parameter1: float = ...,
        parameter2: float = ...,

        bin_height: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            edges=edges,
            bin_height=bin_height,
            left_notch=left_notch,
            right_notch=right_notch,
            absolute_notch=absolute_notch,
        )
        raise NotImplementedError("Not implented yet, just a placeholder")
        # Store kernel-specific parameters
        self.parameter1 = torch.as_tensor(parameter1)
        self.parameter2 = torch.as_tensor(parameter2)

        # Optional: precompute constants
        self._constant = ...

        self.checks()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def checks(self):
        super().checks()

        # Add kernel-specific assertions
        # torch._assert(...)
        pass

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _compute_normalization(self):
        """
        Return multiplicative normalization factor.
        Return torch.tensor(1.0) if no normalization is required.
        """
        return torch.tensor(1.0)

    # ------------------------------------------------------------------
    # Transition functions
    # ------------------------------------------------------------------

    def left_transition_fn(self, x: torch.Tensor) -> torch.Tensor:
        left = self.left_transition_coordinate

        # Implement left tail
        return ...

    def right_transition_fn(self, x: torch.Tensor) -> torch.Tensor:
        right = self.right_transition_coordinate

        # Implement right tail
        return ...

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    # Only override if the kernel needs behaviour beyond BaseKernel.
    #
    # def kernel(self, x):
    #     y = super().kernel(x)
    #     ...
    #     return y


class LinearUnderflowKernel(UnderflowKernel, LinearKernel):
    """One-sided version extending to -∞."""
    pass


class LinearOverflowKernel(OverflowKernel, LinearKernel):
    """One-sided version extending to +∞."""
    pass
