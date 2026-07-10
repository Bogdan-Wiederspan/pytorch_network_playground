
from .kernels.base_kernel import BaseKernel, UnderflowKernel, OverflowKernel
from .kernels.gaussian_kernel import GaussianKernelFinal
from .binning import BinningLayer

# backwards compatibility
from .kernels.gaussian_kernel import GaussianKernelFinal as GaussianKernel
from .kernels.tanh_kernel import TanhKernel, TanhUnderflowKernel, TanhOverflowKernel


KERNEL_MAP = {
    "Tanh":
        {
        "underflow" : TanhUnderflowKernel,
        "overflow" : TanhOverflowKernel,
        "normal" : TanhKernel,
    },
}

__all__ = [
    "KERNEL_MAP"
    "TanhKernel",
    "TanhUnderflow"
    "TanhOverflow"
    "GaussianKernel",
    "BaseKernel",
    "create_kernels",
    "BinningLayer",
    "GaussianKernelFinal"
    "UnderflowKernel",
    "OverflowKernel",
]
