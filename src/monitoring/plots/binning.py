import matplotlib.pylab as plt
import numpy as np

from ..register import register_plot
from ..utils.tensor import to_numpy

# def visualize_bins(kernels=None):
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#     # when no kernels exist simply return empty fig
#     if kernels is None:
#         return dummy()
#     x = torch.linspace(-0.2,1.2,200)

#     for kernel in kernels:
#         y = kernel(x)
#         ax.plot(x,y)

#     ax.set_xlabel("Kernel input", size=25)
#     ax.set_ylabel("Kernel output", size=25)
#     ax.grid()
#     return fig, ax

@register_plot("bin_edges", requires={"binning_edges"})
def plot_bin_edges(ctx, **kwargs):
    binning_edges = ctx.get("binning_edges")
    binning_edges = to_numpy(binning_edges)

    fig, ax = plt.subplots(figsize=(8, 1.5))

    # All points at y=0
    ax.scatter(binning_edges, np.zeros_like(binning_edges), s=50, marker="|", )

    # Hide y-axis
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Put x-axis slightly below the points
    ax.spines["bottom"].set_position(("data", -0.05))
    ax.set_ylim(-0.1, 0.1)
    return fig, ax
