import matplotlib.pyplot as plt
import numpy as np

from bbTT.monitoring.register import register_plot
from bbTT.monitoring.utils.tensor import to_numpy


@register_plot(
    "bin_edges",
    requires={"active_edges"},
    optional=True,
    )
def plot_bin_edges(ctx, **kwargs):
    """
    Visualize the spatial distribution of the model's active bin edges.

    Draws each edge as a tick mark along a single horizontal line, so
    that edges bunching together (bin collapse) are visible as overlapping marks, which is
    harder to spot in numeric tick labels (e.g. on the `asimov` plot).

    Args:
        ctx (EvalContext): Must provide "active_edges". Only available
            for models with an actual binning layer — this plot should
            be registered as optional.
        **kwargs: Unused, present for registry call-signature consistency.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
        figure and axes containing the bin-edge tick visualization.
    """
    binning_edges = ctx.get("active_edges")
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
