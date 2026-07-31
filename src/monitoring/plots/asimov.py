import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

from ..metrics.physics.asimov import (
    asimov_metric,
    asimov_no_background_metric,
    asimov_small_signal_and_no_background_metric,
)
from ..register import register_plot, register_plot_variant
from ..utils.tensor import prepare_tensor

_map = {
    "small_signal" : asimov_small_signal_and_no_background_metric,
    "approximation" : asimov_no_background_metric,
    "full" : asimov_metric,
}

@register_plot(
    "asimov",
    requires=("active_edges" , "s_hist", "b_hist"),
    optional=True,
    )
def plot_asimov_per_bin(
    ctx,
    which_asimov="small_signal",
    **kwargs,
) -> tuple[Figure, Axes]:
    s = ctx.get("s_hist")
    b = ctx.get("b_hist")
    binning_edges = ctx.get("active_edges")
    # Attention, values need to be tensors, since fn is using torch functions mostly

    _label_map = {
        "small_signal" : "Asimov Significance (s << b & w/o uncertainty)",
        "approximation" : "Asimov Significance (w/o uncertainty)",
        "full" : "Asimov Significance (full)"
    }

    fn = _map[which_asimov]
    score = fn(s=s,b=b, **kwargs)
    # setting nan to 0
    nan_mask = ~torch.isnan(score)
    score = torch.where(nan_mask, score, torch.zeros_like(score))

    score, binning_edges = prepare_tensor(score, binning_edges, device="cpu")
    total_asimov = np.sum(score**2)**0.5

    # plot the score as bar plot and add as ticks the edge values
    fig, axes = plt.subplots(1, 1, figsize=(8 * 1, 8 * 1))

    # first bar plot is with equal distant bars
    # second bar plot is with width equal to bin width
    bin_width = 1
    x = np.arange(len(score)) + bin_width / 2 # shift by half bin to be between edges
    axes.bar(x, height=score, width=1., edgecolor="black", facecolor="orange", linewidth=1.5)
    axes.set_xticks(np.arange(len(binning_edges)))
    axes.set_xticklabels([f"{float(edge):.5f}" for edge in binning_edges])

    fig.suptitle(f"Total Asimov $\\sqrt{{\\sum A^2}}$: {total_asimov:.5f}")
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.15)

    axes.tick_params(axis="x", labelrotation=90)
    axes.set_xlabel("Bin Edges")
    axes.set_ylabel(f"{_label_map[which_asimov]}")
    axes.grid()
    return fig, axes

register_plot_variant(
    name="asimov_full",
    base="asimov",
    which_asimov="full",
)

register_plot_variant(
    name="asimov_small_signal",
    base="asimov",
    which_asimov="small_signal",
)

register_plot_variant(
    name="asimov_approximation",
    base="asimov",
    which_asimov="approximation",
)
