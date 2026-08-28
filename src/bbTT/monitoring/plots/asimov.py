import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bbTT.monitoring.metrics.physics.asimov import (
    asimov_metric,
    asimov_no_background_metric,
    asimov_small_signal_and_no_background_metric,
)
from bbTT.monitoring.register import register_plot, register_plot_variant
from bbTT.monitoring.utils.tensor import prepare_tensor

_ASIMOV_VARIANTS = {
    "small_signal": (
        asimov_small_signal_and_no_background_metric,
        "Asimov Significance (s << b & w/o uncertainty)",
    ),
    "approximation": (
        asimov_no_background_metric,
        "Asimov Significance (w/o uncertainty)",
    ),
    "full": (
        asimov_metric,
        "Asimov Significance (full)",
    ),
}


@register_plot(
    "asimov",
    requires=("evaluation_state.binning_edges", "s_hist", "b_hist"),
    optional=False,
)
def plot_asimov_per_bin(
    ctx,
    which_asimov="small_signal",
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plot per-bin Asimov significance as a bar chart.

    Uses the model's active bin edges when available (via
    `evaluation_state.binning_edges`), and the config-driven default
    edges otherwise — so this plot runs for both binning and
    non-binning models.

    Args:
        ctx (EvalContext): Must provide "s_hist", "b_hist",
            "evaluation_state.binning_edges".
        which_asimov (str): One of "small_signal", "approximation",
            "full" — selects the Asimov metric variant (see
            `_ASIMOV_VARIANTS`). Defaults to "small_signal".
        **kwargs: Forwarded to the selected Asimov metric function.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
        figure and axes containing the per-bin Asimov bar chart.
    """
    s = ctx.get("s_hist")
    b = ctx.get("b_hist")
    binning_edges = ctx.get("evaluation_state.binning_edges")
    # Attention, values need to be tensors, since fn is using torch functions mostly

    fn, label = _ASIMOV_VARIANTS[which_asimov]
    score = fn(s=s, b=b, **kwargs)

    # Bins with no meaningful significance (e.g empty background) come out as
    # nan from the metrics - treated as zero contribution
    nan_mask = ~torch.isnan(score)
    score = torch.where(nan_mask, score, torch.zeros_like(score))

    score, binning_edges = prepare_tensor(score, binning_edges, device="cpu")
    total_asimov = np.sum(score**2) ** 0.5

    fig, axes = plt.subplots(1, 1, figsize=(8 * 1, 8 * 1))
    bin_width = 1
    # score has N bins, binning_edges has N+1 edges (one more than bars).
    # Bars sit at integer positions [0, N); ticks/labels sit at edge positions
    # [0, N] — so ticks must be placed at bar *boundaries*, not bar centers,
    # for the label under each tick to correctly show that bin's edge value.

    x = np.arange(len(score)) + bin_width / 2  # shift by half bin to be between edges
    axes.bar(x, height=score, width=1.0, edgecolor="black", facecolor="orange", linewidth=1.5)
    axes.set_xticks(np.arange(len(binning_edges)))
    axes.set_xticklabels([f"{float(edge):.5f}" for edge in binning_edges])

    fig.suptitle(f"Total Asimov $\\sqrt{{\\sum A^2}}$: {total_asimov:.5f}")
    fig.subplots_adjust(hspace=0.15, left=None, bottom=None, right=None, top=None, wspace=None)

    axes.tick_params(axis="x", labelrotation=90)
    axes.set_xlabel("Bin Edges")
    axes.set_ylabel(label)
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
