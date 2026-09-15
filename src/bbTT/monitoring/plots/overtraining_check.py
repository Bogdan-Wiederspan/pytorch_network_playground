import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import ks_2samp

from bbTT.monitoring.register import register_plot
from bbTT.monitoring.utils.tensor import make_plotable


def _torch_histogram_density(values, bins, value_range=(0.0, 1.0)):
    """
    Compute a density-normalized histogram without leaving torch.

    Args:
        values (torch.Tensor): 1D tensor of values to histogram.
        bins (int): Number of bins.
        value_range (tuple[float, float], optional): (min, max) range.
            Defaults to (0.0, 1.0).

    Returns:
        torch.Tensor: Per-bin density, length `bins`, same device as `values`.
    """
    counts = torch.histc(values.float(), bins=bins, min=value_range[0], max=value_range[1])
    bin_width = (value_range[1] - value_range[0]) / bins
    total = counts.sum()
    return counts / (total * bin_width) if total > 0 else counts


@register_plot(
    "overtraining_check",
    requires=None,
    optional=True,
)
def plot_overtraining_check(ctx, ctx_val=None, node="hh", bins=30, **kwargs) -> tuple[Figure, Axes]:
    """
    Compare train vs. validation output-score distributions to spot overtraining.

    Training-set scores are drawn as step-histogram lines per true class,
    Validation-set scores are plotted as points with Poisson error bars on the same
    axes.

    Note for Interpretation: A visible gap between line and points for the same class is
    the overtraining signature to watch for — the per-class KS p-value
    in the legend is a rough compatibility check, not proof on its own.

    Note to myself:
    Two distinct EvalContext instances are necessary. This plot destroys the single-ctx-per-call patern of all other plots.
    The second context `ctx_val` must be supplied explicitly by the caller.

    Args:
        ctx (EvalContext): Training-mode context. Must expose
            `predictions`, `targets`, `target_map`.
        ctx_val (EvalContext): Validation-mode context, same shape.
            Required — this plot has no meaning with only one context.
        node (str, optional): Which target_map class's node score to
            compare. Defaults to "hh".
        bins (int, optional): Number of histogram bins. Defaults to 30.
        **kwargs: May include "title" (str). Other keys are ignored.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
        figure and axes containing the overlaid train/validation scores.
    """
    if ctx_val is None:
        raise ValueError("plot_overtraining_check requires ctx_val (a validation-mode EvalContext).")

    node_idx = ctx.target_map[node]
    bin_edges = torch.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    fig, ax = plt.subplots(figsize=(7, 5))
    ks_pvalues = {}
    centers_np = centers.numpy()

    for process, idx in ctx.target_map.items():
        train_scores = ctx.predictions[ctx.targets[:, idx] == 1, node_idx]
        val_scores = ctx_val.predictions[ctx_val.targets[:, idx] == 1, node_idx]

        train_density = _torch_histogram_density(train_scores, bins)
        val_density = _torch_histogram_density(val_scores, bins)
        val_counts = torch.histc(val_scores.float(), bins=bins, min=0.0, max=1.0) # only calc counts, not np hist conversion as it would be normally the case
        val_err = val_density / torch.sqrt(val_counts + 1)




        # scipy has not torch support, so numpy conversion happens heres
        ks_pvalues[process] = ks_2samp(
            make_plotable(train_scores),
            make_plotable(val_scores),
            ).pvalue
        ax.step(centers, make_plotable(train_density), where="mid", label=f"{process} (train)")
        ax.errorbar(centers, make_plotable(val_density), yerr=make_plotable(val_err), fmt="o", markersize=4, label=f"{process} (val)")


        ax.legend(fontsize=8, title="\n".join(f"KS p({p}) = {v:.3g}" for p, v in ks_pvalues.items()))

    title = kwargs.pop("title", None)
    if title is not None:
        fig.suptitle(title)

    ax.set_xlabel(f"{node} node output score")
    ax.set_ylabel("normalized events")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    return fig, ax
