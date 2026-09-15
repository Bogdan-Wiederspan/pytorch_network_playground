import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bbTT.monitoring.context import EvalContext
from bbTT.monitoring.metrics.physics.asimov import asimov_metric
from bbTT.monitoring.register import register_plot


@register_plot(
    "significance_scan",
    requires=None,
    optional=False,
)
def plot_significance_scan(ctx: EvalContext, node: str="hh", n_thresholds: int=100, **kwargs) -> tuple[Figure, Axes]:
    """
    Plot Asimov significance as a function of a single moving score cut.

    For each threshold t, computes the weighted signal/background yield
    passing `score > t`, then the significance of that single cut.
    Useful as a cross-check against the learned binning's summed
    per-bin significance. Stays entirely in torch (including the scan
    loop and argmax) until the final plotting call.

    Args:
        ctx (EvalContext): Must expose `predictions`, `targets`,
            `target_map`, `event_weights`.
        node (str, optional): Signal class name. Defaults to "hh".
        n_thresholds (int, optional): Number of scan points between 0
            and 1. Defaults to 100.
        **kwargs: May include "title" (str). Other keys are ignored.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
        figure and axes containing the significance-vs-threshold curve.
    """
    node_idx = ctx.target_map[node]
    signal_mask = ctx.targets[:, node_idx] == 1
    background_mask = ~signal_mask

    scores = ctx.predictions[:, node_idx]
    weights = ctx.event_weights

    signal_scores, signal_weights = scores[signal_mask], weights[signal_mask]
    background_scores, background_weights = scores[background_mask], weights[background_mask]

    thresholds = torch.linspace(0.0, 1.0, n_thresholds, device=scores.device)
    significances = torch.zeros(n_thresholds, device=scores.device)

    # loop is fine here: runs once per monitoring interval, not per training step
    for i, t in enumerate(thresholds):
        s = signal_weights[signal_scores > t].sum()
        b = background_weights[background_scores > t].sum()
        significances[i] = asimov_metric(s=s, b=b)

    significances = torch.nan_to_num(significances, nan=0.0)
    best_idx = torch.argmax(significances)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds.cpu().numpy(), significances.cpu().numpy(), color="tab:red", linewidth=2)
    ax.axvline(thresholds[best_idx].item(), color="black", linestyle="--", linewidth=1)
    ax.scatter([thresholds[best_idx].item()], [significances[best_idx].item()], color="black", zorder=5)
    ax.text(
        0.97, 0.95,
        f"best cut = {thresholds[best_idx].item():.3f}\nZ = {significances[best_idx].item():.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    title = kwargs.pop("title", None)
    if title is not None:
        fig.suptitle(title)

    ax.set_xlabel(f"{node} node score cut threshold")
    ax.set_ylabel("significance")
    ax.grid(alpha=0.3)
    return fig, ax
