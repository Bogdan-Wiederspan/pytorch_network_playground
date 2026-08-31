import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import PrecisionRecallDisplay

from bbTT.monitoring.context import EvalContext
from bbTT.monitoring.register import register_plot
from bbTT.monitoring.utils.tensor import make_plotable


@register_plot(
    "precision_recall",
    requires=None,
    optional=False,
)
def plot_precision_recall(ctx: EvalContext, sample_weight: torch.Tensor=None, labels: list[str]=None, colors: list[str]=None, **kwargs) -> tuple[Figure, Axes]:
    """
    Plot one-vs-rest precision-recall curves for each class.

    Complements the ROC plot: PR is more informative under class
    imbalance (e.g. a small "hh" signal fraction), since ROC's false
    positive rate is diluted by a large negative class in a way
    precision is not. sklearn has no torch support, so tensors are
    explicitly moved to CPU/numpy right at that boundary — relying on
    implicit tensor-to-array conversion would raise on GPU tensors
    rather than just being slower.

    Args:
        ctx (EvalContext): Must expose `targets`, `predictions` (both
            shaped [n_samples, n_classes]) and `target_map`.
        sample_weight (torch.Tensor, optional): Per-event weights.
            Defaults to uniform weighting.
        labels (list[str], optional): Class names, in column order.
            Defaults to `ctx.target_map` keys.
        colors (Sequence, optional): Colors to cycle per class. Defaults
            to the "tab10" colormap.
        **kwargs: Unused, present for registry call-signature consistency.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
        figure and axes containing the overlaid PR curves.
    """
    target = ctx.targets
    pred = ctx.predictions
    target_map = ctx.target_map

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.get_cmap("tab10").colors if colors is None else colors

    if sample_weight is None:
        sample_weight = torch.ones(target.shape[0], device=target.device)
    if labels is None:
        labels = list(target_map.keys())
    if not labels:
        raise ValueError("plot_precision_recall requires at least one label to plot.")

    sample_weight_np = make_plotable(sample_weight)

    for class_idx, name in enumerate(labels):
        disp = PrecisionRecallDisplay.from_predictions(
            make_plotable(target[:, class_idx]),
            make_plotable(pred[:, class_idx]),
            sample_weight=sample_weight_np,
            ax=ax,
            name=name,
            color=colors[class_idx % len(colors)],
        )

    ax.set(xlabel="Recall", ylabel="Precision")
    return disp.figure_, disp.ax_
