import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import RocCurveDisplay

from bbTT.monitoring.register import register_plot


@register_plot(
    "roc",
    requires=None,
    optional=False,
)
def roc_curve(ctx, sample_weight=None, labels=None, **kwargs) -> tuple[Figure, Axes]:
    """
    Plot one-vs-rest ROC curves for each class.

    Draws one ROC curve per class (columns of `ctx.targets` /
    `ctx.predictions`), all overlaid on a single axes, colored by a
    cycling colormap.

    Args:
        ctx (EvalContext): Must expose `targets`, `predictions`
            (both shaped [n_samples, n_classes]) and `target_map`.
        sample_weight (torch.Tensor, optional): Per-event weights,
            shape [n_samples]. Defaults to uniform weighting.
        labels (list[str], optional): Class names to plot, in column
            order. Defaults to `ctx.target_map` keys.
        colors (Sequence, optional): Colors to cycle through per class.
            Defaults to the "tab10" colormap.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
        figure and axes containing the overlaid ROC curves.
    """
    target = ctx.targets
    pred = ctx.predictions
    target_map = ctx.target_map

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.get_cmap("tab10").colors if kwargs.get("colors") is None else kwargs["colors"]

    if sample_weight is None:
        sample_weight = torch.ones(target.shape[0])

    if labels is None:
        labels = list(target_map.keys())

    if not labels:
        raise ValueError("roc curve requires at least one label to")

    for class_index, name in enumerate(labels):
        color = colors[class_index % len(colors)]

        disp = RocCurveDisplay.from_predictions(
            target[:, class_index],
            pred[:, class_index],
            sample_weight=sample_weight,
            ax=ax,
            name=name,
            color=color,
        )
    _ = ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    return fig, ax
