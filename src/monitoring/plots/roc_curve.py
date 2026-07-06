import matplotlib.pyplot as plt
import torch
from sklearn.metrics import RocCurveDisplay

from ..register import register_plot

from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

@register_plot("roc")
def roc_curve(ctx, sample_weight=None, labels=None, **kwargs) -> tuple[Figure, Axes]:

    target = ctx.targets
    pred = ctx.predictions
    target_map = ctx.target_map

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.get_cmap("tab10").colors if kwargs.get("colors") is None else kwargs["colors"]

    if sample_weight is None:
        sample_weight = torch.ones(target.shape[0])

    if labels is None:
        labels=list(target_map.keys())

    for c, name in enumerate(labels):
        col = colors[c % len(colors)]

        disp = RocCurveDisplay.from_predictions(
            target[:, c],
            pred[:, c],
            sample_weight=sample_weight.reshape(-1, 1),
            ax=ax,
            name=name,
            # curve_kwargs={"color": col},
            color=col,
        )
    _ = ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    return disp.figure_, disp.ax_
