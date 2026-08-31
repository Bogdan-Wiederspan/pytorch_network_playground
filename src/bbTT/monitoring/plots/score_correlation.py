import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bbTT.monitoring.context import EvalContext
from bbTT.monitoring.register import register_plot
from bbTT.monitoring.utils.tensor import make_plotable


@register_plot(
    "score_correlation_matrix",
    requires=None,
    optional=False,
)
def plot_score_correlation_matrix(ctx: EvalContext, **kwargs) -> tuple[Figure, Axes]:
    """
    Plot the Pearson correlation matrix between output-node scores.

    Note for Interpretation:
    High correlation between two nodes suggests those classes are
    structurally hard to separate. Complements what the confusion
    matrix shows for hard class assignments.

    Args:
        ctx (EvalContext): Must expose `predictions` (shape
            [n_samples, n_classes]) and `target_map`.
        **kwargs: May include "title" (str). Other keys are ignored.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
        figure and axes containing the correlation-matrix heatmap.
    """
    node_names = list(ctx.target_map.keys())
    node_indices = list(ctx.target_map.values())
    pred = ctx.predictions[:, node_indices]

    corr_matrix = torch.corrcoef(pred.T)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    corr_matrix = make_plotable(corr_matrix)
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(node_names)))
    ax.set_xticklabels(node_names)
    ax.set_yticks(range(len(node_names)))
    ax.set_yticklabels(node_names)

    for i in range(len(node_names)):
        for j in range(len(node_names)):
            ax.text(
                j, i, f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                )

    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title(kwargs.pop("title", None) or "Output-node score correlation matrix")
    return fig, ax
