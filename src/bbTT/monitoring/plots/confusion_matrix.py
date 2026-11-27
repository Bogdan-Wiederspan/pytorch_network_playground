from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

from bbTT.monitoring.metrics.classification.matrix import confusion_matrix
from bbTT.monitoring.register import register_plot


@register_plot(
    "confusion_matrix",
    requires=None,
    optional=False,
)
def plot_confusion_matrix(ctx, normalize="true", cmap="Blues", **kwargs) -> tuple[Figure, Axes]:

    """
    Plot a confusion matrix comparing true and predicted classes.

    Reads predictions, targets and class-name mapping directly from *ctx*, which are always present.
    Events are weighted on the composition axis, so a full pass shows the class mixture the network
    is trained on (see EvalContext.weights).

    Args:
        ctx (EvalContext): Must expose `predictions`, `targets`, and
            `target_map` (dict mapping class name to index).
        normalized (str, optional): Normalization mode forwarded to
            sklearn's confusion_matrix as `normalize`. One of "true",
            "pred", "all", or None. Defaults to "true".
        cmap (str, optional): Colormap for the matrix display. Defaults
            to "Blues".
        **kwargs: May include "title" (str) to set a figure suptitle;
            all other keys are ignored.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
        figure and axes containing the confusion matrix display.
    """
    y_pred = ctx.predictions
    y_true = ctx.targets
    target_map = ctx.target_map

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(target_map.values()),
        sample_weight=ctx.weights("composition"),
        normalize=normalize,  # normalize to get probabilities
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(target_map.keys()),
    )
    disp.plot(cmap=cmap)

    title = kwargs.pop("title", None)
    if title is not None:
        disp.figure_.suptitle(title)

    return disp.figure_, disp.ax_
