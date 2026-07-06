import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from ..metrics.classification.matrix import confusion_matrix
from ..register import register_plot

from matplotlib.figure import Figure
from matplotlib.pyplot import Axes


@register_plot("confusion_matrix", requires=None)
def plot_confusion_matrix(
    ctx,
    sample_weight=None,
    normalized="true",
    cmap="Blues",
    **kwargs
    ) -> tuple[Figure, Axes]:
    """
    Calculates a Confusion Matrix using the truth *y_true* and prediction *y_pred* of the model.
    The categories are defined using *target_map* and to weight give *sample_weight*. Styles are
    changed using *cmap*.

    Args:
        y_true (torch.tensor): tensor of true labels
        y_pred (torch.tensor): tensor of predicted labels
        target_map (dict[int]): dict of class name to index mapping
        sample_weight (torch.tensor, optional): tensor of weights on event basis. Defaults to None.
        cmap (str, optional): style of the confusion matrix. Defaults to "Blues".

    Returns:
        tuple: figure, axis, confusion matrix
    """
    y_pred = ctx.predictions
    y_true = ctx.targets
    target_map = ctx.target_map

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(target_map.values()),
        sample_weight=sample_weight,
        normalize=normalized,  # normalize to get probabilities
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=list(target_map.keys()),
    )
    disp.plot(cmap=cmap)
    disp.figure_.suptitle(kwargs.pop("title", None))

    return disp.figure_, disp.ax_
