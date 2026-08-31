import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bbTT.monitoring.context import EvalContext
from bbTT.monitoring.register import register_plot
from bbTT.monitoring.utils.tensor import make_plotable


# TODO requirement checking
# needs mass
@register_plot(
    "score_mass_sculpting",
    requires={"sample_attributes"},
    optional=True,
)
def plot_score_mass_sculpting(ctx: EvalContext, mass_column: str="mass_tautau", node: str="hh", bins: int=30, **kwargs) -> tuple[Figure, Axes]:
    """
    Check whether the classifier score correlates with a mass variable, background only.

    A visible trend between score and mass means cutting on score would
    sculpt the background mass shape — risking a fake signal-like
    feature in a downstream fit. Background = all events whose true
    class is not `node`. Masking and correlation stay in torch; only
    the 2D histogram itself is converted to numpy, since matplotlib's
    hist2d requires array-like input.

    Note: needs a per-event mass variable via
    `ctx.get("sample_attributes")[mass_column]`, which is not yet part
    of the EvalContext contract — requires wiring from
    `full_config.training_config.sample_attributes` before use.

    Args:
        ctx (EvalContext): Must expose `predictions`, `targets`,
            `target_map`, and provide "sample_attributes" containing
            `mass_column`.
        mass_column (str, optional): Name of the mass variable within
            sample_attributes. Defaults to "mass_tautau".
        node (str, optional): Signal class name, defines the background
            mask (all other classes). Defaults to "hh".
        bins (int, optional): Bins per axis for the 2D histogram.
            Defaults to 30.
        **kwargs: May include "title" (str). Other keys are ignored.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
        figure and axes containing the score-vs-mass 2D histogram.
    """
    node_idx = ctx.target_map[node]
    background_mask = ctx.targets[:, node_idx] == 0

    scores = ctx.predictions[background_mask, node_idx]
    mass = ctx.get("sample_attributes")[mass_column][background_mask]

    corr = torch.corrcoef(torch.stack([scores, mass]))[0, 1]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.hist2d(
        make_plotable(scores),
        make_plotable(mass),
        bins=bins,
        cmap="viridis",
        cmin=1,
    )
    ax.set_title(f"Score-vs-mass correlation (background), Pearson r = {corr.item():.2f}")

    title = kwargs.pop("title", None)
    if title is not None:
        fig.suptitle(title)

    ax.set_xlabel(f"{node} node output score")
    ax.set_ylabel(mass_column)
    return fig, ax
