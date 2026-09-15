import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bbTT.monitoring.register import register_plot


@register_plot(
    "dummy",
    requires=None,
    optional=True,
    )
def plot_dummy(ctx, **kwargs) -> tuple[Figure, Axes]:
    """
    Placeholder plot, intended as a copy-paste template for new plots.

    Draws a plain "Not existing" figure. Not wired in as an automatic
    fallback for other plots — the require/provide system has no
    built-in "try X, else Y" behavior, so any fallback use would need
    explicit logic elsewhere.

    Args:
        **kwargs: Unused, present for registry call-signature consistency.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The
        figure and axes containing the placeholder text.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.text(0.5, 0.5, "Not existing", fontsize="large", color="red")
    return fig, ax
