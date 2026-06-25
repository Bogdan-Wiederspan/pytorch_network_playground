import matplotlib.pyplot as plt

from ..register import register_plot

from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

@register_plot("dummy", requires=None)
def plot_dummy(**kwargs) -> tuple[Figure, Axes]:
    """
    Dummy figure that should be used as placeholder

    Returns:
        tuple(plt.Figure, plt.Axes):
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.text((0.5, 0.5), "Not existing", fontsize="large", color="red")
    return fig, ax
