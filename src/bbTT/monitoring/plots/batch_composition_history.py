import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bbTT.monitoring.register import register_plot


@register_plot(
    "batch_composition_history",
    requires=("composition_steps", "composition_by_parent", "composition_by_uid"),
    optional=True,
)
def plot_batch_composition_history(ctx, **kwargs) -> tuple[Figure, Axes]:
    """
    Absolute per-sub-process event counts over training steps.

    Plots absolute counts rather than fractions on purpose: truncation is a drop
    in the total, which a normalized view would hide. Flat lines mean the
    sampler is honouring its quotas; periodic dips mean a cursor is truncating
    at wraparound instead of refilling from the start.

    Args:
        ctx (EvalContext): Must carry ``batch_composition``.
        **kwargs: Unused, present for registry call-signature consistency.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The figure and
        axes containing the composition history.
    """
    steps = ctx.get("composition_steps")
    by_parent = ctx.get("composition_by_parent")
    counts_per_sub_process = ctx.get("composition_by_uid")

    axis_map = ctx.target_map

    fig, ax = plt.subplots(1, 1 + len(axis_map), figsize=(20, 12))
    for parent, values in by_parent.items():
        axis = ax[0]
        line, = axis.plot(steps, values, label=parent, lw=1)


    for parent, index in axis_map.items():
        for sub_process, values in counts_per_sub_process.items():
            if sub_process[0] == parent:
                axis = ax[1 + index]
                line, = axis.plot(steps, values, label=sub_process[1], lw=1)
                axis.set_title(parent)

    for axis in ax:
        axis.set_xlabel("step")
        axis.set_ylabel("events in batch")
        axis.set_ylim(bottom=0)
        # axis.legend() # deactived, only necessary to see big jumps or flat line
    return fig, ax
