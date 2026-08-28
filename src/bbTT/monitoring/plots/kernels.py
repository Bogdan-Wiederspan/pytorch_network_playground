import matplotlib.pyplot as plt
import torch

from bbTT.monitoring.register import register_plot
from bbTT.monitoring.utils.plotting import add_number_legend


def _resolve_kernel_plot_range(kernels, kernel_num, _range, offset=1):
    """
    Resolve which kernels to plot and the x-range to scan them over.

    Args:
        kernels (list): The model's fuzzy-binning kernel objects.
        kernel_num (Sequence[int] | None): Kernel indices to include.
            Negative indices count from the end. None selects all kernels.
        _range (tuple[float, float] | None): Explicit (low, high) scan
            range. None derives it from the selected kernels' own edges,
            padded by 10%.

    Returns:
        tuple[tuple[int, ...], float, float]: Deduplicated, sorted kernel
        indices; the low and high edge of the scan range.
    """
    num_kernels = len(kernels)

    # all kernels are chosen
    if kernel_num is None:
        kernel_num = range(num_kernels)

    # resolve negative indices
    kernel_num = tuple(sorted({(i if i >= 0 else num_kernels + i) for i in kernel_num}))

    if _range is None:
        lowest_edge = min([kernels[num].lower_edge for num in kernel_num]) * offset
        highest_edge = max([kernels[num].upper_edge for num in kernel_num]) * offset
    else:
        lowest_edge, highest_edge = _range
    return kernel_num, lowest_edge, highest_edge


def _draw_kernel_scan(ax, kernels, which_numbers, x, device):
    """Draw each selected kernel's response curve onto `ax`.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        kernels (list): The model's fuzzy-binning kernel objects.
        which_numbers (tuple[int, ...]): Indices of kernels to draw.
        x (torch.Tensor): Input values to evaluate each kernel at.
    """
    x = x.to(device)
    for kernel_num in which_numbers:
        active_kernel = kernels[kernel_num]
        y = active_kernel(x)
        ax.plot(x, y)


def _draw_weighted_prediction_hist(ax, transformed_prediction, targets, target_map, event_weights, x_hist):
    """
    Draw log-scale weighted signal/background prediction histograms.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        transformed_prediction (torch.Tensor): Signal-node scores mapped
            into bin-space via the model's binning_fn.
        signal_mask (torch.Tensor): Boolean mask selecting true-signal events.
        bg_mask (torch.Tensor): Boolean mask selecting true-background events.
        event_weights (torch.Tensor): Per-event weights.
        x_hist (torch.Tensor): Histogram bin edges.
    """
    signal_mask, bg_mask = _get_signal_background_mask(target_map=target_map, targets=targets)

    ax.hist(
        transformed_prediction[signal_mask],
        bins=x_hist,
        histtype="step",
        label="signal",
        color="blue",
        weights=event_weights[signal_mask],
    )
    ax.hist(
        transformed_prediction[bg_mask],
        bins=x_hist,
        histtype="step",
        label="background",
        color="orange",
        weights=event_weights[bg_mask],
    )
    ax.set_yscale("log")
    lines, labels = add_number_legend(ax, "Weighted Prediction")
    ax.legend(lines, labels, fontsize="small", handlelength=1.5, labelspacing=0.3)

def _draw_weighted_stacked_hist(ax, target_map, targets, predictions, event_weights, x_hist, signal_scale=1):
    """
    Draw log-scale weighted signal/background prediction histograms.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        transformed_prediction (torch.Tensor): Signal-node scores mapped
            into bin-space via the model's binning_fn.
        ctx (Context): Context with data
        event_weights (torch.Tensor): Per-event weights.
        x_hist (torch.Tensor): Histogram bin edges.
    """
    order_process = ("tt", "dy")
    process_idx = [(target_map[p]) for p in order_process]

    values = []
    weights = []

    signal_mask = (targets[:, target_map.get("hh")] == 1)
    s_predictions = predictions[signal_mask]
    s_weights = event_weights[signal_mask]

    for pidx in process_idx:
        mask = (targets[:, pidx] == 1)
        weights.append(event_weights[mask])
        values.append(predictions[mask])

    signal_label = "signal" if signal_scale == 1.0 else f"signal (×{signal_scale:g})"

    ax.hist(
            values,
            bins=x_hist,
            histtype="stepfilled",
            stacked=True,
            label=order_process,
            weights=weights,
            linewidth=4,
            alpha=0.7,
        )
    ax.hist(
            s_predictions,
            bins=x_hist,
            histtype="step",
            label=signal_label,
            weights=s_weights * signal_scale,
            alpha=0.7,
            color="blue"
        )

    ax.set_yscale("log")
    lines, labels = add_number_legend(ax, "Weighted Prediction", position=0)
    ax.legend(lines, labels, fontsize="small", handlelength=1.5, labelspacing=0.3)
    return ax


def _draw_asimov_bars(ax, edges, binned_sig, total_sig):
    """
    Draw per-bin Asimov significance as bars, with total significance in the legend.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        edges (torch.Tensor): Bin edges, length N+1.
        binned_sig (torch.Tensor): Per-bin Asimov significance, length N.
        total_sig (torch.Tensor): Scalar total significance, sqrt(sum(binned_sig**2)).
    """
    bin_width = edges[1:] - edges[:-1]
    bin_centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(
        x=bin_centers,
        height=binned_sig,
        width=bin_width,
        bottom=torch.zeros_like(binned_sig),
        label="Asimov per Bin",
        edgecolor="black",
        fill=False,
    )
    ax.set_yscale("log")
    lines, labels = add_number_legend(ax, r"$\sqrt{\sum{A_{i}^{2}}}$" + f"{total_sig.item():.5f}")
    ax.legend(lines, labels, fontsize="small", handlelength=1.5, labelspacing=0.3)


def _get_signal_background_mask(targets, target_map):
    signal_mask = targets[:, target_map["hh"]] == 1
    bg_mask = ~signal_mask
    return signal_mask, bg_mask


@register_plot(
    "kernels_monitor",
    requires=("binning_fn", "kernels", "active_edges", "monitored_tensor.binned_significance"),
    optional=True,
)
def plot_kernels_monitor(
    ctx,
    which_numbers=None,
    _range=None,
):
    """Plot fuzzy-binning kernel shapes, weighted predictions, and per-bin Asimov significance.

    Three stacked panels sharing an x-axis: (1) each active kernel's
    response curve, (2) weighted signal/background prediction histograms
    in bin-space, (3) per-bin Asimov significance.

    Args:
        ctx (EvalContext): See `_prepare_kernel_distribution_data` for
            required fields.
        which_numbers (Sequence[int], optional): Kernel indices to plot.
            Defaults to all kernels.
        _range (tuple[float, float], optional): Explicit x-scan range.
            Defaults to derived from selected kernels' edges.

    Returns:
        tuple[matplotlib.figure.Figure, numpy.ndarray]: The figure and
        its array of 3 axes.
    """
    prediction = ctx.predictions
    kernels = ctx.get("kernels")
    binning_fn = ctx.get("binning_fn")

    edges = ctx.get("active_edges")
    binned_sig = ctx.get("monitored_tensor.binned_significance")
    total_sig = torch.sqrt(torch.sum(binned_sig**2))
    event_weights = ctx.event_weights.flatten()


    # get signal node and apply transform fn
    signal_idx = ctx.target_map["hh"]
    signal_node_prediction = prediction[:, signal_idx]
    transformed_prediction = binning_fn.forward(signal_node_prediction)

    which_numbers, lowest_edge, highest_edge = _resolve_kernel_plot_range(
        kernels=kernels, kernel_num=which_numbers, _range=_range, offset=1.1
    )

    # do actual plot, be careful about device placement
    x = torch.linspace(start=lowest_edge, end=highest_edge, steps=3000)

    x_hist = torch.linspace(start=lowest_edge, end=highest_edge, steps=50)

    fig, ax = plt.subplots(3, 1, sharex=True)

    _draw_kernel_scan(ax[0], kernels=kernels, which_numbers=which_numbers, x=x, device=prediction.device)
    _draw_weighted_stacked_hist(
        ax[1],
        predictions=transformed_prediction,
        target_map=ctx.target_map,
        event_weights=event_weights,
        targets=ctx.targets,
        x_hist=x_hist,
    )

    _draw_asimov_bars(ax[2], edges=edges, binned_sig=binned_sig, total_sig=total_sig)
    fig.suptitle(f"Step: {ctx.global_step}")
    return fig, ax
