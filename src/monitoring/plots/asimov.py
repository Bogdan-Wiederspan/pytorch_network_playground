import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

from ..metrics.physics.asimov import (
    asimov_metric,
    asimov_no_background_metric,
    asimov_small_signal_and_no_background_metric,
)
from ..register import register_plot
from ..utils.tensor import prepare_tensor

_map = {
    "small_signal" : asimov_small_signal_and_no_background_metric,
    "approximation" : asimov_no_background_metric,
    "full" : asimov_metric,
}

@register_plot(
    "asimov",
    requires=("binning_edges" , "s_hist", "b_hist")
    )
def plot_asimov_per_bin(
    ctx,
    which_asimov="approximation",
    **kwargs,
) -> tuple[Figure, Axes]:
    s = ctx.get("s_hist")
    b = ctx.get("b_hist")
    binning_edges = ctx.get("binning_edges")

    # Attention, values need to be tensors, since fn is using torch functions mostly

    _label_map = {
        "small_signal" : "Asimov Significance (s << b & w/o uncertainty)",
        "approximation" : "Asimov Significance (w/o uncertainty)",
        "full" : "Asimov Significance (full)"
    }

    fn = _map[which_asimov]
    score = fn(s=s,b=b, **kwargs)
    # setting nan to 0
    nan_mask = ~torch.isnan(score)
    score = torch.where(nan_mask, score, torch.zeros_like(score))

    score, binning_edges = prepare_tensor(score, binning_edges, device="cpu")
    total_asimov = np.sum(score**2)**0.5

    # plot the score as bar plot and add as ticks the edge values
    fig, axes = plt.subplots(1, 1, figsize=(8 * 1, 8 * 1))

    # first bar plot is with equal distant bars
    # second bar plot is with width equal to bin width
    bin_width = 1
    x = np.arange(len(score)) + bin_width / 2 # shift by half bin to be between edges
    axes.bar(x, height=score, width=1., edgecolor="black", facecolor="orange", linewidth=1.5)
    axes.set_xticks(np.arange(len(binning_edges)))
    axes.set_xticklabels([f"{float(edge):.5f}" for edge in binning_edges])

    fig.suptitle(f"Total Asimov $\\sqrt{{\\sum A^2}}$: {total_asimov:.2f}")
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.15)

    axes.tick_params(axis="x", labelrotation=90)
    axes.set_xlabel("Bin Edges")
    axes.set_ylabel(f"{_label_map[which_asimov]}")
    axes.grid()
    return fig, axes

# @register_plot("asimov", requires=("binning_edges"))
# def asimov_per_bin(
#     ctx,
#     which_asimov="approximation",
# ) -> tuple[Figure, Axes]:
#     binning_edges = ctx.data["binning_edges"]
#     s, b = build_asimov_inputs(ctx, binning_edges)
#     # pick correct asimov function

    # # Attention, values need to be tensors, since fn is using torch functions mostly
    # with torch.no_grad():
    #     possible_values=("small_signal_approximation", "approximation", "full")
    #     if which_asimov in possible_values:
    #         raise ValueError(f"type can only be: {possible_values}")

    #     _map = {
    #         "small_signal" : "asimov_small_signal_approximation",
    #         "approximation" : "asimov_no_uncertainty",
    #         "full" : "asimov_full"
    #     }

    #     _label_map = {
    #         "small_signal" : "Asimov Significance (s << b & w/o uncertainty)",
    #         "approximation" : "Asimov Significance (w/o uncertainty)",
    #         "full" : "Asimov Significance (full)"
    #     }

    #     fn = getattr(loss.loss_functions.SignalEfficiency, _map[which_asimov])

    #     # fill the histogram and calculate the score
    #     hh_node = find_events_of_node(y_true=y_true, y_pred=y_pred, target_map=target_map, node_name = "hh")
    #     hh_node["background"] = torch.concatenate([hh_node["dy"], hh_node["tt"]], axis=0)

    #     binning_edges = binning_edges.flatten()
    #     s_hist, _ = torch.histogram(hh_node["hh"], bins=binning_edges)
    #     b_hist, _ = torch.histogram(hh_node["background"], bins=binning_edges)

    #     score = fn(s=s_hist,b=b_hist)
    #     # setting nan to 0
    #     nan_mask = ~torch.isnan(score)
    #     score = torch.where(nan_mask, score, torch.zeros_like(score))

    # score, binning_edges = prepare_tensor(score, binning_edges, device="cpu")
    # total_asimov = np.sum(score**2)**0.5

    # # plot the score as bar plot and add as ticks the edge values
    # fig, axes = plt.subplots(1, 1, figsize=(8 * 1, 8 * 1))

    # # first bar plot is with equal distant bars
    # # second bar plot is with width equal to bin width
    # bin_width = 1
    # x = np.arange(len(score)) + bin_width / 2 # shift by half bin to be between edges
    # axes.bar(x, height=score, width=1., edgecolor="black", facecolor="orange", linewidth=1.5)
    # axes.set_xticks(np.arange(len(binning_edges)))
    # axes.set_xticklabels([f"{float(edge):.5f}" for edge in binning_edges])

    # fig.suptitle(f"Total Asimov $\\sqrt{{\\sum A^2}}$: {total_asimov:.2f} Iteration: {current_iteration}")
    # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.15)

    # axes.tick_params(axis="x", labelrotation=90)
    # axes.set_xlabel("Bin Edges")
    # axes.set_ylabel(f"{_label_map[which_asimov]}")
    # axes.grid()
    # return fig, axes
