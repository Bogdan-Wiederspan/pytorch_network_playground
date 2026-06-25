import matplotlib.pyplot as plt
import numpy as np

from ..register import register_plot
from ..utils.plotting import append_text_to_legend

from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

@register_plot("output_score", requires=None)
def plot_network_predictions(
    ctx,
    normalize=True,
    single_legend=False,
    **kwargs
    ) -> tuple[Figure, Axes]:
    # create a figure with subplots for each node, where the score is shown
    # nodes are defined over target_map order

    y_true = ctx.predictions
    y_pred = ctx.targets
    target_map = ctx.target_map

    fig, axes = plt.subplots(1,len(target_map), figsize=(8 * len(target_map), 8))
    fig.suptitle(kwargs.pop("title", None))
    weight = None
    # get events that are predicted correctly for each class
    for node, node_idx in target_map.items():
        for data_cls, data_idx in target_map.items():
            y_label = "frequency"

            if not normalize:
                axes[node_idx].set_yscale("log")
                axes[node_idx].set_ylim(top=len(y_pred))
            else:
                axes[node_idx].set_yscale("linear")
                axes[node_idx].set_ylim(top=1.)
                y_label += " normalized"

            axes[node_idx].set_xlabel(f"{node} node", )
            axes[node_idx].set_ylabel(y_label)
            axes[node_idx].grid()


            # get events of specific cls (e.g. hh)
            correct_cls_mask = y_true[:, data_idx] == 1
            # get predictions for cls
            filtered_predictions = y_pred[correct_cls_mask][:, node_idx]

            if normalize:
                weight = np.full(filtered_predictions.shape, 1 / len(filtered_predictions))

            _ = axes[node_idx].hist(
                filtered_predictions,
                bins=kwargs.get("bins", 20),
                histtype=kwargs.get("histtype", "step"),
                alpha=kwargs.get("alpha", 0.7),
                label=data_cls,
                weights=weight,
                **kwargs,
        )
        if not single_legend:
            axes[node_idx].legend()
    if single_legend:
        lines_labels = [fig.axes[0].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels)
    return fig, axes


@register_plot("output_score_hh_node", requires={"binning_edges"})
def plot_network_predictions_hh(
    ctx,
    normalize=True,
    **kwargs
    ) -> tuple[Figure, Axes]:

    y_true = ctx.targets
    y_pred = ctx.predictions

    target_map = ctx.target_map
    binning_edges = ctx.get("binning_edges").flatten()

    # create a figure with subplots for hh node in normal and log
    signal_idx = target_map["hh"]
    num_plots = 2
    fig, axes = plt.subplots(1, num_plots, figsize=(12 * num_plots, 12))
    fig.suptitle(kwargs.pop("title", None))
    # identify events uses TRUTH information to create masks
    masks = {process: (y_true[:, idx] == 1) for process, idx in target_map.items()}
    # to get node information apply index filtering on PREDICTION
    hh_node = {process: y_pred[masks[process]][:, signal_idx] for process, idx in target_map.items()}
    hh_node["background"] = np.concatenate([hh_node["dy"], hh_node["tt"]], axis=0)

    # number of events are not evenly distributed outside of sampler
    # normalize is a factor that also shows up in the legend
    # by default weight is set to 1
    weights = {process: None for process in target_map}
    if normalize:
        weights = {process: np.full(value.shape, 1 / len(value)) for process, value in hh_node.items()}

    # plotting
    # first for the 2 plots with combined background,
    # last for the one with separated background
    plt_cfg = {
        "histtype" : kwargs.get("histtype", "step"),
        "alpha" : kwargs.get("alpha", 0.7),
        "bins" : binning_edges, # needs to be given from outside, since this is a dynamic variable
    }

    _ = axes[0].hist(hh_node["hh"], label="signal", weights=weights["hh"], hatch="/", **plt_cfg)
    _ = axes[0].hist(hh_node["background"], label="background", weights=weights["background"], hatch="\\", **plt_cfg)
    _ = axes[1].hist(hh_node["hh"], label="hh", weights=weights["hh"], hatch="/", **plt_cfg)
    _ = axes[1].hist( hh_node["dy"],label="dy",weights=weights["dy"],hatch="\\",**plt_cfg)
    _ = axes[1].hist(hh_node["tt"], label="tt", weights=weights["tt"], hatch="*", **plt_cfg)

    # apply extra setting per axis
    additional_legend_infos = (
        # f"batch: {current_iteration}",
        "\n".join([f"{process}: {len(hh_node[process])}" for process in target_map.keys()])
    )
    for ax in axes:
        # legend, adding iteration number and weigh factor
        append_text_to_legend(ax, additional_legend_infos)
        ax.set_xlabel("HH node", size=25)
        ax.set_ylabel("frequency", size=25)
        ax.set_ylim((-0.1,1.1))
        ax.set_xlim((-0.1,1.1))
        ax.grid()
    return fig, axes
