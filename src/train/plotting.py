from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import loss.loss_functions

PLOT_REGISTRY = {}

def register_plot(name):
    def wrapper(cls):
        PLOT_REGISTRY[name] = cls
        return cls
    return wrapper

def prepare_tensor(*tensor, device="cpu"):
    return [t.detach().to(device).numpy() for t in tensor]

def add_number_legend(ax, string):
    lines, labels = ax.get_legend_handles_labels()
    dummy_line = plt.Line2D([], [], linestyle="", marker="")
    lines.append(dummy_line)
    labels.append(string)
    return lines, labels

def append_text_to_legend(ax, information):
    lines, labels = ax.get_legend_handles_labels()
    dummy_line = plt.Line2D([], [], linestyle="", marker="")
    if isinstance(information, (list ,tuple)):
        information = (information, )

    for info in information:
        lines.append(dummy_line)
        labels.append(info)
    # manipulate given ax
    ax.legend(lines, labels)
    return ax

def find_events_of_node(y_true, y_pred, target_map, node_name="hh"):
    signal_idx = target_map[node_name]
    # identify events uses TRUTH information to create masks
    masks = {process: (y_true[:, idx] == 1) for process, idx in target_map.items()}
    # to get node information apply index filtering on PREDICTION
    picked_node = {process: y_pred[masks[process]][:, signal_idx] for process, idx in target_map.items()}
    return picked_node


def to_numpy(values):
    if torch.is_tensor(values):
        values = values.numpy()
    elif isinstance(values, (tuple, list)):
        values = np.array(values)
    return values


@register_plot("edges")
def plot_edges_number_line(values, show=False):

    values = to_numpy(values)

    fig, ax = plt.subplots(figsize=(8, 1.5))

    # All points at y=0
    ax.scatter(values, np.zeros_like(values), s=50, marker="|", )

    # Hide y-axis
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Put x-axis slightly below the points
    ax.spines["bottom"].set_position(("data", -0.05))
    ax.set_ylim(-0.1, 0.1)
    if show:
        plt.show()
    return fig, ax

@register_plot("dummy")
def dummy_figure() -> tuple[plt.Figures, plt.Axes]:
    """
    Dummy figure that should be used as placeholder

    Returns:
        tuple(plt.Figure, plt.Axes):
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.text((0.5, 0.5), "Not existing", fontsize="large", color="red")
    return fig, ax

# TODO plotting input features is currently not used, but can be used for control plots to check the input data distribution.
# def plot_input_features(data_map, columns):
#     all_data = ak.concatenate([x.data for x in data_map])
#     columns = list(map(Route, columns))
#     # layout plots
#     num_features = len(columns)
#     num_cols = 4
#     num_row = int(np.ceil(num_features / num_cols))
#     fig_size = (5 * num_cols, 4 * num_row) # wide, tall

#     fig, axes = plt.subplots(nrows=num_row, ncols=num_cols, figsize=fig_size)
#     for ax, _route in zip(axes.flatten(), columns):
#         data = _route.apply(all_data).to_numpy().astype(np.float32)
#         # mask NaNs
#         empty_mask = data == utils.EMPTY_FLOAT

#         ax.set_xlabel(_route.column)
#         ax.set_ylabel("frequency")
#         ax.set_yscale("log")

#         # get lowest value without empty values and add offset
#         _bin = 20
#         bins = np.linspace(
#             np.min(data[~empty_mask]),
#             data.max(),
#             _bin,
#         )
#         # set offset to 3 bins to display underflow, clip data to lower edge and preserve bin width
#         lower_edge = bins[0] - 3 * (bins[1] - bins[0])
#         bins = np.linspace(lower_edge, bins[-1], _bin + 3)

#         # plot without empty values, set empty values to underflow bin
#         _ = ax.hist(np.clip(data, a_min=lower_edge, a_max=None), bins=bins)
#     return fig, axes

# def network_binary_predictions(y_true, y_pred, target_map, normalize=True, single_legend=False, **kwargs):
#     # create a figure with subplots for each node, where the score is shown
#     # nodes are defined over target_map order

#     fig, axes = plt.subplots(1,1, figsize=(8, 8))
#     fig.suptitle(kwargs.pop("title", None))
#     weight = None
#     # get events that are predicted correctly for each class

#     y_label = "frequency"

#     if not normalize:
#         axes[0].set_yscale("log")
#         axes[0].set_ylim(top=len(y_pred))
#     else:
#         axes[0].set_yscale("linear")
#         axes[0].set_ylim(top=1.)
#         y_label += " normalized"

#     axes[0].set_xlabel("Prediction", )
#     axes[0].set_ylabel(y_label)
#     axes[0].grid()

#     from IPython import embed; embed(header = " line: 70 in plotting.py")

#     correct_cls_mask = y_true[:, data_idx] == 1
#     # get predictions for cls
#     filtered_predictions = y_pred[correct_cls_mask][:, node_idx]

#     if normalize:
#         weight = np.full(filtered_predictions.shape, 1 / len(filtered_predictions))

#     _ = axes[node_idx].hist(
#         filtered_predictions,
#         bins=kwargs.get("bins", 20),
#         histtype=kwargs.get("histtype", "step"),
#         alpha=kwargs.get("alpha", 0.7),
#         label=data_cls,
#         weights=weight,
#         **kwargs,
# )
#         if not single_legend:
#             axes[node_idx].legend()
#     if single_legend:
#         lines_labels = [fig.axes[0].get_legend_handles_labels()]
#         lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#         fig.legend(lines, labels)
#     return fig, axes


def network_predictions(
    y_true,
    y_pred,
    target_map,
    normalize=True,
    single_legend=False,
    **kwargs
    ) -> tuple[plt.Figures, plt.Axes]:
    # create a figure with subplots for each node, where the score is shown
    # nodes are defined over target_map order

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

class BinningTimeSeries():
    def __init__(self):
        self.bin_edges_per_iteration = {} # iteration: edges

    def __setitem__(self, key, value):
        self.bin_edges_per_iteration[key] = value

    def plot(self):
        # xlabel="Batch Iteration"
        # ylabel="Bin Population"

        bin_width = 1.0
        bottom = np.zeros(len(self.bin_edges_per_iteration.keys()))
        # x = np.array([int(key) for key in self.bin_edges_per_iteration.keys()])
        x = self.bin_edges_per_iteration.keys()
        # need to invert data from iteration: edges -> bins : edges
        y = np.array(self.bin_edges_per_iteration.values()).transpose()
        bins = {_x:_y for _x,_y in zip(x,y)}
        fig, ax = plt.subplots()


    #     bottom = np.zeros(5)
    # ...:         bins = {_x:_y for _x,_y in zip(x,y)}
    # ...:         fig, ax = plt.subplots()
    # ...:
    # ...:         for iteration, bin_value in bins.items():
    # ...:             p = ax.bar(x, bin_value, 50, bottom=bottom)
    # ...:             bottom = bin_value


        for _, bin_value in bins.items():
            p = ax.bar(x, bin_value, bin_width, bottom=bottom)
            bottom += bin_value

            ax.bar_label(p, label_type='center')

        ax.set_title('Number of penguins by sex')
        ax.legend()

        plt.show()

def visualize_bins(kernels=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # when no kernels exist simply return empty fig
    if kernels is None:
        return dummy_figure()
    x = torch.linspace(-0.2,1.2,200)

    for kernel in kernels:
        y = kernel(x)
        ax.plot(x,y)

    ax.set_xlabel("Kernel input", size=25)
    ax.set_ylabel("Kernel output", size=25)
    ax.grid()
    return fig, ax

@register_plot("1D_hh_node")
def network_predictions_hh(
    y_true,
    y_pred,
    target_map,
    binning_edges,
    normalize,
    current_iteration,
    **kwargs
    ) -> tuple[plt.Figures, plt.Axes]:
    # create a figure with subplots for hh node in normal and log
    signal_idx = target_map["hh"]
    binning_edges = binning_edges.flatten()
    num_plots = 2
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8))
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
        f"batch: {current_iteration}",
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


@register_plot("confusion_matrix")
def confusion_matrix(
    y_true,
    y_pred,
    target_map,
    sample_weight=None,
    normalized="true",
    cmap="Blues",
    **kwargs
    ) -> tuple[plt.Figures, plt.Axes]:
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
    # convert prediction to class indices
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    y_true = torch.argmax(y_true, dim=1).cpu()

    cm = sklearn.metrics.confusion_matrix(
        y_true,
        y_pred,
        labels=list(target_map.values()),
        sample_weight=sample_weight,
        normalize=normalized,  # normalize to get probabilities
    )
    disp = sklearn.metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=list(target_map.keys()),
    )
    disp.plot(cmap=cmap)
    disp.figure_.suptitle(kwargs.pop("title", None))

    return disp.figure_, disp.ax_, disp.confusion_matrix


@register_plot("roc")
def roc_curve(
    target,
    pred,
    sample_weight=None,
    labels=None,
    **kwargs
    ) -> tuple[plt.Figures, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.get_cmap("tab10").colors if kwargs.get("colors") is None else kwargs["colors"]

    if sample_weight is None:
        sample_weight = torch.ones(target.shape[0])

    for c, name in enumerate(labels):
        col = colors[c % len(colors)]

        disp = sklearn.metrics.RocCurveDisplay.from_predictions(
            target[:, c],
            pred[:, c],
            sample_weight=sample_weight.reshape(-1, 1),
            ax=ax,
            name=name,
            # curve_kwargs={"color": col},
            color=col,
            )
    _ = ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    return disp.figure_, disp.ax_


def plot_2D(x, y, bins=10, **kwargs):
    fig, ax = plt.subplots()
    hist = ax.hist2d(
        x,
        y,
        bins=bins,
        cmap=kwargs.get("cmap"),
    )
    ax.set_xlabel(kwargs.get("xlabel"))
    ax.set_xlabel(kwargs.get("ylabel"))
    fig.colorbar(hist[3], ax=ax)
    ax.set_title(kwargs.get("title"))
    if kwargs.get("savepath"):
        fig.savefig(kwargs.get("savepath"))
    return fig, ax

def plot_1D(x, annotations=None, **kwargs):
    fig, ax = plt.subplots()
    _ = ax.hist(
        x.item(),
        bins=kwargs.get("bins"),
        histtype="stepfilled",
        alpha=0.7,
        color=kwargs.get("color"),
        label=kwargs.get("label"),
        density=kwargs.get("density"),
    )
    ax.set_xlabel(kwargs.get("xlabel"))
    ax.set_xlabel(kwargs.get("ylabel"))
    ax.set_title(kwargs.get("title"))
    if kwargs.get("savepath"):
        fig.savefig(kwargs.get("savepath"))
    ax.legend()

    if annotations:
        loss = annotations.get("loss")
        target = annotations.get("target")
        pred = annotations.get("pred")
        num_0s = np.sum(target == 0)
        num_1s = np.sum(target == 1)

        ax.annotate(f"num: 0s: {num_0s:.2f}", (0.5, 0.70), xycoords="axes fraction")
        ax.annotate(f"num: 1s: {num_1s:.2f}", (0.5, 0.65), xycoords="axes fraction")
        ax.annotate(f"loss: {loss:.2f}", (0.5, 0.60), xycoords="axes fraction")
        iteration = annotations.get("iteration")
        ax.annotate(f"iteration {iteration}", (0.5, 0.55))

        tp = np.sum((target == 1) & (pred > 0.5))
        tn = np.sum((target == 0) & (pred < 0.5))
        fp = np.sum((target == 0) & (pred > 0.5))
        fn = np.sum((target == 1) & (pred < 0.5))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)

        ax.annotate(f"accuracy: {accuracy:.2f}", (0.5, 0.50), xycoords="axes fraction")
        ax.annotate(f"sensitivity: {sensitivity:.2f}", (0.5, 0.45), xycoords="axes fraction")

    return fig, ax


def plot_batch(self, input, target, loss, iteration, target_map=None, labels=None):
    input_per_feature = input.to("cpu").transpose(0, 1).detach().numpy()
    input, target = input.to("cpu").detach().numpy(), target.to("cpu").detach().numpy()

    fig, ax = plt.subplots(1, len(self.categorical_inputs), figsize=(8 * len(self.categorical_inputs), 8))
    fig.tight_layout()

    for ind, cat in enumerate(self.categorical_inputs):
        signal_target = target[:, self.categorical_target_map["hh"]]

        background_mask, signal_mask = signal_target.flatten() == 0, signal_target.flatten() == 1
        # background_prediction = detach_pred[zero_s_mask]
        # signal_prediction = detach_pred[zero_s_mask]
        _input = input_per_feature[ind]
        background_input = _input[background_mask]
        signal_input = _input[signal_mask]
        cax = ax[ind]

        cax.hist(
            [background_input, signal_input],
            bins=10,
            histtype="barstacked",
            alpha=0.5,
            label=["tt & dy", "hh"],
            density=True,
        )
        cax.set_xlabel(cat)
        cax.annotate(f"loss: {loss:.2f}", (0.8, 0.60), xycoords="figure fraction")
        cax.annotate(f"iteration {iteration}", (0.75, 0.55))
        cax.legend()
    fig.savefig(f"1d_cats_{iteration}.png")

@register_plot("asimov")
def asimov_per_bin(
    y_true,
    y_pred,
    target_map,
    binning_edges,
    which_asimov="approximation",
    current_iteration=None,
) -> tuple[plt.Figures, plt.Axes]:
    # pick correct asimov function

    # Attention, values need to be tensors, since fn is using torch functions mostly
    with torch.no_grad():
        possible_values=("small_signal_approximation", "approximation", "full")
        if which_asimov in possible_values:
            raise ValueError(f"type can only be: {possible_values}")

        _map = {
            "small_signal" : "asimov_small_signal_approximation",
            "approximation" : "asimov_no_uncertainty",
            "full" : "asimov_full"
        }

        _label_map = {
            "small_signal" : "Asimov Significance (s << b & w/o uncertainty)",
            "approximation" : "Asimov Significance (w/o uncertainty)",
            "full" : "Asimov Significance (full)"
        }

        fn = getattr(loss.loss_functions.SignalEfficiency, _map[which_asimov])

        # fill the histogram and calculate the score
        hh_node = find_events_of_node(y_true=y_true, y_pred=y_pred, target_map=target_map, node_name = "hh")
        hh_node["background"] = torch.concatenate([hh_node["dy"], hh_node["tt"]], axis=0)

        binning_edges = binning_edges.flatten()
        s_hist, _ = torch.histogram(hh_node["hh"], bins=binning_edges)
        b_hist, _ = torch.histogram(hh_node["background"], bins=binning_edges)

        score = fn(s=s_hist,b=b_hist)
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
    # widths = np.diff(binning_edges)
    # axes[1].bar(
    #     binning_edges[:-1],
    #     height=score,
    #     width=widths,
    #     align="edge", # each bin starts at end of previous
    #     edgecolor="black",
    #     facecolor="orange",
    #     linewidth=1.5
    #     )
    # axes[1].set_xticklabels([f"{float(edge):.5f}" for edge in binning_edges])

    fig.suptitle(f"Total Asimov $\\sqrt{{\\sum A^2}}$: {total_asimov:.2f} Iteration: {current_iteration}")
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.15)

    axes.tick_params(axis="x", labelrotation=90)
    axes.set_xlabel("Bin Edges")
    axes.set_ylabel(f"{_label_map[which_asimov]}")
    axes.grid()
    return fig, axes
    #
