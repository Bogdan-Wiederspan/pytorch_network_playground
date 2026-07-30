import matplotlib.pyplot as plt
import torch

from ..register import register_plot
from monitoring.utils.plotting import add_number_legend

@register_plot(
    "active_kernels",
    requires=(
        "evaluation_state.binning.binning_fn",
        "evaluation_state.binning.kernels",
        )
)
def plot_kernel_distribution(
    ctx,
    which_numbers=None,
    _range=None,
    prediction=None,
    binning_fn=None):
    kernels = ctx.get("evaluation_state.binning.kernels")
    prediction = ctx.predictions
    binning_fn = ctx.get("evaluation_state.binning.kernels")

    # prepare kernels
    num_kernels = len(kernels)
    # all kernels are chosen
    if which_numbers is None:
        which_numbers = range(num_kernels)

    # resolve negative values
    which_numbers = [
        (i if i >= 0 else num_kernels + i)
        for i in which_numbers
    ]
    # remove duplicates and sort
    which_numbers = tuple(set(which_numbers))

    # pick range of plot by get highest number, assuming kernels are ordered right.
    # get highest values from chosen kernels
    if _range is None:
        lowest_edge = min([kernels[num].lower_edge for num in which_numbers])
        highest_edge = max([kernels[num].upper_edge for num in which_numbers])

        # small offset 5%
        lowest_edge = lowest_edge * 1.1
        highest_edge = highest_edge * 1.1
    else:
        lowest_edge, highest_edge = _range

    # do actual plot, be careful about device placement
    x = torch.linspace(
        start=lowest_edge,
        end=highest_edge,
        steps=3000
        )

    x_hist = torch.linspace(
        start=lowest_edge,
        end=highest_edge,
        steps=100
        )

    # 2 plots staying on top of each, first showing kernels, second showing prediction
    fig, ax = plt.subplots(2,1)

    # transform the prediction into the space of the bins
    prediction = binning_fn.forward(prediction)

    # run the kernels
    # kernels live on gpu/cpu
    for kernel_num in which_numbers:
        active_kernel = kernels[kernel_num]
        y = active_kernel(x.to(prediction.device))
        ax[0].plot(x ,y )

    background = torch.concatenate(
        [
            prediction[:,1],
            prediction[:,2]
        ],
    )

    prediction = prediction
    background = background

    _ = ax[1].hist(prediction[:,0], bins=x_hist, histtype="step", label="signal")
    _ = ax[1].hist(background, bins=x_hist, histtype="step", label="background")
    ax[1].legend()
    fig.suptitle(f"Step: {ctx.global_step}")
    return fig, ax


@register_plot(
    "active_kernels_advance",
    requires=(
        "evaluation_state.binning.binning_fn",
        "evaluation_state.binning.kernels",
        "evaluation_state.binning.active_edges",
        "monitored_tensor.signal_yield",
        "monitored_tensor.background_yield",
        "monitored_gradient.weighted_dnn_score_bin_*",
        "monitored_tensor.binned_significance"
        )
)
def plot_kernel_distribution_advance(
    ctx,
    which_numbers=None,
    _range=None,
    prediction=None,
    binning_fn=None):

    prediction = ctx.predictions
    kernels = ctx.get("evaluation_state.binning.kernels")
    binning_fn = ctx.get("evaluation_state.binning.binning_fn")
    edges = ctx.get("evaluation_state.binning.active_edges")
    bin_gradients = {ctx.get(_bin) for _bin in sorted(ctx.expand("monitored_gradient.weighted_dnn_score_bin_*"))}
    num_kernels = len(kernels)
    s_yield = ctx.get("monitored_tensor.signal_yield")
    b_yield = ctx.get("monitored_tensor.background_yield")
    binned_sig = ctx.get("monitored_tensor.binned_significance")
    total_sig = torch.sqrt(torch.sum(binned_sig**2))
    event_weights = ctx.event_weights

    target_map = ctx.target_map.copy()
    sig_idx = target_map.pop("hh")
    background_idx = list(target_map.values())
    signal_node_prediction = prediction[:, sig_idx]

    targets = ctx.targets
    signal_mask = targets[:, 0] == 1
    bg_mask = ~signal_mask

    # transform the prediction into the space of the bins
    transformed_prediction = binning_fn.forward(signal_node_prediction)

    # prepare kernels

    # all kernels are chosen
    if which_numbers is None:
        which_numbers = range(num_kernels)

    # resolve negative values
    which_numbers = [
        (i if i >= 0 else num_kernels + i)
        for i in which_numbers
    ]
    # remove duplicates and sort
    which_numbers = tuple(set(which_numbers))

    # pick range of plot by get highest number, assuming kernels are ordered right.
    # get highest values from chosen kernels
    if _range is None:
        lowest_edge = min([kernels[num].lower_edge for num in which_numbers])
        highest_edge = max([kernels[num].upper_edge for num in which_numbers])

        # small offset 5%
        lowest_edge = lowest_edge * 1.1
        highest_edge = highest_edge * 1.1
    else:
        lowest_edge, highest_edge = _range

    # do actual plot, be careful about device placement
    x = torch.linspace(
        start=lowest_edge,
        end=highest_edge,
        steps=3000
        )

    x_hist = torch.linspace(
        start=lowest_edge,
        end=highest_edge,
        steps=100
        )

    # 2 plots staying on top of each, first showing kernels, second showing prediction
    fig, ax = plt.subplots(3,1,sharex=True)

    # run the kernels
    # kernels live on gpu/cpu
    for kernel_num in which_numbers:
        active_kernel = kernels[kernel_num]
        y = active_kernel(x.to(transformed_prediction.device))
        ax[0].plot(x ,y )



    # --- plot predictions
    _ = ax[1].hist(
        transformed_prediction[signal_mask],
        bins=x_hist,
        histtype="step",
        label="signal",
        color="blue",
        weights=event_weights[signal_mask],
        )
    _ = ax[1].hist(
        transformed_prediction[bg_mask],
        bins=x_hist,
        histtype="step",
        label="background",
        color="orange",
        weights=event_weights[bg_mask],
        )
    ax[1].set_yscale("log")
    lines, labels = add_number_legend(ax[1], "Weighted Prediction")
    ax[1].legend(lines, labels)

    # --- plot yield
    bin_width = edges[1:] - edges[:-1]
    bottom = torch.zeros_like(s_yield)
    # ax[2].bar(x=edges[:-1], height=b_yield, width=bin_width, bottom=None, label="Background Yield", color="orange")
    # bottom += b_yield
    # ax[2].bar(x=edges[:-1], height=s_yield, width=bin_width, bottom=bottom, label="Signal Yield", color="blue")
    bin_centers= (edges[:-1] + edges[1:]) / 2
    ax[2].bar(
        x=bin_centers,
        height=binned_sig,
        width=bin_width,
        bottom=bottom,
        label=r"Asimov per Bin",
        edgecolor="black", fill=False
        )
    lines, labels = add_number_legend(ax[2], r"$\sqrt{\sum{A_{i}^{2}}}$" + str(round(total_sig.item(),5)))
    ax[2].set_yscale("log")
    ax[2].legend(lines, labels)

    # --- plot mean gradients of bin
    # from IPython import embed; embed(header="MESSAGE Line 107 | File: /afs/desy.de/user/w/wiedersb/xxl/pytorch_network_playground/src/monitoring/plots/kernels.py")

    # debug_weight = ctx.get("gradient_monitor").debug
    # debug_weight = []

    # mean_gradients = {}
    # for bin_num in sorted(gradients.keys()):
    #     # only signal produces an gradient
    #     gradient_of_bin = gradients[bin_num][:,0]
    #     mean_gradients[bin_num] = gradient_of_bin.mean()

    # mean_gradients_signal = {}
    # for bin_num in sorted(gradients.keys()):
    #     # only signal produces an gradient
    #     gradient_of_bin = gradients[bin_num][:,0][signal_mask]
    #     mean_gradients_signal[bin_num] = gradient_of_bin.mean()

    # mean_gradients_bg = {}
    # for bin_num in sorted(gradients.keys()):
    #     # only signal produces an gradient
    #     gradient_of_bin = gradients[bin_num][:,0][bg_mask]
    #     mean_gradients_bg[bin_num] = gradient_of_bin.mean()



    # ax[2].bar(edges, mean_gradients, width=bin_width)
    fig.suptitle(f"Step: {ctx.global_step}")
    return fig, ax
