import matplotlib.pyplot as plt
import torch

from ..register import register_plot


@register_plot(
    "active_kernels",
    requires=(
        "binning_fn",
        "kernels",
        )
)
def plot_kernel_distribution(
    ctx,
    which_numbers=None,
    _range=None,
    prediction=None,
    binning_fn=None):
    kernels = ctx.get("kernels")
    prediction = ctx.predictions
    binning_fn = ctx.get("binning_fn")

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
