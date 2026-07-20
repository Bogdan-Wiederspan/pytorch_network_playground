import torch

from ...register import register_builder


@register_builder(
    "kernel_state",
    provides={"kernels", "binning_fn", "active_edges", "original_edges"},
    requires={"evaluation_state.binning"}
    )
def build_kernel_state(ctx, **kwargs):
    state = ctx.get("evaluation_state.binning")

    return {
        "kernels": state["kernels"],
        "binning_fn": state["binning_fn"],
        "active_edges": state["active_edges"],
        "original_edges": state["original_edges"],
        }
