from bbTT.monitoring.register import register_builder
from bbTT.monitoring.runner import RequirementNotMet


@register_builder(
    "kernel_state",
    provides={"kernels", "binning_fn", "active_edges", "original_edges"},
    requires={"evaluation_state.binning"}
    )
def build_kernel_state(ctx, **kwargs):
    """
    Provide raw binning-layer internals (kernels, binning_fn, edges).

    Unlike `evaluation_state.binning_edges`, this artifact has NO fallback —
    it only exists for models with an actual binning layer.
    Raises RequirementNotMet when absent so dependent plots are skipped as optional,
    not silently given garbage data.
    """
    raw_state = ctx.get("evaluation_state.binning")
    if raw_state is None:
        raise RequirementNotMet("kernel_state", requester="build_kernel_state")

    return {
        "kernels": raw_state["kernels"],
        "binning_fn": raw_state["binning_fn"],
        "active_edges": raw_state["active_edges"],
        "original_edges": raw_state["original_edges"],
        }
