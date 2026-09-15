import torch

from bbTT.monitoring.register import register_builder


@register_builder("score_bins", provides={"evaluation_state.binning_edges",}, requires=None)
def build_binning_edges(ctx, **kwargs):
    """
    Provide bin edges for output-score histograms.

    Always succeeds: uses the model's edges if a binning layer exists,
    otherwise falls back to equal-width edges from ctx.default_n_bins.

    This artifact is safe for any plot that just needs edges to histogram
    into — it carries no information about kernels/binning_fn.
    """
    raw_state = ctx.evaluation_state.get("binning") # None if no binning layer exist

    if raw_state is None:
        edges = torch.linspace(0, 1, steps = ctx.optional_defaults.get("default_n_bins") + 1)
    else:
        edges = raw_state["active_edges"]
    return {"evaluation_state.binning_edges": edges}
