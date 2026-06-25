import torch

from ...register import register_builder


@register_builder("asimov", provides={"s_hist", "b_hist"}, requires={"binning_edges"})
def build_asimov_inputs(ctx, **kwargs):
    # TODO multi signal not implemented yet!
    # prediction is used to determine in which bin a truth value would land
    # ex. bins are [0,0.3,1] and prediction said 0, while truth is hh -> first bin signal count goes up by 1.
    y_true = ctx.targets
    y_pred = ctx.predictions
    target_map = ctx.target_map
    binning_edges = ctx.get("binning_edges")

    signal_idx = target_map["hh"]
    # mask that says: you belong to this process
    event_classification_masks = {
        name: (y_true[: , idx] == 1)
        for name, idx in target_map.items()
    }

    # score for each process, but onl the signal node
    scores = {
        name: y_pred[event_classification_masks[name]][:, signal_idx]
        for name in target_map
    }

    # calculate s and b in signal node
    s, _ = torch.histogram(
        scores["hh"],
        bins=binning_edges
    )
    # combine b scores
    b = (
        torch.histogram(scores["dy"], bins=binning_edges)[0] +
        torch.histogram(scores["tt"], bins=binning_edges)[0]
    )
    return {"s_hist": s, "b_hist": b}
