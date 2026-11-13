from __future__ import annotations

import torch

from bbTT.monitoring.context.eval_context import EvalContext

# lower clamp for probabilities, prevents log(0) for a confidently wrong prediction
PROBABILITY_EPSILON = 1e-7

def cross_entropy_from_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cross entropy of class probabilities, as a weighted mean over events.
    The Monitoring head always return probabilities, thus pytorch internal functions can't be used.

    Args:
        probabilities (torch.Tensor): Class probabilities of shape (num_events, num_classes).
        targets (torch.Tensor): One-hot targets of shape (num_events, num_classes).
        weights (torch.Tensor | None, optional): Per-event weights, any shape with num_events entries. Defaults to None.

    Returns:
        torch.Tensor: Scalar cross entropy.
    """
    log_probabilities = torch.log(probabilities.clamp_min(PROBABILITY_EPSILON)) # preventing log(0)
    per_event = -torch.sum(targets * log_probabilities, dim=-1)

    if weights is None:
        return torch.mean(per_event)

    weights = weights.reshape(per_event.shape)
    return torch.sum(weights * per_event) / torch.sum(weights)

def build_cross_entropy(ctx: EvalContext, **kwargs) -> dict[str, torch.Tensor]:
    """
    Cross entropy as it is expected in a training batch, independent of the loss that is trained.

    Uses the composition weights of the context so a full pass reproduces the batch composition,
    see Process.per_event_composition_weight. For a sampled batch these are None and the plain mean
    is used. Works on the probabilities the context always holds, not on logits.

    Args:
        ctx (EvalContext): Evaluation context with predictions, targets and optional composition_weights.

    Returns:
        dict[str, torch.Tensor]: {"cross_entropy": scalar}.
    """
    composition_weights = ctx.get("composition_weights") if ctx.has("composition_weights") else None
    cross_entropy = cross_entropy_from_probabilities(
        probabilities=ctx.predictions,
        targets=ctx.targets,
        weights=composition_weights,
    )
    return {"cross_entropy": cross_entropy}
