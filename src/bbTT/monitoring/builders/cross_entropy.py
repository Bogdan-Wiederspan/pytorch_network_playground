from __future__ import annotations

import torch

from bbTT.monitoring.context.eval_context import EvalContext
from bbTT.monitoring.register import register_builder

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

    weights = weights.reshape(per_event.shape).to("cpu")
    return torch.sum(weights * per_event) / torch.sum(weights)

@register_builder("cross_entropy", provides={"cross_entropy",}, requires=None)
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
    cross_entropy = cross_entropy_from_probabilities(
        probabilities=ctx.predictions,
        targets=ctx.targets,
        weights=ctx.weights("composition"),

    )
    return {"cross_entropy": cross_entropy}

@register_builder("cross_entropy_per_class", provides={"cross_entropy_per_class"}, requires=None)
def build_cross_entropy_per_class(ctx: EvalContext, **kwargs) -> dict[str, dict[str, torch.Tensor]]:
    """
    Cross entropy separately for the events of every true class, composition weighted like the total.

    Shows which separation the training is currently working on: the total cross entropy is the
    per-class values weighted by the class shares, so one class at 0.05 and two at 0.5 average to
    the same number as three at 0.35.

    Args:
        ctx (EvalContext): Evaluation context with predictions, targets, target_map and composition_weights.

    Returns:
        dict[str, dict[str, torch.Tensor]]: {"cross_entropy_per_class": {class name: scalar}}.
    """
    weights = ctx.weights("composition")
    per_class = {}
    for name, idx in ctx.target_map.items():
        mask = ctx.targets[:, idx] == 1
        per_class[name] = cross_entropy_from_probabilities(
            probabilities=ctx.predictions[mask],
            targets=ctx.targets[mask],
            weights=weights[mask],
        )
    return {"cross_entropy_per_class": per_class}
