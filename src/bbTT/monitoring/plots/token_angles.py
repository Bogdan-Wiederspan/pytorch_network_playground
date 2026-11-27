from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from bbTT.monitoring.context import EvalContext
from bbTT.monitoring.register import register_plot

ANGLE_PREFIX = "angle_centered_"
CROSS_GROUP = "cross"


def _flatten_pairs(quantity: dict[str, Any]) -> tuple[dict[tuple[str, str], float], dict[str, list[str]]]:
    """
    Read a pair quantity of CatEmbeddingLayer.create_evaluation_state, grouped or flat.

    Args:
        quantity (dict[str, Any]): {"<a>_vs_<b>": float} or {"<group>": {"<a>_vs_<b>": float}}.

    Returns:
        tuple[dict[tuple[str, str], float], dict[str, list[str]]]: (a, b) -> value, and group -> values of
            the group in first-seen order; a flat quantity has the single group "" with every value.
    """
    grouped = all(isinstance(v, dict) for v in quantity.values())
    groups: dict[str, dict[str, float]] = quantity if grouped else {"": quantity}
    pairs: dict[tuple[str, str], float] = {}
    members: dict[str, list[str]] = {}
    for group, group_pairs in groups.items():
        for pair, value in group_pairs.items():
            a, b = pair.split("_vs_")
            pairs[(a, b)] = value
            if group == CROSS_GROUP:
                continue
            for token in (a, b):
                if token not in members.setdefault(group, []):
                    members[group].append(token)
    # a group with one value has no pair of its own and shows up in cross pairs only
    seen = {t for tokens in members.values() for t in tokens}
    for a, b in pairs:
        for token in (a, b):
            if token not in seen:
                members.setdefault("", []).append(token)
                seen.add(token)
    return pairs, members


def _draw_category(ax: Axes, category: str, quantity: dict[str, Any], norm: TwoSlopeNorm) -> None:
    """
    Draw the lower triangle of the pair-angle matrix of one category, tokens sorted by group.

    Args:
        ax (Axes): Axes to draw on.
        category (str): Category name, used as title.
        quantity (dict[str, Any]): Pair angles, see _flatten_pairs.
        norm (TwoSlopeNorm): Color normalisation shared by all categories.
    """
    pairs, members = _flatten_pairs(quantity)
    tokens = [t for group in members.values() for t in group]
    n = len(tokens)
    matrix = np.full((n, n), np.nan)
    for (a, b), value in pairs.items():
        i, j = tokens.index(a), tokens.index(b)
        matrix[max(i, j), min(i, j)] = value  # lower triangle, row > column

    ax.imshow(matrix, cmap="RdYlBu", norm=norm)
    for i in range(n):
        for j in range(i):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8,
                    color="black" if norm.vmin + 25 < value < norm.vmax - 25 else "white")

    # separate the groups: same-group pairs form blocks along the diagonal
    edge = 0
    for group, group_tokens in list(members.items())[:-1]:
        edge += len(group_tokens)
        ax.axhline(edge - 0.5, color="black", lw=1)
        ax.axvline(edge - 0.5, color="black", lw=1)

    values = np.array(list(pairs.values()))
    ax.set_xticks(range(n)); ax.set_xticklabels(tokens)
    ax.set_yticks(range(n)); ax.set_yticklabels(tokens)
    ax.set_title(f"{category}\nmin {values.min():.0f}°  mean {values.mean():.0f}°  max {values.max():.0f}°", fontsize=9)


@register_plot(
    "token_angles",
    requires=("evaluation_state.monitored_blocks.input_layer",),
    optional=True,
    # same tag prefix as the pair scalars, so the heatmap card sits next to their curves in the dashboard
    tag="MonitoredBlocks/input_layer/angle_centered",
)
def plot_token_angles(ctx: EvalContext, block: str = "input_layer", **kwargs) -> tuple[Figure, Axes]:
    """
    Heatmap of the centered angles between the embedding vectors of each categorical feature.

    One panel per category with three or more values, lower triangle of the token x token matrix, 90° is
    orthogonal (neutral) in the middle of the color scale, small angles (collapsing tokens) are red. Tokens
    are ordered by their value groups and the groups are separated by lines, so the same-group pairs are the
    blocks along the diagonal and the cross-group pairs the rectangle below them.

    The scalars ``MonitoredBlocks/<block>/angle_centered_<category>/...`` show *when* an angle moves, this
    plot shows *which* pair at that step; it is logged under the same prefix so both sit next to each other.
    A variant for another block needs its own ``tag`` and ``block`` in register_plot_variant.

    Args:
        ctx (EvalContext): Must carry the evaluation state of the block holding the embedding layer.
        block (str, optional): Monitored block that exposes the embedding state. Defaults to "input_layer".
        **kwargs: Unused, present for registry call-signature consistency.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The figure and its last axes.
    """
    state: dict[str, Any] = ctx.get(f"evaluation_state.monitored_blocks.{block}")
    categories = {k[len(ANGLE_PREFIX):]: v for k, v in state.items() if k.startswith(ANGLE_PREFIX)}
    if not categories:
        raise KeyError(f"block '{block}' exposes no '{ANGLE_PREFIX}*' quantity")

    norm = TwoSlopeNorm(vmin=30.0, vcenter=90.0, vmax=150.0)
    fig, axes = plt.subplots(1, len(categories), figsize=(4.2 * len(categories), 4.2), squeeze=False)
    for ax, (category, quantity) in zip(axes[0], categories.items()):
        _draw_category(ax, category, quantity, norm)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdYlBu"), ax=axes[0].tolist(), fraction=0.02,
                 label="centered angle [°]  (90° = orthogonal)")
    return fig, axes[0][-1]
