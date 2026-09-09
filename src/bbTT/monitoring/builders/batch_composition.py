from bbTT.monitoring.register import register_builder


@register_builder(
    "batch_composition_parts",
    provides={"composition_steps", "composition_by_parent", "composition_by_uid"},
    requires={"batch_composition"},
)
def build_batch_composition_parts(ctx, **kwargs):
    """
    Aggregate the recorded batch history to per-parent event counts per step.

    Aggregated to hh/tt/dy rather than kept per sub-process: fifty-odd series
    are unreadable, and a sub-process that truncates still shows up as a dip in
    its parent's total. Counts stay absolute, since truncation is a loss of
    events that a normalised view would hide.

    Args:
        ctx (EvalContext): Must resolve ``batch_composition_history``.
        **kwargs: Unused, present for registry call-signature consistency.

    Returns:
        dict: ``composition_steps`` (list[int]) and ``composition_by_parent``
        (dict[str, list[int]]).
    """
    history = ctx.get("batch_composition")

    # uid is (parent_process, subprocess); sorting by it groups by parent, and
    # str() on the second element guards against mixed int/str subprocess ids.
    uids = sorted(history.counts[0], key=lambda uid: (uid[0], str(uid[1])))
    parents = sorted({uid[0] for uid in uids})

    by_uid = {uid: [counts[uid] for counts in history.counts] for uid in uids}
    by_parent = {
        parent: [
            sum(counts[uid] for uid in uids if uid[0] == parent)
            for counts in history.counts
        ]
        for parent in parents
    }

    return {
        "composition_steps": history.steps,
        "composition_by_parent": by_parent,
        "composition_by_uid": by_uid,
    }
