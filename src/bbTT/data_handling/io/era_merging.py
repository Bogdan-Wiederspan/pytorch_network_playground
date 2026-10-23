from __future__ import annotations

import torch

from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)

# keys holding a per-uid scalar (0-d tensor) instead of a per-event tensor: summed instead of concatenated
SCALAR_KEYS = frozenset({"total_product_of_weights", "total_normalization_weights", "total_evaluation_weight"})

Uid = tuple[str, int]
EventDict = dict[Uid, dict[str, torch.Tensor]]


def merge_era_events(era_events: dict[str, EventDict]) -> EventDict:
    """
    Merge the per-era event dicts into one dict over all eras.

    Per uid and key, the per-event tensors of all eras are concatenated in the iteration
    order of *era_events*; scalar keys (see SCALAR_KEYS) are summed. Every output tensor
    is allocated exactly once, so the peak memory is the final size plus whatever the
    inputs occupy. With memory-mapped inputs (see DataCacher.load_era) that is only
    file-backed page cache, not process memory.

    Args:
        era_events (dict[str, EventDict]): Mapping era -> {uid: {key: tensor}}.

    Returns:
        EventDict: {uid: {key: tensor}} containing the events of all eras.
    """
    # uids in order of first appearance, so the output order is deterministic
    uids: dict[Uid, None] = {}
    for events in era_events.values():
        uids.update(dict.fromkeys(events))

    merged: EventDict = {}
    for uid in uids:
        chunks = [events[uid] for events in era_events.values() if uid in events]
        merged[uid] = _merge_uid_chunks(chunks)
        logger_inst.debug(f"Merged {uid}: {merged[uid]['continuous'].shape[0]} events from {len(chunks)} eras")
    return merged


def _merge_uid_chunks(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Merge the chunks (one per era) of a single uid.

    Args:
        chunks (list[dict[str, torch.Tensor]]): Per-era {key: tensor} dicts of one uid, all with the same keys.

    Returns:
        dict[str, torch.Tensor]: {key: tensor} with concatenated (or summed) tensors.
    """
    merged: dict[str, torch.Tensor] = {}
    for key in chunks[0]:
        parts = [chunk[key] for chunk in chunks]
        if key in SCALAR_KEYS:
            merged[key] = torch.stack(parts).sum()
        else:
            merged[key] = torch.cat(parts, dim=0)
    return merged
