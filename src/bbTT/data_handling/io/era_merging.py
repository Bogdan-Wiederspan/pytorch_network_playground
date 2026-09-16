import torch

from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)


def merge_era_events(all_events, era_events):
    # incremental concat era tensors to all_events
    scalar_keys = {"total_product_of_weights", "total_normalization_weights", "total_evaluation_weight"}

    for uid, tensors in era_events.items():
        # when uid not there set base
        if uid not in all_events:
            all_events[uid] = tensors
            continue

        # concatenate data from era to all
        existing = all_events[uid]
        merged = {}
        for key, value in tensors.items():
            if key == "mask":
                merged["mask"] = {
                    mask_key: torch.cat([existing["mask"][mask_key], value[mask_key]], dim=0) for mask_key in value
                }
            elif key in scalar_keys:
                merged[key] = existing[key] + value
            else:
                merged[key] = torch.cat([existing[key], value], dim=0)
        all_events[uid] = merged

    return all_events
