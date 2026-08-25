import torch


class BatchSizeAllocator:
    """
    Encapsulates the largest-remainder-style rounding algorithm used to split a
    sub-batch size across process ids proportional to their normalization weight,
    respecting a minimum per-process size. Pure function of plain data in,
    (sizes, relative_weights) out — testable without constructing Process objects.
    """

    def __init__(self, min_size: int = 0):
        self.min_size = min_size

    def allocate(
        self,
        weights_by_pid: dict[str, float],
        sub_sample_ratio: dict[str, float],
        sub_batch_size: int,
        sample_ratio_for_type: float,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """
        Returns:
            sizes: process_id -> integer event count, summing to sub_batch_size.
            relative_weights: process_id -> relative contribution to the batch.
        """
        pids = list(weights_by_pid.keys())
        raw_weights = torch.tensor([
            weights_by_pid[pid] * sub_sample_ratio.get(pid, 1) for pid in pids
        ], dtype=torch.float32)
        total_weight = raw_weights.sum()

        relative_weights = {
            pid: (w / total_weight * sample_ratio_for_type).item()
            for pid, w in zip(pids, raw_weights)
        }

        ideal = sub_batch_size * raw_weights / total_weight
        floored = torch.maximum(torch.floor(ideal), torch.tensor(float(self.min_size)))
        overflow = int(floored.sum()) - sub_batch_size

        above_mask = floored > self.min_size
        above = floored[above_mask]
        overflow_shares = above / above.sum() * overflow
        overflow_shares, order = torch.sort(overflow_shares)

        median = order.median()

        below_median = order <= median if len(order) % 2 == 0 else order < median
        above_median = ~below_median

        overflow_shares[below_median] = torch.floor(overflow_shares[below_median])
        overflow_shares[above_median] = torch.ceil(overflow_shares[above_median])

        above_indices = torch.arange(len(floored))[above_mask][order]
        floored[above_indices] -= overflow_shares

        if (floored.to(torch.int32).sum() - floored.sum()) != 0:
            raise TypeError("Rounding changed the total value unexpectedly")
        floored = floored.to(torch.int32)

        remainder = int(floored.sum()) - sub_batch_size
        floored[above_indices[0]] -= remainder

        sizes = {pid: n.item() for pid, n in zip(pids, floored)}
        return sizes, relative_weights
