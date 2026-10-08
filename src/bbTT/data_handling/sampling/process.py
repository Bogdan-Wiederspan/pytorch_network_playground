from typing import Optional

import torch
import torch.utils.data as t_data

from bbTT.data_handling.sampling.weight import WeightStatistics
from bbTT.monitoring.logger.logger import get_logger
from bbTT.utils.utils import CPU_DEVICE

logger_inst = get_logger(__name__)


class Process(t_data.Dataset):
    """
    Stateless container for all data belonging to one process (process_id, process_type).
    Knows its own data/weights/statistics and how to gather a batch from itself.
    Holds NO cursor/iteration state — that lives in ProcessSampleCursor.
    """

    def __init__(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        target: torch.Tensor,
        normalization_weights: torch.Tensor,
        product_of_weights: torch.Tensor,
        weights_statistics: WeightStatistics,
        evaluation_space_mask: torch.Tensor,
        process_id: str = None,
        process_type: str = None,
    ):
        """
        A Process represents all data connected to a process identified by *process_id*,
        which itself belongs to a *process_type* (e.g. "tt", "dy", "hh").

        Args:
            continuous: Tensor of continuous features, shape [num_events, num_continuous_features]
            categorical: Tensor of categorical features, shape [num_events, num_categorical_features]
            target: One-hot target tensor, shape [num_events, num_classes]
            normalization_weights: Per-event normalization weights (MC oversampling correction), shape [num_events]
            product_of_weights: Per-event product of generation/reconstruction weights, shape [num_events]
            weights_statistics: Aggregated weight statistics for this process (see WeightAggregator)
            evaluation_space_mask: Bool mask selecting events in the evaluation phase space
            process_id: Unique id of the process
            process_type: Physics process type this process belongs to (e.g. "hh")
        """
        self.continuous = continuous.to(torch.float32)
        self.categorical = categorical.to(torch.int32)
        self.targets = target.to(torch.float32)

        # product of all weights assigned during generation and reconstruction;
        # necessary to transfer mc to real yield (real events one would see in data)
        self.product_of_weights = product_of_weights.to(torch.float32)

        # mc samples are generated with higher lumi (to reduce stat. unc.);
        # normalization weights scale down to correct lumi
        self.normalization_weights = normalization_weights.to(torch.float32)

        self.weights_statistics = weights_statistics
        self.evaluation_space_mask = evaluation_space_mask.to(torch.bool)

        self.process_id = process_id
        self.process_type = process_type

        # cross-process bookkeeping set by BatchSizeAllocator;
        # only exist to write into, default is set 0
        # Both are batch-independent (expected shares), the per-batch sample size lives in the allocator.
        self.relative_weight: Optional[float] = 0
        self.post_relative_weight: Optional[float] = 0

    @property
    def uid(self) -> tuple[str, str]:
        return (self.process_type, self.process_id)

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.continuous[idx], self.categorical[idx], self.targets[idx]

    def per_event_sample_weight(self, n: int, denominator: int) -> torch.Tensor:
        """
        Single source of truth for the sample_weights value, used by sample/peek/generator alike.
        The sample weight describes how much each event contributes to the total yield of the process, given that only a subset of events is sampled.

        Args:
            n: Number of events to create weights for.
            denominator: Number of events that should represent the whole process yield. For a training
                batch this is the drawn number of events, for a full pass it is ``len(self)``.

        Returns:
            Tensor of shape [n, 1].
        """
        # n == 0 -> empty batch, nothing to weight (and denominator == 0 would divide by zero)
        if n == 0:
            return torch.empty((0, 1))
        return torch.full((n, 1), self.weights_statistics.normalization_whole_sum / denominator)

    def gather(self, idx: torch.Tensor, sample_from: tuple[str, ...], weight_denominator: int, device=CPU_DEVICE) -> dict[str, torch.Tensor]:
        """
        Pure lookup: given indices + attribute names, return the corresponding batch dict.
        No mutation, no cursor state — sample()/peek()/generator on ProcessSampleCursor all
        reduce to calling this.

        Args:
            idx: Indices to gather.
            sample_from: Attribute names to gather (e.g. "continuous", "categorical", "targets").
            weight_denominator: Passed to per_event_sample_weight.
            device: Device to move tensors to.

        Returns:
            Dict of tensors for each requested attribute, plus "sample_weights".
        """
        out = {attribute: getattr(self, attribute)[idx].to(device) for attribute in sample_from}
        out["sample_weights"] = self.per_event_sample_weight(len(idx), weight_denominator).to(device)
        return out


class ProcessSampleCursor:
    """
    Owns the mutable iteration state (current position, shuffle order) for repeatedly
    sampling from one Process.
    Kept separate from Process so:
    - Process stays a stateless, shareable data container
    - multiple independent cursors can sample the same Process without interfering
        (e.g. a train cursor and a monitoring/peek cursor over the same data)
    """

    def __init__(self, process: Process, randomize: bool = True):
        self.process = process
        self.randomize = randomize
        self.reset()

    def reset(self):
        """Reset the cursor to the start, reshuffling indices if randomize=True."""
        self.current_idx = 0
        self.last_idx = None
        n = len(self.process)
        self.indices = torch.randperm(n) if self.randomize else torch.arange(n)

    def sample(
        self,
        sample_from: tuple[str, ...],
        number: int,
        device=CPU_DEVICE,
    ) -> dict[str, torch.Tensor]:
        """
        Sample *number* events from the process.

        Always returns exactly *number* events. When the cursor reaches the end of
        the process it wraps around (reshuffling first if randomize=True) and keeps
        filling from the start, so the per-process quota in a batch is never
        silently truncated. Note that if *number* exceeds the process size, the
        returned batch necessarily contains repeated events.

        Args:

        sample_from (tuple[str]): Attribute names to sample.
        number (int): Number of events to sample, decided per batch by the BatchSizeAllocator.
        device (torch.device, optional): Device to move the gathered tensors to.

        Returns (dict[str, torch.Tensor]): Sampled tensors keyed by attribute name, plus ``"sample_weights"``.
        """
        max_events = len(self.process)
        remaining = number
        if remaining < 0:
            raise ValueError(f"Number of events to sample must be positive, got {remaining}.")

        # Accumulate index chunks across wraparounds.
        # A loop is used to cover multiple spans
        chunks: list[torch.Tensor] = []
        while remaining > 0:
            if self.current_idx >= max_events:
                self.reset()
            take = min(remaining, max_events - self.current_idx)
            chunks.append(self.indices[self.current_idx : self.current_idx + take])
            self.current_idx += take
            remaining -= take

        # number == 0 --> the allocator gave this process no events in this batch. In that case, chunks is empty
        # In that case, return an empty tensor of the correct shape.
        idx = torch.cat(chunks) if chunks else self.indices[:0]

        self.last_idx = idx

        # The batch-based weight: the drawn events represent the whole process yield, so denominator = len(idx).
        # peek() relies on this, it can rebuild the identical weights from last_idx alone.
        return self.process.gather(idx=idx, sample_from=sample_from, weight_denominator=len(idx), device=device)

    def peek(self, sample_from: tuple[str, ...], device=CPU_DEVICE) -> dict[str, torch.Tensor]:
        """Return the last sampled batch again, without advancing the cursor."""
        if self.last_idx is None:
            raise RuntimeError("No batch has been sampled yet. Call sample() before peek().")
        return self.process.gather(idx=self.last_idx, sample_from=sample_from, weight_denominator=len(self.last_idx), device=device) #noqa

    def generator(self, sample_from: tuple[str, ...], batch_size: int = -1, device=CPU_DEVICE):
        """
        Iterate over the whole process once, in order (no shuffling), independent of
        sample()/peek() cursor state. If batch_size == -1, everything is returned in one batch.

        Args:
            sample_from: Attribute names to gather per batch.
            batch_size: Events per yielded batch; -1 for the whole process at once.
            device: Device to move tensors to.

        Yields:
            Dict of tensors for each requested attribute, plus "sample_weights".
        """
        n = len(self.process)
        batch_size = n if batch_size == -1 else batch_size
        bounds = [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]

        for start, end in bounds:
            # chunks of one pass all represent the whole process, so the denominator is the process size,
            # not the chunk size. Otherwise the weights would depend on the chosen batch_size.
            yield self.process.gather(torch.arange(start, end), sample_from, weight_denominator=n, device=device)
