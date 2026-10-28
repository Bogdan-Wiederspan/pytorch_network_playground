from dataclasses import dataclass


@dataclass(frozen=True)
class EpochStatistic:
    """
    Epoch bookkeeping of one process (or one aggregated process_type).

    Attributes:
        num_events (int): Number of events available in the dataset.
        expected_per_batch (float): Expected number of events drawn per batch, see ``Process.expected_sample_size``.
    """

    num_events: int
    expected_per_batch: float

    @property
    def iterations_per_epoch(self) -> float:
        """
        Number of batches until every event has been drawn once in expectation.

        Returns:
            float: num_events / expected_per_batch, ``inf`` if the process is never drawn.
        """
        if self.expected_per_batch <= 0:
            return float("inf")
        return self.num_events / self.expected_per_batch

@dataclass(frozen=True)
class EpochOverview:
    """
    Epoch statistics of a whole sampler, per process_id and aggregated per process_type.

    Attributes:
        per_pid (dict[tuple[str, str], EpochStatistic]): uid -> statistic of one process.
        per_type (dict[str, EpochStatistic]): process_type -> statistic summed over its pids.
    """

    per_pid: dict[tuple[str, str], EpochStatistic]
    per_type: dict[str, EpochStatistic]

    @property
    def limiting_pid(self) -> tuple[str, str]:
        """
        Process that finishes its epoch first, i.e. the one that is oversampled the most.

        Returns:
            tuple[str, str]: uid with the smallest iterations_per_epoch.
        """
        return min(self.per_pid, key=lambda uid: self.per_pid[uid].iterations_per_epoch)
