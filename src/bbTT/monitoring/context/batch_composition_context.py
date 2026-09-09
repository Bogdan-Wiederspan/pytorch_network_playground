from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bbTT.data_handling.sampling.process import ProcessSampleCursor


@dataclass
class BatchCompositionHistory:
    """
    Per-step record of how many events each sub-process contributed to a batch.

    Accumulated in the training loop and passed into the eval context as
    metadata. Exists to make sampler truncation visible: a cursor that hits the
    end of its process without wrapping hands out a short batch, which shows up
    here as a dip in the affected sub-process while the others stay flat.

    Args:
        steps (list[int]): Global step at which each record was taken.
        counts (list[dict]): One dict per record, mapping sub-process name to event count.
    """

    steps: list[int] = field(default_factory=list)
    counts: list[dict[str, int]] = field(default_factory=list)

    def record(self, step: int, cursors: dict[str, "ProcessSampleCursor"]) -> None:
        """
        Append the composition of the batch that was just sampled.

        Reads ``cursor.last_idx``, which already holds exactly the indices the
        cursor handed out, so nothing needs to be threaded through the batch
        assembly itself.

        Args:
            step (int): Current global step.
            cursors (dict[str, ProcessSampleCursor]): Cursors keyed by sub-process name.
        """
        self.steps.append(step)
        self.counts.append({name: len(c.last_idx) for name, c in cursors.items()})
