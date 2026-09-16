import time
from collections import defaultdict
from contextlib import contextmanager

registered_timings: dict[str, list[float]] = defaultdict(list)


@contextmanager
def time_block(name: str):
    """
    Measure and record the wall-clock duration of a code block.


    Args:
        name: Label under which the duration is stored, e.g. a plot's name.

    Yields:
        None. Timing happens as a side effect on exit.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        registered_timings[name].append(time.perf_counter() - start)
