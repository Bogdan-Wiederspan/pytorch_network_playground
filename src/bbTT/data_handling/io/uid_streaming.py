from typing import Union

import numpy as np

from bbTT.data_handling.io.root_reading import load_root_and_convert_to_numpy
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)

def group_by_process_id(array):
    # helper to extract data by process id and group them by this
    pids = array["process_id"]
    for uid in np.unique(pids):
        yield int(uid), array[pids == uid]


def stream_events_by_uid(
    dataset_paths: list[str],
    columns: Union[list[str], str],
    cut: Union[list[str], None] = None,
    flush_threshold_rows: int = 1_000_000,
) -> dict[tuple[str, int], np.typing.ArrayLike]:
    """
    Load root files in *dataset_paths*, extract *columns* with applied *cut* on it.
    The data is then rearranged by their process_id, removing year and era information.
    To reduce peak memory *flush_threshold_rows* can be adjusted. The higher, the bigger is peak memory impact vs. CPU time.
    Returns dictionary with np.arrays where key is (dataset, uid), ex. ('tt', 1100).

    Args:
        dataset_paths (list[str]): pattern describing dataset e.g. "dy_*" -> all drell-yan datasets
        branches (Union[list[str], str]): columns that should be loaded e.g. ["events", "run"]. If None loads all columns . Defaults to None.
        cut (Union[list[str], None], optional): list of cuts to be applied on top of baseline cut, which are defined in root_to_numpy. Defaults to None.
        flush_threshold_rows (int, optional): Number of rows inside array before flush is started. Trade of between CPU and peak memory. Defaults to 1_000_000.

    Returns:
        dict[tuple[str, int], np.typing.ArrayLike]: Dict of arrays where key is tuple of dataset name and id.
    """


    buffers = {}  # uid -> list of not-yet-merged fragments
    buffered_rows = {}  # uid -> sum of rows currently buffered (unmerged)
    data = {}  # uid -> merged running array

    def flush(uid, buffers):
        # helper to flush buffer and integrate into data
        fragments = buffers.get(uid)
        if not fragments:
            return

        # concatenate data or just unpack
        fragment = fragments[0]
        if len(fragments) > 1:
            fragment = np.concatenate(fragments, axis=0)

        # reset buffer
        buffers[uid] = []
        buffered_rows[uid] = 0

        # when uid fresh just add as base, else append fragment to array
        if uid not in data:
            data[uid] = fragment
        else:
            data[uid] = np.concatenate([data[uid], fragment], axis=0)

    num_events_per_dataset = {}
    num_events_per_pid = {}
    for dataset, files in dataset_paths.items():
        logger_inst.info(f"Start loading and conversion of root files: {dataset}")
        events_bucket, num_events_of_files = load_root_and_convert_to_numpy(files, branches=columns, cut=cut)
        num_events_per_dataset[dataset] = num_events_of_files
        for events in events_bucket:
            for pid, p_array in group_by_process_id(events):
                if pid not in num_events_per_pid:
                    num_events_per_pid[pid] = 0
                num_events_per_pid[pid] += len(p_array)

                uid = (dataset[:2], pid)
                buffers.setdefault(uid, []).append(p_array)
                buffered_rows[uid] = buffered_rows.get(uid, 0) + len(p_array)
                if buffered_rows[uid] >= flush_threshold_rows:
                    flush(uid=uid, buffers=buffers)
            del events

    # final flush for any remaining buffered fragments:
    for uid in list(buffers.keys()):
        flush(uid=uid, buffers=buffers)
        logger_inst.debug(f"UID: {uid} | NUM: {len(data[uid])}")

    return data, num_events_per_dataset, num_events_per_pid
