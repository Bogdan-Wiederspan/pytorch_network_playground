import awkward as ak
import numpy as np
import torch
import uproot


def flatten_lists(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_lists(item)
        else:
            yield item

def depthCount(lst):
    # takes an arbitrarily nested list as a parameter and returns the maximum depth to which the list has nested sub-lists
    if isinstance(lst, list):
        return 1 + max(depthCount(x) for x in lst)
    else:
        return 0

def pad_empty(array, target_length, pad_value=0):
    array = ak.pad_none(array, target_length, axis=1, clip=True)
    array = ak.fill_none(array, pad_value)
    return array

def get_statitics_over_years(events: dict[dict[str:ak.Array]]) -> dict[str:float]:
    # {era : {process_id : array}}
    # get statitics:
    bucket = {}
    for year, data_by_year in events.items():
        for name, array in data_by_year.items():
            if name not in bucket:
                bucket[name] = []
            bucket[name].append(len(array))
    bucket = {name: sum(num_events) for name,num_events in bucket.items()}
    return bucket

def calculate_frquency_weights(events):
    stats = get_statitics_over_years(events)
    return {name:1/stat for name, stat in stats.items()}
