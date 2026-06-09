import awkward as ak

def flatten_lists(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_lists(item)
        else:
            yield item

def depthCount(lst):
    """Takes an arbitrarily nested list as a parameter and returns the maximum depth to which the list has nested sub-lists"""
    if isinstance(lst, list):
        return 1 + max(depthCount(x) for x in lst)
    else:
        return 0

def pad_empty(array: ak.Array, target_length: int, pad_value: float = 0.0):
    """Pads an akward array of lists to a specified target length with a given pad value."""
    array = ak.pad_none(array, target_length, axis=1, clip=True)
    array = ak.fill_none(array, pad_value)
    return array
