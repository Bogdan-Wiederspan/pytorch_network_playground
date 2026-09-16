from __future__ import annotations

import numpy as np
import numpy.lib.recfunctions as rfn
import torch

from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)

def struct_to_group_tensor(arr: np.typing.NDArray, fields: tuple[str], dtype: torch.dtype = torch.float32):
    """
    Small helper convert struct *arr* *fields* into torch tensor of given *dtype*.

    Args:
        arr (np.typing.NDArray): structured array
        fields (tuple[str]): fields one wants to extract
        dtype (torch.dtype, optional): final dtype. Defaults to torch.float32.

    Returns:
        torch.Tensor: Tensor of extracted fields in given dtype
    """
    # get numpy equivalent dtype
    np_dtype = torch.empty(0, dtype=dtype).numpy().dtype
    dense = rfn.structured_to_unstructured(arr[fields], dtype=np_dtype)
    # some arrays have negative strides for some reason, which torch cannot handle -> cast to contiguous array first
    dense = np.ascontiguousarray(dense)
    return torch.from_numpy(dense)


def hash_dictionary(dictionary: dict):
    """
    Create hash from objects defined in a dictionary.

    Args:
        dictionary (dict): Any dictionary

    Returns:
        str: Hash of the given dict content.
    """
    import hashlib

    hashable_dict = sorted(dictionary.items(), key=lambda item: item[0])
    h = tuple(hashable_dict)
    h = hashlib.sha256(str(h).encode("utf-8")).hexdigest()[:10]
    return h
