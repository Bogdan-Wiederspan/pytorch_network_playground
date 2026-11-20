from __future__ import annotations

import numpy as np
import numpy.lib.recfunctions as rfn
import torch

from bbTT.monitoring.logger.logger import get_logger
from bbTT.utils.utils import EMPTY_FLOAT

logger_inst = get_logger(__name__)


def markers_by_position(features: tuple[str, ...], markers: dict[str, float], fill: float = EMPTY_FLOAT) -> list[float]:
    """
    One missing-value marker per feature in *features* order, for tensor-wise comparisons.

    Args:
        features (tuple[str, ...]): Feature names in tensor column order.
        markers (dict[str, float]): Marker per feature that has one, see ``DataConfig.dummy_values``.
        fill (float, optional): Sentinel for features without marker, must never occur in the data.
            Defaults to EMPTY_FLOAT.

    Returns:
        list[float]: Marker per column, JSON-serializable.
    """
    return [markers.get(f, fill) for f in features]


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
