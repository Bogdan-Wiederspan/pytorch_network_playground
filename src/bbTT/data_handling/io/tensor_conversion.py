import numpy as np
import torch

from bbTT.data_handling.utils import struct_to_group_tensor
from bbTT.monitoring.logger.logger import get_logger

logger_inst = get_logger(__name__)


def filter_nan(array, features, uid):
    event_mask = np.zeros(array.size, dtype=np.bool)
    for f in features:
        event_mask |= np.isnan(array[f])
    num_filter = np.sum(event_mask)
    if num_filter:
        logger_inst.warning(f"Filtered {num_filter} Nan events from pid {uid}")
    return array[~event_mask]

def filter_and_convert_to_torch(events: np.array, continuous_features: list[str], categorical_features: list[str]):
    """
    Calculates final weights, extract masks aswell as extract all *continuous_features* and *categorical_features* from structured numpy array *events*.
    Converts all arrays to torch tensors and returns a dictionary containing these.

    Args:
        events (np.array): _description_
        continuous_features (list[str]): List of continuous features to be extracted
        categorical_features (list[str]): List of categorical features to be extracted
        dtype (torch.dtype, optional): Torch dtype. Defaults to None.
    """

    for uid in list(events.keys()):
        arr = events.pop(uid)

        # filter all nans out, when result is empty array, skip whole uid
        arr = filter_nan(arr, continuous_features + categorical_features, uid)

        if arr.size == 0:
            logger_inst.warning(f"Skipping {uid} due to zero elements - which can happen after filtering nans")
            continue

        # extract features
        continuous_tensor = struct_to_group_tensor(arr, continuous_features, dtype=torch.float32)
        categorical_tensor = struct_to_group_tensor(arr, categorical_features, dtype=torch.float32)

        # extract evaluation masks
        masks_tensor = struct_to_group_tensor(arr, ["bjet_mask", "di_tau_mask", "di_bjet_mask"], torch.bool)
        final_mask = masks_tensor[:, 0] & masks_tensor[:, 1] & masks_tensor[:, 2]

        # extract weights
        weights_tensor = struct_to_group_tensor(arr, ["normalization_weight", "combined_weight"], dtype=torch.float32)
        normalization_weights = weights_tensor[:, 0]
        sum_of_normalization_weights = torch.sum(normalization_weights)

        product_of_all_weights = weights_tensor[:, 1]
        sum_of_combined_weights = torch.sum(product_of_all_weights)
        total_evaluation_weight = torch.sum(product_of_all_weights[final_mask])

        # extract meta fields
        event_id = struct_to_group_tensor(arr, ["event"], dtype=torch.int64).flatten()

        events[uid] = {
            "continuous": continuous_tensor,
            "categorical": categorical_tensor,
            "event_id": event_id,
            "normalization_weights": normalization_weights,
            "product_of_weights": product_of_all_weights,
            "total_product_of_weights": sum_of_combined_weights,
            "total_normalization_weights": sum_of_normalization_weights,
            "total_evaluation_weight": total_evaluation_weight,
            "evaluation_mask": final_mask,
            "mask_bjet": masks_tensor[:, 0],
            "mask_di_tau": masks_tensor[:, 1],
            "mask_di_bjet": masks_tensor[:, 2],
            }
        del arr
    return events
