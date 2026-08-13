import numpy as np
import torch


def prepare_tensor(*tensor, device="cpu"):
    return [t.detach().to(device).numpy() for t in tensor]

def find_events_of_node(y_true, y_pred, target_map, node_name="hh"):
    signal_idx = target_map[node_name]
    # identify events uses TRUTH information to create masks
    masks = {process: (y_true[:, idx] == 1) for process, idx in target_map.items()}
    # to get node information apply index filtering on PREDICTION
    picked_node = {process: y_pred[masks[process]][:, signal_idx] for process, idx in target_map.items()}
    return picked_node


def to_numpy(values):
    if torch.is_tensor(values):
        values = values.numpy()
    elif isinstance(values, (tuple, list)):
        values = np.array(values)
    return values
