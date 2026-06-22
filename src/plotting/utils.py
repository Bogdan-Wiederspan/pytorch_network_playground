import matplotlib.pyplot as plt
import numpy as np
import torch


def prepare_tensor(*tensor, device="cpu"):
    return [t.detach().to(device).numpy() for t in tensor]

def add_number_legend(ax, string):
    lines, labels = ax.get_legend_handles_labels()
    dummy_line = plt.Line2D([], [], linestyle="", marker="")
    lines.append(dummy_line)
    labels.append(string)
    return lines, labels

def append_text_to_legend(ax, information):
    lines, labels = ax.get_legend_handles_labels()
    dummy_line = plt.Line2D([], [], linestyle="", marker="")
    if isinstance(information, (list ,tuple)):
        information = (information, )

    for info in information:
        lines.append(dummy_line)
        labels.append(info)
    # manipulate given ax
    ax.legend(lines, labels)
    return ax

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
