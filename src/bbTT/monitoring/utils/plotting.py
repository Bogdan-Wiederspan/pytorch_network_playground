import matplotlib.pyplot as plt


def add_number_legend(ax, string, position=None):
    lines, labels = ax.get_legend_handles_labels()
    dummy_line = plt.Line2D([], [], linestyle="", marker="")
    if position is None:
        lines.append(dummy_line)
        labels.append(string)
    else:
        lines.insert(position, dummy_line)
        labels.insert(position, string)
    return lines, labels


def append_text_to_legend(ax, information, **kwargs):
    lines, labels = ax.get_legend_handles_labels()
    dummy_line = plt.Line2D([], [], linestyle="", marker="")
    if isinstance(information, str):
        information = (information,)

    for info in information:
        lines.append(dummy_line)
        labels.append(info)
    # manipulate given ax
    ax.legend(lines, labels, **kwargs)
    return ax
