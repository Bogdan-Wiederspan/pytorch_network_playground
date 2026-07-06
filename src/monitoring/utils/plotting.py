import matplotlib.pyplot as plt


def add_number_legend(ax, string):
    lines, labels = ax.get_legend_handles_labels()
    dummy_line = plt.Line2D([], [], linestyle="", marker="")
    lines.append(dummy_line)
    labels.append(string)
    return lines, labels

def append_text_to_legend(ax, information):
    lines, labels = ax.get_legend_handles_labels()
    dummy_line = plt.Line2D([], [], linestyle="", marker="")
    if isinstance(information, str):
        information = (information, )

    for info in information:
        lines.append(dummy_line)
        labels.append(info)
    # manipulate given ax
    ax.legend(lines, labels)
    return ax
