import os
import numpy as np
import matplotlib.pyplot as plt


def plot_exploration(path, all_data, bounded=True,
                     width=0.5, figsize=(10, 6), fontsize=16):

    plt.style.use("ggplot")

    for current_key in all_data.keys():

        path_save = os.path.join(path, "%s.png" % current_key)
        title = "Parameter Exploration, %s" % current_key

        data = all_data[current_key]

        fig, ax = plt.subplots(figsize=figsize)

        x_vals = np.arange(len(data["x_vals"]))
        x_labels = data["x_vals"]

        y_vals = data["y_vals"]
        y_labels = np.arange(0, 1.05, 0.1).round(2)

        ax.bar(x_vals, y_vals, width=width, color="coral")

        ax.set_xlabel("Parameter", fontsize=fontsize + 2)
        ax.set_ylabel("Measure", fontsize=fontsize + 2)
        ax.set_title(title, fontsize=fontsize + 2)

        if bounded:
            ax.set_ylim([-0.01, 1.05])
            y_labels = np.arange(0, 1.05, 0.1).round(2)
            ax.set_yticks(y_labels, labels=y_labels, fontsize=fontsize)
        else:
            for label in ax.get_yticklabels():
                label.set_fontsize(fontsize)

        ax.set_xticks(x_vals, labels=x_labels, fontsize=fontsize)

        fig.tight_layout()
        fig.savefig(path_save)
        plt.close()


def plot_bars(path, data, tag=None,
              figsize=(10, 6), fontsize=16):

    fig, ax = plt.subplots(figsize=figsize)

    categories = list(data.keys())
    values = list(data.values())

    if tag is None:

        height = 0.4
        ax.set_xlim([-0.01, 1.05])
        x_labels = np.arange(0, 1.05, 0.1).round(2)
        ax.set_xticks(x_labels, labels=x_labels, fontsize=fontsize)
    elif tag == "SS":

        height = 0.1
        ax.set_xlim([0, 2])
        ax.set_ylim(-0.1, 0.1)
        x_labels = np.arange(-0.01, 2.01, 0.2).round(2)
        ax.set_xticks(x_labels, labels=x_labels, fontsize=fontsize)
        values = values[0] + 1
    else:

        height = 0.1
        ax.set_xlim([0, 5])
        ax.set_ylim(-0.1, 0.1)
        for label in ax.get_xticklabels():
            label.set_fontsize(fontsize)

    ax.barh(categories, values, height=height, color="lightgreen")

    ax.set_yticks(categories, labels=categories, fontsize=fontsize)

    ax.set_xlabel("Measures", fontsize=fontsize+2)
    ax.set_ylabel("Value", fontsize=fontsize+2)

    fig.tight_layout()
    fig.savefig(path)
