

from utils.general import load_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime


plt.style.use("ggplot")


def plot_data(dataset, fig_size=(14, 8), font_size=14, show_plots=0):

    
    colors = ['darkred', 'darkblue', 'darkgreen']
    type_labels = np.unique(dataset.labels)
    num_type_labels = len(type_labels)
    colors = ["darkred", "darkblue", "darkgreen", "darkorange"][:num_type_labels]
    color_label = dict(zip(type_labels, colors))
    scatter_colors = [color_label[label] for label in dataset.labels]

    fig, ax = plt.subplots(figsize=fig_size)

    # for classification
    ax.scatter(dataset.samples[:, 0], dataset.samples[:, 1],
               c=scatter_colors)
       
    # for regression
    #ax.scatter(dataset.samples.reshape(-1), dataset.labels)

    ax.set_title("2D Dataset: %s" % dataset.name, fontsize=font_size)
    ax.set_xlabel("X0", fontsize=font_size)
    ax.set_ylabel("X1", fontsize=font_size)

    fig.tight_layout()
    path = "/Users/june/Documents/results/NNclass/regression/plots"
    save_path = f"{path}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    plt.savefig(save_path, bbox_inches='tight')
    if show_plots:
        plt.show()



def plot_evaluation(path_list):

    target_names = ["f1", "recall", "accuracy", "precision"]
    results = []
    x_vals = ("1", "2", "3", "4", "5")

    for fold in path_list:
        data = load_data(fold)
        metric_sums = {metric: 0 for metric in target_names}

        for metric in target_names:
            if f"{metric}_epoch" in data.columns:
                metric_sums[metric] += data[f"{metric}_epoch"].sum()
    
        averages = {metric: metric_sums[metric] / data["epoch"].max() for metric in target_names}
        results.append(averages)

    x = np.arange(len(path_list))

    fig, ax = plt.subplots()

    width = 0.2  # Set the width of the bars
    for i, metric in enumerate(target_names):
        metric_values = [result[metric] for result in results]
        rects = ax.bar(x + i * width, metric_values, width=width, label=metric)

    ax.set_ylabel('Performance')
    ax.set_title('Evaluation measures per Fold')
    ax.set_xticks(x + 1.5 * width)  # Use set_xticks to define the locations of the ticks
    ax.set_xticklabels(x_vals)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()


def plot_decision_boundary(model, X, all_predictions):
    from IPython import embed
    embed()
    x_min, x_max = X[:,0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    meshgrid = np.c_[xx.ravel(), yy.ravel()]
    
    with torch.no_grad():
        logits = model(torch.tensor(meshgrid, dtype=torch.float32))
        all_predictions = torch.argmax(logits, dim=1)
    Z = all_predictions.numpy().reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title('Decision Boundary for 3-class classification (PyTorch Lightning)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()








