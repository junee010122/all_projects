"""
Purpose: Plot tools
Author: June
"""


import os
import shutil
import numpy as np
import matplotlib.pyplot as plt


plt.style.use("ggplot")


def create_folder(path):

    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


def plot_data(dataset, results, path, figsize=(9, 6), fontsize=14):

    path_scatter = os.path.join(path, "scatterplots")
    path_pcs = os.path.join(path, "pcs")

    create_folder(path_scatter)
    create_folder(path_pcs)

    class_results = results["classes"]

    num_features = dataset.samples.shape[-1]
    

    test_data = np.asarray([[1.7569, 3.3501, 2.9871, 5.8192, 7.1915],
                            [3.2561, 2.7053, -0.3155, 5.5450, 4.8105],
                            [0.4990, 4.0318, 1.0987, 4.9764, 9.1189],
                            [0.8943, 2.6107, 1.6978, 6.7883, 7.5238]])


    for i in range(num_features):

        x0 = dataset.samples[:, i]

        for j in range(num_features):

            if i == j:
                continue

            x1 = dataset.samples[:, j]

            fig, ax = plt.subplots(figsize=figsize)

            ax.scatter(x0, x1, c=dataset.labels)

            title = "Visualizing: Features %s vs %s" % (i, j)
            ax.set_title(title, fontsize=fontsize)

            ax.set_xlabel("x0", fontsize=fontsize)
            ax.set_ylabel("x1", fontsize=fontsize)

            fig.tight_layout()

            path_save = os.path.join(path_scatter, "%s_vs_%s.png" % (i, j))
            fig.savefig(path_save)

            fig, ax = plt.subplots(figsize=figsize)

            indices = np.asarray((i, j))

            c = ["darkred", "darkblue", "green", "purple"]

            for label in np.unique(dataset.labels):
                vals, vecs = class_results[label]["eigen"]
                vals = vals[indices]
                vecs = vecs[:, indices]

                pc = vecs[np.argmax(vals)]
                # ax.scatter(test_data[i], test_data[j], color=c[label])
                ax.quiver(*(0, 0), *pc, color=c[label])

            title = "Visualizing: Features %s vs %s" % (i, j)
            ax.set_title(title, fontsize=fontsize)

            ax.set_xlabel("x0", fontsize=fontsize)
            ax.set_ylabel("x1", fontsize=fontsize)

            fig.tight_layout()

            path_save = os.path.join(path_pcs, "%s_vs_%s.png" % (i, j))
            fig.savefig(path_save)
