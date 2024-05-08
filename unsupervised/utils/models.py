import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture as GMM

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import davies_bouldin_score, silhouette_score

from utils.general import save_data, create_folder


def calculate_analytics(preds, labels, refs):

    s_measures = {}
    s_measures["ARI"] = adjusted_rand_score(labels, preds)
    s_measures["NMI"] = normalized_mutual_info_score(labels, preds)

    u_measures = {}
    u_measures["DBI"] = davies_bouldin_score(refs, preds)
    u_measures["SS"] = silhouette_score(refs, preds)

    return s_measures, u_measures


def compare(x, y):

    matrix = np.zeros((x.shape[0], y.shape[0]))

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            matrix[i, j] = np.linalg.norm(x[i] - y[j])

    return matrix


def get_and_save_predictions(path, m_meta, model, data,
                             count, grid_num_samples=25):

    m_name = m_meta["name"]
    v_name = m_meta["version"]

    if m_name == "DBSCAN":

        if count == 0:
            eps = 8
        else:
            eps = 5.6

        # from sklearn.neighbors import NearestNeighbors as NN

        # nn = NN(n_neighbors=10)
        # nn.fit(data.samples)

        # distances, indices = nn.kneighbors(data.samples)

        # k_distances = np.sort(distances[:, 9])[::-1]

        # plt.plot(k_distances)
        # plt.xlabel('Points sorted')
        # plt.ylabel('Distance to 10 nearest neighbor')
        # plt.title('k-Distance Plot')
        # plt.grid(True)
        # plt.show()

        model.set_params(**{"eps": eps,
                            "min_samples": 10,
                            "metric": "euclidean"})

        model.fit(data.samples)
        preds = model.labels_.astype(int)

        indices = np.where(preds != -1)
        samples = data.samples[indices]
        labels = data.labels[indices]
        refs = data.refs[indices]
        preds = preds[indices]

    elif m_name == "GMM":

        model.fit(data.samples)
        probs = model.predict_proba(data.samples)

        preds = []
        for relation in probs:
            if (relation < 0.8).all():
                preds.append(-1)
            else:
                preds.append(np.argmax(relation))

        preds = np.asarray(preds)

        indices = np.where(preds != -1)
        samples = data.samples[indices]
        labels = data.labels[indices]
        refs = data.refs[indices]
        preds = preds[indices]

    elif m_name == "K-Means":

        model.fit(data.samples)
        centers = model.cluster_centers_
        matrix = compare(centers, data.samples)

        thresh = 0.95
        all_sims, all_indices = [], []
        for z in matrix:
            z = 1 - ((z - z.min()) / (z.max() - z.min()))

            sims, indices = [], []
            for i, ele in enumerate(z):
                if ele >= thresh:
                    indices.append(i)
                    sims.append(ele)
                else:
                    indices.append(-1)
                    sims.append(-1)

            all_indices.append(indices)
            all_sims.append(sims)

        all_indices = np.asarray(all_indices)
        all_sims = np.asarray(all_sims)

        preds = []
        for i in range(all_sims.shape[1]):
            relation = all_sims[:, i]
            if (relation == -1).all():
                preds.append(-1)
            else:
                preds.append(np.argmax(relation))
        preds = np.asarray(preds)

        indices = np.where(preds != -1)
        samples = data.samples[indices]
        labels = data.labels[indices]
        refs = data.refs[indices]
        preds = preds[indices]

    analytics = calculate_analytics(preds, labels, samples)

    save_data(analytics, path)

    grid_num_samples = 25
    grid_num_rows = int(np.sqrt(grid_num_samples))

    n_size, f_size = refs.shape
    f_size = int(np.sqrt(f_size))

    all_refs = refs.reshape(n_size, 1, f_size, f_size)

    tag = path.split("/")[-1]
    path_root = path.replace(tag, "")

    for current_label in np.unique(preds):

        path_images = "%s/images/%s/%s/%s" % (path_root, m_name, v_name, current_label)
        create_folder(path_images)

        path_save = "%s/%s.png" % (path_images, current_label)

        indices = np.where(current_label == preds)
        refs_tensor = torch.tensor(all_refs[indices][:grid_num_samples])

        grid = make_grid(refs_tensor, nrow=grid_num_rows,
                         normalize=True, scale_each=True)

        grid = np.transpose(grid.numpy(), (1, 2, 0)).squeeze()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid, cmap="gray")
        ax.axis(False)
        fig.tight_layout()
        fig.savefig(path_save)
        plt.close()


def optimize_and_save(path, all_models, data, count):

    for current_key in all_models:

        model_params = all_models[current_key]

        model = model_params["model"]

        if "search" in model_params.keys():

            search = model_params["search"]
            target_key = list(search.keys())[0]

            min_val, max_val, step_size = search[target_key]

            all_values = np.arange(min_val, max_val, step_size)

            for i, value in enumerate(all_values):

                name = "%s_%s_%s.joblib" % (current_key, target_key, value)

                path_save = os.path.join(path, name)

                start = time.time()

                params = {target_key: value}
                model.set_params(**params)

                meta = {"name": current_key, "version": value}

                get_and_save_predictions(path_save, meta,
                                         model, data, count)

                stop = time.time() - start

                desc = "%s Optimization | Iteration = %s | Time (Seconds) = %s"

                print(desc % (current_key, i, stop))

        else:

            name = "%s.joblib" % (current_key)

            path_save = os.path.join(path, name)

            start = time.time()

            model.fit(data.samples, data.labels)

            meta = {"name": current_key, "version": 0}

            get_and_save_predictions(path_save, meta,
                                     model, data, count)

            stop = time.time() - start

            desc = "%s Optimization | Time (Seconds) = %s"

            print(desc % (current_key, stop))


def select_models(params):

    all_models = {"K-Means": {"model": KMeans(n_init="auto"),
                              "search": {"n_clusters": [3, 15, 1]}},

                  "GMM": {"model": GMM(),
                          "search": {"n_components": [3, 15, 1]}},

                  "DBSCAN": {"model": DBSCAN()}}

    return all_models
