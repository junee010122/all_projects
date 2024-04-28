import pickle
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def select_models():
    return {
        'KMeans': {'n_clusters': [2, 3, 4, 5, 6]},
        'GMM': {'n_components': [2, 3, 4, 5, 6]},
        'DBSCAN': {'eps': [0.5, 0.6, 0.7, 0.8, 0.9], 'min_samples': [5, 10, 15]}
    }

def compute_metrics(X, labels, measures):
    results = {}
    if 'silhouette' in measures:
        results['silhouette_score'] = silhouette_score(X, labels, metric='euclidean') if len(set(labels)) > 1 else -1
    if 'davies_bouldin' in measures:
        results['davies_bouldin_score'] = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else float('inf')
    return results

def train_clustering_models(choices, train_pca, valid_pca, measures, path_save):
    model_params = select_models()
    results = {}
    for model_name, params in model_params.items():
        if model_name not in choices:
            continue
        model_results = {}
        for param_key, values in params.items():
            for value in values:
                if model_name == 'KMeans':
                    model = KMeans(n_clusters=value)
                elif model_name == 'GMM':
                    model = GaussianMixture(n_components=value)
                elif model_name == 'DBSCAN':
                    model = DBSCAN(eps=value, min_samples=params['min_samples'][0])

                model.fit(train_pca)
                train_labels = model.labels_
                train_metrics = compute_metrics(train_pca, train_labels, measures)

                model.fit(valid_pca)
                valid_labels = model.labels_
                valid_metrics = compute_metrics(valid_pca, valid_labels, measures)

                model_results[f'{param_key}_{value}'] = {
                    'params': {param_key: value},
                    'train_metrics': train_metrics,
                    'valid_metrics': valid_metrics
                }

        results[model_name] = model_results
        path_model_save = f"{path_save}/{model_name}_results.pkl"
        with open(path_model_save, "wb") as writer:
            pickle.dump(model_results, writer)

        print(f"Results for {model_name} saved successfully.")
    return results


