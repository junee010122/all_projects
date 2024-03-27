"""
Purpose: Example of hand crafted covariance :)
Author: June
"""


import numpy as np

from scipy.spatial import distance

from utils.plots import plot_data
from utils.data import load_mat_file


def classify(test_data, results):

    num_classes = len(results["classes"].keys())

    all_euclidean, all_mahal = [], []

    for sample in test_data:

        class_exemplars = np.asarray([results["classes"][i]["mean"]
                                      for i in range(num_classes)])

        class_matrices = np.asarray([results["classes"][i]["covariance"]
                                     for i in range(num_classes)])

        euclidean = np.sqrt(np.sum((class_exemplars - sample) ** 2, axis=1))
        all_euclidean.append(euclidean)

        inv_matrices = np.linalg.inv(class_matrices)
        mahal = []
        for matrix, exemplar in zip(inv_matrices, class_exemplars):
            mahal.append(distance.mahalanobis(exemplar, sample, matrix))

        all_mahal.append(mahal)

    all_euclidean = np.asarray(all_euclidean)
    labels = np.argmin(all_euclidean, axis=1)

    print("\nEuclidean Results\n")
    print(all_euclidean)
    print(labels)

    all_mahal = np.asarray(all_mahal)
    labels = np.argmin(all_mahal, axis=1)

    print("\nMahalanobis Results\n")
    print(all_mahal)
    print(labels)


def covariance(x, y):
    """
    Calculate covariance between two variables

    Arguments:
    - x (np.ndarray[float]): variable 1
    - y (np.ndarray[float]): variable 2

    Returns:
    - (float): covariance between the two variables
    """

    return np.sum(((x - np.mean(x)) * (y - np.mean(y)))) / (x.shape[0] - 1)


def get_cov_matrix(data):
    """
    Create a covariance matrix using the data

    Arguments:
    - data (np.ndarray[float]): dataset defined by shape (n x m)

    Returns:
    - (np.ndarray[float]): covariance matrix of shape (m x m)
    """

    num_features = data.shape[-1]

    matrix = np.zeros((num_features, num_features))

    for i in range(num_features):
        for j in range(num_features):
            matrix[i][j] = covariance(data[:, i], data[:, j])

    return matrix


def get_data_stats(dataset):
    """
    Gather dataset mean, covariances, and eigen information

    Arguments:
    - dataset (Dataset): supervised dataset

    Returns:
    - (dict[str, any]): dataset statistics
    """

    # Calculate: Single Covariance
    # - This will be for entire dataset

    matrix = get_cov_matrix(dataset.samples).round(7)
    eigen = np.linalg.eig(matrix)

    data_stats = {"mean": np.mean(dataset.samples, axis=0),
                  "covariance": matrix,
                  "eigen": eigen}

    # Calculate: Covariances, Each Class

    class_stats = {}

    for label in np.unique(dataset.labels):

        indices = np.where(label == dataset.labels)
        class_samples = dataset.samples[indices]
        matrix = get_cov_matrix(class_samples).round(7)
        eigen = np.linalg.eig(matrix)

        class_stats[label] = {"mean": np.mean(class_samples, axis=0),
                              "covariance": matrix,
                              "eigen": eigen}

    print(class_stats)

    return {"all": data_stats, "classes": class_stats}


def experiment(params):

    # Generate: Dataset

    dataset = load_mat_file(params["paths"]["data"])

    # Calculate: Dataset Statistics

    all_results = get_data_stats(dataset)

    # Plot: Results

    plot_data(dataset, all_results, params["paths"]["results"])

    # Classify: Dataset

    test_data = np.asarray([[1.7569, 3.3501, 2.9871, 5.8192, 7.1915],
                            [3.2561, 2.7053, -0.3155, 5.5450, 4.8105],
                            [0.4990, 4.0318, 1.0987, 4.9764, 9.1189],
                            [0.8943, 2.6107, 1.6978, 6.7883, 7.5238]])

    distance_results = classify(test_data, all_results)


if __name__ == "__main__":
    """
    Example of project
    """
    params = {"paths": {"data": "data_class4.mat", "results": "results"}}

    experiment(params)
