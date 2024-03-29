

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def within_class_scatter_matrix(class1_data, class2_data):
    Sw_class1 = get_cov_matrix(class1_data)
    Sw_class2 = get_cov_matrix(class2_data)
    
    return 1/2*Sw_class1.T + 1/2*Sw_class2.T

def between_class_scatter_matrix(class1_data, class2_data):
    mean1 = get_mean(class1_data)
    mean2 = get_mean(class2_data, axis=0)
    diff = mean2 - mean1
    Sb = np.outer(diff, diff)
    return Sb

def multiclass_fisher_linear_discriminant(training_set):
    
    X = training_set.samples
    y = training_set.labels

    classes = np.unique(y)
    means = {c: get_mean(X[y == c]) for c in classes}
    overall_mean = get_mean(X)
    
    S_W = np.sum([np.dot((X[y == c] - means[c]).T, (X[y == c] - means[c])) for c in classes], axis=0)
    S_B = np.sum([len(X[y == c]) * np.dot((means[c] - overall_mean).reshape(-1, 1), 
                                          (means[c] - overall_mean).reshape(1, -1)) for c in classes], axis=0)
    
    # Solve the generalized eigenvalue problem
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B)) 

    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx[:1]]
    print(W)
    return W


def hand_coded_eigenvectors_eigenvalues(Sw_inv, Sb):
    M = np.dot(Sw_inv, Sb)
    eigen_values = np.zeros(Sw_inv.shape[0])
    eigen_vectors = np.zeros_like(Sw_inv)
    for i in range(Sw_inv.shape[0]):
        v = np.random.rand(Sw_inv.shape[0])
        for _ in range(50):
            v_new = np.dot(M, v)
            v = v_new
        eigen_values[i] = np.dot(v, np.dot(M, v))
        eigen_vectors[:, i] = v
        M -= eigen_values[i] * np.outer(v, v)
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]
    return eigen_values, eigen_vectors


def get_mean(data):
    num_features = data.shape[1]

    sum_data = np.zeros(num_features)
    total = 0
    for sample in data:
        total += sample
    mu = total / len(data)

    return mu

def linear_classifier(x, mu1, mu2, Sw_inv):
    diff1 = x - mu1
    diff2 = x - mu2

    if np.dot(np.dot(diff1,Sw_inv),diff1)<np.dot(np.dot(diff2,Sw_inv), diff2):
        return 1
    else:
        return 0

def apply_classifier(testing_set, mu1, mu2, Sw_inv):
    predicted_labels = []
    
    for i, sample in enumerate(testing_set):
        decision = linear_classifier(sample, mu1, mu2, Sw_inv)

        if decision > 0:
            predicted_labels.append(1)
        else:
            predicted_labels.append(2)


    return np.array(predicted_labels)



def calculate_error_rate(true_labels, predicted_labels):
    
    misclassified = np.sum(true_labels != predicted_labels)
    error_rate = (misclassified / len(true_labels)) * 100
    return error_rate


def project_data(W, dataset):
    return np.dot(W, dataset.T).T









