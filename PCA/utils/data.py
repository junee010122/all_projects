"""
Purpose: Data tools
Author: June
"""

import scipy
import numpy as np


class Dataset:
    """
    Template for supervised learning dataset
    """

    def __init__(self, samples, labels):
        """
        Assign dataset information

        Arguments:
        - samples (np.ndarray[float]): dataset samples
        - labels (np.ndarray[int]): dataset labels
        """

        self.samples = samples
        self.labels = labels


def load_mat_file(path):

    raw_data = scipy.io.loadmat(path)
    raw_data = raw_data["Data"].reshape(-1)

    all_samples, all_labels = [], []
    for i, class_samples in enumerate(raw_data):
        all_samples.append(class_samples.T)
        all_labels.append([i] * class_samples.shape[1])

    all_samples = np.vstack(all_samples)
    all_labels = np.hstack(all_labels).reshape(-1)

    return Dataset(all_samples, all_labels)
