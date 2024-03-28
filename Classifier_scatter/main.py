import numpy as np

from utils.data import load_mat_file
from utils.models import *
from utils.plots import plot_data, plot_projected



path = "/Users/june/Desktop/study/Pattern Recognition/MP3/"

if __name__ == "__main__":
    
    training_set, testing_set =load_mat_file(path)
    
    classes = np.unique(training_set.labels)
    class_samples = {label: [] for label in classes}

    for label, sample in zip(training_set.labels, training_set.samples):
        class_samples[label].append(sample) 

    # 1.1: Compute Within-Class Scatter Matrix
    class1 = np.asarray(class_samples[1])
    class2 = np.asarray(class_samples[2])
    Sw = within_class_scatter_matrix(class1, class2)
    Sw_inv = np.linalg.inv(Sw)
    mu1 = get_mean(class1)
    mu2 = get_mean(class2)
    W = np.dot(Sw_inv, (mu2 - mu1))


    predicted_labels=apply_classifier(testing_set.samples, mu1, mu2, Sw_inv)
    true_labels = testing_set.labels

    error_rate = calculate_error_rate(true_labels, predicted_labels)
    print("error rate:", error_rate, "%")

    plot_data(testing_set,true_labels,mu1,mu2, Sw_inv,W)

    # 1.2 Project points to each 2D surface - XY, YZ, ZX
    W_XY = np.array([[1, 0, 0], [0, 1, 0]])
    W_XZ = np.array([[1, 0, 0], [0, 0, 1]])
    W_YZ = np.array([[0, 1, 0], [0, 0, 1]])

    projected_data = [project_data(W, testing_set.samples) for W in (W_XY, W_XZ, W_YZ)]

    subspace_labels = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]

    for i, projected_data_subspace in enumerate(projected_data):
        classes = np.unique(training_set.labels)
        class_samples = {label: [] for label in classes}
        for label, sample in zip(testing_set.labels, projected_data_subspace):
            class_samples[label].append(sample)
        class1 = np.asarray(class_samples[1])
        class2 = np.asarray(class_samples[2])
        Sw = within_class_scatter_matrix(class1, class2)
        Sw_inv = np.linalg.inv(Sw)
        mu1 = get_mean(class1)
        mu2 = get_mean(class2)
        W = np.dot(Sw_inv, (mu2 - mu1))


        predicted_labels=apply_classifier(projected_data_subspace, mu1, mu2, Sw_inv)
        true_labels = testing_set.labels

        error_rate = calculate_error_rate(true_labels, predicted_labels)
        print("Error Rate:", error_rate, "%")
        plot_projected(subspace_labels[i], projected_data_subspace, true_labels, Sw_inv)

    # 1.3 Project points to each axis - X, Y, Z

    W_X = np.array([[1, 0, 0]])
    W_Y = np.array([[0, 1, 0]])
    W_Z = np.array([[0, 0, 1]])
    
    projected_data = [project_data(W, testing_set.samples) for W in (W_X, W_Y, W_Z)]
    subspace_labels_1D = ['X', 'Y', 'Z']
    for i, projected_data_subspace in enumerate(projected_data):
        classes = np.unique(training_set.labels)
        class_samples = {label: [] for label in classes}
        for label, sample in zip(testing_set.labels, projected_data_subspace):
            class_samples[label].append(sample)
        class1 = np.asarray(class_samples[1])
        class2 = np.asarray(class_samples[2])
        Sw = within_class_scatter_matrix(class1, class2)
        Sw_inv = np.linalg.inv(Sw)
        mu1 = get_mean(class1)
        mu2 = get_mean(class2)
        W = np.dot(Sw_inv, (mu2 - mu1))


        predicted_labels=apply_classifier(projected_data_subspace, mu1, mu2, Sw_inv)
        true_labels = testing_set.labels

        error_rate = calculate_error_rate(true_labels, predicted_labels)
        print("Error Rate:", error_rate, "%")

        plot_projected(subspace_labels[i], projected_data_subspace, true_labels, Sw_inv)


    # 2.1
    W = multiclass_fisher_linear_discriminant(training_set)
    W = np.expand_dims(np.array([1.0,2.0,-1.5]).T, axis=1)
    
    projected_testing_samples = np.dot(testing_set.samples, W)
    classes = np.unique(testing_set.labels)
    class_samples = {label: [] for label in classes}

    for label, sample in zip(testing_set.labels, projected_testing_samples):
        class_samples[label].append(sample)


    class1 = np.asarray(class_samples[1])
    class2 = np.asarray(class_samples[2])
    Sw = within_class_scatter_matrix(class1, class2)
    Sw_inv = np.linalg.inv(Sw)
    mu1 = get_mean(class1)
    mu2 = get_mean(class2)

    predicted_labels = apply_classifier(projected_testing_samples, mu1, mu2, Sw_inv)
    true_labels = testing_set.labels
    
    plot_projected(None,projected_testing_samples, true_labels, Sw_inv)
    error_rate = calculate_error_rate(true_labels, predicted_labels)
    print("error rate:", error_rate, "%")

    
        






