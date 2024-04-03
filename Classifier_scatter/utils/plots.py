import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from utils.models import linear_classifier


def plot_data(testing_set, true_labels,mu1,mu2,Sw_inv,W):


    classes = np.unique(testing_set.labels)
    class_samples = {label: [] for label in classes}

    for label, sample in zip(testing_set.labels, testing_set.samples):
        class_samples[label].append(sample) 
    class1 = np.asarray(class_samples[1])
    class2 = np.asarray(class_samples[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = testing_set.samples[:, 0].min() - 1, testing_set.samples[:, 0].max() + 1
    y_min, y_max = testing_set.samples[:, 1].min() - 1, testing_set.samples[:, 1].max() + 1
    z_min, z_max = testing_set.samples[:, 2].min() - 1, testing_set.samples[:, 2].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    W = mu1 - mu2
    midpoint = (mu1 + mu2) / 2
    Z = (-W[0] * xx - W[1] * yy + W.dot(midpoint)) / W[2]

    ax.plot_surface(xx, yy, Z, alpha=0.5, color='gray')

    ax.scatter(testing_set.samples[:, 0], testing_set.samples[:, 1], testing_set.samples[:, 2], c=true_labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('3D Dataset with Decision Boundary')

    plt.show()

def plot_projected(subspace, projected_data, true_labels,Sw_inv):
    if projected_data.shape[1] == 1:
        # Plot projected data in one-dimensional subspace
        plt.scatter(projected_data[true_labels == 1], np.zeros_like(projected_data[true_labels == 1]), c='b', label='Class 1')
        plt.scatter(projected_data[true_labels == 2], np.zeros_like(projected_data[true_labels == 2]), c='r', label='Class 2')
        mu1 = np.mean(projected_data[true_labels == 1])
        mu2 = np.mean(projected_data[true_labels == 2])

        boundary_point = None
        for i in range(1, len(projected_data)):
            if linear_classifier(projected_data[i], mu1, mu2, Sw_inv) != linear_classifier(projected_data[i - 1], mu1, mu2, Sw_inv):
                boundary_point = projected_data[i]
                break

        # Plot the dataset and decision boundary change point
        plt.scatter(projected_data, np.zeros_like(projected_data), c=true_labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        plt.scatter(boundary_point, 0, color='green', label='Decision Boundary', s=50)
        plt.xlabel('Feature')
        plt.title('1D Dataset with Decision Boundary Change Point')
        plt.legend()
        plt.show()

    elif projected_data.shape[1] == 2:
        # Plot projected data in two-dimensional subspace
        plt.scatter(projected_data[true_labels == 1, 0], projected_data[true_labels == 1, 1], c='b', label='Class 1')
        plt.scatter(projected_data[true_labels == 2, 0], projected_data[true_labels == 2, 1], c='r', label='Class 2')
    
        mu1 = np.mean(projected_data[true_labels == 1], axis=0)
        mu2 = np.mean(projected_data[true_labels == 2], axis=0)
        
        x_min, x_max = projected_data[:, 0].min() - 1, projected_data[:, 0].max() + 1
        y_min, y_max = projected_data[:, 1].min() - 1, projected_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
        mesh_data = np.c_[xx.ravel(), yy.ravel()]
        predictions = np.array([linear_classifier(point, mu1, mu2, Sw_inv) for point in mesh_data])
        predictions = predictions.reshape(xx.shape)

        # Plot the dataset and decision boundary
        plt.contourf(xx, yy, predictions, alpha=0.3)
        plt.scatter(projected_data[:, 0], projected_data[:, 1], c=true_labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('2D Dataset with Decision Boundary')
        plt.show()
    plt.show()



