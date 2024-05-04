import numpy as np
from sklearn.linear_model import Perceptron

def hypercube_kernel(x, h):
    """
    Hypercube (Rectangular) kernel function.
    
    Parameters:
        x (numpy array): Input data point.
        h (float): Bandwidth.
        
    Returns:
        float: Kernel function value.
    """
    kernel_values = np.zeros_like(x)
    mask = np.abs(x) <= h / 2
    kernel_values[mask] = 1 / h
    return kernel_values

def gaussian_kernel(x, sigma):
    """
    Gaussian kernel function.
    
    Parameters:
        x (numpy array): Input data point or array.
        sigma (float): Standard deviation.
        
    Returns:
        numpy array: Kernel function values.
    """
    return np.exp(-0.5 * (x / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


def parzen_window_estimate(train_set, test_point,kernel, h):
    """
    Compute Parzen Window estimate for a single test point.
    
    Parameters:
        train_set (tuple): Tuple containing training samples and labels.
        test_point (numpy array): Test point for which to compute the estimate.
        h (float): Bandwidth.
        
    Returns:
        numpy array: Density estimate for each class.
    """
    train_samples, train_labels = train_set
    classes = np.unique(train_labels)
    num_classes = len(classes)
    class_density = np.zeros(num_classes)

    for i, c in enumerate(classes):
        # Select samples belonging to class c
        class_samples = train_samples[train_labels == c]
        
        # Compute density estimate for class c using the specified kernel
        num_samples_c = len(class_samples)
        distances = np.abs(test_point - class_samples)
        kernel_values = kernel(distances, h)
        
        # Sum kernel values to compute density estimate
        class_density[i] += np.sum(kernel_values)
    
    # Normalize density estimate by total number of samples
    class_density /= len(train_samples)
    
    return class_density


def classify_parzen_window(train_set, test_set, kernel,h):
    """
    Classify test set using Parzen Window method.
    
    Parameters:
        train_set (object): Object containing training samples and labels.
        test_set (object): Object containing test samples.
        h (float): Bandwidth.
        
    Returns:
        numpy array: Predicted labels for test set.
    """
    train_samples = train_set.samples
    train_labels = train_set.labels
    test_samples = test_set.samples
    num_test_samples = len(test_samples)
    predicted_labels = np.zeros(num_test_samples)
    
    for i, test_point in enumerate(test_samples):
        # Compute Parzen Window estimate for test point
        density_estimate = parzen_window_estimate((train_samples, train_labels), test_point,kernel, h)
        
        # Calculate posterior probability using Bayes' theorem
        # P(class | data) = P(data | class) * P(class) / P(data)
        # Since P(data) is constant for all classes, we only need to compare the numerator
        posterior_probs = density_estimate / np.sum(density_estimate)
        
        # Predict class label with highest posterior probability
        predicted_labels[i] = np.argmax(posterior_probs)
    
    return predicted_labels


def predict_one_vs_all(samples, classifiers):
    # Compute scores for each classifier
    scores = [np.dot(samples, weights) + bias for weights, bias in classifiers]
    # Determine the class with the highest score
    return np.argmax(scores, axis=0)

def train_perceptron(samples, labels, eta=0.01, max_iter=1000, dynamic_eta=False):
    n_samples, n_features = samples.shape
    weights = np.zeros(n_features)
    bias = 0
    for k in range(1, max_iter + 1):
        if dynamic_eta:
            current_eta = 1 / np.sqrt(k)  # Dynamic learning rate based on iteration number
        else:
            current_eta = eta  # Use constant learning rate

        has_converged = True
        for idx in range(n_samples):
            if labels[idx] * (np.dot(weights, samples[idx]) + bias) <= 0:
                weights += current_eta * labels[idx] * samples[idx]
                bias += current_eta * labels[idx]
                has_converged = False
        if has_converged:
            break
    return weights, bias

def train_one_vs_all(samples, labels, eta=0.01, max_iter=1000, dynamic_eta=False):
    n_classes = len(np.unique(labels))
    classifiers = []
    for current_class in range(n_classes):
        binary_labels = np.where(labels == current_class, 1, -1)  # Convert to binary labels for the current class
        weights, bias = train_perceptron(samples, binary_labels, eta, max_iter, dynamic_eta)
        classifiers.append((weights, bias))
    return classifiers
