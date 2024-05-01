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

def train_perceptron(train_data, train_labels, eta):
    """
    Train a Perceptron model with a given learning rate.
    
    Parameters:
        train_data (numpy array): Training data.
        train_labels (numpy array): Labels corresponding to the training data.
        eta (float): Learning rate.
        
    Returns:
        Perceptron: Trained Perceptron model.
    """
    perceptron = Perceptron(eta0=eta, max_iter=1000)  # Set maximum number of iterations
    perceptron.fit(train_data, train_labels)
    return perceptron

def evaluate_perceptron(perceptron, test_data, test_labels):
    """
    Evaluate a trained Perceptron model on test data.
    
    Parameters:
        perceptron (Perceptron): Trained Perceptron model.
        test_data (numpy array): Test data.
        test_labels (numpy array): Labels corresponding to the test data.
        
    Returns:
        float: Classification accuracy.
    """
    predicted_labels = perceptron.predict(test_data)
    accuracy = np.mean(predicted_labels == test_labels)
    return accuracy

