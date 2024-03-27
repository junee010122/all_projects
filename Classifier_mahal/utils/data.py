import numpy as np
import yaml

class Dataset:

    def __init__(self, samples, labels):

        self.samples = samples
        self.labels = labels

def make_class_data(num_samples, num_features, all_u, all_sig):
    
    all_samples, all_labels = [], []

    with open('/Users/june/Desktop/study/Pattern Recognition/MP2/configs/params.yaml', 'r') as file:
        data = yaml.safe_load(file)

    all_u = np.array(data['elements']['mean'])

    all_sig = np.array(data['elements']['covariance'])

    for i, (u, sig) in enumerate(zip(all_u, all_sig)):

        samples = np.random.multivariate_normal(np.array([0,0]), np.array([[1,0],[0,1]]), num_samples)
        samples = dewhitening(samples, u, sig)
        labels = [i] * num_samples
        all_samples.append(samples)
        all_labels.append(labels)

    all_samples = np.vstack(all_samples)

    all_labels = np.vstack(all_labels).reshape(-1)

    return Dataset(all_samples, all_labels)

def dewhitening(data, u, sig):

    covariance_matrix = sig
    eigenvalue_estimate, eigenvector_estimate = compute_eigenvectors_eigenvalues(u,sig)

    data = np.dot(np.linalg.inv(np.dot(np.linalg.inv(np.sqrt(eigenvalue_estimate)),eigenvector_estimate.T)), data.T) + np.tile(u, (1000, 1)).T

    return data.T


def load_datasets(params):

    num_samples = params["datasets"]["num_samples"]

    num_features = params["datasets"]["num_features"]

    all_u = params["elements"]["mean"]

    all_sig = params["elements"]["covariance"]

    return make_class_data(num_samples, num_features, all_u, all_sig)

def compute_covariance_matrix(data):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    covariance_matrix = np.zeros((centered_data.shape[1], centered_data.shape[1]))
    
    for i in range(centered_data.shape[1]):
        for j in range(i, centered_data.shape[1]):
            covariance_matrix[i, j] = np.sum(centered_data[:, i] * centered_data[:, j]) / (len(centered_data) - 1)
            covariance_matrix[j, i] = covariance_matrix[i, j]  # The covariance matrix is symmetric
            
    return covariance_matrix


def compute_eigenvectors_eigenvalues(mu, C):

    """
    solve for det([[C11- lambda, C12],[C21, C22-lambda]]) = 0
    (C11-lambda)(C22-lambda)-C12*C21=0
    lambda**2-(C11+C22)*lambda + C11*C22-C12*C21=0
    """
    C11, C12, C21, C22 = C.flatten()

    lambda1 = (C11 + C22 + np.sqrt((C11 + C22)**2 - 4*(C11 * C22 - C12*C21))) / 2
    lambda2 = (C11 + C22 - np.sqrt((C11 + C22)**2 - 4*(C11 * C22 - C12*C21))) / 2

    # Compute the eigenvectors
    if C12 != 0:
        # Calculate the first eigenvector using the first eigenvalue
        x1 = 1
        y1 = (lambda1 - C11) / C12
        # Calculate the second eigenvector using the second eigenvalue
        x2 = 1
        y2 = (lambda2 - C11) / C12
    else:
        # Use standard basis vectors if C12 is zero
        x1 = 1 
        y1 = 0
        x2 = 0        
        y2 = 1

    # Normalize the eigenvectors
    v1 = np.array([x1, y1])
    v2 = np.array([x2, y2])
    magnitude_v1 = np.sqrt(x1 ** 2 + y1 ** 2)
    magnitude_v2 = np.sqrt(x2 ** 2 + y2 ** 2)
    v1 = v1 / magnitude_v1
    v2 = v2 / magnitude_v2

    
    # Arrange the eigenvalues into a diagonal matrix
    eigenvalues = np.array([[lambda1, 0], [0, lambda2]])

    # Arrange the eigenvectors into a matrix
    eigenvectors = np.array([v1, v2])

    return eigenvalues, eigenvectors


