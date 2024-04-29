import numpy as np

def covariance(x, y):

    return np.sum(((x - np.mean(x)) * (y - np.mean(y)))) / (x.shape[0] - 1)

def compute_class_means(train_samples, train_labels):
    unique_labels = np.unique(train_labels)
    class_means = {}
    
    for label in unique_labels:
        class_samples = train_samples[train_labels == label]
        class_mean = np.mean(class_samples, axis=0)
        class_means[label] = class_mean
    
    return class_means

def get_cov_matrix(data):

    num_features = data.shape[-1]

    matrix = np.zeros((num_features, num_features))

    for i in range(num_features):
        for j in range(num_features):
            matrix[i][j] = covariance(data[:, i], data[:, j])

    print("Covariance Matrix:")
    print(matrix)

    return matrix

def Discriminant_function(m_dist,dim, cov, prior):
    discriminant_val =-0.5 * (m_dist ** 2) - 0.5 * dim * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)) + np.log(prior)
    return discriminant_val

def bayes_decision_rule(test_samples, test_labels, class_means, class_probabilities):
    covariance_matrices = {}
    for label in np.unique(test_labels):
        class_samples = test_samples[test_labels == label]
        covariance_matrices[label] = get_cov_matrix(class_samples)
    
    discriminant_values = []
    for label in np.unique(test_labels):
        mean = class_means[label]
        covariance = covariance_matrices[label]
        prior = class_probabilities[label]
        
        diff = test_samples - mean
        m_dist = np.sqrt(np.sum(np.dot(diff, np.linalg.inv(covariance)) * diff, axis=1))
        
        discriminant_val = Discriminant_function(m_dist, len(mean), covariance, prior)
        discriminant_values.append(discriminant_val)
    
    predicted_labels = np.argmax(discriminant_values, axis=0)
    
    return predicted_labels

