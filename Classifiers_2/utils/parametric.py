import numpy as np

def covariance(x, y):

    return np.sum(((x - np.mean(x)) * (y - np.mean(y)))) / (x.shape[0] - 1)

def compute_class_covariances(train_samples, train_labels):
    unique_labels = np.unique(train_labels)
    class_covariances = {}
    for label in unique_labels:
        class_samples = train_samples[train_labels == label]
        class_covariances[label] = get_cov_matrix(class_samples)
    return class_covariances


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

    return matrix

def Discriminant_function(m_dist,dim, cov, prior):
    discriminant_val =-0.5 * (m_dist ** 2) - 0.5 * dim * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)) + np.log(prior)
    return discriminant_val

def ml_estimation(samples, class_means, class_covariances, class_probabilities):
    discriminant_values = []
    for label in class_means.keys():
        mean = class_means[label]
        covariance = class_covariances[label]
        prior = class_probabilities[label]
        diff = samples - mean
        m_dist = np.sqrt(np.sum(np.dot(diff, np.linalg.inv(covariance)) * diff, axis=1))
        discriminant_val = Discriminant_function(m_dist, len(mean), covariance, prior)
        discriminant_values.append(discriminant_val)
    predicted_labels = np.argmax(np.column_stack(discriminant_values), axis=1)
    return predicted_labels

def bayes_decision_rule(samples, labels, class_means, class_probabilities):
    # Compute a single covariance matrix for the whole dataset
    total_cov_matrix = get_cov_matrix(samples)
    
    discriminant_values = []

    for label in np.unique(labels):
        mean = class_means[label]
        prior = class_probabilities[label]
        
        diff = samples - mean
        m_dist = np.sqrt(np.sum(np.dot(diff, np.linalg.inv(total_cov_matrix)) * diff, axis=1))
        discriminant_val = Discriminant_function(m_dist, len(mean), total_cov_matrix, prior)
        discriminant_values.append(discriminant_val)
    
    predicted_labels = np.argmax(discriminant_values, axis=0)
    
    return predicted_labels
def get_mean(data):
    num_features = data.shape[1]

    sum_data = np.zeros(num_features)
    total = 0
    for sample in data:
        total += sample
    mu = total / len(data)

    return mu

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
    W = eigvecs[:, idx[:2]]
    print(W)
    return W




