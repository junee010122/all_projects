import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from utils.data import load_datasets, make_class_data
from utils.general import load_config
from utils.parametric import *
from utils.non_parametric import *

def run_experiment(params):

    data = load_datasets(params)
    train_1, train_2, test_1, test_2= data[0], data[1], data[2], data[3] 
    priors = params["elements"]["prior"]
    train_list = [train_1, train_2]
    test_list = [test_1, test_2]
    from IPython import embed
    embed()
    # part 1
    # (a,b)
    for t_set in test_list:
        test_samples = t_set.samples
        test_labels = t_set.labels
        class_means = compute_class_means(test_samples, test_labels)  
        predicted_labels = bayes_decision_rule(test_samples, test_labels, class_means, priors)
        print("Predicted labels:", predicted_labels)
        conf_matrix = confusion_matrix(test_labels, predicted_labels)
        print("Confusion Matrix:\n", conf_matrix)



    # (c)
    class_means = compute_class_means(train_1.samples, train_1.labels)
    class_covariances = compute_class_covariances(train_1.samples, train_1.labels)
    predicted_labels = ml_estimation(test_2.samples, class_means, class_covariances, priors)
    conf_matrix = confusion_matrix(test_2.labels, predicted_labels)
    print("Confusion Matrix:\n", conf_matrix)
    # (d)
    class_means = compute_class_means(train_2.samples, train_2.labels)
    class_covariances = compute_class_covariances(train_2.samples, train_2.labels)
    predicted_labels = ml_estimation(test_2.samples, class_means, class_covariances, priors)
    conf_matrix = confusion_matrix(test_2.labels, predicted_labels)
    print("Confusion Matrix:\n", conf_matrix)


    # part 2
    # (a) """ How should I select the number of dimensions? """
    W = multiclass_fisher_linear_discriminant(train_1)
    projected_testing_samples = np.dot(train_1.samples, W)
    mu = compute_class_means(projected_testing_samples, train_1.labels) 
    cov = compute_class_covariances(projected_testing_samples, train_1.labels)

    # (b)
    W = multiclass_fisher_linear_discriminant(test_2)
    projected_testing_samples_2 = np.dot(test_2.samples, W)
    mu_2 = compute_class_means(projected_testing_samples_2, test_2.labels)
    predicted_labels = bayes_decision_rule(projected_testing_samples_2, test_2.labels, mu_2, priors)
    conf_matrix = confusion_matrix(test_2.labels, predicted_labels)
    print("Confusion Matrix:\n", conf_matrix)

    # (c)
    W = multiclass_fisher_linear_discriminant(train_2)
    projected_testing_samples = np.dot(train_2.samples, W)
    mu = compute_class_means(projected_testing_samples, train_2.labels)
    cov = compute_class_covariances(projected_testing_samples, train_2.labels)
    projected_testing_samples = np.dot(test_2.samples, W)
    predicted_labels = bayes_decision_rule(projected_testing_samples, test_2.labels, mu, priors)
    conf_matrix = confusion_matrix(test_2.labels, predicted_labels)
    print("Confusion Matrix:\n", conf_matrix)

    
    # part 3
    # (a)
    predicted_labels = classify_parzen_window(train_1, test_2, hypercube_kernel,0.7)
    # (b)
    hn = [0.1, 0.7, 5]
    for h in hn:
        classify_parzen_window(train_2, test_2, hypercube_kernel,h)
    
    #(c)
    sigma = [0.1, 0.7, 5]
    for sig in sigma :
        predicted_labels_gaussian = classify_parzen_window(train_2, test_2, gaussian_kernel, sig)


    # part 4
    # (a)
    n_neighbors = int(np.sqrt(len(train_1.samples)))
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(train_1.samples, train_1.labels)
    predicted_labels = knn_classifier.predict(test_2.samples)
    #accuracy = np.mean(predicted_labels == test_label)
    #print("Classification Accuracy:", accuracy)
    
    # (b)
    n_neighbors = int(np.sqrt(len(train_2.samples)))
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(train_2.samples, train_2.labels)
    predicted_labels = knn_classifier.predict(test_2.samples)

    
    
    # part 5
    eta_1 = 1/2
    perceptron_1 = train_perceptron(train_1.samples, train_1.labels, eta_1)
    accuracy_1 = evaluate_perceptron(perceptron_1, test_2.samples, test_2.labels)
    print("Part 1) Classification Accuracy:", accuracy_1)
    eta_2 = 1/np.sqrt(perceptron_1.n_iter_)  # Use number of iterations from part a) for k
    perceptron_2 = train_perceptron(train_2.samples, train_2.labels, eta_2)
    accuracy_2 = evaluate_perceptron(perceptron_2, test_2.samples, test_2.labels)
    print("Part 2) Classification Accuracy:", accuracy_2)



if __name__ == "__main__":

    params = load_config(sys.argv)

    run_experiment(params)

