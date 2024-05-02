import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from utils.data import load_datasets, make_class_data
from utils.general import load_config
from utils.parametric import *
from utils.non_parametric import *
from utils.plots import plot_confusion_matrix, plot_performance_metrics, plot_projected, plot_decision_boundary
np.random.seed(42) 
def run_experiment(params):

    data = load_datasets(params)
    train_1, train_2, test_1, test_2= data[0], data[1], data[2], data[3] 
    priors = params["elements"]["prior"]
    train_list = [train_1, train_2]
    test_list = [test_1, test_2]
    class_names = [0, 1, 2]
    path = '/Users/june/Desktop/'
     

    from IPython import embed
    # part 1
    # (a,b)
    for i, t_set in enumerate(test_list):
        test_samples = t_set.samples
        test_labels = t_set.labels
        class_means = compute_class_means(test_samples, test_labels)  
        predicted_labels = bayes_decision_rule(test_samples, test_labels, class_means, priors)
        plot_confusion_matrix(test_labels, predicted_labels, class_names, f'Part1_({i})_confusion', path)
        plot_performance_metrics(test_labels, predicted_labels, f'Part1_({i})_evaluation', path) 
            # (c)
    class_means = compute_class_means(train_1.samples, train_1.labels)
    class_covariances = compute_class_covariances(train_1.samples, train_1.labels)
    predicted_labels = ml_estimation(test_2.samples, class_means, class_covariances, priors)
    plot_confusion_matrix(test_2.labels, predicted_labels, class_names, 'Part1_(c)_confusion', path)
    plot_performance_metrics(test_2.labels, predicted_labels, 'Part1_(c)_evaluation', path) 

    # (d)
    class_means = compute_class_means(train_2.samples, train_2.labels)
    class_covariances = compute_class_covariances(train_2.samples, train_2.labels)
    predicted_labels = ml_estimation(test_2.samples, class_means, class_covariances, priors)
    plot_confusion_matrix(test_2.labels, predicted_labels, class_names, 'Part1_(d)_confusion', path)
    plot_performance_metrics(test_2.labels, predicted_labels, 'Part1_(d)_evaluation', path) 
 


    # part 2
    # (a) """ How should I select the number of dimensions? """
    W = multiclass_fisher_linear_discriminant(train_1)
    projected_testing_samples = np.dot(train_1.samples, W)
    mu = compute_class_means(projected_testing_samples, train_1.labels) 
    cov = compute_class_covariances(projected_testing_samples, train_1.labels)
    print(f"part2 (a) projected_data: {projected_testing_samples.shape}")
    print(mu)
    print(cov)
    plot_projected(projected_testing_samples, train_1.labels)
    from IPython import embed
    embed()


    # (b)
    W = multiclass_fisher_linear_discriminant(test_2)
    projected_testing_samples_2 = np.dot(test_2.samples, W)
    mu_2 = compute_class_means(projected_testing_samples_2, test_2.labels)
    predicted_labels = bayes_decision_rule(projected_testing_samples_2, test_2.labels, mu_2, priors)
    plot_confusion_matrix(test_2.labels, predicted_labels, class_names, 'Part2_(b)_confusion', path)

    # (c)
    W = multiclass_fisher_linear_discriminant(train_2)
    projected_testing_samples = np.dot(train_2.samples, W)
    mu = compute_class_means(projected_testing_samples, train_2.labels)
    cov = compute_class_covariances(projected_testing_samples, train_2.labels)
    projected_testing_samples = np.dot(test_2.samples, W)
    print(f"part2 (c) projected_data: {projected_testing_samples.shape}")
    print(mu)
    print(cov)
    predicted_labels = bayes_decision_rule(projected_testing_samples, test_2.labels, mu, priors)
    plot_confusion_matrix(test_2.labels, predicted_labels, class_names, 'Part2_(c)_confusion', path)
    #print("Confusion Matrix:\n", conf_matrix)
    plot_projected(projected_testing_samples, test_2.labels) 

    
    # part 3
    # (a)
    hn = [0.1, 0.7, 5]
  
    for h in hn:
 
        predicted_labels = classify_parzen_window(train_1, test_2, hypercube_kernel,h)
        plot_confusion_matrix(test_2.labels, predicted_labels, class_names, f'Part3_(a)_confusion_{h}', path)
        plot_performance_metrics(test_2.labels, predicted_labels,f'Part3_(a)_evaluation_{h}', path) 

    # (b)

    for h in hn:
 
        predicted_labels = classify_parzen_window(train_2, test_2, hypercube_kernel,h)
        plot_confusion_matrix(test_2.labels, predicted_labels, class_names, f'Part3_(b)_confusion_{h}', path)
        plot_performance_metrics(test_2.labels, predicted_labels,f'Part3_(b)_evaluation_{h}', path) 
    
    #(c)
   
    sigma = [0.1, 0.7, 5]
    for sig in sigma :
        predicted_labels_gaussian = classify_parzen_window(train_2, test_2, gaussian_kernel, sig)
        plot_confusion_matrix(test_2.labels, predicted_labels, class_names,  f'Part3_(c)_confusion_{sig}', path)
        plot_performance_metrics(test_2.labels, predicted_labels,f'Part3_(c)_evaluation_{sig}', path)


    # part 4
    # (a)
    n_neighbors = int(np.sqrt(len(train_1.samples)))
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(train_1.samples, train_1.labels)
    predicted_labels = knn_classifier.predict(test_2.samples)
    plot_confusion_matrix(test_2.labels, predicted_labels, class_names,  f'Part4_(a)_confusion', path)
    plot_performance_metrics(test_2.labels, predicted_labels,  f'Part4_(a)_evaluation', path)
    #plot_decision_boundary(test_2.samples, test_2.labels, knn_classifier, 'k-NN Classifier Decision Boundary and Testing Data')

    
    # (b)
    n_neighbors = int(np.sqrt(len(train_2.samples)))
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(train_2.samples, train_2.labels)
    predicted_labels = knn_classifier.predict(test_2.samples)
    plot_confusion_matrix(test_2.labels, predicted_labels, class_names,  f'Part4_(b)_confusion', path)
    plot_performance_metrics(test_2.labels, predicted_labels, f'Part4_(b)_evaluation', path)
    #plot_decision_boundary(test_2.samples, test_2.labels, knn_classifier, 'k-NN Classifier Decision Boundary and Testing Data')

    # (c)
    n_neighbors = int(np.cbrt(len(train_1.samples)))
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(train_1.samples, train_1.labels)
    predicted_labels = knn_classifier.predict(test_2.samples)
    plot_confusion_matrix(test_2.labels, predicted_labels, class_names, f'Part4_(c)_confusion', path)
    plot_performance_metrics(test_2.labels, predicted_labels, f'Part4_(c)_evaluation', path)
    #plot_decision_boundary(test_2.samples, test_2.labels, knn_classifier, 'k-NN Classifier Decision Boundary and Testing Data')



      # part 5
      # eta : learning rate 
    eta_1 = 0.5
    weights_1, bias_1, n_iter_1 = train_perceptron(train_1.samples, train_1.labels, eta_1)
    predicted_labels_1 = predict(test_2.samples, weights_1, bias_1)
    plot_confusion_matrix(test_2.labels, predicted_labels_1, class_names, f'Part5_(1)_confusion', path)
    plot_performance_metrics(test_2.labels, predicted_labels_1, f'Part5_(1)_evaluation', path)

    eta_2 = 1/np.sqrt(n_iter_1)
    weights_2, bias_2, n_iter_2 = train_perceptron(train_1.samples, train_1.labels, eta_2)
    predicted_labels_2 = predict(test_2.samples, weights_2, bias_2)
    plot_confusion_matrix(test_2.labels, predicted_labels_2, class_names, f'Part5_(2)_confusion', path)
    plot_performance_metrics(test_2.labels, predicted_labels_2, f'Part5_(2)_evaluation', path)
    from IPython import embed
    embed()

if __name__ == "__main__":

    params = load_config(sys.argv)

    run_experiment(params)
#
