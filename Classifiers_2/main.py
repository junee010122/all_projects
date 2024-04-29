import sys
import numpy as np

from utils.data import load_datasets, make_class_data
from utils.general import load_config
from utils.tools import *

def run_experiment(params):

    data = load_datasets(params)
    train_1, train_2, test_1, test_2= data[0], data[1], data[2], data[3]
    
    # part 1
    priors = params["elements"]["prior"]
    test_samples = test_1.samples
    test_labels = test_1.labels
    class_means = compute_class_means(test_samples, test_labels) # Dictionary mapping class labels to mean vectors
    class_probabilities = priors  

    predicted_labels = bayes_decision_rule(test_samples, test_labels, class_means, class_probabilities)
    print("Predicted labels:", predicted_labels)
    # part 2

    # part 3

    # part 4

    # part 5

if __name__ == "__main__":

    params = load_config(sys.argv)

    run_experiment(params)

