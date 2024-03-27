
import sys
import numpy as np

from utils.model import decision_model
from utils.general import load_config
from utils.data import load_datasets, make_class_data
from utils.plot import plot_data, plot_pdf, plot_pdf_on_mesh

def run_experiment(params):
    
    dataset=load_datasets(params)
    all_sample = dataset.samples
    all_label = dataset.labels

    if(params["datasets"]["show"]):
        plot_data(dataset)

    model = decision_model(params)

    num_samples = params["datasets"]["num_samples"]
    num_features = params["datasets"]["num_features"]

    mean = params["elements"]["mean"]
    covariance = params["elements"]["covariance"]
    prior = np.asarray(params["elements"]["prior"])

    # 1 --> compute Mahalanobis distance 

    point = np.random.multivariate_normal(mean[0], covariance[0])
    M_distance = decision_model.get_M_distance(point, Sig=covariance[0], mu=mean[0])

    print(f"The Mahalanobis distance between {point} and {mean[0]} is {M_distance}")
        
    # 2 ---> compute discriminant function
    
    point1 = np.random.multivariate_normal(mean[0],covariance[0]) #--> class 1
    point2 = np.random.multivariate_normal(mean[1], covariance[1]) #--> class2
    M_distance1= decision_model.get_M_distance(point1, mu=mean[1], Sig=covariance[1])
    M_distance2= decision_model.get_M_distance(point2, mu=mean[1], Sig=covariance[1])
    dec_1 = decision_model.Discriminant_function(M_distance1, num_features, covariance[0], prior[0])
    dec_2 = decision_model.Discriminant_function(M_distance2, num_features, covariance[1], prior[1])
    
    if dec_1 - dec_2 > 0:
        print("Class 1")
    else:
        print("Class 2")
    

    # 3,5 --> plot prob density function + dewhitening transform & posterior probability
    indices = np.concatenate((np.zeros(num_samples, dtype=int), np.ones(num_samples, dtype=int)), axis=0)
    pdf_vals=[]
    posterior_vals=[]
    for i, (sample, index) in enumerate(zip(all_sample,indices)): 
        pdf=decision_model.probability_density(sample, mean[index], covariance[index], prior[index])
        pdf_vals.append(pdf)
    plot_pdf(dataset, pdf_vals)
    
    x_min = dataset.samples[0].min()
    x_max = dataset.samples[0].max()
    y_min = dataset.samples[1].min()
    y_max = dataset.samples[1].max()

    plot_pdf_on_mesh(mean,covariance,prior, -5, 10, -5, 10,100)
    decision_model.posterior_prob(mean[0],covariance[0], mean[1], covariance[1], prior[0], prior[1])

    # 4 --> plot decision boundary
    indices = np.concatenate((np.zeros(num_samples, dtype=int), np.ones(num_samples, dtype=int)), axis=0)
    dec = decision_model.decision_boundary(all_sample, mean, covariance, prior)

    # 6,7,8 --> change test set

if __name__ == "__main__":

    params = load_config(sys.argv)

    run_experiment(params)


