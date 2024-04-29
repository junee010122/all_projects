import numpy as np

class Dataset:

    def __init__(self, samples, labels):
        
        self.samples = samples
        self.labels = labels

def make_class_data(num_samples, num_features, all_u, all_sig):

    data = []

    all_u = np.asarray(all_u)
    all_sig = np.asarray(all_sig)
    num_samples = np.asarray(num_samples)
    
    for num in num_samples:
        all_samples, all_labels = [],[]
        for i, (u,sig) in enumerate(zip(all_u, all_sig)):
            samples = np.random.multivariate_normal(u,sig,num)
            labels = [i] * num

            all_samples.append(samples)
            all_labels.append(labels)
        
        all_samples = np.vstack(all_samples)
        all_labels = np.vstack(all_labels).reshape(-1)
        dataset = Dataset(all_samples, all_labels)
        data.append(dataset)

    return data


    #all_u = np.array(data[][])
    #all_sig = 
    #num_samples = 


def load_datasets(params):

    num_features = params["datasets"]["num_features"]
    num_samples = params["datasets"]["num_samples"]
    all_u = params["elements"]["mean"]
    all_sig = params["elements"]["covariance"]
    prior = params["elements"]["prior"]

    return make_class_data(num_samples, num_features, all_u, all_sig)
#
