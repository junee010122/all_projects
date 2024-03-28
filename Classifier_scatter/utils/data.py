import scipy
import numpy as np

class Dataset:

    def __init__(self, samples, labels):
                
        self.samples = samples
        self.labels = labels

    
def load_mat_file(path):
    
    
    train = path + 'Train_Data'
    test = path + 'Test_Data'
    train_data = scipy.io.loadmat(train)
    test_data = scipy.io.loadmat(test)

    train_samples = np.vstack(train_data['data'])
    train_labels = np.hstack(train_data['labels'])
    test_samples = np.vstack(test_data['data'])
    test_labels = np.hstack(test_data['labels'])

    return Dataset(train_samples, train_labels), Dataset(test_samples, test_labels) 
            
