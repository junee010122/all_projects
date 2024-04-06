
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, name):
        self.name = name
        self.labels = labels
        self.samples = samples.astype(np.float32) 


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

def preprocess_secom_data(data_path, labels_path):

    data = pd.read_csv(data_path, sep=' ', header=None)
    labels = pd.read_csv(labels_path, sep=' ', header=None, usecols=[0])
    
    feature_names = ['Feature_' + str(i) for i in range(1, data.shape[1] + 1)]
    data.columns = feature_names
    labels.columns = ['Outcome']
    labels = (labels + 1) / 2

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    

    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data_imputed.columns)
    
    return data_scaled, labels.values.flatten()

def load_datasets(train_samples, train_labels, test_samples, test_labels):

    train_dataset = Dataset(train_samples, train_labels, "Train Dataset")
    test_dataset = Dataset(test_samples, test_labels, "Test Dataset")

    batch_size = 64
    num_workers = 5


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=True)

    return {"train": train_loader, "test": test_loader}

