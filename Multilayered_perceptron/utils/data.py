import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.datasets import make_blobs, make_regression
from utils.plots import plot_data, plot_decision_boundary


class Dataset:
    def __init__(self, samples, labels, name):

        self.name = name
        self.labels = labels

        self.samples = samples
        # if regression problem, turn this on
        #self.samples = self.samples.to(torch.float32)
        self.samples = self.samples.astype(np.float32)
    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        return (self.samples[index], self.labels[index])


def load_datasets(params):

    # Load: Dataset

    num_samples = params["datasets"]["num_samples"]
    num_classes = params["datasets"]["num_classes"]
    num_features = params["datasets"]["num_features"]
    type_problem = params["network"]["type"]
    
    if type_problem == 'classification': 
        centers = np.array([[0.5, 0], [2, 0.0]])
        cluster_std = cluster_std = [0.5, 0.7]
        samples, labels = make_blobs(n_samples=num_samples, n_features=num_features, centers=centers, cluster_std=cluster_std)


    if type_problem == 'regression':
        samples = np.random.rand(num_samples, 1) * 10
        labels = -0.5 * samples**2 + 2 * samples - 3 + np.random.randn(num_samples, 1) * 0.5
        #samples, labels = make_regression(n_samples=num_samples, n_features=1, n_informative=1)
        samples = torch.tensor(samples, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
    # Partition: Data --> Datasets (Training, Testing)

    num_train = int(samples.shape[0] * 0.80)
    train_samples, test_samples = samples[:num_train], samples[num_train:]
    train_labels, test_labels = labels[:num_train], labels[num_train:]

    # Format: Datasets --> Supervised Learning Template

    train = Dataset(train_samples, train_labels, "Train Dataset")
    test = Dataset(test_samples, test_labels, "Test Dataset")


    # Plot: Datasets
    if params["datasets"]["show_plots"]:
        plot_data(train)
        plot_data(test, show_plots=1)

    # Convert: Datasets --> DataLoaders

    batch_size = params["network"]["batch_size"]
    num_workers = params["datasets"]["num_workers"]

    train = DataLoader(train, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, persistent_workers=True)

    test = DataLoader(test, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, persistent_workers=True)

    return {"train": train, "test": test}

# testing
#if __name__=="__main__":

#    params = yaml.load(open('../configs/params.yaml'), Loader = yaml.FullLoader)
#    dataset = load_dataset(params) 

#    batch = next(iter(dataset.train_dataloader())

def track_weights(model):
    """
    Track weights
    """

    weights = []
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.weight.data)
            weights.append(layer.bias.data)

    #if len(weights[-1]) == 2:
        #plot_data(weights=weights[-1])

    return weights



