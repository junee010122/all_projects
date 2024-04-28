import numpy as np
import os
from PIL import Image
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA

from utils.general import create_folder
from utils.plots import plot_images

class Dataset:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample = self.samples[index].reshape(28,28)
        label = self.labels[index].astype(np.int32)
        sample = Image.fromarray(sample)

        return sample, label

    def __len__(self):
        return len(self.samples)

def apply_pca(train_dataset, valid_dataset, n_components=100):
    train_samples = np.array([x[0] for x in train_dataset]).reshape(len(train_dataset), -1)
    valid_samples = np.array([x[0] for x in valid_dataset]).reshape(len(valid_dataset), -1)

    # Apply PCA
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_samples)
    valid_pca = pca.transform(valid_samples)

    # Return transformed data as new Dataset instances
    return Dataset(train_pca, train_dataset.labels), Dataset(valid_pca, valid_dataset.labels), pca




def binarize_data(data):

    binarized = (data > 0).astype(np.uint8)

    return binarized


def load_mnist(path):
    train = torchvision.datasets.MNIST(root=path, train=True, download=True)
    valid = torchvision.datasets.MNIST(root=path, train=False, download=True)

    # Convert MNIST data from torch dataset to numpy array and add channel dimension

    train_samples = train.data.unsqueeze(dim=-1).numpy()
    train_labels = train.targets.numpy()
    valid_samples = valid.data.unsqueeze(dim=-1).numpy()
    valid_labels = valid.targets.numpy()

    return train_samples, train_labels, valid_samples, valid_labels


#def save_dataset(path, dataset):
#    print("\nSaving Data To: %s\n" % path)
#    desc = "Saving Class Data"
#    for label in tqdm(np.unique(dataset.labels), desc=desc):
#        path_folder = os.path.join(path, str(label).zfill(3))
#        os.makedirs(path_folder, exist_ok=True)
#        indices = np.where(dataset.labels == label)[0]
#        class_samples = dataset.samples[indices]
#
#        if class_samples.shape[-1] == 1:  # Squeeze if grayscale image
#            class_samples = class_samples.squeeze(axis=-1)
#
#        for i, sample in enumerate(class_samples):
#            path_file = os.path.join(path_folder, str(i).zfill(5) + ".png")
#            Image.fromarray(sample).save(path_file)

def load_data(params):
    path = params["paths"]["data"]
    choice = params["dataset"]["type"]
    save_data = params["dataset"]["save"]
    batch_size = params["network"]["batch_size"]
    num_workers = params["system"]["num_workers"]
    binarize = params["dataset"]["binarize"]


    if choice == 0:
        path = os.path.join(path, "mnist")
        train_samples, train_labels, valid_samples, valid_labels = load_mnist(path)

    else:
        raise NotImplementedError

    if save_data:
        path_save = os.path.join(path, "train")
        save_dataset(path_save, train)
        path_save = os.path.join(path, "valid")
        save_dataset(path_save, valid)

    orig = train_samples

    if binarize:
        train = binarize_data(train_samples)
        valid = binarize_data(valid_samples)

    train_dataset = Dataset(train, train_labels)
    valid_dataset = Dataset(valid, valid_labels)

        # plot_images(orig, train)
    
    return train_dataset, valid_dataset


