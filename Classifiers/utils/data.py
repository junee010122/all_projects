import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader
import torch
from torchvision import transforms

from utils.general import create_folder
from utils.plots import plot_images

class Dataset:
    def __init__(self, samples, labels, transform=None):
        self.transform = transform  # This will be torch_standardize or None
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index].astype(np.int32)

        # Convert numpy array to PIL Image for processing
        sample = Image.fromarray(sample)

        if self.transform:
            sample = self.transform(sample)  # Apply the torch_standardize

        return sample, label

    def __len__(self):
        return len(self.samples)

def torch_standardize(x, binarize=False):

    desc = "Data must be formatted: [H, W, C], where C = 1 or C = 3"
    assert x.mode in ['RGB', 'L'], desc  # Assuming x is a PIL Image

    if binarize:
        x = Image.fromarray(np.where(np.array(x) > 0, 255, 0).astype('unit8'))

    num_channels = 3 if x.mode == 'RGB' else 1
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)
    ])
    return transformation(x)

def load_cifar(path, transform):
    train = torchvision.datasets.CIFAR10(root=path, train=True, download=True)
    valid = torchvision.datasets.CIFAR10(root=path, train=False, download=True)

    train_samples = train.data
    train_labels = np.asarray(train.targets)
    valid_samples = valid.data
    valid_labels = np.asarray(valid.targets)

    train_dataset = Dataset(train_samples, train_labels, transform)
    valid_dataset = Dataset(valid_samples, valid_labels, transform)

    return train_dataset, valid_dataset

def load_mnist(path, transform):
    train = torchvision.datasets.MNIST(root=path, train=True, download=True)
    valid = torchvision.datasets.MNIST(root=path, train=False, download=True)

    # Convert MNIST data from torch dataset to numpy array and add channel dimension
    train_samples = train.data.unsqueeze(dim=-1).numpy()
    train_labels = train.targets.numpy()
    valid_samples = valid.data.unsqueeze(dim=-1).numpy()
    valid_labels = valid.targets.numpy()

    train_dataset = Dataset(train_samples, train_labels, transform)
    valid_dataset = Dataset(valid_samples, valid_labels, transform)

    return train_dataset, valid_dataset

def save_dataset(path, dataset):
    print("\nSaving Data To: %s\n" % path)
    desc = "Saving Class Data"
    for label in tqdm(np.unique(dataset.labels), desc=desc):
        path_folder = os.path.join(path, str(label).zfill(3))
        os.makedirs(path_folder, exist_ok=True)
        indices = np.where(dataset.labels == label)[0]
        class_samples = dataset.samples[indices]

        if class_samples.shape[-1] == 1:  # Squeeze if grayscale image
            class_samples = class_samples.squeeze(axis=-1)

        for i, sample in enumerate(class_samples):
            path_file = os.path.join(path_folder, str(i).zfill(5) + ".png")
            Image.fromarray(sample).save(path_file)

def load_data(params):
    path = params["paths"]["data"]
    choice = params["dataset"]["type"]
    save_data = params["dataset"]["save"]
    batch_size = params["network"]["batch_size"]
    num_workers = params["system"]["num_workers"]
    binarize = params["dataset"]["binarize"]
    transform = None

    if params["dataset"]["augmentations"]:
        transform = lambda x: torch_standardize(x, binarize=binarize)

    if choice == 0:
        path = os.path.join(path, "mnist")
        train, valid = load_mnist(path, transform)
    elif choice == 1:
        path = os.path.join(path, "cifar")
        train, valid = load_cifar(path, transform)
    else:
        raise NotImplementedError

    if save_data:
        path_save = os.path.join(path, "train")
        save_dataset(path_save, train)
        path_save = os.path.join(path, "valid")
        save_dataset(path_save, valid)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

    return train_loader, valid_loader


