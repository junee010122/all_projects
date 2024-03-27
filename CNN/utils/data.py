
import os
import numpy as np
import torchvision

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.general import create_folder


class Dataset:

    def __init__(self, samples, labels, transforms):

        self.transforms = transforms
        self.samples = samples
        self.labels = labels

    def augmentations(self, sample):

        # Normalize image [0, 1] by dividing by (2^8 - 1)

        sample = sample / 255

        # Re-organize sample shape (Pytorch likes this for images)
        # - Before = H W C
        # - After = C H W

        sample = sample.transpose(2, 0, 1)

        return sample

    def __getitem__(self, index):

        if self.transforms:
            sample = self.augmentations(self.samples[index])
        else:
            sample = self.samples[index]

        sample = sample.astype(np.float32)
        label = self.labels[index].astype(np.int32)

        return sample, label

    def __len__(self):

        return self.samples.shape[0]


def save_dataset(path, dataset):

    print("\nSaving Data To: %s\n" % path)

    desc = "Saving Class Data"
    for label in tqdm(np.unique(dataset.labels), desc=desc):

        path_folder = os.path.join(path, str(label).zfill(3))
        create_folder(path_folder)

        indices = np.where(label == dataset.labels)
        class_samples = dataset.samples[indices]

        if class_samples.shape[1] == 1:
            class_samples = class_samples.squeeze(axis=1)

        for i in range(class_samples.shape[0]):

            path_file = os.path.join(path_folder, str(i).zfill(5) + ".png")

            sample = class_samples[i]

            # Convert image (the numpy array) to a PIL Object and save it :)

            Image.fromarray(sample).save(path_file)


def load_cifar(path, transforms):

    train = torchvision.datasets.CIFAR10(root=path,
                                         train=True,
                                         download=True)

    valid = torchvision.datasets.CIFAR10(root=path,
                                         train=False,
                                         download=True)
    # Organize: Dataset

    train_samples = train.data
    train_labels = np.asarray(train.targets)

    valid_samples = valid.data
    valid_labels = np.asarray(valid.targets)

    # Template: Dataset

    train = Dataset(train_samples, train_labels, transforms)
    valid = Dataset(valid_samples, valid_labels, transforms)

    return train, valid


def load_mnist(path, transforms):

    # Download: Dataset

    train = torchvision.datasets.MNIST(root=path,
                                       train=True,
                                       download=True)

    valid = torchvision.datasets.MNIST(root=path,
                                       train=False,
                                       download=True)

    # Organize: Dataset

    train_samples = train.data.unsqueeze(dim=-1).numpy()
    train_labels = train.targets.numpy()

    valid_samples = valid.data.unsqueeze(dim=-1).numpy()
    valid_labels = valid.targets.numpy()

    # Template: Dataset

    train = Dataset(train_samples, train_labels, transforms)
    valid = Dataset(valid_samples, valid_labels, transforms)

    return train, valid


def load_data(params):

    path = params["paths"]["data"]
    choice = params["dataset"]["type"]
    save_data = params["dataset"]["save"]
    batch_size = params["network"]["batch_size"]
    num_workers = params["system"]["num_workers"]
    transforms = params["dataset"]["augmentations"]

    # Load: MNIST

    if choice == 0:
        path = os.path.join(path, "mnist")
        train, valid = load_mnist(path, transforms)

    # Load: CIFAR

    elif choice == 1:
        from IPython import embed
        embed()
        path = os.path.join(path, "cifar")
        train, valid = load_cifar(path, transforms)

    else:

        raise NotImplementedError

    # Save: Dataset

    if save_data:
        path_save = os.path.join(path, "train")
        save_dataset(path_save, train)

        path_save = os.path.join(path, "valid")
        save_dataset(path_save, valid)

    # Format: DataLoader

    train = DataLoader(train, batch_size=batch_size,
                       shuffle=True, num_workers=num_workers,
                       persistent_workers=True)

    valid = DataLoader(valid, batch_size=batch_size,
                       shuffle=False, num_workers=num_workers,
                       persistent_workers=True)

    return train, valid
