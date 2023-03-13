"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

NUM_WORKERS = 0

def get_datasets(batch_size=32, train_indices=None):
    """
    Loads the CIFAR10 dataset and calculates the mean and standard deviation of the training data.

    Examples:
    mean, std = load_data(64)

    Args:

    Returns:
    :returns: (tuple) A tuple containing the mean and standard deviation of the training data.

    .. _link:
    https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-standardized.ipynb
    """
    # Setup training data
    train_dataset = datasets.CIFAR10(
        root="CIFAR10",  # where to download data to?
        train=True,  # get training data
        download=True,  # download data if it doesn't exist on disk
        transform=transforms.ToTensor()  # images come as PIL format, we want to turn into Torch tensors
    )

    # Setup test data
    test_dataset = datasets.CIFAR10(
        root="CIFAR10",  # where to download data to?
        train=False,  # get test data
        download=True,  # download data if it doesn't exist on disk
        transform=transforms.ToTensor()  # images come as PIL format, we want to turn into Torch tensors
    )
    # Get class names to idx mapping
    classes_to_idx = train_dataset.class_to_idx

    if train_indices is not None:
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, train_indices)

    return train_dataset, test_dataset, classes_to_idx

def train_mean_std(train_dataset, batch_size=32):
    """
    Calculates the mean and standard deviation of the training data.
    :param train_dataset:
    :param batch_size:
    :return:
    """

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=False)
    train_mean = []
    train_std = []
    for i, image in enumerate(train_loader, 0):
        numpy_image = image[0].numpy()

        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std = np.std(numpy_image, axis=(0, 2, 3))

        train_mean.append(batch_mean)
        train_std.append(batch_std)
    train_mean = torch.tensor(np.mean(train_mean, axis=0))
    train_std = torch.tensor(np.mean(train_std, axis=0))
    # convert mean and std to tuple
    print('Mean:', train_mean)
    print('Std Dev:', train_std)
    return train_mean, train_std


def load_data(batch_size, train_transform, test_transform, train_indices=None):
    """
    Loads the CIFAR10 dataset and creates PyTorch DataLoaders for the training and test data.

    Args:
    :param test_transform:
    :param batch_size (int): The number of samples in each batch.
    :param transform (transforms.Compose): The transformation to apply to the data.

    Returns:
    :returns: (tuple) A tuple containing the training and test DataLoaders.

    Example Usage:
    train_loader, test_loader = load_data(64)
    """

    # Setup training data
    train_dataset = datasets.CIFAR10(
        root="CIFAR10",  # where to download data to?
        train=True,  # get training data
        download=False,  # download data if it doesn't exist on disk
        transform=train_transform
    )

    # Setup test data
    test_dataset = datasets.CIFAR10(
        root="CIFAR10",  # where to download data to?
        train=False,  # get test data
        download=False,  # download data if it doesn't exist on disk
        transform=test_transform
    )

    if train_indices is not None:
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, train_indices)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers= NUM_WORKERS,
                              pin_memory=True)


    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers= NUM_WORKERS,
                             pin_memory=True)

    return train_dataset, test_dataset, train_loader, test_loader