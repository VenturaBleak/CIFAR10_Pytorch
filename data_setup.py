"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

def get_num_workers(device):
    """
    :param device: torch.device: The device to use for the DataLoader.
    :return:the number of workers to use for the DataLoader.
    """
    if str(device) != 'cpu':
        # In case of GPU is available: Use half the number of CPU cores, but at least 2
        num_workers = max(2, int((os.cpu_count())/2))
    else:
        # Use 0 workers for CPU
        num_workers = 0
    print('Number of workers:', num_workers)
    return num_workers

def train_mean_std(train_dataset,
                   train_sampler,
                   batch_size=32):
    """
    Calculates the mean and standard deviation of the training data.
    :param train_dataset:
    :param batch_size:
    :return:
    """

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              sampler=train_sampler,
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

def get_dataloaders_cifar10(batch_size: int,
                            device: torch.device,
                            resolution=None,
                            augmentation=False,
                            validation_fraction=None):
    """
    Returns the train, validation (if specified), and test data loaders for the CIFAR10 dataset using PyTorch.

    Args:
    :param batch_size: The batch size to use for the data loaders.
    :param device: The device to use for loading the data. This can be either 'cpu' or 'cuda'.
    :param resolution: The size of the images in pixels. Must be specified.
    :param augmentation: Whether to apply data augmentation to the training data.
    :param validation_fraction: The fraction of the training data to use for validation. If not specified, no validation
    data will be returned.

    Returns:
    If validation_fraction is None:
    :return: A tuple containing the train loader augmented, the train loader, the test loader, and a dictionary
    mapping class names to indices.
    If validation_fraction is not None:
    :return: A tuple containing the train loader augmented, the train loader, the validation loader, the test loader,
    and a dictionary mapping class names to indices.

    :raises ValueError: If resolution is not specified.
    """

    # raise error if resolution is not specified
    if resolution is None:
        raise ValueError('Resolution must be specified')

    root = "CIFAR10"

    num_workers = get_num_workers(device)

    initial_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(root=root,
                                     train=True,
                                     transform=initial_transform,
                                     download=True)

    valid_dataset = datasets.CIFAR10(root=root,
                                     train=True,
                                     download=False,
                                     transform=initial_transform)

    test_dataset = datasets.CIFAR10(root=root,
                                    train=False,
                                    download=True,
                                    transform=initial_transform)

    # Get class names to idx mapping
    classes_to_idx = train_dataset.class_to_idx

    # set indices for training and validation
    if str(device) == 'cpu':
        # if using cpu, only use 400 samples
        train_data_length = 200
        train_indices = torch.arange(0, train_data_length)
        if validation_fraction is not None:
            valid_indices = torch.arange(train_data_length, train_data_length*2)
    else:
        # if using gpu, use all samples
        train_data_length = len(train_dataset)
        if validation_fraction is not None:
            # split the training data into training and validation
            num = int(validation_fraction * len(train_dataset))
            train_indices = torch.arange(0, train_data_length - num)
            valid_indices = torch.arange(train_data_length - num, train_data_length)
        else:
            train_indices = torch.arange(0, train_data_length)

    # specify the sampler for training and validation
    train_sampler = SubsetRandomSampler(train_indices)
    if validation_fraction is not None:
        valid_sampler = SubsetRandomSampler(valid_indices)

    # get the mean and standard deviation of the training data
    train_mean, train_std = train_mean_std(train_dataset, train_sampler)

    # define the transforms
    test_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])
    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std) # results in images with mean = 0 and std = 1
        ])
        print('Augmentation will be applied.')
    else:
        train_transform = test_transform

    train_dataset_aug = datasets.CIFAR10(root=root,
                                     train=True,
                                     transform=train_transform,
                                     download=False)

    train_dataset = datasets.CIFAR10(root=root,
                                     train=True,
                                     transform=test_transform,
                                     download=False)

    valid_dataset = datasets.CIFAR10(root=root,
                                     train=True,
                                     download=False,
                                     transform=test_transform)

    test_dataset = datasets.CIFAR10(root=root,
                                    train=False,
                                    download=False,
                                    transform=test_transform)
    if str(device) == 'cpu':
        test_dataset = Subset(test_dataset, train_indices)

    if validation_fraction is not None:
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader_aug = DataLoader(dataset=train_dataset_aug,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)

    else:
        train_loader_aug = DataLoader(dataset=train_dataset_aug,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    if validation_fraction is None:
        return train_loader_aug, train_loader, test_loader, classes_to_idx
    else:
        return train_loader_aug, train_loader, valid_loader, test_loader, classes_to_idx