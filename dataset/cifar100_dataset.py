import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dataset.cifar100_config import *


def cifar100_train_loader(path, task_id, train_batch_size, num_workers=4, pin_memory=True, normalize=None):
    dataset_name = dataset[task_id]
    if normalize is None:
        normalize = transforms.Normalize(
            mean=mean[dataset_name], std=std[dataset_name])
    # train_transform = transform_train_cifar
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(f'{path}/train/{dataset_name}', train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def cifar100_val_loader(path, task_id, val_batch_size, num_workers=4, pin_memory=True, normalize=None):
    dataset_name = dataset[task_id]
    # test_transform = transform_test_cifar
    if normalize is None:
        normalize = transforms.Normalize(
            mean=mean[dataset_name], std=std[dataset_name])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


    val_dataset = \
        datasets.ImageFolder(f'{path}/test/{dataset_name}', test_transform
                )

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)
