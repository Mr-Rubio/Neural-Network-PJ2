"""CIFAR-10 data loaders with augmentation."""

import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _default_num_workers():
    return 0 if sys.platform == 'win32' else 4


def get_cifar_loader(root='../../data/', batch_size=128, train=True,
                     shuffle=True, num_workers=None, augment=True, fake_data=False,
                     n_items=-1):
    if num_workers is None:
        num_workers = _default_num_workers()
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    if train and augment:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    if fake_data:
        size = n_items if n_items > 0 else 5000
        dataset = datasets.FakeData(
            size=size, image_size=(3, 32, 32), num_classes=10, transform=transform)
    else:
        dataset = datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader
