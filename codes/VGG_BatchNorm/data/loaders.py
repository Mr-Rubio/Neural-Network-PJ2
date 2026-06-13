"""
Data loaders
"""
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets



class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def _default_num_workers():

    return 0 if sys.platform == 'win32' else 4


def get_cifar_loader(root='../data/', batch_size=128, train=True, shuffle=True,
                     num_workers=None, n_items=-1, fake_data=False):
    if num_workers is None:
        num_workers = _default_num_workers()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    data_transforms = transforms.Compose(
        [transforms.ToTensor(),
         normalize])

    if fake_data:
        size = n_items if n_items > 0 else 5000
        dataset = datasets.FakeData(
            size=size, image_size=(3, 32, 32), num_classes=10,
            transform=data_transforms)
    else:
        dataset = datasets.CIFAR10(
            root=root, train=train, download=True, transform=data_transforms)
        if n_items > 0:
            dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader

if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break