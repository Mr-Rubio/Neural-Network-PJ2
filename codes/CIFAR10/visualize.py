"""Visualization utilities for CIFAR-10 models."""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.loaders import CIFAR10_MEAN, CIFAR10_STD, get_cifar_loader
from models.custom_cnn import build_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_PATH = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIGURES_PATH = os.path.join(HOME_PATH, 'reports', 'figures', 'cifar10')
MODELS_PATH = os.path.join(HOME_PATH, 'reports', 'models')
DATA_PATH = os.path.join(HOME_PATH, 'data')

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def denormalize(tensor):
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return tensor.cpu() * std + mean


def plot_filters(model, save_path, max_filters=64):
    conv = model.first_conv
    weights = conv.weight.data.cpu().numpy()
    n_filters = min(weights.shape[0], max_filters)
    grid_size = int(np.ceil(np.sqrt(n_filters)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            filt = np.transpose(weights[i], (1, 2, 0))
            filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
            ax.imshow(filt)
        ax.axis('off')
    fig.suptitle('First Conv Layer Filters')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_activation_maps(model, image, device, save_path):
    model.eval()
    activations = []

    def hook_fn(module, inp, out):
        activations.append(out.detach())

    handle = model.stage1[0].block[0].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(image.unsqueeze(0).to(device))
    handle.remove()

    if not activations:
        return

    feat = activations[0][0]
    n_maps = min(16, feat.shape[0])
    grid = int(np.ceil(np.sqrt(n_maps)))
    fig, axes = plt.subplots(grid, grid, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < n_maps:
            ax.imshow(feat[i].cpu().numpy(), cmap='viridis')
        ax.axis('off')
    fig.suptitle('Stage-1 Activation Maps')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sample_predictions(model, loader, device, save_path, n_samples=8):
    model.eval()
    images, labels, preds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            pred = out.argmax(1)
            for i in range(min(x.size(0), n_samples - len(images))):
                images.append(denormalize(x[i]))
                labels.append(y[i].item())
                preds.append(pred[i].item())
            if len(images) >= n_samples:
                break

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, img, gt, pr in zip(axes.flat, images, labels, preds):
        ax.imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 1))
        color = 'green' if gt == pr else 'red'
        ax.set_title(f'gt:{CIFAR10_CLASSES[gt]}\npred:{CIFAR10_CLASSES[pr]}', color=color)
        ax.axis('off')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='base')
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--model-path', default=None)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(variant=args.variant, activation=args.activation)
    model_path = args.model_path or os.path.join(
        MODELS_PATH, f'cifar10_{args.variant}_{args.activation}_adam_wd0.0.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f'Loaded weights from {model_path}')
    model.to(device)

    tag = f'{args.variant}_{args.activation}'
    os.makedirs(FIGURES_PATH, exist_ok=True)
    plot_filters(model, os.path.join(FIGURES_PATH, f'{tag}_filters.png'))

    val_loader = get_cifar_loader(
        root=DATA_PATH, batch_size=64, train=False, shuffle=True, augment=False)
    for x, y in val_loader:
        plot_activation_maps(
            model, x[0], device,
            os.path.join(FIGURES_PATH, f'{tag}_activations.png'))
        break

    plot_sample_predictions(
        model, val_loader, device,
        os.path.join(FIGURES_PATH, f'{tag}_predictions.png'))
    print(f'Visualizations saved to {FIGURES_PATH}')


if __name__ == '__main__':
    main()
