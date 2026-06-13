import argparse
import math
import os
import random
import sys
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_PATH = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIGURES_PATH = os.path.join(HOME_PATH, 'reports', 'figures')
MODELS_PATH = os.path.join(HOME_PATH, 'reports', 'models')
DATA_PATH = os.path.join(HOME_PATH, 'data')

BATCH_SIZE = 128
NUM_WORKERS = 0 if sys.platform == 'win32' else 4
LEARNING_RATES = [1e-3, 2e-3, 1e-4, 5e-4]
SEED = 2020


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if str(device) != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_accuracy(model, data_loader, device, show_progress=False, desc='val'):
    model.eval()
    correct = 0
    total = 0
    loader = data_loader
    if show_progress:
        loader = tqdm(data_loader, desc=desc, leave=False, unit='batch')
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            correct += (prediction.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def estimate_total_minutes(epochs_n, train_size, batch_size, num_runs, device):
    """Rough ETA for RTX 4090 / CPU (validation included per epoch)."""
    batches_per_epoch = math.ceil(train_size / batch_size)
    steps_per_run = epochs_n * batches_per_epoch * 2  # train + val
    total_steps = steps_per_run * num_runs
    if torch.cuda.is_available() and 'cpu' not in str(device):
        sec_per_step = 0.04  # ~4090, VGG-A, batch=128
        gpu = torch.cuda.get_device_name(0).lower()
        if '4090' in gpu:
            sec_per_step = 0.035
        elif '3080' in gpu or '3090' in gpu:
            sec_per_step = 0.05
    else:
        sec_per_step = 0.25
    return total_steps * sec_per_step / 60


def train(model, optimizer, criterion, train_loader, val_loader, device,
          scheduler=None, epochs_n=20, save_figures=False, run_name='',
          epoch_bar=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    batches_n = len(train_loader)
    losses_list = []
    grads = []

    epoch_iter = range(epochs_n)
    if epoch_bar is None:
        epoch_iter = tqdm(epoch_iter, unit='epoch', desc=run_name, position=1, leave=False)

    for epoch in epoch_iter:
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []
        grad = []
        epoch_loss = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f'{run_name} e{epoch + 1}/{epochs_n}',
            leave=False,
            unit='batch',
            position=2,
        )
        for x, y in batch_bar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()

            if model.classifier[4].weight.grad is not None:
                grad.append(model.classifier[4].weight.grad.clone().detach().cpu().numpy())
            loss_list.append(loss.item())
            epoch_loss += loss.item()

            optimizer.step()
            batch_bar.set_postfix(loss=f'{loss.item():.4f}')

        losses_list.append(loss_list)
        grads.append(grad)
        learning_curve[epoch] = epoch_loss / batches_n
        val_acc = get_accuracy(
            model, val_loader, device, show_progress=True,
            desc=f'{run_name} val e{epoch + 1}')
        val_accuracy_curve[epoch] = val_acc

        if epoch_bar is not None:
            epoch_bar.set_postfix(
                loss=f'{learning_curve[epoch]:.4f}', val_acc=f'{val_acc:.4f}')
        elif hasattr(epoch_iter, 'set_postfix'):
            epoch_iter.set_postfix(
                loss=f'{learning_curve[epoch]:.4f}', val_acc=f'{val_acc:.4f}')

        if save_figures:
            os.makedirs(FIGURES_PATH, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(15, 3))
            axes[0].plot(learning_curve)
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[1].plot(val_accuracy_curve)
            axes[1].set_title('Validation Accuracy')
            axes[1].set_xlabel('Epoch')
            fig.savefig(os.path.join(FIGURES_PATH, f'{run_name}_epoch_{epoch:03d}.png'))
            plt.close(fig)

    return losses_list, grads, learning_curve, val_accuracy_curve


def flatten_step_losses(all_losses):
    """Flatten per-epoch batch losses into a single step-wise list."""
    flat = []
    for epoch_losses in all_losses:
        flat.extend(epoch_losses)
    return flat


def aggregate_min_max_curves(all_losses):
    """For each training step, take min/max loss across models with different LRs."""
    min_len = min(len(flatten_step_losses(losses)) for losses in all_losses)
    min_curve = []
    max_curve = []
    for step in range(min_len):
        step_values = []
        for losses in all_losses:
            flat = flatten_step_losses(losses)
            step_values.append(flat[step])
        min_curve.append(min(step_values))
        max_curve.append(max(step_values))
    return min_curve, max_curve


def plot_loss_landscape(min_curve, max_curve, title, save_path):
    steps = np.arange(len(min_curve))
    plt.figure(figsize=(10, 5))
    plt.plot(steps, min_curve, label='min loss', linewidth=1.5)
    plt.plot(steps, max_curve, label='max loss', linewidth=1.5)
    plt.fill_between(steps, min_curve, max_curve, alpha=0.3)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison_landscape(min_no_bn, max_no_bn, min_bn, max_bn, save_path):
    steps = np.arange(len(min_no_bn))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].plot(steps, min_no_bn, label='min', linewidth=1.2)
    axes[0].plot(steps, max_no_bn, label='max', linewidth=1.2)
    axes[0].fill_between(steps, min_no_bn, max_no_bn, alpha=0.3)
    axes[0].set_title('VGG-A (without BN)')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(steps, min_bn, label='min', linewidth=1.2)
    axes[1].plot(steps, max_bn, label='max', linewidth=1.2)
    axes[1].fill_between(steps, min_bn, max_bn, alpha=0.3)
    axes[1].set_title('VGG-A + BatchNorm')
    axes[1].set_xlabel('Training Step')
    axes[1].legend()

    fig.suptitle('Loss Landscape Comparison (multiple learning rates)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def gradient_predictiveness(grads):
    """Average cosine similarity between consecutive gradient snapshots."""
    flat_grads = []
    for epoch_grads in grads:
        flat_grads.extend(epoch_grads)
    if len(flat_grads) < 2:
        return []
    similarities = []
    for i in range(1, len(flat_grads)):
        g0 = flat_grads[i - 1].reshape(-1)
        g1 = flat_grads[i].reshape(-1)
        denom = np.linalg.norm(g0) * np.linalg.norm(g1)
        if denom < 1e-12:
            similarities.append(0.0)
        else:
            similarities.append(float(np.dot(g0, g1) / denom))
    return similarities


def run_multi_lr_experiment(model_cls, train_loader, val_loader, device, epochs_n, lrs,
                            phase_bar=None):
    all_losses = []
    all_grads = []
    for lr in lrs:
        run_name = f'{model_cls.__name__}_lr_{lr}'
        if phase_bar is not None:
            phase_bar.set_description(f'LR={lr}')
        set_random_seeds(SEED, device)
        model = model_cls()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        losses, grads, _, _ = train(
            model, optimizer, criterion, train_loader, val_loader, device,
            epochs_n=epochs_n, run_name=run_name, epoch_bar=phase_bar)
        all_losses.append(losses)
        all_grads.append(grads)
        if phase_bar is not None:
            phase_bar.update(1)
    return all_losses, all_grads


def run_vgg_comparison(train_loader, val_loader, device, epochs_n=20, phase_bar=None):
    """Train VGG-A and VGG-A+BN with the same hyperparameters for comparison."""
    results = {}
    for model_cls in (VGG_A, VGG_A_BatchNorm):
        name = model_cls.__name__
        if phase_bar is not None:
            phase_bar.set_description(f'Compare {name}')
        set_random_seeds(SEED, device)
        model = model_cls()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        losses, grads, learning_curve, val_acc_curve = train(
            model, optimizer, criterion, train_loader, val_loader, device,
            epochs_n=epochs_n, save_figures=False, run_name=name, epoch_bar=phase_bar)
        if phase_bar is not None:
            phase_bar.update(1)
        results[name] = {
            'learning_curve': learning_curve,
            'val_accuracy_curve': val_acc_curve,
            'final_val_acc': val_acc_curve[-1],
            'grad_similarity': gradient_predictiveness(grads),
        }
        os.makedirs(MODELS_PATH, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(MODELS_PATH, f'{name}.pth'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, res in results.items():
        axes[0].plot(res['learning_curve'], label=name)
        axes[1].plot(res['val_accuracy_curve'], label=name)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    fig.tight_layout()
    os.makedirs(FIGURES_PATH, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_PATH, 'vgg_bn_comparison.png'), dpi=150)
    plt.close(fig)

    return results


def main():
    parser = argparse.ArgumentParser(description='VGG-A Loss Landscape experiment')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--n-items', type=int, default=-1,
                        help='Use partial dataset for quick runs (e.g. 5000)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 5 epochs, 5000 samples')
    parser.add_argument('--fake-data', action='store_true',
                        help='Use FakeData when CIFAR-10 is unavailable')
    parser.add_argument('--compare-only', action='store_true',
                        help='Only run VGG-A vs BN comparison (skip landscape)')
    parser.add_argument('--plots-only', action='store_true',
                        help='Regenerate PNGs from saved min/max curve txt files')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 5
        args.n_items = 5000

    os.makedirs(FIGURES_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)

    device = get_device()
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    train_loader = get_cifar_loader(
        root=DATA_PATH, batch_size=BATCH_SIZE, train=True,
        num_workers=NUM_WORKERS, n_items=args.n_items, fake_data=args.fake_data)
    val_loader = get_cifar_loader(
        root=DATA_PATH, batch_size=BATCH_SIZE, train=False,
        shuffle=False, num_workers=NUM_WORKERS, n_items=args.n_items,
        fake_data=args.fake_data)

    # Verify dataloader
    train_size = len(train_loader.dataset)
    for X, y in train_loader:
        print(f'Sample batch shape: {X.shape}, labels: {y[:4].tolist()}')
        break

    if args.plots_only:
        min_no_bn = np.loadtxt(os.path.join(FIGURES_PATH, 'min_curve_no_bn.txt'))
        max_no_bn = np.loadtxt(os.path.join(FIGURES_PATH, 'max_curve_no_bn.txt'))
        min_bn = np.loadtxt(os.path.join(FIGURES_PATH, 'min_curve_bn.txt'))
        max_bn = np.loadtxt(os.path.join(FIGURES_PATH, 'max_curve_bn.txt'))
        plot_loss_landscape(min_no_bn, max_no_bn, 'VGG-A Loss Landscape',
                            os.path.join(FIGURES_PATH, 'loss_landscape_no_bn.png'))
        plot_loss_landscape(min_bn, max_bn, 'VGG-A + BN Loss Landscape',
                            os.path.join(FIGURES_PATH, 'loss_landscape_bn.png'))
        min_len = min(len(min_no_bn), len(min_bn))
        plot_comparison_landscape(
            min_no_bn[:min_len], max_no_bn[:min_len],
            min_bn[:min_len], max_bn[:min_len],
            os.path.join(FIGURES_PATH, 'loss_landscape_comparison.png'))
        print(f'Plots regenerated in {FIGURES_PATH}')
        return

    num_runs = 2 if args.compare_only else 2 + len(LEARNING_RATES) * 2
    est_min = estimate_total_minutes(
        args.epochs, train_size, BATCH_SIZE, num_runs, device)
    print(f'\nWorkload: {num_runs} training runs x {args.epochs} epochs '
          f'({train_size} train samples, batch={BATCH_SIZE})')
    print(f'Estimated time on this GPU: ~{est_min:.0f} min ({est_min / 60:.1f} h)')

    t0 = time.time()
    main_bar = tqdm(total=num_runs, desc='Overall', unit='run', position=0)

    print('\n=== VGG-A vs VGG-A+BN comparison ===')
    comparison = run_vgg_comparison(
        train_loader, val_loader, device, epochs_n=args.epochs, phase_bar=main_bar)
    for name, res in comparison.items():
        print(f'{name}: final val acc = {res["final_val_acc"]:.4f}')

    if args.compare_only:
        main_bar.close()
        elapsed = time.time() - t0
        print(f'\nTotal elapsed: {elapsed / 60:.1f} min')
        print(f'Figures saved to {FIGURES_PATH}')
        print(f'Models saved to {MODELS_PATH}')
        return

    print('\n=== Loss landscape (VGG-A, multiple LRs) ===')
    main_bar.set_description('Landscape no-BN')
    losses_no_bn, grads_no_bn = run_multi_lr_experiment(
        VGG_A, train_loader, val_loader, device, args.epochs, LEARNING_RATES,
        phase_bar=main_bar)
    min_no_bn, max_no_bn = aggregate_min_max_curves(losses_no_bn)
    plot_loss_landscape(
        min_no_bn, max_no_bn, 'VGG-A Loss Landscape',
        os.path.join(FIGURES_PATH, 'loss_landscape_no_bn.png'))

    print('\n=== Loss landscape (VGG-A+BN, multiple LRs) ===')
    main_bar.set_description('Landscape +BN')
    losses_bn, grads_bn = run_multi_lr_experiment(
        VGG_A_BatchNorm, train_loader, val_loader, device, args.epochs, LEARNING_RATES,
        phase_bar=main_bar)
    main_bar.close()
    min_bn, max_bn = aggregate_min_max_curves(losses_bn)
    plot_loss_landscape(
        min_bn, max_bn, 'VGG-A + BN Loss Landscape',
        os.path.join(FIGURES_PATH, 'loss_landscape_bn.png'))

    min_len = min(len(min_no_bn), len(min_bn))
    plot_comparison_landscape(
        min_no_bn[:min_len], max_no_bn[:min_len],
        min_bn[:min_len], max_bn[:min_len],
        os.path.join(FIGURES_PATH, 'loss_landscape_comparison.png'))

    np.save(os.path.join(FIGURES_PATH, 'losses_no_bn.npy'), losses_no_bn, allow_pickle=True)
    np.save(os.path.join(FIGURES_PATH, 'losses_bn.npy'), losses_bn, allow_pickle=True)
    np.savetxt(os.path.join(FIGURES_PATH, 'min_curve_no_bn.txt'), min_no_bn)
    np.savetxt(os.path.join(FIGURES_PATH, 'max_curve_no_bn.txt'), max_no_bn)
    np.savetxt(os.path.join(FIGURES_PATH, 'min_curve_bn.txt'), min_bn)
    np.savetxt(os.path.join(FIGURES_PATH, 'max_curve_bn.txt'), max_bn)

    sim_no_bn = gradient_predictiveness(grads_no_bn[0])
    sim_bn = gradient_predictiveness(grads_bn[0])
    if sim_no_bn and sim_bn:
        min_sim_len = min(len(sim_no_bn), len(sim_bn))
        plt.figure(figsize=(10, 4))
        plt.plot(sim_no_bn[:min_sim_len], label='VGG-A (lr=1e-3)', alpha=0.8)
        plt.plot(sim_bn[:min_sim_len], label='VGG-A+BN (lr=1e-3)', alpha=0.8)
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Cosine Similarity')
        plt.title('Gradient Predictiveness')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, 'gradient_predictiveness.png'), dpi=150)
        plt.close()

    elapsed = time.time() - t0
    print(f'\nTotal elapsed: {elapsed / 60:.1f} min ({elapsed / 3600:.2f} h)')
    print(f'Figures saved to {FIGURES_PATH}')
    print(f'Models saved to {MODELS_PATH}')


if __name__ == '__main__':
    main()
