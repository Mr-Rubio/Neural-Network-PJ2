"""Training script for CIFAR-10 custom CNN."""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from data.loaders import get_cifar_loader
from evaluate import get_accuracy, measure_epoch_time
from models.custom_cnn import build_model, get_number_of_parameters
from utils.nn import set_random_seeds

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_PATH = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(HOME_PATH, 'data')
FIGURES_PATH = os.path.join(HOME_PATH, 'reports', 'figures', 'cifar10')
MODELS_PATH = os.path.join(HOME_PATH, 'reports', 'models')
LOGS_PATH = os.path.join(HOME_PATH, 'reports', 'logs')


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def build_optimizer(model, name, lr, weight_decay):
    params = model.parameters()
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f'Unknown optimizer: {name}')


def train(model, train_loader, val_loader, device, optimizer, criterion,
          scheduler=None, epochs=30, run_name='run'):
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [], 'epoch_time': [],
    }
    best_val_acc = 0.0
    best_state = None

    for epoch in tqdm(range(epochs), desc=run_name):
        if scheduler is not None:
            scheduler.step()

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        batch_bar = tqdm(train_loader, desc=f'{run_name} e{epoch+1}', leave=False, unit='batch')
        for x, y in batch_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
            batch_bar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss = running_loss / total
        train_acc = correct / total
        val_acc = get_accuracy(model, val_loader, device)

        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                val_loss_sum += criterion(outputs, y).item() * y.size(0)
                val_total += y.size(0)
        val_loss = val_loss_sum / val_total

        epoch_time = time.time() - epoch_start
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f'Epoch {epoch+1}/{epochs} | '
              f'train loss {train_loss:.4f} acc {train_acc:.4f} | '
              f'val loss {val_loss:.4f} acc {val_acc:.4f} | '
              f'time {epoch_time:.1f}s')

    if best_state is not None:
        model.load_state_dict(best_state)
    history['best_val_acc'] = best_val_acc
    return history


def save_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'], label='val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(history['train_acc'], label='train')
    axes[1].plot(history['val_acc'], label='val')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='base', choices=['base', 'wide'])
    parser.add_argument('--activation', default='relu', choices=['relu', 'gelu'])
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true', help='5 epochs for smoke test')
    parser.add_argument('--fake-data', action='store_true',
                        help='Use FakeData when CIFAR-10 is unavailable')
    parser.add_argument('--n-items', type=int, default=-1)
    args = parser.parse_args()

    if args.quick:
        args.epochs = 5

    default_lrs = {'sgd': 0.1, 'adam': 1e-3, 'adamw': 1e-3}
    lr = args.lr if args.lr is not None else default_lrs[args.optimizer]

    os.makedirs(FIGURES_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)

    device = get_device()
    set_random_seeds(args.seed, device)

    train_loader = get_cifar_loader(
        root=DATA_PATH, batch_size=args.batch_size, train=True, augment=True,
        fake_data=args.fake_data, n_items=args.n_items)
    val_loader = get_cifar_loader(
        root=DATA_PATH, batch_size=args.batch_size, train=False,
        shuffle=False, augment=False, fake_data=args.fake_data,
        n_items=args.n_items)

    model = build_model(variant=args.variant, activation=args.activation)
    model.to(device)
    n_params = get_number_of_parameters(model)
    print(f'Model: {args.variant}, activation={args.activation}, params={n_params:,}')

    optimizer = build_optimizer(model, args.optimizer, lr, args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    if args.optimizer == 'sgd':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f'{args.variant}_{args.activation}_{args.optimizer}_wd{args.weight_decay}'
    history = train(model, train_loader, val_loader, device, optimizer, criterion,
                    scheduler=scheduler, epochs=args.epochs, run_name=run_name)

    model_path = os.path.join(MODELS_PATH, f'cifar10_{run_name}.pth')
    torch.save(model.state_dict(), model_path)

    fig_path = os.path.join(FIGURES_PATH, f'{run_name}_curves.png')
    save_training_curves(history, fig_path)

    log = {
        'variant': args.variant,
        'activation': args.activation,
        'optimizer': args.optimizer,
        'lr': lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'n_params': n_params,
        'best_val_acc': history['best_val_acc'],
        'avg_epoch_time': sum(history['epoch_time']) / len(history['epoch_time']),
        'model_path': model_path,
        'figure_path': fig_path,
    }
    log_path = os.path.join(LOGS_PATH, f'{run_name}.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)

    print(f'\nBest val accuracy: {history["best_val_acc"]:.4f}')
    print(f'Model saved: {model_path}')
    print(f'Log saved: {log_path}')


if __name__ == '__main__':
    main()
