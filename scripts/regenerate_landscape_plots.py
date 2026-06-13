"""Regenerate loss landscape PNGs from saved min/max curve txt files."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(HOME, 'reports', 'figures')


def main():
    for tag, title in [('no_bn', 'VGG-A Loss Landscape'), ('bn', 'VGG-A + BN Loss Landscape')]:
        mn = np.loadtxt(os.path.join(FIG_DIR, f'min_curve_{tag}.txt'))
        mx = np.loadtxt(os.path.join(FIG_DIR, f'max_curve_{tag}.txt'))
        steps = np.arange(len(mn))
        plt.figure(figsize=(10, 5))
        plt.plot(steps, mn, label='min')
        plt.plot(steps, mx, label='max')
        plt.fill_between(steps, mn, mx, alpha=0.3)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'loss_landscape_{tag}.png'), dpi=150)
        plt.close()

    mn0 = np.loadtxt(os.path.join(FIG_DIR, 'min_curve_no_bn.txt'))
    mx0 = np.loadtxt(os.path.join(FIG_DIR, 'max_curve_no_bn.txt'))
    mn1 = np.loadtxt(os.path.join(FIG_DIR, 'min_curve_bn.txt'))
    mx1 = np.loadtxt(os.path.join(FIG_DIR, 'max_curve_bn.txt'))
    n = min(len(mn0), len(mn1))
    steps = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mn, mx, title in [
        (axes[0], mn0, mx0, 'VGG-A (without BN)'),
        (axes[1], mn1, mx1, 'VGG-A + BatchNorm'),
    ]:
        ax.plot(steps, mn[:n], label='min')
        ax.plot(steps, mx[:n], label='max')
        ax.fill_between(steps, mn[:n], mx[:n], alpha=0.3)
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel('Training Step')
    axes[0].set_ylabel('Loss')
    fig.suptitle('Loss Landscape Comparison')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'loss_landscape_comparison.png'), dpi=150)
    plt.close(fig)
    print(f'Plots saved to {FIG_DIR}')


if __name__ == '__main__':
    main()
