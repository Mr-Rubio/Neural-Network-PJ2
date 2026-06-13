"""Plot comparison charts from experiment logs."""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_PATH = os.path.dirname(os.path.dirname(SCRIPT_DIR))
LOGS_PATH = os.path.join(HOME_PATH, 'reports', 'logs')
FIGURES_PATH = os.path.join(HOME_PATH, 'reports', 'figures', 'cifar10')


def main():
    summary_path = os.path.join(LOGS_PATH, 'summary.json')
    if not os.path.exists(summary_path):
        print(f'No summary found at {summary_path}')
        return

    with open(summary_path, encoding='utf-8') as f:
        summary = json.load(f)

    os.makedirs(FIGURES_PATH, exist_ok=True)

    labels = [f"{r['variant']}_{r['activation']}_{r['optimizer']}" for r in summary]
    accs = [r['best_val_acc'] for r in summary]
    params = [r['n_params'] for r in summary]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(labels, accs, color='steelblue')
    axes[0].set_xlabel('Best Val Accuracy')
    axes[0].set_title('Optimizer / Architecture Comparison')
    axes[1].barh(labels, [p / 1e6 for p in params], color='coral')
    axes[1].set_xlabel('Parameters (millions)')
    axes[1].set_title('Model Size')
    fig.tight_layout()
    out = os.path.join(FIGURES_PATH, 'experiment_comparison.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
