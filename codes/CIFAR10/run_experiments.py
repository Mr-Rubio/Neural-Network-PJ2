"""Run all Part-1 comparison experiments and summarize results."""

import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_PATH = os.path.dirname(os.path.dirname(SCRIPT_DIR))
LOGS_PATH = os.path.join(HOME_PATH, 'reports', 'logs')


# Use: conda activate newenv && python run_experiments.py
EXPERIMENTS = [
    # variant, activation, optimizer, weight_decay, extra_args
    ('base', 'relu', 'adam', 0.0, []),
    ('base', 'relu', 'adamw', 1e-2, []),
    ('base', 'relu', 'sgd', 5e-4, []),
    ('base', 'gelu', 'adam', 0.0, []),
    ('wide', 'relu', 'adam', 0.0, []),
    ('base', 'relu', 'adam', 5e-4, []),  # L2 regularization via weight decay
]


def main():
    quick = '--quick' in sys.argv
    fake_data = '--fake-data' in sys.argv
    os.makedirs(LOGS_PATH, exist_ok=True)
    train_script = os.path.join(SCRIPT_DIR, 'train.py')

    for variant, activation, optimizer, wd, extra in EXPERIMENTS:
        cmd = [
            sys.executable, train_script,
            '--variant', variant,
            '--activation', activation,
            '--optimizer', optimizer,
            '--weight-decay', str(wd),
        ] + extra
        if quick:
            cmd.append('--quick')
        if fake_data:
            cmd.append('--fake-data')
        print('\n>>>', ' '.join(cmd))
        subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)

    # Summarize
    summary = []
    for fname in sorted(os.listdir(LOGS_PATH)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(LOGS_PATH, fname), encoding='utf-8') as f:
            summary.append(json.load(f))

    summary.sort(key=lambda x: x.get('best_val_acc', 0), reverse=True)
    summary_path = os.path.join(LOGS_PATH, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\n=== Experiment Summary (by val accuracy) ===')
    for row in summary:
        print(f"{row['variant']:5s} {row['activation']:4s} {row['optimizer']:5s} "
              f"wd={row['weight_decay']:<6} acc={row['best_val_acc']:.4f} "
              f"params={row['n_params']:,}")
    print(f'\nSummary saved to {summary_path}')

    if summary:
        best = summary[0]
        import shutil
        models_path = os.path.join(HOME_PATH, 'reports', 'models')
        src = best.get('model_path')
        dst = os.path.join(models_path, 'best_cifar10.pth')
        if src and os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'Best model copied to {dst} (acc={best["best_val_acc"]:.4f})')


if __name__ == '__main__':
    main()
