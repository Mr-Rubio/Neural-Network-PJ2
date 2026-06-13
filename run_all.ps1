# Run all Project-2 experiments in conda env newenv
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== Download CIFAR-10 (if needed) ==="
conda run -n newenv python "$ROOT\scripts\download_cifar10.py"

Write-Host "=== Part 2: VGG-A / BN / Loss Landscape ==="
Push-Location "$ROOT\codes\VGG_BatchNorm"
conda run -n newenv python VGG_Loss_Landscape.py --epochs 20
Pop-Location

Write-Host "=== Part 1: CIFAR-10 experiments ==="
Push-Location "$ROOT\codes\CIFAR10"
conda run -n newenv python run_experiments.py
conda run -n newenv python plot_comparison.py
conda run -n newenv python visualize.py --variant base --activation relu
Pop-Location

Write-Host "=== Done. See reports/ for outputs ==="
