# Project-2 运行说明

环境：`conda activate newenv`

## Part 2：VGG-A + BatchNorm + Loss Landscape

```powershell
cd codes\VGG_BatchNorm
python VGG_Loss_Landscape.py --epochs 20          # 完整实验
python VGG_Loss_Landscape.py --compare-only --epochs 20  # 仅 VGG 对比
python VGG_Loss_Landscape.py --plots-only         # 从 txt 重绘 Loss Landscape
python VGG_Loss_Landscape.py --quick              # 快速调试
```

## Part 1：自定义 CIFAR-10 CNN

```powershell
cd codes\CIFAR10
python train.py --variant base --activation gelu --optimizer adam --epochs 30
python run_experiments.py                         # 全部对照实验
python plot_comparison.py                         # 优化器/结构对比图
python visualize.py --variant base --activation gelu --model-path ..\..\reports\models\best_cifar10.pth
```

## 实验结果摘要

### Part 2（VGG-A, 20 epochs）
| 模型 | Val Accuracy |
|------|-------------|
| VGG_A | 76.12% |
| VGG_A_BatchNorm | 83.29% |

### Part 1（CustomCNN, 30 epochs, 最佳配置）
| 配置 | Val Accuracy | 参数量 |
|------|-------------|--------|
| base + GELU + Adam | **89.82%** | 3.25M |
| base + ReLU + SGD | 89.20% | 3.25M |
| base + ReLU + Adam | 89.19% | 3.25M |
| wide + ReLU + Adam | 88.29% | 8.78M |

## 输出目录

```
reports/
├── figures/
│   ├── vgg_bn_comparison.png
│   ├── loss_landscape_*.png
│   ├── gradient_predictiveness.png
│   └── cifar10/
│       ├── experiment_comparison.png
│       ├── *_curves.png
│       ├── base_gelu_filters.png
│       └── base_gelu_predictions.png
├── models/
│   ├── VGG_A.pth / VGG_A_BatchNorm.pth
│   └── best_cifar10.pth
└── logs/
    └── summary.json
```

数据集：`data/cifar-10-batches-py/`（torchvision 自动下载）
