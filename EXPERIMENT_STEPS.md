# CIFAR-10 + ResNet-18 + FGSM/PGD + PGD 对抗训练实验步骤

本文档用于在远程 RTX 3090 平台上复现实验。当前仓库只提供代码与运行说明，本地不需要安装依赖或启动训练。

## 1. 实验目标

本实验以 CIFAR-10 图像分类为任务，训练一个适配 CIFAR-10 的 ResNet-18 模型，并使用 FGSM 与 PGD 生成 `L_inf` 范数约束下的对抗样本。随后通过 PGD 对抗训练获得防御模型，比较自然训练模型与防御模型在干净样本和对抗样本上的表现。

核心指标包括：

- Clean Accuracy：干净测试集准确率。
- Robust Accuracy：攻击后仍分类正确的比例。
- Attack Success Rate：原本分类正确但攻击后分类错误的比例。
- Average Confidence Drop：真实类别置信度下降。
- Average `L_inf` / `L2` Perturbation：平均扰动大小。

## 2. 代码结构

```text
.
├── requirements.txt
├── EXPERIMENT_STEPS.md
├── README.md
└── src
    ├── attacks.py      # FGSM / PGD 攻击实现
    ├── constants.py    # CIFAR-10 均值方差与类别名
    ├── data.py         # CIFAR-10 数据加载与 train/val/test 划分
    ├── evaluate.py     # Clean / FGSM / PGD 评估
    ├── models.py       # CIFAR-style ResNet-18
    ├── train.py        # 自然训练与 PGD 对抗训练
    ├── utils.py
    └── visualize.py    # 训练曲线、鲁棒性曲线、扰动图、混淆矩阵
```

图像在数据加载阶段保持 `[0, 1]` 像素空间，CIFAR-10 标准化被放入模型内部。因此 FGSM 和 PGD 的 `epsilon=8/255` 等参数直接对应像素扰动强度，便于解释和可视化。

## 3. 远程 3090 环境配置

建议使用 Conda：

```bash
conda create -n adv_exp python=3.10 -y
conda activate adv_exp
```

安装 PyTorch。请根据远程机器 CUDA 版本选择官方命令。若 CUDA 12.1 可用：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装其余依赖：

```bash
pip install -r requirements.txt
```

检查 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 4. 数据集划分

代码使用 CIFAR-10 官方划分：

- 原始训练集：50,000 张。
- 验证集：从训练集中固定随机划分 5,000 张。
- 实际训练集：45,000 张。
- 测试集：10,000 张。

默认随机种子为 `42`。第一次运行时加入 `--download` 自动下载 CIFAR-10，之后可以去掉。

## 5. 训练自然模型

推荐先训练自然模型，作为攻击基线：

```bash
python -m src.train \
  --mode natural \
  --data-dir data \
  --output-dir checkpoints \
  --run-name natural_resnet18 \
  --epochs 30 \
  --batch-size 256 \
  --lr 0.1 \
  --amp \
  --download
```

输出文件：

```text
checkpoints/natural_resnet18/
├── args.json
├── history.csv
├── best.pt
└── last.pt
```

`best.pt` 基于验证集准确率保存。30 epoch 通常足以快速验证完整流程；若希望更高 clean accuracy，可将 `--epochs` 提升到 50 或 100。

## 6. 训练 PGD 对抗训练模型

轻量 PGD 对抗训练配置如下：

```bash
python -m src.train \
  --mode pgd-at \
  --data-dir data \
  --output-dir checkpoints \
  --run-name pgd_at_resnet18 \
  --epochs 30 \
  --batch-size 256 \
  --lr 0.1 \
  --epsilon 8/255 \
  --pgd-alpha 2/255 \
  --pgd-steps 3 \
  --amp
```

说明：

- 训练阶段使用 PGD-3，控制计算开销。
- 评估阶段建议使用 PGD-10 或 PGD-20，检验更强攻击下的鲁棒性。
- 对抗训练模型的 clean accuracy 通常低于自然训练模型，但 PGD/FGSM 下的 robust accuracy 应明显更高。

## 7. 评估自然模型的攻击效果

对自然模型运行 FGSM 和 PGD：

```bash
python -m src.evaluate \
  --checkpoint checkpoints/natural_resnet18/best.pt \
  --data-dir data \
  --output-dir logs/eval \
  --batch-size 256 \
  --attacks fgsm,pgd \
  --epsilons 0,2/255,4/255,8/255,16/255 \
  --pgd-alpha 2/255 \
  --pgd-steps 10
```

快速调试时可以只跑部分测试集：

```bash
python -m src.evaluate \
  --checkpoint checkpoints/natural_resnet18/best.pt \
  --data-dir data \
  --output-dir logs/eval \
  --batch-size 256 \
  --attacks fgsm,pgd \
  --epsilons 0,8/255 \
  --pgd-steps 10 \
  --max-samples 1000
```

评估结果保存在：

```text
logs/eval/natural_resnet18/metrics.csv
logs/eval/natural_resnet18/confusion_*.npy
```

## 8. 评估 PGD 对抗训练模型

使用相同攻击配置评估防御模型：

```bash
python -m src.evaluate \
  --checkpoint checkpoints/pgd_at_resnet18/best.pt \
  --data-dir data \
  --output-dir logs/eval \
  --batch-size 256 \
  --attacks fgsm,pgd \
  --epsilons 0,2/255,4/255,8/255,16/255 \
  --pgd-alpha 2/255 \
  --pgd-steps 10
```

为了更严格，可额外使用 PGD-20：

```bash
python -m src.evaluate \
  --checkpoint checkpoints/pgd_at_resnet18/best.pt \
  --data-dir data \
  --output-dir logs/eval_pgd20 \
  --batch-size 256 \
  --attacks pgd \
  --epsilons 8/255 \
  --pgd-alpha 2/255 \
  --pgd-steps 20
```

## 9. 生成可视化结果

### 9.1 训练曲线

自然模型：

```bash
python -m src.visualize \
  --history checkpoints/natural_resnet18/history.csv \
  --output-dir figures/natural_resnet18
```

对抗训练模型：

```bash
python -m src.visualize \
  --history checkpoints/pgd_at_resnet18/history.csv \
  --output-dir figures/pgd_at_resnet18
```

生成：

```text
training_curves.png
```

### 9.2 Robust Accuracy vs Epsilon

自然模型：

```bash
python -m src.visualize \
  --metrics logs/eval/natural_resnet18/metrics.csv \
  --output-dir figures/natural_resnet18
```

防御模型：

```bash
python -m src.visualize \
  --metrics logs/eval/pgd_at_resnet18/metrics.csv \
  --output-dir figures/pgd_at_resnet18
```

生成：

```text
robust_accuracy_vs_epsilon.png
```

### 9.3 原图、对抗图、扰动热力图

自然模型 PGD 攻击可视化：

```bash
python -m src.visualize \
  --checkpoint checkpoints/natural_resnet18/best.pt \
  --data-dir data \
  --output-dir figures/natural_resnet18 \
  --attack pgd \
  --epsilon 8/255 \
  --pgd-alpha 2/255 \
  --pgd-steps 10 \
  --num-examples 8
```

防御模型 PGD 攻击可视化：

```bash
python -m src.visualize \
  --checkpoint checkpoints/pgd_at_resnet18/best.pt \
  --data-dir data \
  --output-dir figures/pgd_at_resnet18 \
  --attack pgd \
  --epsilon 8/255 \
  --pgd-alpha 2/255 \
  --pgd-steps 10 \
  --num-examples 8
```

生成：

```text
adversarial_examples_pgd_eps_8-255.png
confusion_clean_vs_pgd_eps_8-255.png
```

## 10. 推荐最小实验流程

如果只想快速完成可报告的最小闭环，按以下顺序执行：

```bash
python -m src.train --mode natural --run-name natural_resnet18 --epochs 30 --batch-size 256 --amp --download

python -m src.train --mode pgd-at --run-name pgd_at_resnet18 --epochs 30 --batch-size 256 --epsilon 8/255 --pgd-alpha 2/255 --pgd-steps 3 --amp

python -m src.evaluate --checkpoint checkpoints/natural_resnet18/best.pt --attacks fgsm,pgd --epsilons 0,2/255,4/255,8/255,16/255 --pgd-steps 10

python -m src.evaluate --checkpoint checkpoints/pgd_at_resnet18/best.pt --attacks fgsm,pgd --epsilons 0,2/255,4/255,8/255,16/255 --pgd-steps 10

python -m src.visualize --history checkpoints/natural_resnet18/history.csv --metrics logs/eval/natural_resnet18/metrics.csv --checkpoint checkpoints/natural_resnet18/best.pt --output-dir figures/natural_resnet18 --attack pgd --epsilon 8/255 --pgd-steps 10

python -m src.visualize --history checkpoints/pgd_at_resnet18/history.csv --metrics logs/eval/pgd_at_resnet18/metrics.csv --checkpoint checkpoints/pgd_at_resnet18/best.pt --output-dir figures/pgd_at_resnet18 --attack pgd --epsilon 8/255 --pgd-steps 10
```

## 11. 实验报告建议表格

最终可整理如下结果表：

| Model | Attack | Epsilon | Clean Acc | Robust Acc | Attack Success Rate | Avg L_inf | Avg L2 |
|---|---|---:|---:|---:|---:|---:|---:|
| Natural ResNet-18 | FGSM | 8/255 | - | - | - | - | - |
| Natural ResNet-18 | PGD-10 | 8/255 | - | - | - | - | - |
| PGD-AT ResNet-18 | FGSM | 8/255 | - | - | - | - | - |
| PGD-AT ResNet-18 | PGD-10 | 8/255 | - | - | - | - | - |

重点图包括：

- `training_curves.png`：训练损失和验证准确率。
- `robust_accuracy_vs_epsilon.png`：扰动强度与鲁棒准确率关系。
- `adversarial_examples_*.png`：原图、对抗图、扰动热力图。
- `confusion_clean_vs_*.png`：干净样本与攻击样本混淆矩阵对比。

## 12. 预期现象

自然训练模型在干净测试集上通常能达到较高准确率，但在 `epsilon=8/255` 的 PGD-10 攻击下准确率会显著下降。PGD 对抗训练模型的干净准确率通常略有下降，但在 FGSM 和 PGD 攻击下的鲁棒准确率会明显高于自然训练模型。这一结果可用于讨论标准准确率与对抗鲁棒性之间的权衡。
