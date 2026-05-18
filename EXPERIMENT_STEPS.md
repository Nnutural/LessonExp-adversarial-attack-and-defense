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

原因基本是：你的 shell 里没有加载 Conda 的 shell hook，所以当前执行的是普通的 `conda` 可执行文件，而不是带有 `activate` 功能的 shell 函数。

`conda activate` 不是一个普通子命令，它需要 shell 初始化脚本支持。你现在直接调用 `conda activate skq_adv_exp`，Conda 把 `activate` 当成普通 command 解析，于是报：

```text
invalid choice: 'activate'
```

**临时解决**

在当前终端执行：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skq_adv_exp
```

如果成功，后面就可以运行：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**永久解决**

先看你当前 shell：

```bash
echo $SHELL
```

如果是 `zsh`：

```bash
conda init zsh
exec zsh
conda activate skq_adv_exp
```

如果是 `bash`：

```bash
conda init bash
exec bash
conda activate skq_adv_exp
```

如果你不想改 shell 配置，也可以每次进入项目后手动执行：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skq_adv_exp
```

**无需 activate 的替代方式**

也可以直接用：

```bash
conda run -n skq_adv_exp python -m src.train --help
```

或者直接调用环境里的 Python：

```bash
/home/admin2/anaconda3/envs/skq_adv_exp/bin/python -m src.train --help
```

建议你先用这个最小检查：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skq_adv_exp
which python
python -V
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

安装 PyTorch。请根据远程机器 CUDA 版本选择官方命令。若 CUDA 12.1 可用：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

可以按下面流程操作。你这台机器建议先用已有的 `skq_adv_exp` 环境，不要在 `base` 里装。

PyTorch 官方当前 Linux pip 安装页要求 Python 3.10+，并提供 CUDA 11.8 / 12.6 / 12.8 等预编译包；有 NVIDIA GPU 时，应选择适合本机驱动的 CUDA 版本安装。[PyTorch 官方安装页](https://pytorch.org/get-started/locally/)

**1. 先正确进入 Conda 环境**

你之前 `conda activate` 报错，是因为 shell 没有加载 Conda hook。先执行：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skq_adv_exp
```

确认环境：

```bash
which python
python -V
which pip
```

如果 `python -V` 低于 3.10，建议重建或升级环境。PyTorch 官方当前稳定版要求 Python 3.10+。

**2. 查看远程机器支持的 CUDA 版本**

运行：

```bash
nvidia-smi
```

重点看右上角类似：

```text
CUDA Version: 12.6
```

注意：这里显示的是 NVIDIA 驱动最高支持的 CUDA 运行时版本，不一定是你系统安装的 `nvcc` 版本。安装 PyTorch 时主要看 `nvidia-smi` 这个值。

**3. 根据 `nvidia-smi` 选择安装命令**

先卸载可能已有的 CPU 版或冲突版本：

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --upgrade pip
```

然后按情况选一个。

如果 `nvidia-smi` 显示 `CUDA Version: 12.8` 或更高：

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

如果显示 `CUDA Version: 12.6` 或 `12.7`：

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

如果显示 `CUDA Version: 12.1`、`12.2`、`12.3`、`12.4`、`12.5`，建议用 CUDA 11.8 版 PyTorch，兼容性更稳：

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

如果你明确想装旧版 `cu121`，可以用旧版本 PyTorch，例如：

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

但我更建议优先用当前官方稳定版的 `cu118 / cu126 / cu128`。

**4. 验证 PyTorch 是否能调用 3090**

安装后运行：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version in torch:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
PY
```

理想输出应类似：

```text
cuda available: True
gpu: NVIDIA GeForce RTX 3090
```

**5. 再安装项目其余依赖**

进入你的项目目录：

```bash
cd ~/skq/Lesson/DLS/adversarial-attack-and-defense
python -m pip install -r requirements.txt
```

如果担心 `requirements.txt` 触发 PyTorch 重装，也可以只装非 PyTorch 依赖：

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn tqdm
```

**6. 最小检查项目代码**

```bash
python -m src.train --help
python -m src.evaluate --help
python -m src.visualize --help
```

如果这三条能正常显示参数说明，环境基本就配好了。

你这台机器如果 `nvidia-smi` 里显示 CUDA 12.1，我建议直接执行：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skq_adv_exp
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy pandas matplotlib seaborn scikit-learn tqdm
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

(skq_adv_exp) admin2@admin2-MS-7D25:~/skq/Lesson/DLS/adversarial-attack-and-defense/LessonExp-adversarial-attack-and-defense$ python -m src.train \
>   --mode natural \
>   --data-dir data \
>   --output-dir checkpoints \
>   --run-name natural_resnet18 \
>   --epochs 30 \
>   --batch-size 256 \
>   --lr 0.1 \
>   --amp \
>   --download
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 170M/170M [00:17<00:00, 9.49MB/s]
/home/admin2/skq/Lesson/DLS/adversarial-attack-and-defense/LessonExp-adversarial-attack-and-defense/src/train.py:152: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = GradScaler(enabled=args.amp and device.type == "cuda")
train/natural:   0%|                                                                                                | 0/175 [00:00<?, ?it/s]/home/admin2/skq/Lesson/DLS/adversarial-attack-and-defense/LessonExp-adversarial-attack-and-defense/src/train.py:96: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast(enabled=args.amp and device.type == "cuda"):
epoch=001 train_loss=2.4980 train_acc=0.1634 val_loss=1.9027 val_acc=0.2636                                                                 
epoch=002 train_loss=1.7245 train_acc=0.3410 val_loss=1.6194 val_acc=0.3960                                                                 
epoch=003 train_loss=1.4761 train_acc=0.4542 val_loss=1.3281 val_acc=0.5174                                                                 
epoch=004 train_loss=1.2748 train_acc=0.5330 val_loss=1.2595 val_acc=0.5484                                                                 
epoch=005 train_loss=1.0812 train_acc=0.6131 val_loss=1.1371 val_acc=0.6044                                                                 
epoch=006 train_loss=0.9287 train_acc=0.6689 val_loss=1.0184 val_acc=0.6390                                                                 
epoch=007 train_loss=0.8181 train_acc=0.7064 val_loss=0.8226 val_acc=0.7096                                                                 
epoch=008 train_loss=0.7216 train_acc=0.7464 val_loss=0.7633 val_acc=0.7400                                                                 
epoch=009 train_loss=0.6463 train_acc=0.7752 val_loss=0.7313 val_acc=0.7484                                                                 
epoch=010 train_loss=0.5690 train_acc=0.8008 val_loss=0.6612 val_acc=0.7734                                                                 
epoch=011 train_loss=0.5156 train_acc=0.8203 val_loss=0.5876 val_acc=0.7966                                                                 
epoch=012 train_loss=0.4706 train_acc=0.8389 val_loss=0.5106 val_acc=0.8218                                                                 
epoch=013 train_loss=0.4254 train_acc=0.8542 val_loss=0.5455 val_acc=0.8204                                                                 
epoch=014 train_loss=0.3854 train_acc=0.8668 val_loss=0.8397 val_acc=0.7546                                                                 
epoch=015 train_loss=0.3507 train_acc=0.8791 val_loss=0.4622 val_acc=0.8430                                                                 
epoch=016 train_loss=0.3301 train_acc=0.8859 val_loss=0.4599 val_acc=0.8450                                                                 
epoch=017 train_loss=0.2966 train_acc=0.8960 val_loss=0.4117 val_acc=0.8606                                                                 
epoch=018 train_loss=0.2668 train_acc=0.9069 val_loss=0.4216 val_acc=0.8562                                                                 
epoch=019 train_loss=0.2362 train_acc=0.9184 val_loss=0.3575 val_acc=0.8826                                                                 
epoch=020 train_loss=0.2112 train_acc=0.9269 val_loss=0.3257 val_acc=0.8928                                                                 
epoch=021 train_loss=0.1787 train_acc=0.9368 val_loss=0.3078 val_acc=0.9022                                                                 
epoch=022 train_loss=0.1553 train_acc=0.9465 val_loss=0.3211 val_acc=0.8980                                                                 
epoch=023 train_loss=0.1300 train_acc=0.9550 val_loss=0.3309 val_acc=0.9006                                                                 
epoch=024 train_loss=0.1081 train_acc=0.9633 val_loss=0.2845 val_acc=0.9140                                                                 
epoch=025 train_loss=0.0869 train_acc=0.9721 val_loss=0.2844 val_acc=0.9154                                                                 
epoch=026 train_loss=0.0701 train_acc=0.9781 val_loss=0.2652 val_acc=0.9222                                                                 
epoch=027 train_loss=0.0584 train_acc=0.9823 val_loss=0.2730 val_acc=0.9194                                                                 
epoch=028 train_loss=0.0514 train_acc=0.9850 val_loss=0.2688 val_acc=0.9228                                                                 
epoch=029 train_loss=0.0458 train_acc=0.9875 val_loss=0.2659 val_acc=0.9250                                                                 
epoch=030 train_loss=0.0423 train_acc=0.9881 val_loss=0.2643 val_acc=0.9242                                                                 
Finished. Best validation accuracy: 0.9250. Run directory: checkpoints/natural_resnet18

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

python -m src.train \
  --mode pgd-at \
  --data-dir data \
  --output-dir checkpoints \
  --run-name pgd_at_resnet18 \
  --epochs 2 \
  --batch-size 256 \
  --lr 0.1 \
  --epsilon 8/255 \
  --pgd-alpha 2/255 \
  --pgd-steps 3 \
  --amp

说明：
/home/admin2/skq/Lesson/DLS/adversarial-attack-and-defense/LessonExp-adversarial-attack-and-defense/src/train.py:152: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = GradScaler(enabled=args.amp and device.type == "cuda")
train/pgd-at:   0%|                                                                                                 | 0/175 [00:00<?, ?it/s]/home/admin2/skq/Lesson/DLS/adversarial-attack-and-defense/LessonExp-adversarial-attack-and-defense/src/train.py:96: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast(enabled=args.amp and device.type == "cuda"):
epoch=001 train_loss=2.6588 train_acc=0.1242 val_loss=2.1057 val_acc=0.2008                                                                 
epoch=002 train_loss=2.1551 train_acc=0.1849 val_loss=2.0244 val_acc=0.2230                                                                 
Finished. Best validation accuracy: 0.2230. Run directory: checkpoints/pgd_at_resnet18

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
(skq_adv_exp) admin2@admin2-MS-7D25:~/skq/Lesson/DLS/adversarial-attack-and-defense/LessonExp-adversarial-attack-and-defense$ python -m src.evaluate \
>   --checkpoint checkpoints/natural_resnet18/best.pt \
>   --data-dir data \
>   --output-dir logs/eval \
>   --batch-size 256 \
>   --attacks fgsm,pgd \
>   --epsilons 0,8/255 \
>   --pgd-steps 10 \
>   --max-samples 1000
fgsm eps=0/255  clean_acc=0.9320 robust_acc=0.9320 asr=0.0000                                                                               
fgsm eps=8/255  clean_acc=0.9320 robust_acc=0.0740 asr=0.9217                                                                               
 pgd eps=0/255  clean_acc=0.9320 robust_acc=0.9320 asr=0.0000                                                                               
 pgd eps=8/255  clean_acc=0.9320 robust_acc=0.0000 asr=1.0000                                                                               
Saved metrics to logs/eval/natural_resnet18/metrics.csv

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
(skq_adv_exp) admin2@admin2-MS-7D25:~/skq/Lesson/DLS/adversarial-attack-and-defense/LessonExp-adversarial-attack-and-defense$ python -m src.evaluate \
>   --checkpoint checkpoints/pgd_at_resnet18/best.pt \
>   --data-dir data \
>   --output-dir logs/eval \
>   --batch-size 256 \
>   --attacks fgsm,pgd \
>   --epsilons 0,2/255,4/255,8/255,16/255 \
>   --pgd-alpha 2/255 \
>   --pgd-steps 10
fgsm eps=0/255  clean_acc=0.2157 robust_acc=0.2157 asr=0.0000                                                                               
fgsm eps=2/255  clean_acc=0.2157 robust_acc=0.2082 asr=0.0723                                                                               
fgsm eps=4/255  clean_acc=0.2157 robust_acc=0.1972 asr=0.1428                                                                               
fgsm eps=8/255  clean_acc=0.2157 robust_acc=0.1708 asr=0.2707                                                                               
fgsm eps=16/255 clean_acc=0.2157 robust_acc=0.1215 asr=0.4905                                                                               
 pgd eps=0/255  clean_acc=0.2157 robust_acc=0.2157 asr=0.0000                                                                               
 pgd eps=2/255  clean_acc=0.2157 robust_acc=0.2085 asr=0.0723                                                                               
 pgd eps=4/255  clean_acc=0.2157 robust_acc=0.1986 asr=0.1442                                                                               
 pgd eps=8/255  clean_acc=0.2157 robust_acc=0.1734 asr=0.2735                                                                               
 pgd eps=16/255 clean_acc=0.2157 robust_acc=0.1316 asr=0.4525                                                                               
Saved metrics to logs/eval/pgd_at_resnet18/metrics.csv

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
