# CIFAR-10 对抗样本攻击与 PGD 对抗训练实验报告

> 注：
>
> - 实验一利用论文《FGSM Explainer: An Interactive Visualization for Understanding Adversarial Attack》搭建的FGSM和对抗训练实验可视化网站进行介绍（https://visxai-aml.vercel.app/）
> - 实验二利用自行准备的实验（即本实验报告内容），实验完整代码和一些实验数据均上传github仓库（https://github.com/Nnutural/LessonExp-adversarial-attack-and-defense）

[TOC]



## 1. 实验目的

本实验基于 CIFAR-10 图像分类任务，构建 ResNet-18 分类模型，并在 `L_inf` 约束下使用 FGSM 与 PGD 生成对抗样本，验证标准深度模型对微小扰动的敏感性。同时引入 PGD 对抗训练作为防御方法，观察其训练过程和对抗鲁棒性变化。

## 2. 实验方法

**实验采用 CIFAR-10 数据集**，训练集与验证集从官方训练集划分，测试集使用官方测试集。模型采用适配 32x32 图像输入的 **ResNet-18**：首层卷积修改为 `3x3, stride=1, padding=1`，去除原始 ImageNet 结构中的 maxpool，并将最终分类层设为 10 类输出。

**攻击方法包括 FGSM 与 PGD**。二者均在像素空间 `[0, 1]` 内生成 `L_inf` 约束扰动，主要攻击强度为 `epsilon=8/255`。PGD 评估采用多步迭代攻击，实验中使用 `pgd_steps=10`。

**防御方法采用 PGD 对抗训练**。训练阶段对每个 batch 先生成 PGD 对抗样本，再用对抗样本更新模型参数。当前实验中自然模型完成 30 个 epoch；PGD 对抗训练模型仅完成 2 个 epoch，因此其结果主要用于流程验证，不作为充分训练后的最终防御性能结论。

## 3. 实验流程

实验流程如下：

1. 配置 Conda 与 PyTorch GPU 环境。
2. 下载并加载 CIFAR-10 数据集。
3. 训练自然 ResNet-18 模型。
4. 使用 FGSM 与 PGD 评估自然模型的对抗鲁棒性。
5. 使用 PGD 对抗训练训练防御模型。
6. 使用相同攻击配置评估防御模型。
7. 生成训练曲线、鲁棒准确率曲线、对抗样本图和混淆矩阵。

## 4. 自然模型训练结果

自然训练模型完成 30 个 epoch。训练损失从 `2.4980` 降至 `0.0423`，训练准确率从 `0.1634` 提升至 `0.9881`；验证准确率最高达到 `0.9250`，最终 epoch 验证准确率为 `0.9242`。

![自然模型训练曲线](figures/natural_resnet18/training_curves.png)

从训练曲线可以看出，自然模型在前 10 个 epoch 内快速收敛，随后训练准确率持续上升并接近 1.0。验证准确率最终稳定在约 0.92 左右，说明模型已经较好拟合 CIFAR-10 分类任务。训练准确率与验证准确率后期存在一定差距，体现出轻微过拟合，但总体训练结果可作为有效攻击基线。

## 5. 自然模型攻击结果

在 1000 张测试样本子集上，自然模型评估结果如下：

| Model | Attack | Epsilon | Clean Acc | Robust Acc | Attack Success Rate |
|---|---|---:|---:|---:|---:|
| Natural ResNet-18 | FGSM | 0/255 | 0.9320 | 0.9320 | 0.0000 |
| Natural ResNet-18 | FGSM | 8/255 | 0.9320 | 0.0740 | 0.9217 |
| Natural ResNet-18 | PGD-10 | 0/255 | 0.9320 | 0.9320 | 0.0000 |
| Natural ResNet-18 | PGD-10 | 8/255 | 0.9320 | 0.0000 | 1.0000 |

![自然模型鲁棒准确率曲线](figures/natural_resnet18/robust_accuracy_vs_epsilon.png)

结果表明，自然模型在干净样本上准确率较高，但在 `epsilon=8/255` 的扰动下鲁棒准确率急剧下降。FGSM 攻击后准确率仅为 `0.0740`，PGD-10 攻击后鲁棒准确率降至 `0.0000`，说明自然训练模型几乎无法抵抗强一阶迭代攻击。

## 6. 自然模型对抗样本可视化

![自然模型 PGD 对抗样本](figures/natural_resnet18/adversarial_examples_pgd_eps_8-255.png)

可视化结果展示了原始图像、PGD 对抗图像以及扰动热力图。扰动强度为 `L_inf=0.0314`，即约 `8/255`。从视觉上看，对抗图像与原图差异较小，但模型预测结果发生明显变化，例如：

- `cat -> dog`
- `ship -> automobile`
- `airplane -> ship`
- `frog -> bird / dog / cat`
- `automobile -> truck`

这说明对抗扰动虽然幅度有限，但能够有效改变模型决策边界附近样本的分类结果。

## 7. 自然模型混淆矩阵分析

![自然模型混淆矩阵](figures/natural_resnet18/confusion_clean_vs_pgd_eps_8-255.png)

干净样本混淆矩阵中，对角线颜色明显更深，表示自然模型在正常测试样本上分类较准确。PGD 攻击后，对角线显著减弱，错误预测分布扩散到多个类别，说明攻击破坏了模型原有的类别判别结构。特别是部分类别被集中误分到视觉相近或模型偏置较强的类别中，体现出 PGD 攻击的强破坏性。

## 8. PGD 对抗训练过程

PGD 对抗训练模型当前完成 2 个 epoch。训练日志如下：

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---:|---:|---:|---:|---:|
| 1 | 2.6588 | 0.1242 | 2.1057 | 0.2008 |
| 2 | 2.1551 | 0.1849 | 2.0244 | 0.2230 |

![PGD 对抗训练曲线](figures/pgd_at_resnet18/training_curves.png)

从曲线看，PGD 对抗训练在前 2 个 epoch 内 loss 开始下降，accuracy 有小幅提升，但整体准确率仍然很低。由于对抗训练比自然训练困难得多，2 个 epoch 远不足以得到稳定的鲁棒模型。因此后续防御结果应解释为“对抗训练流程已跑通”，而非“防御方法已充分收敛”。

## 9. PGD 对抗训练模型评估

PGD 对抗训练模型在测试集上的评估结果如下：

| Model | Attack | Epsilon | Clean Acc | Robust Acc | Attack Success Rate |
|---|---|---:|---:|---:|---:|
| PGD-AT ResNet-18 | FGSM | 0/255 | 0.2157 | 0.2157 | 0.0000 |
| PGD-AT ResNet-18 | FGSM | 2/255 | 0.2157 | 0.2082 | 0.0723 |
| PGD-AT ResNet-18 | FGSM | 4/255 | 0.2157 | 0.1972 | 0.1428 |
| PGD-AT ResNet-18 | FGSM | 8/255 | 0.2157 | 0.1708 | 0.2707 |
| PGD-AT ResNet-18 | FGSM | 16/255 | 0.2157 | 0.1215 | 0.4905 |
| PGD-AT ResNet-18 | PGD-10 | 0/255 | 0.2157 | 0.2157 | 0.0000 |
| PGD-AT ResNet-18 | PGD-10 | 2/255 | 0.2157 | 0.2085 | 0.0723 |
| PGD-AT ResNet-18 | PGD-10 | 4/255 | 0.2157 | 0.1986 | 0.1442 |
| PGD-AT ResNet-18 | PGD-10 | 8/255 | 0.2157 | 0.1734 | 0.2735 |
| PGD-AT ResNet-18 | PGD-10 | 16/255 | 0.2157 | 0.1316 | 0.4525 |

![PGD 对抗训练模型鲁棒准确率曲线](figures/pgd_at_resnet18/robust_accuracy_vs_epsilon.png)

随着 `epsilon` 从 `0/255` 增大到 `16/255`，模型鲁棒准确率逐步下降，符合扰动强度越大、攻击越强的基本规律。但由于该模型仅训练 2 个 epoch，clean accuracy 仅为 `0.2157`，分类能力尚未充分形成，因此不能直接与 30 epoch 的自然模型做最终性能比较。

## 10. PGD 对抗训练模型可视化

![PGD 对抗训练模型对抗样本](figures/pgd_at_resnet18/adversarial_examples_pgd_eps_8-255.png)

对抗样本图显示，在 `epsilon=8/255` 下，PGD 攻击仍可导致模型预测错误，例如 `horse -> truck`、`frog -> truck`、`ship -> automobile` 等。扰动热力图相较自然模型更稀疏，但由于模型尚未充分训练，预测结果本身稳定性不足。

![PGD 对抗训练模型混淆矩阵](figures/pgd_at_resnet18/confusion_clean_vs_pgd_eps_8-255.png)

混淆矩阵显示，模型在干净样本和 PGD 样本上的预测均存在明显类别偏置，部分类别被频繁预测为 `frog` 或 `ship`。这进一步说明当前 PGD-AT 模型仍处于早期训练状态，需要继续训练至 30 epoch 或更长时间后再进行正式防御效果比较。

## 11. 结论

本实验完成了 CIFAR-10 上 ResNet-18 自然训练、FGSM/PGD 攻击、PGD 对抗训练与可视化分析的完整流程。自然模型在干净样本上达到较高准确率，验证准确率最高为 `0.9250`；但在 `epsilon=8/255` 的 PGD-10 攻击下，鲁棒准确率降至 `0.0000`，说明标准训练模型对对抗扰动高度敏感。

PGD 对抗训练部分已经成功运行并生成结果，但当前仅训练 2 个 epoch，模型准确率较低，因此只能说明流程有效，尚不能作为充分防御结论。后续应继续完成 30 epoch PGD 对抗训练，再使用相同 FGSM/PGD 配置评估其 clean accuracy 与 robust accuracy，从而严谨比较自然训练与对抗训练之间的鲁棒性差异。

## 12. 后续完善建议

1. 将 PGD 对抗训练继续运行至 30 epoch。
2. 对自然模型和 PGD-AT 模型均使用完整测试集评估。
3. 对自然模型补充 `2/255, 4/255, 16/255` 多扰动强度评估。
4. 在最终报告中统一比较 `Natural` 与 `PGD-AT` 的 `Clean Acc`、`FGSM Robust Acc` 和 `PGD Robust Acc`。
5. 如时间允许，增加 PGD-20 或 AutoAttack 作为更严格鲁棒性评估。
