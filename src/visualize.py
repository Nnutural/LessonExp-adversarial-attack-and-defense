"""Generate experiment figures for training, attacks, and defenses."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

from .attacks import fgsm_attack, pgd_attack
from .constants import CIFAR10_CLASSES
from .data import build_cifar10_test_loader
from .models import build_resnet18
from .utils import ensure_dir, format_epsilon, load_model_state, parse_float_or_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visualizations for CIFAR-10 adversarial experiments.")
    parser.add_argument("--history", type=Path, default=None, help="Path to history.csv produced by train.py.")
    parser.add_argument("--metrics", type=Path, default=None, help="Path to metrics.csv produced by evaluate.py.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint for adversarial examples/confusion plots.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--attack", choices=["fgsm", "pgd"], default="pgd")
    parser.add_argument("--epsilon", type=parse_float_or_fraction, default=parse_float_or_fraction("8/255"))
    parser.add_argument("--pgd-alpha", type=parse_float_or_fraction, default=parse_float_or_fraction("2/255"))
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--max-confusion-samples", type=int, default=1000)
    return parser.parse_args()


def plot_history(history_path: Path, output_dir: Path) -> None:
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)

    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["epoch"], history["train_acc"], label="Train")
    axes[1].plot(history["epoch"], history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_robust_curve(metrics_path: Path, output_dir: Path) -> None:
    metrics = pd.read_csv(metrics_path)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=160)
    sns.lineplot(data=metrics, x="epsilon", y="robust_acc", hue="attack", marker="o", ax=ax)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Robust Accuracy")
    ax.set_title("Robust Accuracy vs Epsilon")
    ax.grid(alpha=0.25)

    ticks = sorted(metrics["epsilon"].unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_epsilon(float(value)) for value in ticks])
    fig.tight_layout()
    fig.savefig(output_dir / "robust_accuracy_vs_epsilon.png", bbox_inches="tight")
    plt.close(fig)


def _generate_attack(
    attack: str,
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    pgd_alpha: float,
    pgd_steps: int,
) -> torch.Tensor:
    if attack == "fgsm":
        return fgsm_attack(model, images, labels, epsilon=epsilon, set_eval=True)
    return pgd_attack(
        model,
        images,
        labels,
        epsilon=epsilon,
        alpha=pgd_alpha,
        steps=pgd_steps,
        random_start=True,
        set_eval=True,
    )


def _to_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def plot_adversarial_examples(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    attack: str,
    epsilon: float,
    pgd_alpha: float,
    pgd_steps: int,
    num_examples: int,
) -> None:
    model.eval()
    selected = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            clean_logits = model(images)
            clean_predictions = clean_logits.argmax(dim=1)

        adv_images = _generate_attack(attack, model, images, labels, epsilon, pgd_alpha, pgd_steps)

        with torch.no_grad():
            adv_logits = model(adv_images)
            adv_predictions = adv_logits.argmax(dim=1)

        success_mask = (clean_predictions == labels) & (adv_predictions != labels)
        candidate_indices = success_mask.nonzero(as_tuple=False).flatten().tolist()
        if len(candidate_indices) < num_examples:
            fallback_indices = torch.arange(labels.size(0), device=device).tolist()
            candidate_indices.extend([idx for idx in fallback_indices if idx not in candidate_indices])

        for index in candidate_indices:
            selected.append(
                (
                    images[index].detach().cpu(),
                    adv_images[index].detach().cpu(),
                    int(labels[index].detach().cpu()),
                    int(clean_predictions[index].detach().cpu()),
                    int(adv_predictions[index].detach().cpu()),
                )
            )
            if len(selected) >= num_examples:
                break
        if len(selected) >= num_examples:
            break

    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 3, figsize=(7.5, 2.3 * len(selected)), dpi=160)
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (clean_image, adv_image, label, clean_pred, adv_pred) in enumerate(selected):
        delta = adv_image - clean_image
        heatmap = delta.abs().mean(dim=0).numpy()

        axes[row, 0].imshow(_to_image(clean_image))
        axes[row, 0].set_title(f"Original\ntrue={CIFAR10_CLASSES[label]}, pred={CIFAR10_CLASSES[clean_pred]}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(_to_image(adv_image))
        axes[row, 1].set_title(f"Adversarial\npred={CIFAR10_CLASSES[adv_pred]}", fontsize=8)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(heatmap, cmap="magma", vmin=0.0, vmax=max(epsilon, 1e-8))
        axes[row, 2].set_title(f"|delta| heatmap\nL_inf={delta.abs().max().item():.4f}", fontsize=8)
        axes[row, 2].axis("off")

    fig.suptitle(f"{attack.upper()} adversarial examples, epsilon={format_epsilon(epsilon)}", y=1.0)
    fig.tight_layout()
    fig.savefig(output_dir / f"adversarial_examples_{attack}_eps_{format_epsilon(epsilon).replace('/', '-')}.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    attack: str,
    epsilon: float,
    pgd_alpha: float,
    pgd_steps: int,
    max_samples: int,
) -> None:
    clean_confusion = np.zeros((len(CIFAR10_CLASSES), len(CIFAR10_CLASSES)), dtype=np.int64)
    adv_confusion = np.zeros_like(clean_confusion)
    total = 0
    model.eval()

    for images, labels in loader:
        if max_samples > 0 and total >= max_samples:
            break
        if max_samples > 0:
            remaining = max_samples - total
            images = images[:remaining]
            labels = labels[:remaining]

        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)

        adv_images = _generate_attack(attack, model, images, labels, epsilon, pgd_alpha, pgd_steps)

        with torch.no_grad():
            adv_predictions = model(adv_images).argmax(dim=1)

        for true_label, clean_pred, adv_pred in zip(
            labels.detach().cpu().numpy(),
            clean_predictions.detach().cpu().numpy(),
            adv_predictions.detach().cpu().numpy(),
        ):
            clean_confusion[int(true_label), int(clean_pred)] += 1
            adv_confusion[int(true_label), int(adv_pred)] += 1

        total += labels.size(0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)
    for ax, matrix, title in [
        (axes[0], clean_confusion, "Clean"),
        (axes[1], adv_confusion, f"{attack.upper()} eps={format_epsilon(epsilon)}"),
    ]:
        sns.heatmap(
            matrix,
            annot=False,
            cmap="Blues",
            xticklabels=CIFAR10_CLASSES,
            yticklabels=CIFAR10_CLASSES,
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=45)
        ax.tick_params(axis="y", labelrotation=0)

    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_clean_vs_{attack}_eps_{format_epsilon(epsilon).replace('/', '-')}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    sns.set_theme(style="whitegrid")

    if args.history is not None:
        plot_history(args.history, output_dir)
        print(f"Saved training curves to {output_dir}")

    if args.metrics is not None:
        plot_robust_curve(args.metrics, output_dir)
        print(f"Saved robust accuracy curve to {output_dir}")

    if args.checkpoint is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_resnet18(num_classes=10).to(device)
        model.load_state_dict(load_model_state(args.checkpoint, map_location=device))
        loader = build_cifar10_test_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.download,
        )
        plot_adversarial_examples(
            model=model,
            loader=loader,
            device=device,
            output_dir=output_dir,
            attack=args.attack,
            epsilon=args.epsilon,
            pgd_alpha=args.pgd_alpha,
            pgd_steps=args.pgd_steps,
            num_examples=args.num_examples,
        )
        plot_confusion_matrices(
            model=model,
            loader=loader,
            device=device,
            output_dir=output_dir,
            attack=args.attack,
            epsilon=args.epsilon,
            pgd_alpha=args.pgd_alpha,
            pgd_steps=args.pgd_steps,
            max_samples=args.max_confusion_samples,
        )
        print(f"Saved adversarial examples and confusion matrices to {output_dir}")


if __name__ == "__main__":
    main()
