"""Evaluate clean and adversarial robustness metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .attacks import fgsm_attack, pgd_attack
from .constants import CIFAR10_CLASSES
from .data import build_cifar10_test_loader
from .models import build_resnet18
from .utils import ensure_dir, format_epsilon, load_model_state, parse_float_list, parse_float_or_fraction, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 clean and adversarial robustness.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/eval"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--attacks", type=str, default="fgsm,pgd", help="Comma-separated attacks: fgsm,pgd")
    parser.add_argument("--epsilons", type=str, default="0,2/255,4/255,8/255,16/255")
    parser.add_argument("--pgd-alpha", type=parse_float_or_fraction, default=parse_float_or_fraction("2/255"))
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full test set.")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def _attack_images(
    attack_name: str,
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    pgd_alpha: float,
    pgd_steps: int,
) -> torch.Tensor:
    if epsilon <= 0:
        return images.detach()
    if attack_name == "fgsm":
        return fgsm_attack(model, images, labels, epsilon=epsilon, set_eval=True)
    if attack_name == "pgd":
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
    raise ValueError(f"Unsupported attack: {attack_name}")


def evaluate_attack(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    attack_name: str,
    epsilon: float,
    pgd_alpha: float,
    pgd_steps: int,
    max_samples: int,
    confusion_path: Path,
) -> dict:
    model.eval()
    total = 0
    clean_correct = 0
    adv_correct = 0
    clean_correct_for_success = 0
    attack_success = 0
    confidence_drop_sum = 0.0
    linf_sum = 0.0
    l2_sum = 0.0
    confusion = np.zeros((len(CIFAR10_CLASSES), len(CIFAR10_CLASSES)), dtype=np.int64)

    progress = tqdm(loader, desc=f"eval/{attack_name}/{format_epsilon(epsilon)}", leave=False)
    for images, labels in progress:
        if max_samples > 0 and total >= max_samples:
            break
        if max_samples > 0:
            remaining = max_samples - total
            images = images[:remaining]
            labels = labels[:remaining]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            clean_logits = model(images)
            clean_probs = F.softmax(clean_logits, dim=1)
            clean_predictions = clean_logits.argmax(dim=1)
            clean_true_conf = clean_probs.gather(1, labels.view(-1, 1)).squeeze(1)

        adv_images = _attack_images(attack_name, model, images, labels, epsilon, pgd_alpha, pgd_steps)

        with torch.no_grad():
            adv_logits = model(adv_images)
            adv_probs = F.softmax(adv_logits, dim=1)
            adv_predictions = adv_logits.argmax(dim=1)
            adv_true_conf = adv_probs.gather(1, labels.view(-1, 1)).squeeze(1)

        batch_size = labels.size(0)
        clean_mask = clean_predictions == labels
        adv_mask = adv_predictions == labels
        clean_correct += clean_mask.sum().item()
        adv_correct += adv_mask.sum().item()
        clean_correct_for_success += clean_mask.sum().item()
        attack_success += (clean_mask & ~adv_mask).sum().item()
        confidence_drop_sum += (clean_true_conf - adv_true_conf).sum().item()

        delta = (adv_images - images).detach().flatten(start_dim=1)
        linf_sum += delta.abs().amax(dim=1).sum().item()
        l2_sum += torch.linalg.vector_norm(delta, ord=2, dim=1).sum().item()

        for true_label, predicted_label in zip(labels.detach().cpu().numpy(), adv_predictions.detach().cpu().numpy()):
            confusion[int(true_label), int(predicted_label)] += 1

        total += batch_size
        progress.set_postfix(robust_acc=f"{adv_correct / max(1, total):.4f}")

    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(confusion_path, confusion)

    return {
        "attack": attack_name,
        "epsilon": epsilon,
        "epsilon_label": format_epsilon(epsilon),
        "pgd_steps": pgd_steps if attack_name == "pgd" else 0,
        "pgd_alpha": pgd_alpha if attack_name == "pgd" else 0.0,
        "num_samples": total,
        "clean_acc": clean_correct / max(1, total),
        "robust_acc": adv_correct / max(1, total),
        "attack_success_rate": attack_success / max(1, clean_correct_for_success),
        "avg_confidence_drop": confidence_drop_sum / max(1, total),
        "avg_linf": linf_sum / max(1, total),
        "avg_l2": l2_sum / max(1, total),
        "confusion_path": str(confusion_path),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_resnet18(num_classes=10).to(device)
    model.load_state_dict(load_model_state(args.checkpoint, map_location=device))
    model.eval()

    loader = build_cifar10_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )

    run_name = args.run_name or args.checkpoint.parent.name
    output_dir = ensure_dir(args.output_dir / run_name)
    attacks = [item.strip().lower() for item in args.attacks.split(",") if item.strip()]
    epsilons = parse_float_list(args.epsilons)

    rows = []
    for attack_name in attacks:
        for epsilon in epsilons:
            confusion_file = output_dir / f"confusion_{attack_name}_eps_{format_epsilon(epsilon).replace('/', '-')}.npy"
            row = evaluate_attack(
                model=model,
                loader=loader,
                device=device,
                attack_name=attack_name,
                epsilon=epsilon,
                pgd_alpha=args.pgd_alpha,
                pgd_steps=args.pgd_steps,
                max_samples=args.max_samples,
                confusion_path=confusion_file,
            )
            rows.append(row)
            print(
                f"{attack_name:>4s} eps={row['epsilon_label']:<6s} "
                f"clean_acc={row['clean_acc']:.4f} robust_acc={row['robust_acc']:.4f} "
                f"asr={row['attack_success_rate']:.4f}"
            )

    metrics_path = output_dir / "metrics.csv"
    write_csv(metrics_path, rows)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
