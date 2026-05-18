"""Train natural or PGD adversarially trained ResNet-18 on CIFAR-10."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .attacks import pgd_attack
from .data import build_cifar10_loaders
from .models import build_resnet18
from .utils import (
    AverageMeter,
    accuracy_from_logits,
    ensure_dir,
    parse_float_or_fraction,
    save_checkpoint,
    save_json,
    seed_everything,
    write_csv,
)


def _serializable_args(args: argparse.Namespace) -> dict:
    payload = vars(args).copy()
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 ResNet-18 with optional PGD adversarial training.")
    parser.add_argument("--mode", choices=["natural", "pgd-at"], default="natural")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if it is not present.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision for the training update.")
    parser.add_argument("--epsilon", type=parse_float_or_fraction, default=parse_float_or_fraction("8/255"))
    parser.add_argument("--pgd-alpha", type=parse_float_or_fraction, default=parse_float_or_fraction("2/255"))
    parser.add_argument("--pgd-steps", type=int, default=3, help="PGD steps used only in pgd-at training.")
    parser.add_argument("--no-random-start", action="store_true", help="Disable random start for PGD adversarial training.")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    progress = tqdm(loader, desc=f"train/{args.mode}", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if args.mode == "pgd-at":
            images_for_update = pgd_attack(
                model,
                images,
                labels,
                epsilon=args.epsilon,
                alpha=args.pgd_alpha,
                steps=args.pgd_steps,
                random_start=not args.no_random_start,
                set_eval=True,
            )
        else:
            images_for_update = images

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp and device.type == "cuda"):
            logits = model(images_for_update)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy_from_logits(logits.detach(), labels), batch_size)
        progress.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return {"train_loss": loss_meter.avg, "train_acc": acc_meter.avg}


@torch.no_grad()
def evaluate_clean(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in tqdm(loader, desc="val/clean", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy_from_logits(logits, labels), batch_size)

    return {"val_loss": loss_meter.avg, "val_acc": acc_meter.avg}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"{args.mode}_resnet18_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(args.output_dir / run_name)
    args_payload = _serializable_args(args)
    save_json(run_dir / "args.json", args_payload | {"device": str(device), "run_name": run_name})

    train_loader, val_loader, _ = build_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        seed=args.seed,
        download=args.download,
    )

    model = build_resnet18(num_classes=10).to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    history: list[dict] = []
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        val_stats = evaluate_clean(model, val_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "mode": args.mode,
            "lr": optimizer.param_groups[0]["lr"],
            **train_stats,
            **val_stats,
        }
        history.append(row)
        write_csv(run_dir / "history.csv", history)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": max(best_val_acc, val_stats["val_acc"]),
            "args": args_payload,
            "history": history,
        }
        save_checkpoint(run_dir / "last.pt", checkpoint)
        if val_stats["val_acc"] > best_val_acc:
            best_val_acc = val_stats["val_acc"]
            save_checkpoint(run_dir / "best.pt", checkpoint)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_stats['train_loss']:.4f} train_acc={train_stats['train_acc']:.4f} "
            f"val_loss={val_stats['val_loss']:.4f} val_acc={val_stats['val_acc']:.4f}"
        )

    print(f"Finished. Best validation accuracy: {best_val_acc:.4f}. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
