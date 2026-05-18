"""Utility helpers used by training, evaluation, and visualization scripts."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def parse_float_or_fraction(value: str | float | int) -> float:
    """Parse values such as 0.031 or 8/255 into float."""
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(text)


def parse_float_list(text: str) -> list[float]:
    return [parse_float_or_fraction(item) for item in text.split(",") if item.strip()]


def format_epsilon(value: float) -> str:
    scaled = value * 255.0
    if abs(round(scaled) - scaled) < 1e-6:
        return f"{int(round(scaled))}/255"
    return f"{value:.6f}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    """Track a running average."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

    def update(self, value: float, count: int) -> None:
        self.sum += float(value) * int(count)
        self.count += int(count)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_model_state(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")
