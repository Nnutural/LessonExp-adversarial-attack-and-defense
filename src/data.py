"""CIFAR-10 data loading utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def _eval_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


def build_cifar10_loaders(
    data_dir: str | Path,
    batch_size: int,
    val_size: int = 5000,
    num_workers: int = 4,
    seed: int = 42,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/validation/test loaders.

    Images are kept in [0, 1]. The model owns CIFAR-10 normalization so attacks can
    operate directly in pixel space.
    """
    data_dir = Path(data_dir)
    train_full_for_indices = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=_train_transform(),
        download=download,
    )
    num_train = len(train_full_for_indices)
    if not 0 <= val_size < num_train:
        raise ValueError(f"val_size must be in [0, {num_train}), got {val_size}")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_train, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(
        datasets.CIFAR10(root=data_dir, train=True, transform=_train_transform(), download=False),
        train_indices,
    )
    val_dataset = Subset(
        datasets.CIFAR10(root=data_dir, train=True, transform=_eval_transform(), download=False),
        val_indices,
    )
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=_eval_transform(), download=download)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader


def build_cifar10_test_loader(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 4,
    download: bool = False,
) -> DataLoader:
    data_dir = Path(data_dir)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=_eval_transform(), download=download)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
