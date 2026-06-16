"""FGSM and PGD attacks in pixel space under L_inf constraint."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


@contextmanager
def _temporary_eval(model: nn.Module, enabled: bool = True):
    previous_training_state = model.training
    if enabled:
        model.eval()
    try:
        yield
    finally:
        if enabled:
            model.train(previous_training_state)


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    set_eval: bool = True,
) -> torch.Tensor:
    """Generate untargeted FGSM adversarial examples."""
    if epsilon <= 0:
        return images.detach()

    images = images.detach()
    with _temporary_eval(model, enabled=set_eval):
        adv_images = images.clone().detach().requires_grad_(True)
        logits = model(adv_images)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, adv_images, only_inputs=True)[0]

        adv_images = adv_images + epsilon * grad.sign()
        adv_images = torch.clamp(adv_images, 0.0, 1.0)
    return adv_images.detach()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
    set_eval: bool = True,
) -> torch.Tensor:
    """Generate untargeted PGD adversarial examples under L_inf constraint."""
    if epsilon <= 0 or steps <= 0:
        return images.detach()

    images = images.detach()
    if random_start:
        adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)
    else:
        adv_images = images.clone()

    with _temporary_eval(model, enabled=set_eval):
        for _ in range(steps):
            adv_images = adv_images.detach().requires_grad_(True)
            logits = model(adv_images)
            loss = F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, adv_images, only_inputs=True)[0]
            
            adv_images = adv_images + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
            adv_images = torch.clamp(images + delta, 0.0, 1.0)

    return adv_images.detach()


def make_attack(
    name: str,
    epsilon: float,
    alpha: float | None = None,
    steps: int = 10,
    random_start: bool = True,
    set_eval: bool = True,
) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]:
    attack_name = name.lower()
    if attack_name == "fgsm":
        return lambda model, images, labels: fgsm_attack(model, images, labels, epsilon, set_eval=set_eval)
    if attack_name == "pgd":
        pgd_alpha = alpha if alpha is not None else max(epsilon / 4.0, 1.0 / 255.0)
        return lambda model, images, labels: pgd_attack(
            model,
            images,
            labels,
            epsilon,
            pgd_alpha,
            steps,
            random_start=random_start,
            set_eval=set_eval,
        )
    raise ValueError(f"Unsupported attack: {name}")
