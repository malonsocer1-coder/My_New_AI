"""Programación de la tasa de aprendizaje."""

from __future__ import annotations

import math


def lr_at(step: int, lr: float, min_lr: float, warmup: int, max_steps: int,
          schedule: str = "cosine", wsd_decay_frac: float = 0.2) -> float:
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if schedule == "constant":
        return lr
    span = max(1, max_steps - warmup)
    progress = min(1.0, (step - warmup) / span)
    if schedule == "cosine":
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))
    if schedule == "linear":
        return lr + (min_lr - lr) * progress
    if schedule == "wsd":
        # Warmup-Stable-Decay: meseta larga y caída final. Permite prolongar el
        # entrenamiento desde un checkpoint de la meseta sin reiniciar el coseno.
        decay_steps = max(1, int(wsd_decay_frac * max_steps))
        decay_start = max_steps - decay_steps
        if step < decay_start:
            return lr
        frac = min(1.0, (step - decay_start) / decay_steps)
        return min_lr + (lr - min_lr) * (1 - math.sqrt(frac))
    raise ValueError(f"Programación desconocida: {schedule}")
