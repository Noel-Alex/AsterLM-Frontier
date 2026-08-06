from __future__ import annotations

import math


def _decay_curve(progress: float, min_ratio: float, shape: str) -> float:
    progress = min(max(progress, 0.0), 1.0)
    if shape == "linear":
        factor = 1.0 - progress
    elif shape == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif shape == "sqrt":
        factor = math.sqrt(max(0.0, 1.0 - progress))
    else:
        raise ValueError(f"Unsupported decay shape: {shape}")
    return min_ratio + (1.0 - min_ratio) * factor


def learning_rate_multiplier(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.1,
    schedule_type: str = "wsd",
    decay_fraction: float = 0.1,
    decay_shape: str = "cosine",
) -> float:
    """Return a warmup/cosine or warmup-stable-decay learning-rate multiplier.

    WSD is useful for pretraining because a long stable phase permits extending a run
    before committing to the final cooldown. The total horizon should still reflect the
    actual token budget; ``Trainer`` derives it from ``max_tokens`` when supplied.
    """

    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, (step + 1) / warmup_steps)
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be between 0 and 1")

    if schedule_type == "constant":
        return 1.0
    if schedule_type == "cosine":
        progress = (step + 1 - warmup_steps) / max(1, total_steps - warmup_steps)
        return _decay_curve(progress, min_ratio, "cosine")
    if schedule_type != "wsd":
        raise ValueError(f"Unsupported schedule_type: {schedule_type}")
    if not 0.0 < decay_fraction <= 1.0:
        raise ValueError("decay_fraction must be in (0, 1]")

    decay_steps = max(1, round(total_steps * decay_fraction))
    stable_end = max(warmup_steps, total_steps - decay_steps)
    if step < stable_end:
        return 1.0
    progress = (step + 1 - stable_end) / max(1, total_steps - stable_end)
    return _decay_curve(progress, min_ratio, decay_shape)


def cosine_warmup_multiplier(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.1,
) -> float:
    """Backward-compatible cosine schedule wrapper."""

    return learning_rate_multiplier(
        step,
        warmup_steps,
        total_steps,
        min_ratio=min_ratio,
        schedule_type="cosine",
    )
