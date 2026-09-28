"""Result-file parsing helpers for saved paper evidence."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


VALIDATION_LOSS_RE = re.compile(r"step:(\d+)/(\d+)\s+val_loss:([0-9.eE+-]+)")


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already sorted list."""

    if not sorted_values:
        raise ValueError("empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    fraction = position - lo
    return sorted_values[lo] * (1.0 - fraction) + sorted_values[hi] * fraction


def read_validation_loss(path: Path, *, min_step: int = 0) -> dict[int, float]:
    """Read validation losses from either a raw text log or a JSONL file."""

    text = path.read_text(errors="replace")
    values: dict[int, float] = {}
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            step = int(record["step"])
            if step >= min_step:
                values[step] = float(record["val_loss"])
    else:
        for match in VALIDATION_LOSS_RE.finditer(text):
            step = int(match.group(1))
            if step >= min_step:
                values[step] = float(match.group(3))
    if not values:
        raise RuntimeError(f"No validation losses at step >= {min_step} in {path}")
    return values


def value_at_step(values: dict[int, float], step: int) -> float:
    """Return an exact value or linearly interpolate between neighboring steps."""

    if step in values:
        return values[step]
    available_steps = sorted(values)
    lower = max((candidate for candidate in available_steps if candidate < step), default=None)
    upper = min((candidate for candidate in available_steps if candidate > step), default=None)
    if lower is None or upper is None:
        raise RuntimeError(f"Cannot interpolate validation loss at step {step}")
    weight = (step - lower) / (upper - lower)
    return values[lower] + weight * (values[upper] - values[lower])


def summarize_by_seed(
    values_by_seed: dict[int, dict[int, float]],
    *,
    target_steps: list[int] | None = None,
) -> list[dict[str, float | int]]:
    """Aggregate per-seed validation losses into table rows."""

    if target_steps is None:
        target_steps = sorted(set.intersection(*(set(vals) for vals in values_by_seed.values())))
    if not target_steps:
        raise RuntimeError("No shared validation steps across seeds")

    rows: list[dict[str, float | int]] = []
    for step in target_steps:
        values = sorted(value_at_step(values_by_seed[seed], step) for seed in sorted(values_by_seed))
        mean = sum(values) / len(values)
        rows.append(
            {
                "step": step,
                "seed_count": len(values),
                "min": values[0],
                "q25": quantile(values, 0.25),
                "mean": mean,
                "median": quantile(values, 0.50),
                "q75": quantile(values, 0.75),
                "max": values[-1],
                "std": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
            }
        )
    return rows
