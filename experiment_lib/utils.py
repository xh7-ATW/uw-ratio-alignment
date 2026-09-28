"""Low-level utilities shared by the paper experiment code."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

DIAGNOSTIC_TYPE_ORDER = ["attn_k", "attn_proj", "attn_q", "attn_v", "mlp_fc", "mlp_proj"]

TYPE_TO_LR_GROUP = {
    "attn_q": "qkv",
    "attn_k": "qkv",
    "attn_v": "qkv",
    "attn_proj": "attn_proj",
    "mlp_fc": "mlp_fc",
    "mlp_proj": "mlp_proj",
}

def iclr_code_dir(start: Path | None = None) -> Path:
    """Return the nearest parent directory named ``iclr_code``."""

    current = (start or Path(__file__)).resolve()
    for path in [current, *current.parents]:
        if path.name == "iclr_code":
            return path
    raise RuntimeError(f"Could not find iclr_code directory from {current}")


def repository_root(iclr_dir: Path | None = None) -> Path:
    """Return the enclosing repo root, or ``iclr_code`` for standalone archives."""

    root = (iclr_dir or iclr_code_dir()).resolve()
    for path in [root, *root.parents]:
        if (path / "records" / "track_3_optimization").exists():
            return path
    return root


def resolve_inside_iclr(relative_path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve a path relative to ``iclr_code`` unless it is absolute."""

    path = Path(relative_path)
    if path.is_absolute():
        return path
    return (base or iclr_code_dir()) / path


def diagnostic_type_sort_key(type_name: str) -> tuple[int, str]:
    """Sort matrix types in the diagnostic CSV order."""

    try:
        return (DIAGNOSTIC_TYPE_ORDER.index(type_name), type_name)
    except ValueError:
        return (len(DIAGNOSTIC_TYPE_ORDER), type_name)


def is_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def as_float(value: object) -> float:
    return float(value) if is_finite(value) else float("nan")


def safe_div(numerator: object, denominator: object) -> float:
    num = as_float(numerator)
    den = as_float(denominator)
    if not is_finite(num) or not is_finite(den) or den == 0.0:
        return float("nan")
    return num / den


def json_clean(value: object) -> object:
    """Replace non-finite floats with JSON null while preserving structure."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_clean(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_clean(val) for val in value]
    return value


def weighted_mean(values: Iterable[object], weights: Iterable[object]) -> float:
    pairs = [(as_float(value), as_float(weight)) for value, weight in zip(values, weights)]
    pairs = [
        (value, weight)
        for value, weight in pairs
        if is_finite(value) and is_finite(weight) and weight > 0.0
    ]
    denominator = sum(weight for _, weight in pairs)
    if denominator == 0.0:
        return float("nan")
    return sum(value * weight for value, weight in pairs) / denominator


def weighted_covariance(
    xs: Iterable[object],
    ys: Iterable[object],
    weights: Iterable[object],
) -> float:
    xs = list(xs)
    ys = list(ys)
    weights = list(weights)
    x_mean = weighted_mean(xs, weights)
    y_mean = weighted_mean(ys, weights)
    if not is_finite(x_mean) or not is_finite(y_mean):
        return float("nan")
    triples = [
        (as_float(x), as_float(y), as_float(weight))
        for x, y, weight in zip(xs, ys, weights)
    ]
    triples = [
        (x, y, weight)
        for x, y, weight in triples
        if is_finite(x) and is_finite(y) and is_finite(weight) and weight > 0.0
    ]
    denominator = sum(weight for _, _, weight in triples)
    if denominator == 0.0:
        return float("nan")
    return sum(weight * (x - x_mean) * (y - y_mean) for x, y, weight in triples) / denominator


def weighted_variance(values: Iterable[object], weights: Iterable[object]) -> float:
    values = list(values)
    weights = list(weights)
    mean = weighted_mean(values, weights)
    if not is_finite(mean):
        return float("nan")
    pairs = [(as_float(value), as_float(weight)) for value, weight in zip(values, weights)]
    pairs = [
        (value, weight)
        for value, weight in pairs
        if is_finite(value) and is_finite(weight) and weight > 0.0
    ]
    denominator = sum(weight for _, weight in pairs)
    if denominator == 0.0:
        return float("nan")
    return sum(weight * (value - mean) * (value - mean) for value, weight in pairs) / denominator


def basic_stats(values: Iterable[object]) -> dict[str, float | int]:
    vals = [as_float(value) for value in values]
    vals = [value for value in vals if is_finite(value)]
    if not vals:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "max_over_min": float("nan"),
        }
    mean = sum(vals) / len(vals)
    variance = sum((value - mean) * (value - mean) for value in vals) / len(vals)
    value_min = min(vals)
    value_max = max(vals)
    return {
        "count": len(vals),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": value_min,
        "max": value_max,
        "max_over_min": safe_div(value_max, value_min),
    }
