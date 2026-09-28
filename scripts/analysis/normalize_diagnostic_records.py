#!/usr/bin/env python3
"""Normalize saved Muon diagnostics to the paper's ideal-ratio convention."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
ICLR_DIR = SCRIPT.parents[2]
sys.path.insert(0, str(ICLR_DIR))

from src.diagnostic_utils import _vector_match_stats
from src.diagnostics import apply_alignment_policy


DEFAULT_PATHS = [
    ICLR_DIR
    / "experiment_results/shared_lr/seed42_diagnostics/raw/seed42_plain_muon_local_quadratic_diag.json",
    ICLR_DIR
    / "experiment_results/type_lr/seed42_diag/muon_diagnostics/raw/seed42_plain_muon_optlr_local_quadratic_diag.json",
]


def refresh_match_summaries(record: dict) -> None:
    valid = [block for block in record["blocks"] if block.get("t_i") is not None]
    t_values = [float(block["t_i"]) for block in valid]
    stats = {
        "plain": _vector_match_stats([float(block["x_plain"]) for block in valid], t_values),
        "typed": _vector_match_stats([float(block["x_typed"]) for block in valid], t_values),
        "all": _vector_match_stats([float(block["x_all"]) for block in valid], t_values),
    }
    record.update(stats)
    for name, values in stats.items():
        record[f"cosine_{name}_t"] = values["cosine"]
        record[f"relative_l2_{name}_t"] = values["relative_l2_after_best_scale"]
        record[f"pearson_{name}_t"] = values["pearson"]

    plain, typed, all_layer = stats["plain"], stats["typed"], stats["all"]
    record["ordering"] = {
        "cosine_typed_gt_plain": typed["cosine"] > plain["cosine"],
        "cosine_all_gt_plain": all_layer["cosine"] > plain["cosine"],
        "cosine_typed_gt_all": typed["cosine"] > all_layer["cosine"],
        "l2_typed_lt_plain": typed["relative_l2_after_best_scale"]
        < plain["relative_l2_after_best_scale"],
        "l2_all_lt_plain": all_layer["relative_l2_after_best_scale"]
        < plain["relative_l2_after_best_scale"],
        "l2_typed_lt_all": typed["relative_l2_after_best_scale"]
        < all_layer["relative_l2_after_best_scale"],
    }


def normalize(path: Path) -> None:
    payload = json.loads(path.read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    for record in records:
        apply_alignment_policy(record)
        refresh_match_summaries(record)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    os.replace(temporary, path)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_PATHS)
    args = parser.parse_args()
    for path in args.paths:
        normalize(path.resolve())


if __name__ == "__main__":
    main()
