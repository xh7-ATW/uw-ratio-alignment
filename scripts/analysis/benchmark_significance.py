#!/usr/bin/env python3
"""Summarize benchmark validation-loss evidence against a target loss."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
ICLR_DIR = SCRIPT.parents[2]
sys.path.insert(0, str(ICLR_DIR))

from scripts.analysis.result_parsing import read_validation_loss, value_at_step


DEFAULT_STEPS = [3250, 3260, 3270, 3280, 3290, 3300]
DEFAULT_RUNS = [
    ("MuonL shared lr, H100", "experiment_results/benchmark"),
]
DEFAULT_BENCHMARK_THRESHOLD = 0.004


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        action="append",
        dest="steps",
        type=int,
        help="Evaluation step to summarize. May be repeated.",
    )
    parser.add_argument("--target-loss", type=float, default=3.28)
    parser.add_argument("--benchmark-threshold", type=float, default=DEFAULT_BENCHMARK_THRESHOLD)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def trial_seed_map(run_dir: Path) -> dict[str, int]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    return {
        str(entry["trial_id"]): int(entry["seed"])
        for entry in manifest
        if "trial_id" in entry and "seed" in entry
    }


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def summarize_run(
    label: str,
    run_dir: Path,
    steps: list[int],
    target_loss: float,
    benchmark_threshold: float,
) -> dict[str, object]:
    seed_by_trial = trial_seed_map(run_dir)
    value_sets: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("*.txt")):
        losses = read_validation_loss(path)
        value_sets.append(
            {
                "log": path.name,
                "seed": seed_by_trial.get(path.stem),
                "losses": losses,
            }
        )
    if not value_sets:
        raise RuntimeError(f"No raw benchmark logs found in {run_dir}")

    rows: list[dict[str, float | int | str]] = []
    first_pass_step: int | None = None
    for step in steps:
        values = [value_at_step(item["losses"], step) for item in value_sets]
        mean = sum(values) / len(values)
        std = sample_std(values)
        se = std / math.sqrt(len(values)) if values else float("nan")
        if se == 0.0:
            t_stat = math.inf if mean < target_loss else -math.inf
        else:
            t_stat = (target_loss - mean) / se
        benchmark_stat = (target_loss - mean) * math.sqrt(len(values))
        benchmark_pass = benchmark_stat >= benchmark_threshold
        if benchmark_pass and first_pass_step is None:
            first_pass_step = step
        rows.append(
            {
                "step": step,
                "n": len(values),
                "mean": mean,
                "std": std,
                "se": se,
                "target_loss": target_loss,
                "target_minus_mean": target_loss - mean,
                "t_stat": t_stat,
                "benchmark_stat": benchmark_stat,
                "benchmark_threshold": benchmark_threshold,
                "benchmark_pass": benchmark_pass,
                "best": min(values),
                "worst": max(values),
            }
        )

    return {
        "label": label,
        "run_dir": str(run_dir.relative_to(ICLR_DIR)),
        "trial_count": len(value_sets),
        "benchmark_formula": "(target_loss - mean_loss) * sqrt(n)",
        "benchmark_threshold": benchmark_threshold,
        "first_pass_step": first_pass_step,
        "rows": rows,
    }


def print_table(summaries: list[dict[str, object]]) -> None:
    header = (
        f"{'run':32s} {'step':>5s} {'n':>3s} {'mean':>10s} {'std':>10s} "
        f"{'se':>10s} {'t':>10s} {'bench':>10s} {'pass':>5s}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        label = str(summary["label"])
        for row in summary["rows"]:
            print(
                f"{label:32s} {int(row['step']):5d} {int(row['n']):3d} "
                f"{float(row['mean']):10.5f} {float(row['std']):10.5f} "
                f"{float(row['se']):10.5f} {float(row['t_stat']):10.3f} "
                f"{float(row['benchmark_stat']):10.5f} {str(bool(row['benchmark_pass'])):>5s}"
            )
        print(f"{label:32s} first_pass_step={summary['first_pass_step']}")


def main() -> None:
    args = parse_args()
    steps = sorted(set(args.steps or DEFAULT_STEPS))
    summaries = [
        summarize_run(label, ICLR_DIR / relative_dir, steps, args.target_loss, args.benchmark_threshold)
        for label, relative_dir in DEFAULT_RUNS
    ]
    print_table(summaries)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
