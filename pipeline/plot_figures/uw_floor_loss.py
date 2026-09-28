#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT = Path(__file__).resolve()
ICLR = SCRIPT.parents[2]
RESULTS = ICLR.parent
sys.path.insert(0, str(ICLR))

from pipeline.analysis.result_parsing import read_validation_loss, summarize_by_seed
from pipeline.plot_figures.style import add_legend, configure_paper_style, style_axis

SHARED_RUN_ROOT = ICLR / "runs" / "shared_lr" / "random5_main"
UWFLOOR_RUN_ROOT = (
    ICLR
    / "runs"
    / "uw_floor"
    / "random5_c035"
    / "muon_c035"
)
FIGURES = ICLR / "figures"
OUT_FIG = FIGURES / "uw_floor_loss.png"

TARGET_STEPS = [3100, 3125, 3150, 3175, 3200, 3225, 3250, 3275, 3300]

SERIES = [
    ("plain_muon", "Muon", "#333333", "txt"),
    ("uw_floor", "u/w floor", "#0072B2", "jsonl"),
    ("alllayer_muonl", "MuonL", "#D55E00", "txt"),
]


def load_seeds(shared_run_root: Path) -> list[int]:
    manifest = json.loads((shared_run_root / "seed_manifest.json").read_text())
    seeds = manifest["paired_design"]["seeds"]
    return [int(seed) for seed in seeds]


def raw_log_path(
    scheme: str,
    seed: int,
    source_kind: str,
    shared_run_root: Path,
    uw_floor_run_root: Path,
) -> Path:
    if source_kind == "txt":
        raw_dir = shared_run_root / scheme / f"seed{seed}" / "raw"
        matches = sorted(raw_dir.glob(f"seed{seed}_{scheme}_*.txt"))
    elif source_kind == "jsonl":
        legacy_dir = uw_floor_run_root / f"seed{seed}" / "raw"
        generated_dir = uw_floor_run_root / f"seed_{seed}" / "raw"
        raw_dir = generated_dir if generated_dir.exists() else legacy_dir
        matches = sorted(raw_dir.glob(f"seed{seed}_*_val_loss.jsonl"))
    else:
        raise ValueError(f"unknown source kind {source_kind!r}")
    if len(matches) != 1:
        raise RuntimeError(f"Expected one raw log under {raw_dir}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-run-root", type=Path, default=SHARED_RUN_ROOT)
    parser.add_argument("--uw-floor-run-root", type=Path, default=UWFLOOR_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=OUT_FIG)
    args = parser.parse_args()
    shared_run_root = args.shared_run_root.resolve()
    uw_floor_run_root = args.uw_floor_run_root.resolve()
    output = args.output.resolve()

    seeds = load_seeds(shared_run_root)
    data: list[tuple[str, str, str, list[dict[str, float | int]]]] = []

    for scheme, label, color, source_kind in SERIES:
        values_by_seed = {
            seed: read_validation_loss(
                raw_log_path(
                    scheme, seed, source_kind, shared_run_root, uw_floor_run_root
                ),
                min_step=min(TARGET_STEPS),
            )
            for seed in seeds
        }
        rows = summarize_by_seed(values_by_seed, target_steps=TARGET_STEPS)
        data.append((scheme, label, color, rows))

    configure_paper_style()
    fig, ax = plt.subplots(figsize=(14.1667, 9.0667), dpi=150)

    all_losses: list[float] = []
    for _scheme, label, color, rows in data:
        rows_by_step = {int(row["step"]): row for row in rows}
        means = [float(rows_by_step[step]["mean"]) for step in TARGET_STEPS]
        stds = [float(rows_by_step[step]["std"]) for step in TARGET_STEPS]
        lower = [mean - std for mean, std in zip(means, stds)]
        upper = [mean + std for mean, std in zip(means, stds)]
        all_losses.extend(lower)
        all_losses.extend(upper)
        ax.fill_between(TARGET_STEPS, lower, upper, color=color, alpha=0.18, linewidth=0)
        ax.plot(
            TARGET_STEPS,
            means,
            marker="o",
            markersize=7,
            linewidth=3.2,
            color=color,
            label=f"{label} mean ({means[-1]:.5f})",
            solid_capstyle="round",
        )

    ymin = min(all_losses)
    ymax = max(all_losses)
    pad = max(0.12 * (ymax - ymin), 0.0002)
    ax.set_xlim(min(TARGET_STEPS) - 8, max(TARGET_STEPS) + 8)
    ax.set_ylim(ymin - pad, ymax + pad)
    style_axis(ax, xlabel="Optimizer step", ylabel="Validation loss")
    add_legend(ax, fontsize=28, loc="upper right")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
