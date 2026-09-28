#!/usr/bin/env python3
"""Plot the shared-learning-rate five-seed validation-loss comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT = Path(__file__).resolve()
ICLR_CODE = SCRIPT.parents[2]
sys.path.insert(0, str(ICLR_CODE))

from scripts.analysis.result_parsing import read_validation_loss, summarize_by_seed
from scripts.plot_figures.style import add_legend, configure_paper_style, style_axis

RUN_ROOT = ICLR_CODE / "experiment_results" / "shared_lr" / "random5_main"
FIGURES = ICLR_CODE / "figures"
OUT_FIG = FIGURES / "shared_lr_random5_loss.png"

MIN_STEP = 3100

SERIES = [
    ("plain_muon", "Muon", "#333333"),
    ("typed_muonl", "t-MuonL", "#0072B2"),
    ("alllayer_muonl", "MuonL", "#D55E00"),
]


def load_seeds(run_root: Path) -> list[int]:
    manifest_path = run_root / "seed_manifest.json"
    if not manifest_path.exists():
        return [161317530, 1490436515, 1391286422, 1892922382, 681062724]
    manifest = json.loads(manifest_path.read_text())
    seeds = manifest["paired_design"]["seeds"]
    return [int(seed) for seed in seeds]


def raw_log_path(run_root: Path, scheme: str, seed: int) -> Path:
    legacy = run_root / scheme / f"seed{seed}" / "raw"
    generated = run_root / scheme / f"seed_{seed}" / "raw"
    raw_dir = generated if generated.exists() else legacy
    matches = sorted(raw_dir.glob(f"seed{seed}_{scheme}*.txt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one raw log under {raw_dir}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output", type=Path, default=OUT_FIG)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    seeds = load_seeds(run_root)
    data: list[tuple[str, str, str, list[dict[str, float | int]]]] = []

    for scheme, label, color in SERIES:
        values_by_seed = {
            seed: read_validation_loss(raw_log_path(run_root, scheme, seed), min_step=MIN_STEP)
            for seed in seeds
        }
        rows = summarize_by_seed(values_by_seed)
        data.append((scheme, label, color, rows))

    shared_steps = sorted(set.intersection(*(set(int(row["step"]) for row in rows) for _, _, _, rows in data)))
    configure_paper_style()
    fig, ax = plt.subplots(figsize=(14.1667, 9.0667), dpi=150)

    all_losses: list[float] = []
    for _scheme, label, color, rows in data:
        rows_by_step = {int(row["step"]): row for row in rows}
        means = [float(rows_by_step[step]["mean"]) for step in shared_steps]
        stds = [float(rows_by_step[step]["std"]) for step in shared_steps]
        lower = [mean - std for mean, std in zip(means, stds)]
        upper = [mean + std for mean, std in zip(means, stds)]
        all_losses.extend(lower)
        all_losses.extend(upper)
        ax.fill_between(shared_steps, lower, upper, color=color, alpha=0.18, linewidth=0)
        ax.plot(
            shared_steps,
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
    ax.set_xlim(min(shared_steps) - 8, max(shared_steps) + 8)
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
