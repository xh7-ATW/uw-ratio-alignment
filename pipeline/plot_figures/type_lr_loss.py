#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT = Path(__file__).resolve()
ICLR_CODE = SCRIPT.parents[2]
sys.path.insert(0, str(ICLR_CODE))

from pipeline.analysis.result_parsing import read_validation_loss, summarize_by_seed
from pipeline.plot_figures.style import add_legend, configure_paper_style, style_axis

RUNS = ICLR_CODE / "runs"
FIGURES = ICLR_CODE / "figures"

OPT_RUN = RUNS / "type_lr" / "random5_muon_vs_tmuonl"
OPT_TYPED_RUN = OPT_RUN / "typed_muonl"
OPT_MUON_RUN = OPT_RUN / "plain_muon"

OUT_FIG = FIGURES / "type_lr_random5_loss.png"

SEEDS = [161317530, 1490436515, 1391286422, 1892922382, 681062724]
MIN_STEP = 3100
TARGET_STEPS = [3100, 3125, 3150, 3175, 3200, 3225, 3250, 3275, 3300]


def read_complete_val_loss(path: Path) -> dict[int, float]:
    vals = read_validation_loss(path, min_step=MIN_STEP)
    if max(vals) < 3300:
        raise RuntimeError(f"Run is not complete to 3300 in {path}")
    return vals


def val_json_path(root: Path, scheme: str, seed: int) -> Path:
    generated = root / f"seed_{seed}" / "raw"
    legacy = root / f"seed{seed}" / "raw"
    raw_dir = generated if generated.exists() else legacy
    matches = sorted(raw_dir.glob(f"seed{seed}_{scheme}_*_val_loss.jsonl"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one val-loss jsonl under {raw_dir}, found {len(matches)}")
    return matches[0]


def typed_muonl_path(seed: int) -> Path:
    return val_json_path(OPT_TYPED_RUN, "typed_muonl", seed)


def plain_muon_path(seed: int) -> Path:
    return val_json_path(OPT_MUON_RUN, "plain_muon", seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain-root", type=Path, default=OPT_MUON_RUN)
    parser.add_argument("--typed-root", type=Path, default=OPT_TYPED_RUN)
    parser.add_argument("--output", type=Path, default=OUT_FIG)
    args = parser.parse_args()
    series = [
        ("plain_muon", "Muon", "#333333", args.plain_root.resolve()),
        ("typed_muonl", "t-MuonL", "#0072B2", args.typed_root.resolve()),
    ]

    data: list[tuple[str, str, str, list[dict[str, float | int]]]] = []

    for scheme, label, color, root in series:
        values_by_seed = {
            seed: read_complete_val_loss(val_json_path(root, scheme, seed))
            for seed in SEEDS
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
