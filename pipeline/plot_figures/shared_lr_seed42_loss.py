#!/usr/bin/env python3
"""Plot the shared-learning-rate seed-42 validation-loss comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT = Path(__file__).resolve()
ICLR_CODE = SCRIPT.parents[2]
sys.path.insert(0, str(ICLR_CODE))

from pipeline.analysis.result_parsing import read_validation_loss, value_at_step
from pipeline.plot_figures.style import add_legend, configure_paper_style, style_axis

RUN_ROOT = ICLR_CODE / "runs" / "shared_lr" / "seed42_main"
FIGURES = ICLR_CODE / "figures"
OUT_FIG = FIGURES / "shared_lr_seed42_loss.png"

TARGET_STEPS = [3100, 3125, 3150, 3175, 3200, 3225, 3250, 3275, 3300]

SERIES = [
    ("plain_muon", "Muon", "#333333"),
    ("typed_muonl", "t-MuonL", "#0072B2"),
    ("alllayer_muonl", "MuonL", "#D55E00"),
]


def raw_log_path(scheme: str) -> Path:
    raw_dir = RUN_ROOT / scheme / "raw"
    matches = sorted(raw_dir.glob(f"seed42_{scheme}_*.txt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one raw log under {raw_dir}, found {len(matches)}")
    return matches[0]


def main() -> None:
    configure_paper_style()
    fig, ax = plt.subplots(figsize=(14.1667, 9.0667), dpi=150)

    all_losses: list[float] = []
    for scheme, label, color in SERIES:
        values = read_validation_loss(raw_log_path(scheme), min_step=min(TARGET_STEPS))
        losses = [value_at_step(values, step) for step in TARGET_STEPS]
        all_losses.extend(losses)
        ax.plot(
            TARGET_STEPS,
            losses,
            marker="o",
            markersize=7,
            linewidth=3.2,
            color=color,
            label=f"{label} ({losses[-1]:.5f})",
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
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(OUT_FIG)


if __name__ == "__main__":
    main()
