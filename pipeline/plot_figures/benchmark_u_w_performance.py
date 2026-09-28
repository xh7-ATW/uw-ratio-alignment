#!/usr/bin/env python3
"""Plot external benchmark tail validation loss for u-w floor comparisons."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT = Path(__file__).resolve()
ICLR_DIR = SCRIPT.parents[2]
sys.path.insert(0, str(ICLR_DIR))

from pipeline.analysis.result_parsing import VALIDATION_LOSS_RE
from pipeline.plot_figures.style import configure_paper_style


RUN_ROOT = ICLR_DIR / "reference_results"
OUT = ICLR_DIR / "figures" / "benchmark_u_w_performance.png"
TARGET_LOSS = 3.28
MIN_STEP = 2500
BENCHMARK_30_KEEP_STEPS = {2500, 2625, 2750, 2875, 2930, 2940, 3000, 3020}
NO_U_W_FLOOR_PRUNING_NAME = "no " + "u/w" + " floor"


def read_validation_loss_trials(path: Path, *, min_step: int = 0) -> list[dict[int, float]]:
    """Read one or more concatenated validation-loss trials from a text log."""

    text = path.read_text(errors="replace")
    trials: list[dict[int, float]] = []
    current: dict[int, float] = {}
    previous_step: int | None = None

    for match in VALIDATION_LOSS_RE.finditer(text):
        step = int(match.group(1))
        if step < min_step:
            continue
        if previous_step is not None and step <= previous_step and current:
            trials.append(current)
            current = {}
        current[step] = float(match.group(3))
        previous_step = step

    if current:
        trials.append(current)
    if not trials:
        raise RuntimeError(f"No validation losses at step >= {min_step} in {path}")
    return trials


def read_mean_val_loss(run_dir: Path) -> tuple[list[int], list[float]]:
    losses_by_step: dict[int, list[float]] = defaultdict(list)
    for path in sorted(run_dir.glob("*.txt")):
        for trial in read_validation_loss_trials(path, min_step=MIN_STEP):
            for step, loss in trial.items():
                losses_by_step[step].append(loss)
    if not losses_by_step:
        raise RuntimeError(f"No validation losses found in {run_dir}")
    steps = sorted(losses_by_step)
    losses = [sum(losses_by_step[step]) / len(losses_by_step[step]) for step in steps]
    return steps, losses


def read_pruning_delta(path: Path, name: str) -> float:
    data = json.loads(path.read_text())
    for item in data["items"]:
        if item["name"] == name:
            return float(item["delta_val_loss"])
    raise RuntimeError(f"No pruning delta named {name!r} in {path}")


def main() -> None:
    benchmark_30_dir = RUN_ROOT / "benchmark_30_projection_pruning"
    no_floor_delta = read_pruning_delta(
        benchmark_30_dir / "pruning_data.json",
        NO_U_W_FLOOR_PRUNING_NAME,
    )
    series = [
        {
            "path": RUN_ROOT / "benchmark_9_normuon_u_w_floor",
            "label": "#9 NorMuon: u/w floor",
            "color": "#1f77b4",
            "linestyle": "-",
            "keep_steps": None,
            "loss_offset": 0.0,
        },
        {
            "path": RUN_ROOT / "benchmark_10_normuon_weight_decay",
            "label": "#10 NorMuon: weight decay",
            "color": "#1f77b4",
            "linestyle": "--",
            "keep_steps": None,
            "loss_offset": 0.0,
        },
        {
            "path": benchmark_30_dir,
            "label": "#30: u/w floor",
            "color": "#2ca02c",
            "linestyle": "-",
            "keep_steps": BENCHMARK_30_KEEP_STEPS,
            "loss_offset": 0.0,
        },
        {
            "path": benchmark_30_dir,
            "label": f"#30: no u/w floor (+{no_floor_delta:.4f})",
            "color": "#2ca02c",
            "linestyle": "--",
            "keep_steps": BENCHMARK_30_KEEP_STEPS,
            "loss_offset": no_floor_delta,
        },
    ]

    configure_paper_style()
    fig, ax = plt.subplots(figsize=(14.1667, 9.0667), dpi=150)

    for item in series:
        steps, losses = read_mean_val_loss(item["path"])
        if item["keep_steps"] is not None:
            kept = [(step, loss) for step, loss in zip(steps, losses) if step in item["keep_steps"]]
            steps = [step for step, _ in kept]
            losses = [loss for _, loss in kept]
        if item["loss_offset"]:
            losses = [loss + item["loss_offset"] for loss in losses]
        plot_kwargs = {}
        if item["linestyle"] == "--":
            plot_kwargs["dashes"] = (8, 5)
        ax.plot(
            steps,
            losses,
            marker="o",
            markersize=7,
            linewidth=3.2,
            linestyle=item["linestyle"],
            color=item["color"],
            label=item["label"],
            solid_capstyle="round",
            dash_capstyle="round",
            **plot_kwargs,
        )

    ax.axhline(TARGET_LOSS, color="gray", linestyle="--", linewidth=2.2)
    ax.annotate(
        "target=3.28",
        xy=(2510, TARGET_LOSS),
        xytext=(0, 12),
        textcoords="offset points",
        color="gray",
        fontsize=18,
    )

    ax.set_xlim(2480, 3335)
    ax.set_ylim(3.27, 3.35)
    ax.set_xlabel("Training steps @ 0.5M bsz", fontsize=26)
    ax.set_ylabel("Tail validation loss", fontsize=26)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, alpha=0.28)
    ax.legend(
        fontsize=22,
        loc="upper right",
        frameon=True,
        borderpad=0.25,
        handlelength=2.4,
        handletextpad=0.5,
        labelspacing=0.25,
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
