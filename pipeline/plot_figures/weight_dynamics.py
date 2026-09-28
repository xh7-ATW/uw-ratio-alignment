#!/usr/bin/env python3
"""Plot per-type weight-norm dynamics for MuonL and OrScale variants."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SCRIPT = Path(__file__).resolve()
ICLR_DIR = SCRIPT.parents[2]
sys.path.insert(0, str(ICLR_DIR))

from pipeline.plot_figures.style import configure_paper_style


RUN_ROOT = ICLR_DIR / "runs" / "weight_dynamics"
OUT = ICLR_DIR / "figures" / "weight_dynamics.png"

SERIES = [
    ("MuonL", RUN_ROOT / "muonl", "#D55E00"),
    ("decoupled-OrScale-V", RUN_ROOT / "decoupled_orscale", "#0072B2"),
    ("OrScale-V", RUN_ROOT / "orscale", "#009E73"),
]

TYPE_ORDER = ["attn_q", "attn_k", "attn_v", "attn_proj", "mlp_fc", "mlp_proj"]
TYPE_LABELS = {
    "attn_q": r"$W_Q$",
    "attn_k": r"$W_K$",
    "attn_v": r"$W_V$",
    "attn_proj": r"$W_O$",
    "mlp_fc": r"$W_{\mathrm{up}}$",
    "mlp_proj": r"$W_{\mathrm{down}}$",
}

STEP_STRIDE = 5


def weight_norm_file(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("*weight_norms.jsonl"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one weight-norm JSONL under {run_dir}, found {len(matches)}")
    return matches[0]


def read_weight_norm_means(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    steps_by_type: dict[str, list[int]] = {layer_type: [] for layer_type in TYPE_ORDER}
    means_by_type: dict[str, list[float]] = {layer_type: [] for layer_type in TYPE_ORDER}

    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            step = int(record["step"])
            if step % STEP_STRIDE != 0:
                continue
            values_by_type: dict[str, list[float]] = defaultdict(list)
            for block in record.get("blocks") or record.get("records"):
                layer_type = str(block["type"])
                if layer_type in TYPE_ORDER:
                    values_by_type[layer_type].append(float(block["w_norm"]))
            for layer_type in TYPE_ORDER:
                values = values_by_type[layer_type]
                if values:
                    steps_by_type[layer_type].append(step)
                    means_by_type[layer_type].append(sum(values) / len(values))

    return {
        layer_type: (steps_by_type[layer_type], means_by_type[layer_type])
        for layer_type in TYPE_ORDER
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--muonl-run", type=Path, default=RUN_ROOT / "muonl")
    parser.add_argument("--decoupled-orscale-run", type=Path, default=RUN_ROOT / "decoupled_orscale")
    parser.add_argument("--orscale-run", type=Path, default=RUN_ROOT / "orscale")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    series = [
        ("MuonL", args.muonl_run.resolve(), "#D55E00"),
        ("decoupled-OrScale-V", args.decoupled_orscale_run.resolve(), "#0072B2"),
        ("OrScale-V", args.orscale_run.resolve(), "#009E73"),
    ]
    configure_paper_style()
    data = [
        (label, read_weight_norm_means(weight_norm_file(run_dir)), color)
        for label, run_dir, color in series
    ]

    fig, axes = plt.subplots(
        2, 3, figsize=(14.1667, 9.0667), dpi=150, sharex=True, sharey=True
    )
    axes_flat = list(axes.ravel())
    global_ymax = max(
        max(means)
        for _, series_by_type, _ in data
        for _, means in series_by_type.values()
        if means
    )

    for ax, layer_type in zip(axes_flat, TYPE_ORDER):
        for label, series_by_type, color in data:
            steps, means = series_by_type[layer_type]
            ax.plot(
                steps,
                means,
                color=color,
                linewidth=3.6,
                label=label,
                solid_capstyle="round",
            )
        ax.set_ylim(0.0, global_ymax * 1.08 if global_ymax > 0.0 else 1.0)
        ax.set_title(TYPE_LABELS[layer_type], fontsize=30, pad=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=24)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    for ax in axes[-1, :]:
        ax.set_xlabel("Optimizer step", fontsize=28)
    fig.supylabel(r"Mean $\|W\|_F$ across blocks", fontsize=34, x=0.018)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.990),
        ncol=3,
        frameon=False,
        fontsize=28,
        borderpad=0.35,
        handlelength=2.7,
        handletextpad=0.6,
        labelspacing=0.25,
    )
    fig.tight_layout(rect=(0.04, 0.0, 1.0, 0.890))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
