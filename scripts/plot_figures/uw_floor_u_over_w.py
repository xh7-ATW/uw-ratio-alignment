#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SCRIPT = Path(__file__).resolve()
DRAFT = SCRIPT.parents[2]
RESULTS = DRAFT.parent
RATIO_JSON = (
    DRAFT
    / "experiment_results"
    / "uw_floor"
    / "seed42_c035_u_over_w"
    / "normuon_c035_u_over_w"
    / "raw"
    / "seed42_normuon_benchmark9_uwfloor_c0p35_u_over_w.json"
)
OUT = RESULTS / "iclr_code" / "figures" / "uw_floor_u_over_w.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, default=RATIO_JSON)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--ratio-field", choices=("blocks", "raw_blocks"), default="blocks")
    parser.add_argument("--kind", default="attnproj")
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=0.1)
    parser.add_argument("--ymax", type=float, default=1.0)
    args = parser.parse_args()

    data = json.loads(args.input_json.read_text())
    target_uw = data["metadata"]["target_uw"]
    records = data["records"]

    steps = [record["step"] for record in records]
    xmax = args.xmax if args.xmax is not None else max(steps)
    block_values = {
        block_idx: [
            record[args.ratio_field][str(block_idx)][args.kind]
            for record in records
        ]
        for block_idx in range(12)
    }

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(14.1667, 9.0667), dpi=150)
    colors = plt.get_cmap("tab20").colors

    for block_idx, values in block_values.items():
        ax.plot(
            steps,
            values,
            label=f"block {block_idx}",
            color=colors[block_idx],
            linewidth=2.5,
        )

    ax.axhline(
        target_uw,
        color="#666666",
        linestyle="--",
        linewidth=2.0,
        label="_nolegend_",
    )

    ax.set_xlim(0, xmax)
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_xlabel("Optimizer step", fontsize=34, labelpad=8)
    ax.set_ylabel("u-w ratio", fontsize=34, labelpad=8)
    ax.tick_params(axis="both", labelsize=28)
    ax.set_xticks([0, 800, 1600, 2400, int(xmax)])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, alpha=0.28)
    ax.legend(
        fontsize=22,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=6,
        frameon=False,
        handlelength=1.7,
        handletextpad=0.5,
        columnspacing=0.85,
        labelspacing=0.45,
    )

    fig.subplots_adjust(left=0.130, right=0.985, top=0.98, bottom=0.285)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
