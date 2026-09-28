#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


SCRIPT = Path(__file__).resolve()
ICLR_DIR = SCRIPT.parents[2]
SOURCE = ICLR_DIR / "runs" / "shared_lr" / "seed42_diagnostics"
RAW_JSON = SOURCE / "raw" / "seed42_plain_muon_local_quadratic_diag.json"
ANALYSIS = SOURCE / "analysis"
FIGURES = ICLR_DIR / "figures"

OUT_FIGS = {
    "typed": FIGURES / "t_muon_l_shape_scaling.png",
    "all_layer": FIGURES / "muon_l_shape_scaling.png",
}
OUT_SUMMARY = ANALYSIS / "shape_scaling_deviation_summary.json"
ALIGNMENT_NAMES = {
    "typed": "t-MuonL",
    "all_layer": "MuonL",
}
TYPE_ORDER = ["attn_q", "attn_k", "attn_v", "attn_proj", "mlp_fc", "mlp_proj"]
TYPE_LABELS = {
    "attn_q": r"$W_Q$",
    "attn_k": r"$W_K$",
    "attn_v": r"$W_V$",
    "attn_proj": r"$W_O$",
    "mlp_fc": r"$W_{\mathrm{up}}$",
    "mlp_proj": r"$W_{\mathrm{down}}$",
}
TYPE_COLORS = {
    "attn_k": "#0072B2",
    "attn_proj": "#D55E00",
    "attn_q": "#009E73",
    "attn_v": "#CC79A7",
    "mlp_fc": "#E69F00",
    "mlp_proj": "#56B4E9",
}


def quantile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def abs_rank_signed(values: list[float], q: float) -> float:
    vals = sorted((v for v in values if math.isfinite(v)), key=lambda v: (abs(v), v))
    if not vals:
        return float("nan")
    idx = int(round(q * (len(vals) - 1)))
    return vals[idx]


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def std(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    mu = mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def collect_rows(alignment: str) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    records = json.loads(RAW_JSON.read_text())
    long_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for record in records:
        step = int(record["step"])
        blocks = record["blocks"]
        r_pdf = [math.sqrt(float(block["m_i"])) / float(block["w_norm"]) for block in blocks]
        q_pdf = [1.0 / value for value in r_pdf]
        indices_by_type: dict[str, list[int]] = defaultdict(list)
        for idx, block in enumerate(blocks):
            indices_by_type[str(block["type"])].append(idx)
        qbar_all = mean(q_pdf)
        wbar_by_type = {
            layer_type: mean([float(blocks[idx]["w_norm"]) for idx in indices])
            for layer_type, indices in indices_by_type.items()
        }

        deviations_by_type: dict[str, list[float]] = defaultdict(list)
        for idx, block in enumerate(blocks):
            layer_type = str(block["type"])
            if alignment == "typed":
                ratio = float(block["w_norm"]) / wbar_by_type[layer_type]
            elif alignment == "all_layer":
                ratio = q_pdf[idx] / qbar_all
            else:
                raise ValueError(f"unknown alignment {alignment!r}")
            deviation = ratio - 1.0
            deviations_by_type[layer_type].append(deviation)
            long_rows.append(
                {
                    "step": step,
                    "type": layer_type,
                    "block_idx": int(block["block_idx"]),
                    "block_name": str(block["block_name"]),
                    f"{alignment}_update_over_sqrt_m": ratio,
                    "signed_deviation": deviation,
                    "abs_deviation": abs(deviation),
                }
            )

        for layer_type in TYPE_ORDER:
            deviations = deviations_by_type[layer_type]
            abs_deviations = [abs(v) for v in deviations]
            summary_rows.append(
                {
                    "step": step,
                    "type": layer_type,
                    "block_count": len(deviations),
                    "min_signed_deviation": min(deviations),
                    "q25_abs_rank_signed_deviation": abs_rank_signed(deviations, 0.25),
                    "median_abs_rank_signed_deviation": abs_rank_signed(deviations, 0.50),
                    "mean_signed_deviation": mean(deviations),
                    "q75_abs_rank_signed_deviation": abs_rank_signed(deviations, 0.75),
                    "max_signed_deviation": max(deviations),
                    "min_abs_deviation": min(abs_deviations),
                    "q25_abs_deviation": quantile(abs_deviations, 0.25),
                    "median_abs_deviation": quantile(abs_deviations, 0.50),
                    "q75_abs_deviation": quantile(abs_deviations, 0.75),
                    "max_abs_deviation": max(abs_deviations),
                    "std_signed_deviation": std(deviations),
                }
            )

    return summary_rows, long_rows


def plot_alignment(
    alignment: str,
    summary_rows: list[dict[str, float | int | str]],
    long_rows: list[dict[str, float | int | str]],
    y_lim: tuple[float, float],
) -> None:
    summary_by_type: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    long_by_type_block: dict[str, dict[int, list[dict[str, float | int | str]]]] = defaultdict(lambda: defaultdict(list))
    for row in summary_rows:
        summary_by_type[str(row["type"])].append(row)
    for row in long_rows:
        long_by_type_block[str(row["type"])][int(row["block_idx"])].append(row)

    plt.style.use("default")
    fig, axes = plt.subplots(2, 3, figsize=(14.1667, 9.0667), dpi=150, sharex=True, sharey=True)
    axes_flat = list(axes.ravel())

    for ax, layer_type in zip(axes_flat, TYPE_ORDER):
        color = TYPE_COLORS[layer_type]
        rows = sorted(summary_by_type[layer_type], key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in rows]
        y_min = [100.0 * float(row["min_abs_deviation"]) for row in rows]
        y_q25 = [100.0 * float(row["q25_abs_deviation"]) for row in rows]
        y_med = [100.0 * float(row["median_abs_deviation"]) for row in rows]
        y_q75 = [100.0 * float(row["q75_abs_deviation"]) for row in rows]
        y_max = [100.0 * float(row["max_abs_deviation"]) for row in rows]

        ax.fill_between(steps, y_min, y_max, color=color, alpha=0.16, linewidth=0)
        ax.fill_between(steps, y_q25, y_q75, color=color, alpha=0.34, linewidth=0)
        for block_idx, block_rows in sorted(long_by_type_block[layer_type].items()):
            block_rows = sorted(block_rows, key=lambda row: int(row["step"]))
            ax.plot(
                [int(row["step"]) for row in block_rows],
                [100.0 * float(row["abs_deviation"]) for row in block_rows],
                color=color,
                linewidth=1.1,
                alpha=0.22,
            )
        ax.plot(steps, y_med, color=color, linewidth=3.0)
        ax.set_title(TYPE_LABELS[layer_type], fontsize=30, pad=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=24)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_ylim(*y_lim)

    for ax in axes[-1, :]:
        ax.set_xlabel("Diagnostic step", fontsize=28)
    ylabel = "Norm deviation, t-MuonL (%)" if alignment == "typed" else "Norm deviation, MuonL (%)"
    fig.supylabel(ylabel, fontsize=34, x=0.004)
    legend_handles = [
        Patch(facecolor="#777777", alpha=0.16, edgecolor="none", label="Min-max"),
        Patch(facecolor="#777777", alpha=0.34, edgecolor="none", label="25-75 quantile |dev|"),
        Line2D([0], [0], color="#333333", linewidth=3.0, label="Median |dev|"),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#d0d0d0",
        fontsize=26,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.000),
        ncol=3,
        borderpad=0.3,
        handlelength=2.2,
        handletextpad=0.55,
        columnspacing=1.25,
        labelspacing=0.25,
    )

    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.87))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGS[alignment], bbox_inches="tight")
    plt.close(fig)

    print(OUT_FIGS[alignment])


def paper_text_statistics(
    summary_rows: list[dict[str, float | int | str]],
) -> dict[str, float | int | str | list[int]]:
    """Statistics reported in the Figure 7 text.

    For each diagnostic step and matrix type, first summarize the 12 block
    deviations by their median and maximum absolute percentage deviation, then
    average those per-type-step summaries over all steps and matrix types.
    """

    median_values = [100.0 * float(row["median_abs_deviation"]) for row in summary_rows]
    maximum_values = [100.0 * float(row["max_abs_deviation"]) for row in summary_rows]
    return {
        "statistic": (
            "mean over diagnostic-step/type summaries; each summary is computed "
            "across the 12 blocks of that matrix type"
        ),
        "type_step_count": len(summary_rows),
        "block_counts_per_type_step": sorted({int(row["block_count"]) for row in summary_rows}),
        "averaged_median_abs_percentage_deviation": mean(median_values),
        "averaged_max_abs_percentage_deviation": mean(maximum_values),
    }


def main() -> None:
    rows_by_alignment = {
        alignment: collect_rows(alignment)
        for alignment in ["typed", "all_layer"]
    }
    all_deviations = [
        100.0 * float(row["abs_deviation"])
        for _, long_rows in rows_by_alignment.values()
        for row in long_rows
    ]
    max_abs = max(all_deviations)
    y_lim = (0.0, 1.08 * max_abs)

    paper_stats = {}
    for alignment, (summary_rows, long_rows) in rows_by_alignment.items():
        plot_alignment(alignment, summary_rows, long_rows, y_lim)
        paper_stats[alignment] = {
            "method": ALIGNMENT_NAMES[alignment],
            **paper_text_statistics(summary_rows),
        }

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(paper_stats, indent=2, sort_keys=True) + "\n")
    for alignment in ["typed", "all_layer"]:
        stats = paper_stats[alignment]
        print(
            f"{stats['method']}: averaged median "
            f"{stats['averaged_median_abs_percentage_deviation']:.6f}%, "
            f"averaged maximum {stats['averaged_max_abs_percentage_deviation']:.6f}%"
        )
    print(OUT_SUMMARY)


if __name__ == "__main__":
    main()
