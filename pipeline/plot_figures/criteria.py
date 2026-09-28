#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator


SCRIPT = Path(__file__).resolve()
ICLR_DIR = SCRIPT.parents[2]
sys.path.insert(0, str(ICLR_DIR))
SOURCE = ICLR_DIR / "runs" / "shared_lr" / "seed42_diagnostics"
ANALYSIS = SOURCE / "analysis"
FIGURES = ICLR_DIR / "figures"
RAW_JSON = SOURCE / "raw" / "seed42_plain_muon_local_quadratic_diag.json"

GLOBAL_COLORS = {
    "typed": "#0072B2",
    "all_layer": "#D55E00",
}
TERM_LABELS = {
    "first_order_gain": "First-order descent gain",
    "curvature_penalty": "Curvature gain",
    "rescaling_cost": "Rescaling cost",
}
TERM_STYLES = {
    "first_order_gain": {"marker": "o", "linestyle": "-"},
    "curvature_penalty": {"marker": "s", "linestyle": "--"},
    "rescaling_cost": {"marker": "^", "linestyle": ":"},
}
CSV_ROW_CACHE: dict[str, list[dict[str, object]]] | None = None


def rebuild_csv_rows_from_raw() -> dict[str, list[dict[str, object]]]:
    from experiment_lib.diagnostics import build_rows, read_records

    records = read_records(RAW_JSON)
    _, _, score_rows, criteria_rows = build_rows(records, score_lr_policy="unit")

    global_all_layer_rows = []
    for row in criteria_rows:
        row = dict(row)
        if row["candidate"] == "all_layer" and row["scope"] == "global":
            global_all_layer_rows.append(row)

    global_by_step: dict[int, dict[str, float]] = defaultdict(dict)
    for row in score_rows:
        if row["scope"] == "global":
            global_by_step[int(row["step"])][str(row["candidate"])] = float(row["score"])

    global_score_rows = []
    for step, scores in sorted(global_by_step.items()):
        plain = scores["plain"]
        global_score_rows.append(
            {
                "step": step,
                "typed_eq9_over_plain_score": scores["typed"] / plain,
                "all_layer_over_plain_score": scores["all_layer"] / plain,
            }
        )

    return {
        "global_all_layer_vs_plain_corollary.csv": global_all_layer_rows,
        "global_scores_eq9_corollary.csv": global_score_rows,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    global CSV_ROW_CACHE
    if path.exists():
        with path.open(newline="") as f:
            return list(csv.DictReader(f))

    if path.parent == ANALYSIS:
        if CSV_ROW_CACHE is None:
            CSV_ROW_CACHE = rebuild_csv_rows_from_raw()
        if path.name in CSV_ROW_CACHE:
            return CSV_ROW_CACHE[path.name]  # type: ignore[return-value]

    raise FileNotFoundError(path)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def style_axis(ax: plt.Axes, ylabel: str, threshold: float = 1.0) -> None:
    ax.set_xlabel("Diagnostic step", fontsize=34)
    ax.set_ylabel(ylabel, fontsize=34)
    ax.tick_params(axis="both", labelsize=28)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, alpha=0.28)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=2.2)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def legend(ax: plt.Axes, **kwargs) -> None:
    leg = ax.legend(
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#d0d0d0",
        borderpad=0.3,
        handlelength=2.4,
        handletextpad=0.5,
        labelspacing=0.25,
        **kwargs,
    )
    leg.set_zorder(10)


def plot_global_criteria_margin() -> None:
    all_rows = sorted(
        read_csv(ANALYSIS / "global_all_layer_vs_plain_corollary.csv"),
        key=lambda row: int(row["step"]),
    )
    steps = [int(row["step"]) for row in all_rows]
    all_margins = [f(row, "lhs_sufficient") - f(row, "rhs_variance") for row in all_rows]
    term_rows = {
        int(row["step"]): row
        for row in read_all_layer_term_rows()
    }

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(14.1667, 9.0667), dpi=150)
    ax.plot(
        steps,
        all_margins,
        color="#111111",
        linewidth=4.2,
        alpha=0.94,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        steps,
        all_margins,
        linestyle="None",
        marker="o",
        markersize=12.8,
        markerfacecolor="#111111",
        markeredgecolor="#111111",
        markeredgewidth=0.0,
        color="#111111",
        label="MuonL margin (LHS - RHS)",
        zorder=5,
    )

    all_values = list(all_margins)
    term_linestyles = {
        "first_order_gain": "--",
        "curvature_penalty": "--",
        "rescaling_cost": ":",
    }
    term_colors = {
        "first_order_gain": "#E68632",
        "curvature_penalty": "#7B5EA7",
        "rescaling_cost": "#4C9F70",
    }
    for term in ("first_order_gain", "curvature_penalty", "rescaling_cost"):
        values = [float(term_rows[step][term]) for step in steps]
        all_values.extend(values)
        style = TERM_STYLES[term]
        ax.plot(
            steps,
            values,
            color=term_colors[term],
            marker=style["marker"],
            markersize=8.8,
            markerfacecolor=term_colors[term],
            markeredgecolor=term_colors[term],
            markeredgewidth=0.0,
            linestyle=term_linestyles[term],
            linewidth=2.3,
            alpha=0.82,
            label=TERM_LABELS[term],
            zorder=3,
        )

    style_axis(ax, "MuonL criterion diagnostic", threshold=0.0)
    span = max(all_values) - min(all_values)
    pad = 0.12 * span if span > 0.0 else 0.01
    ax.set_ylim(min(all_values) - pad, max(all_values) + 1.75 * pad)
    leg = ax.legend(
        fontsize=24,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        borderaxespad=0.0,
        ncol=2,
        frameon=False,
        handlelength=2.4,
        handletextpad=0.5,
        columnspacing=1.5,
        labelspacing=0.25,
    )
    leg.set_zorder(10)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.15, top=0.98)
    save(fig, "shared_lr_criterion_margin")


def plot_global_scores() -> None:
    rows = sorted(
        read_csv(ANALYSIS / "global_scores_eq9_corollary.csv"),
        key=lambda row: int(row["step"]),
    )
    steps = [int(row["step"]) for row in rows]
    typed = [f(row, "typed_eq9_over_plain_score") for row in rows]
    all_layer = [f(row, "all_layer_over_plain_score") for row in rows]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(14.1667, 9.0667), dpi=150)
    ax.plot(
        steps,
        typed,
        color=GLOBAL_COLORS["typed"],
        linewidth=4.2,
        alpha=0.94,
        solid_capstyle="round",
        zorder=2,
    )
    ax.plot(
        steps,
        typed,
        linestyle="None",
        marker="o",
        markersize=12.8,
        markerfacecolor=GLOBAL_COLORS["typed"],
        markeredgecolor=GLOBAL_COLORS["typed"],
        markeredgewidth=0.0,
        color=GLOBAL_COLORS["typed"],
        label=r"$S_{\mathrm{SL}}(x^{\mathrm{typed}})/S_{\mathrm{SL}}(x^{\mathrm{base}})$",
        zorder=3,
    )
    ax.plot(
        steps,
        all_layer,
        color="#111111",
        linewidth=4.2,
        alpha=0.94,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        steps,
        all_layer,
        linestyle="None",
        marker="o",
        markersize=12.8,
        markerfacecolor="#111111",
        markeredgecolor="#111111",
        markeredgewidth=0.0,
        color="#111111",
        label=r"$S_{\mathrm{SL}}(x^{\mathrm{full}})/S_{\mathrm{SL}}(x^{\mathrm{base}})$",
        zorder=5,
    )

    ratios = typed + all_layer
    style_axis(ax, "Score ratio")
    ax.set_ylim(0.988, 1.032)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    legend(
        ax,
        fontsize=28,
        loc="upper right",
    )
    fig.tight_layout()
    save(fig, "shared_lr_score_ratios")


def read_all_layer_term_rows() -> list[dict[str, float]]:
    rows = sorted(
        read_csv(ANALYSIS / "global_all_layer_vs_plain_corollary.csv"),
        key=lambda row: int(row["step"]),
    )
    return [
        {
            "step": int(row["step"]),
            "first_order_gain": 2.0 * f(row, "c1"),
            "curvature_penalty": -f(row, "c2"),
            "rescaling_cost": f(row, "rhs_variance"),
        }
        for row in rows
    ]



def main() -> None:
    plot_global_criteria_margin()
    plot_global_scores()
    print(FIGURES / "shared_lr_criterion_margin.png")
    print(FIGURES / "shared_lr_score_ratios.png")


if __name__ == "__main__":
    main()
