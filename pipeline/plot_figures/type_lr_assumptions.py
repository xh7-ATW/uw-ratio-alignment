#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, LogLocator, MaxNLocator, MultipleLocator, NullFormatter
from matplotlib.ticker import NullLocator

SCRIPT = Path(__file__).resolve()
ICLR_DIR = SCRIPT.parents[2]
sys.path.insert(0, str(ICLR_DIR))

from experiment_lib.diagnostics import build_rows, read_records


TYPE_ORDER = ["attn_q", "attn_k", "attn_v", "attn_proj", "mlp_fc", "mlp_proj"]


def f(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 150,
        }
    )


def style_axis(ax: plt.Axes, ylabel: str, *, log_y: bool) -> None:
    ax.set_xlabel("Diagnostic step", fontsize=34)
    ax.set_ylabel(ylabel, fontsize=34)
    ax.tick_params(axis="both", labelsize=28)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=6))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=2.2)
    ax.grid(True, alpha=0.28, which="major")


def plot_series(
    out_path: Path,
    steps: list[int],
    series: list[tuple[str, list[float], str, str]],
    ylabel: str,
    min_note: str,
    *,
    force_linear: bool = False,
    y_major_step: float | None = None,
    log_tick_exponent_step: int | None = None,
    legend_ncol: int = 1,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    log_lower_factor: float = 0.9,
    log_upper_factor: float = 1.35,
    note_xy: tuple[float, float] = (0.03, 0.04),
) -> None:
    all_values = [value for _, values, _, _ in series for value in values]
    log_y = (
        not force_linear
        and all(value > 0.0 and math.isfinite(value) for value in all_values)
    )

    configure_style()
    fig, ax = plt.subplots(figsize=(14.1667, 9.0667), dpi=150)
    for series_idx, (label, values, color, linestyle) in enumerate(series):
        marker = ("o", "s", "^")[series_idx % 3]
        marker_face = "white" if series_idx == 1 else color
        marker_edge = color
        marker_edge_width = 1.6 if series_idx == 1 else 0.0
        ax.plot(
            steps,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=3.6,
            marker=marker,
            markersize=8.8,
            markerfacecolor=marker_face,
            markeredgecolor=marker_edge,
            markeredgewidth=marker_edge_width,
            label=label,
            solid_capstyle="round",
            zorder=2 + series_idx,
        )

    style_axis(ax, ylabel, log_y=log_y)
    ax.set_xlim(min(steps) - 80, max(steps) + 80)
    if log_y:
        positive_values = [value for value in all_values if value > 0.0 and math.isfinite(value)]
        ymin = min(positive_values)
        ymax = max(positive_values)
        lower = ymin * log_lower_factor
        upper = ymax * log_upper_factor
        ax.set_ylim(lower, upper)
        if log_tick_exponent_step is not None:
            lo_exp = math.floor(math.log10(lower))
            hi_exp = math.ceil(math.log10(upper))
            ticks = [
                10.0**exp
                for exp in range(lo_exp, hi_exp + 1)
                if exp % log_tick_exponent_step == 0
            ]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.text(
            note_xy[0],
            note_xy[1],
            min_note,
            transform=ax.transAxes,
            fontsize=30,
            va="bottom",
            ha="left",
        )
    else:
        ymin = min(all_values)
        ymax = max(all_values)
        if y_major_step is not None:
            ax.yaxis.set_major_locator(MultipleLocator(y_major_step))
            ax.set_ylim(-0.06 * y_major_step, 3.12 * y_major_step)
        else:
            span = ymax - ymin
            pad = 0.12 * span if span > 0.0 else 1e-6
            ax.set_ylim(ymin - pad, ymax + pad)
    leg = ax.legend(
        fontsize=28,
        loc="upper right",
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.55,
        columnspacing=1.0,
        labelspacing=0.3,
    )
    leg.set_zorder(10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def shared_curvature_series(diag_path: Path) -> tuple[list[int], list[tuple[str, list[float], str, str]], dict[str, float]]:
    records = read_records(diag_path)
    _, _, score_rows, _ = build_rows(records, score_lr_policy="unit")
    steps = [int(record["step"]) for record in records]

    by_step_candidate: dict[int, dict[str, float]] = defaultdict(dict)
    global_a_base_abs: list[float] = []

    for row in score_rows:
        step = int(row["step"])
        scope = str(row["scope"])
        candidate = str(row["candidate"])
        curvature = f(row["quadratic_curvature_sum"])
        first_order = f(row["first_order_sum"])
        if scope == "global":
            by_step_candidate[step][candidate] = curvature
            if candidate == "plain":
                global_a_base_abs.append(abs(first_order))

    base = [by_step_candidate[step]["plain"] for step in steps]
    full = [by_step_candidate[step]["all_layer"] for step in steps]

    series = [
        (r"$B(x^{\mathrm{base}})$", base, "#222222", "-"),
        (r"$B(x^{\mathrm{full}})$", full, "#D55E00", "-."),
    ]
    minima = {
        "B_x_base_min": min(base),
        "B_x_full_min": min(full),
        "all_positive": all(value > 0.0 for values in (base, full) for value in values),
        "min_t_abs_A_base_global_shared_lr": min(global_a_base_abs),
    }
    return steps, series, minima


def prop2_terms(record: dict[str, object]) -> dict[str, object]:
    blocks = record["blocks"]
    matrix = np.asarray(record["cross_curvature"]["matrix"], dtype=float)
    types = [type_name for type_name in TYPE_ORDER if any(block["type"] == type_name for block in blocks)]
    by_type = {
        type_name: [idx for idx, block in enumerate(blocks) if block["type"] == type_name]
        for type_name in types
    }

    n_blocks = len(blocks)
    w_norm = np.zeros(n_blocks, dtype=float)
    descent = np.zeros(n_blocks, dtype=float)

    for type_name in types:
        indices = by_type[type_name]
        for idx in indices:
            w_norm[idx] = f(blocks[idx]["w_norm"])
            descent[idx] = f(blocks[idx]["g_dot_v"])

    curv = matrix
    type_count = len(types)
    avec = np.zeros(type_count, dtype=float)
    bmat = np.zeros((type_count, type_count), dtype=float)
    gamma = np.zeros((type_count, type_count), dtype=float)
    cvec = np.zeros(type_count, dtype=float)
    w_unit_by_type: dict[str, np.ndarray] = {}

    for g, type_name in enumerate(types):
        indices = by_type[type_name]
        w_group = w_norm[indices]
        d_group = descent[indices]
        w_unit = w_group / np.mean(w_group)
        d_unit = d_group / np.mean(d_group)
        w_unit_by_type[type_name] = w_unit
        avec[g] = np.mean(d_group)
        cvec[g] = np.mean((w_unit - 1.0) * (d_unit - 1.0))

    for g, type_g in enumerate(types):
        idx_g = by_type[type_g]
        w_g = w_unit_by_type[type_g]
        for h, type_h in enumerate(types):
            idx_h = by_type[type_h]
            w_h = w_unit_by_type[type_h]
            block_curv = curv[np.ix_(idx_g, idx_h)]
            weight_product = w_g[:, None] * w_h[None, :]
            bmat[g, h] = np.mean(block_curv)
            cov_weight_curv = np.mean(
                (weight_product - np.mean(weight_product))
                * (block_curv - np.mean(block_curv))
            )
            gamma[g, h] = (
                (cvec[g] + cvec[h] + cvec[g] * cvec[h]) * bmat[g, h]
                - cov_weight_curv
            )

    bmat = 0.5 * (bmat + bmat.T)
    gamma = 0.5 * (gamma + gamma.T)
    dmat = np.diag(1.0 + cvec)
    typed_curv = dmat @ bmat @ dmat - gamma
    typed_curv = 0.5 * (typed_curv + typed_curv.T)

    b_eigs = np.linalg.eigvalsh(bmat)
    typed_eigs = np.linalg.eigvalsh(typed_curv)
    return {
        "step": int(record["step"]),
        "b_eig": [float(value) for value in b_eigs],
        "typed_eig": [float(value) for value in typed_eigs],
        "b_eig_min": float(b_eigs[0]),
        "typed_eig_min": float(typed_eigs[0]),
        "a_min_abs": float(np.min(np.abs(avec))),
        "one_plus_c_min_abs": float(np.min(np.abs(1.0 + cvec))),
    }


def type_specific_eigenvalues(diag_path: Path) -> tuple[list[int], np.ndarray, np.ndarray, dict[str, float]]:
    diagnostics = [prop2_terms(record) for record in read_records(diag_path)]
    steps = [row["step"] for row in diagnostics]
    b_eigs = np.asarray([row["b_eig"] for row in diagnostics], dtype=float)
    typed_eigs = np.asarray([row["typed_eig"] for row in diagnostics], dtype=float)
    minima = {
        "min_t_G_abs_A_G_type_specific_lr": min(row["a_min_abs"] for row in diagnostics),
        "min_t_G_abs_1_plus_c_G_type_specific_lr": min(row["one_plus_c_min_abs"] for row in diagnostics),
    }
    return steps, b_eigs, typed_eigs, minima


def plot_type_specific_eigenvalues(
    out_path: Path,
    steps: list[int],
    b_eigs: np.ndarray,
    typed_eigs: np.ndarray,
) -> None:
    all_values = np.concatenate([b_eigs.reshape(-1), typed_eigs.reshape(-1)])
    log_y = bool(np.all(np.isfinite(all_values)) and np.all(all_values > 0.0))

    configure_style()
    fig, ax = plt.subplots(figsize=(14.1667, 9.0667), dpi=150)
    eigen_count = b_eigs.shape[1]
    for k in range(eigen_count):
        ax.plot(
            steps,
            b_eigs[:, k],
            color="#222222",
            linestyle="-",
            linewidth=2.4,
            marker="o",
            markersize=6.0,
            markerfacecolor="#222222",
            markeredgewidth=0.0,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            steps,
            typed_eigs[:, k],
            color="#0072B2",
            linestyle=(0, (5.0, 4.0)),
            linewidth=2.8,
            marker="s",
            markersize=6.2,
            markerfacecolor="white",
            markeredgecolor="#0072B2",
            markeredgewidth=1.4,
            dash_capstyle="butt",
            solid_capstyle="round",
            zorder=3,
        )

    style_axis(ax, "Smallest eigenvalue", log_y=log_y)
    ax.set_ylabel("Eigenvalue", fontsize=34)
    ax.set_xlim(min(steps) - 80, max(steps) + 80)
    if log_y:
        ymin = float(np.min(all_values))
        ymax = float(np.max(all_values))
        lower_decade = 10.0 ** math.floor(math.log10(ymin))
        ax.set_ylim(0.9 * lower_decade, ymax * 12.0)
        ax.text(
            0.03,
            0.04,
            f"minimum plotted eigenvalue = {ymin:.3g} > 0",
            transform=ax.transAxes,
            fontsize=30,
            va="bottom",
            ha="left",
        )
    else:
        ymin = float(np.min(all_values))
        ymax = float(np.max(all_values))
        span = ymax - ymin
        pad = 0.12 * span if span > 0.0 else 1e-6
        ax.set_ylim(ymin - pad, ymax + pad)

    handles = [
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="-",
            marker="o",
            linewidth=3.2,
            markersize=7.5,
            label=r"$\lambda_k(\mathbf{K}),\ k=1,\ldots,6$",
        ),
        Line2D(
            [0],
            [0],
            color="#0072B2",
            linestyle=(0, (5.0, 4.0)),
            marker="s",
            linewidth=3.2,
            markersize=7.0,
            markerfacecolor="white",
            markeredgecolor="#0072B2",
            markeredgewidth=1.4,
            label=r"$\lambda_k(\mathbf{D}\mathbf{K}\mathbf{D}-\boldsymbol{\Gamma}),\ k=1,\ldots,6$",
        ),
    ]
    leg = ax.legend(
        handles=handles,
        fontsize=28,
        loc="upper right",
        frameon=False,
        handlelength=2.6,
        handletextpad=0.55,
        labelspacing=0.3,
    )
    leg.set_zorder(10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shared-diag",
        type=Path,
        default=ICLR_DIR
        / "runs/shared_lr/seed42_diagnostics/raw/seed42_plain_muon_local_quadratic_diag.json",
    )
    parser.add_argument(
        "--opt-diag",
        type=Path,
        default=ICLR_DIR
        / "runs/type_lr/seed42_diag/muon_diagnostics/raw/seed42_plain_muon_optlr_local_quadratic_diag.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ICLR_DIR / "figures",
    )
    parser.add_argument(
        "--shared-only",
        action="store_true",
        help="Regenerate Figure 6(a) without recomputing or overwriting Figure 6(b).",
    )
    args = parser.parse_args()

    shared_steps, shared_series, shared_minima = shared_curvature_series(args.shared_diag)
    shared_min_note = (
        "minimum plotted curvature = "
        f"{min(value for _, values, _, _ in shared_series for value in values):.3g} > 0"
    )
    plot_series(
        args.out_dir / "shared_lr_assumptions.png",
        shared_steps,
        shared_series,
        "Curvature coefficient",
        shared_min_note,
        legend_ncol=1,
        log_lower_factor=0.92,
        log_upper_factor=1.08,
        note_xy=(0.03, 0.03),
    )

    shared_summary = {
        "quantity": "global shared-learning-rate curvature coefficient B(x)",
        "diagnostic_steps": shared_steps,
        "values_by_candidate": {
            "x_base": shared_series[0][1],
            "x_full": shared_series[1][1],
        },
        "minima": {
            "x_base": shared_minima["B_x_base_min"],
            "x_full": shared_minima["B_x_full_min"],
        },
        "all_positive": shared_minima["all_positive"],
    }
    shared_summary_path = args.shared_diag.parents[1] / "analysis" / "shared_lr_assumptions_summary.json"
    shared_summary_path.parent.mkdir(parents=True, exist_ok=True)
    shared_summary_path.write_text(json.dumps(shared_summary, indent=2, sort_keys=True) + "\n")

    opt_minima: dict[str, float] = {}
    if not args.shared_only:
        opt_steps, b_eigs, typed_eigs, opt_minima = type_specific_eigenvalues(args.opt_diag)
        plot_type_specific_eigenvalues(
            args.out_dir / "type_lr_assumptions.png",
            opt_steps,
            b_eigs,
            typed_eigs,
        )

    print(
        json.dumps(
            {
                **shared_minima,
                **opt_minima,
                "shared_figure": str(args.out_dir / "shared_lr_assumptions.png"),
                "shared_summary": str(shared_summary_path),
                **(
                    {"type_specific_figure": str(args.out_dir / "type_lr_assumptions.png")}
                    if not args.shared_only
                    else {}
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
