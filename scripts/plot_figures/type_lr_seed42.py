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
from matplotlib.ticker import MaxNLocator

SCRIPT = Path(__file__).resolve()
ICLR_DIR = SCRIPT.parents[2]
sys.path.insert(0, str(ICLR_DIR))

from src.diagnostics import build_rows, read_records


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
    "attn_q": "#009E73",
    "attn_k": "#0072B2",
    "attn_v": "#5E3C99",
    "attn_proj": "#E41A1C",
    "mlp_fc": "#F0A202",
    "mlp_proj": "#56B4E9",
}
COLORS = {
    "opt_muon": "#222222",
    "opt_typed": "#0072B2",
    "typed": "#0072B2",
}
TERM_LABELS = {
    "first_order_gain": "First-order descent gain",
    "curvature_gain": "Curvature gain",
    "rescaling_cost": "Rescaling cost",
}
TERM_COLORS = {
    "first_order_gain": "#E68632",
    "curvature_gain": "#7B5EA7",
    "rescaling_cost": "#4C9F70",
}
TERM_MARKERS = {
    "first_order_gain": "o",
    "curvature_gain": "s",
    "rescaling_cost": "^",
}


def f(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def is_finite(value: object) -> bool:
    return math.isfinite(f(value))


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


def style_axis(
    ax: plt.Axes,
    *,
    xlabel: str = "Diagnostic step",
    ylabel: str,
    threshold: float | None = None,
) -> None:
    ax.set_xlabel(xlabel, fontsize=34)
    ax.set_ylabel(ylabel, fontsize=34)
    ax.tick_params(axis="both", labelsize=28)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, alpha=0.28)
    if threshold is not None:
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=2.2)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


def read_val_jsonl(path: Path, min_step: int) -> dict[int, float]:
    vals: dict[int, float] = {}
    if not path.exists():
        return vals
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step = int(record["step"])
        val = f(record.get("val_loss"))
        if step >= min_step and math.isfinite(val):
            vals[step] = val
    return vals


def interpolate_val_loss(vals: dict[int, float], target_steps: list[int]) -> list[float]:
    available_steps = sorted(vals)
    losses: list[float] = []
    for step in target_steps:
        if step in vals:
            losses.append(vals[step])
            continue
        lower = max((s for s in available_steps if s < step), default=None)
        upper = min((s for s in available_steps if s > step), default=None)
        if lower is None or upper is None:
            raise RuntimeError(f"cannot interpolate validation loss at step {step}")
        weight = (step - lower) / (upper - lower)
        losses.append(vals[lower] + weight * (vals[upper] - vals[lower]))
    return losses


def plot_optlr_muon_typed_tail_loss(
    muon_val_jsonl: Path,
    tmuonl_val_jsonl: Path,
    out_dir: Path,
) -> None:
    series = [
        (
            "Muon",
            muon_val_jsonl,
            COLORS["opt_muon"],
        ),
        (
            "t-MuonL",
            tmuonl_val_jsonl,
            COLORS["opt_typed"],
        ),
    ]

    configure_style()
    fig, ax = plt.subplots(figsize=(14.1667, 9.0667), dpi=150)
    target_steps = [3100, 3125, 3150, 3175, 3200, 3225, 3250, 3275, 3300]
    all_losses: list[float] = []
    for label, path, color in series:
        vals = read_val_jsonl(path, min_step=3100)
        if 3300 not in vals:
            raise RuntimeError(f"missing step 3300 validation loss in {path}")
        steps = target_steps
        losses = interpolate_val_loss(vals, steps)
        all_losses.extend(losses)
        ax.plot(
            steps,
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
    pad = 0.12 * (ymax - ymin)
    ax.set_xlim(min(steps) - 8, max(steps) + 8)
    ax.set_ylim(ymin - pad, ymax + pad)
    style_axis(ax, xlabel="Optimizer step", ylabel="Validation loss")
    legend(
        ax,
        fontsize=28,
        loc="upper right",
    )
    save(fig, out_dir / "type_lr_seed42_loss.png")


def connected_scatter(
    ax: plt.Axes,
    steps: list[int],
    values: list[float],
    *,
    color: str,
    label: str,
) -> None:
    ax.plot(
        steps,
        values,
        color=color,
        linewidth=4.2,
        alpha=0.94,
        solid_capstyle="round",
        zorder=2,
    )
    ax.plot(
        steps,
        values,
        linestyle="None",
        marker="o",
        markersize=12.8,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.0,
        color=color,
        label=label,
        zorder=3,
    )


def aggregate_typed_criteria_components(
    criteria_rows: list[dict[str, object]],
) -> tuple[list[int], dict[str, list[float]], list[float]]:
    by_step: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in criteria_rows:
        if row["candidate"] == "typed_by_type":
            by_step[int(row["step"])].append(row)

    steps = sorted(by_step)
    components = {
        "first_order_gain": [],
        "curvature_gain": [],
        "rescaling_cost": [],
    }
    margins: list[float] = []
    for step in steps:
        rows = by_step[step]
        first_order_gain = sum(2.0 * f(row["c1"]) for row in rows)
        curvature_gain = sum(-f(row["c2"]) for row in rows)
        rescaling_cost = sum(f(row["rhs_variance"]) for row in rows)
        components["first_order_gain"].append(first_order_gain)
        components["curvature_gain"].append(curvature_gain)
        components["rescaling_cost"].append(rescaling_cost)
        margins.append(first_order_gain + curvature_gain - rescaling_cost)
    return steps, components, margins


def criteria_rows_from_diag(diag_json: Path) -> list[dict[str, object]]:
    records = read_records(diag_json)
    _, _, _, criteria_rows = build_rows(records, score_lr_policy="unit")
    return criteria_rows


def plot_aggregate_criteria_margin(criteria_rows: list[dict[str, object]], out_dir: Path) -> None:
    steps, components, margins = aggregate_typed_criteria_components(criteria_rows)

    configure_style()
    fig, ax = plt.subplots(figsize=(14.1667, 9.0667), dpi=150)
    all_values = list(margins)
    ax.plot(
        steps,
        margins,
        color=COLORS["typed"],
        linewidth=4.2,
        alpha=0.94,
        solid_capstyle="round",
        label="sum of t-MuonL within-type margins",
        marker="o",
        markersize=12.8,
        markerfacecolor=COLORS["typed"],
        markeredgecolor=COLORS["typed"],
        markeredgewidth=0.0,
        zorder=5,
    )

    term_linestyles = {
        "first_order_gain": "--",
        "curvature_gain": "--",
        "rescaling_cost": ":",
    }
    for term in ("first_order_gain", "curvature_gain", "rescaling_cost"):
        values = components[term]
        all_values.extend(values)
        color = TERM_COLORS[term]
        ax.plot(
            steps,
            values,
            color=color,
            marker=TERM_MARKERS[term],
            markersize=8.8,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.0,
            linestyle=term_linestyles[term],
            linewidth=2.3,
            alpha=0.82,
            label=TERM_LABELS[term],
            zorder=3,
        )

    style_axis(ax, ylabel="Within-type criteria", threshold=0.0)
    span = max(all_values) - min(all_values)
    lower_pad = 0.10 * span if span > 0.0 else 0.01
    upper_pad = 0.42 * span if span > 0.0 else 0.01
    ax.set_ylim(min(all_values) - lower_pad, max(all_values) + upper_pad)
    leg = ax.legend(
        fontsize=23,
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
    save(fig, out_dir / "type_lr_within_type_sum.png")

def proposition2_criterion(record: dict[str, object]) -> dict[str, object]:
    blocks = record["blocks"]
    matrix = np.asarray(record["cross_curvature"]["matrix"], dtype=float)
    types = [typ for typ in TYPE_ORDER if any(block["type"] == typ for block in blocks)]
    by_type = {
        typ: [idx for idx, block in enumerate(blocks) if block["type"] == typ]
        for typ in types
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

    b_eig_min = float(np.linalg.eigvalsh(bmat)[0])
    typed_eig_min = float(np.linalg.eigvalsh(typed_curv)[0])
    if b_eig_min <= 0.0 or typed_eig_min <= 0.0:
        return {
            "step": int(record["step"]),
            "criterion": float("nan"),
            "score_ratio": float("nan"),
            "b_eig_min": b_eig_min,
            "typed_eig_min": typed_eig_min,
            "rho_min": float("nan"),
            "rho_max": float("nan"),
            "positive_definite": False,
        }

    d_inv = np.diag(1.0 / np.diag(dmat))
    perturb = d_inv @ gamma @ d_inv
    perturb = 0.5 * (perturb + perturb.T)

    chol = np.linalg.cholesky(bmat)
    chol_inv = np.linalg.inv(chol)
    eig_system = chol_inv @ perturb @ chol_inv.T
    eig_system = 0.5 * (eig_system + eig_system.T)
    rho, yvec = np.linalg.eigh(eig_system)
    vmat = np.linalg.solve(chol.T, yvec)
    coeff = vmat.T @ avec
    terms = rho / (1.0 - rho) * coeff * coeff
    criterion = float(np.sum(terms))

    base_score = 0.5 * float(avec @ np.linalg.inv(bmat) @ avec)
    typed_score = 0.5 * float(avec @ np.linalg.inv(bmat - perturb) @ avec)
    return {
        "step": int(record["step"]),
        "criterion": criterion,
        "score_ratio": typed_score / base_score,
        "b_eig_min": b_eig_min,
        "typed_eig_min": typed_eig_min,
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "rho": [float(value) for value in rho],
        "v_dot_a": [float(value) for value in coeff],
        "terms": [float(value) for value in terms],
        "positive_definite": True,
    }


def plot_prop2_criterion_with_contributions(diag_json: Path, out_dir: Path) -> None:
    diagnostics = [proposition2_criterion(record) for record in read_records(diag_json)]
    diagnostics = [row for row in diagnostics if row["positive_definite"]]
    steps = [int(row["step"]) for row in diagnostics]
    overall = [f(row["criterion"]) for row in diagnostics]
    eig_count = len(diagnostics[0]["terms"])
    component_colors = [
        "#111111",
        "#E68632",
        "#7B5EA7",
        "#4C9F70",
        "#C44E52",
        "#8C6D31",
    ]
    line_styles = ["--", ":", "-.", (0, (5, 2, 1, 2)), (0, (3, 2)), (0, (1, 1))]

    configure_style()
    fig, ax = plt.subplots(figsize=(14.1667, 9.0667), dpi=150)
    connected_scatter(
        ax,
        steps,
        overall,
        color=COLORS["typed"],
        label="t-MuonL overall",
    )

    all_values = list(overall)
    for k in range(eig_count):
        color = component_colors[k % len(component_colors)]
        term_values = [f(row["terms"][k]) for row in diagnostics]
        all_values.extend(term_values)
        if k == 0:
            label = r"$k=1$ contrib. (smallest $\rho_k$)"
        elif k == eig_count - 1:
            label = rf"$k={k + 1}$ contrib. (largest $\rho_k$)"
        else:
            label = rf"$k={k + 1}$ contrib."
        ax.plot(
            steps,
            term_values,
            color=color,
            linestyle=line_styles[k % len(line_styles)],
            linewidth=2.0,
            alpha=0.55,
            marker="o",
            markersize=5.8,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.0,
            label=label,
            zorder=2,
        )

    style_axis(ax, ylabel="t-MuonL criterion diagnostic", threshold=0.0)
    span = max(all_values) - min(all_values)
    ax.set_ylim(
        min(all_values) - 0.14 * span,
        max(all_values) + 0.55 * span,
    )
    leg = ax.legend(
        fontsize=24,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        handletextpad=0.45,
        columnspacing=1.15,
        labelspacing=0.25,
    )
    leg.set_zorder(10)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.15, top=0.98)
    save(fig, out_dir / "type_lr_prop2.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "experiment_results"
        / "type_lr"
        / "seed42_diag",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--diag-json",
        type=Path,
        default=None,
        help="Muon diagnostic JSON. Overrides the legacy path derived from --run-root.",
    )
    parser.add_argument(
        "--muon-val-jsonl",
        type=Path,
        default=None,
        help="Muon validation-loss JSONL. Overrides the legacy --run-root path.",
    )
    parser.add_argument(
        "--tmuonl-val-jsonl",
        type=Path,
        default=None,
        help="t-MuonL validation-loss JSONL. Overrides the legacy --run-root path.",
    )
    parser.add_argument(
        "--prop2-criterion-only",
        action="store_true",
        help="Only draw the paper Proposition 2 criterion figure.",
    )
    parser.add_argument(
        "--type-lr-loss-only",
        action="store_true",
        help="Only draw the opt-lr Muon vs t-MuonL validation-loss comparison.",
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else ICLR_DIR / "figures"
    diag_json = (args.diag_json or (
        run_root / "muon_diagnostics/raw/seed42_plain_muon_optlr_local_quadratic_diag.json"
    )).resolve()
    muon_val_jsonl = (args.muon_val_jsonl or (
        run_root / "muon_diagnostics/raw/seed42_plain_muon_optlr_local_quadratic_val_loss.jsonl"
    )).resolve()
    tmuonl_val_jsonl = (args.tmuonl_val_jsonl or (
        run_root / "tmuonl_loss/raw/seed42_typed_muonl_optlr_loss_only_val_loss.jsonl"
    )).resolve()

    if args.prop2_criterion_only:
        plot_prop2_criterion_with_contributions(diag_json, out_dir)
        return
    if args.type_lr_loss_only:
        plot_optlr_muon_typed_tail_loss(muon_val_jsonl, tmuonl_val_jsonl, out_dir)
        return

    criteria_rows = criteria_rows_from_diag(diag_json)
    plot_optlr_muon_typed_tail_loss(muon_val_jsonl, tmuonl_val_jsonl, out_dir)
    plot_aggregate_criteria_margin(criteria_rows, out_dir)
    plot_prop2_criterion_with_contributions(diag_json, out_dir)


if __name__ == "__main__":
    main()
