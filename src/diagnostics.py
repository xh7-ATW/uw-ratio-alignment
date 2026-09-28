"""Theory-diagnostic calculations used in the u-w ratio paper.

The functions in this module are a behavior-preserving extraction from the
original analysis script. Names follow the paper where concise notation helps:
``r_i`` is the Muon-specific ideal ratio ``sqrt(m_i)/||W_i||_F``, ``w_i`` is
the weight norm, and ``C_ij`` is the cross-curvature term.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .utils import (
    TYPE_TO_LR_GROUP,
    as_float,
    basic_stats,
    diagnostic_type_sort_key,
    is_finite,
    json_clean,
    safe_div,
    weighted_covariance,
    weighted_mean,
    weighted_variance,
)

CANDIDATE_X_KEY = {
    "plain": "x_plain",
    "typed": "x_typed",
    "all_layer": "x_all",
}
CANDIDATE_R_KEY = {
    "plain": "r_i_ideal",
    "typed": "r_i_ideal",
    "all_layer": "r_i_ideal",
}


def harmonic_mean(values: list[float]) -> float:
    finite_positive = [value for value in values if is_finite(value) and value > 0.0]
    if not finite_positive:
        return float("nan")
    return len(finite_positive) / sum(1.0 / value for value in finite_positive)


def ideal_muon_norm(block: dict[str, Any]) -> float:
    """Paper convention for the shape-scaled Muon update norm."""

    m_i = as_float(block.get("m_i"))
    return math.sqrt(m_i) if is_finite(m_i) and m_i > 0.0 else float("nan")


def apply_alignment_policy(record: dict[str, Any]) -> dict[str, Any]:
    """Attach paper-side values using the ideal Muon norm convention."""

    blocks = record.get("blocks", [])
    if not blocks:
        return record

    ideal_rs = [
        safe_div(ideal_muon_norm(block), block.get("w_norm"))
        for block in blocks
    ]
    x_all = harmonic_mean(ideal_rs)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        by_type[str(block.get("type", "unknown"))].append(block)

    typed_x_by_type: dict[str, float] = {}
    typed_sqrt_m_by_type: dict[str, float] = {}
    mean_w_by_type: dict[str, float] = {}
    for type_name, type_blocks in by_type.items():
        first_m = as_float(type_blocks[0].get("m_i"))
        sqrt_m = math.sqrt(first_m) if is_finite(first_m) and first_m > 0.0 else float("nan")
        mean_w = sum(as_float(block.get("w_norm")) for block in type_blocks) / len(type_blocks)
        typed_sqrt_m_by_type[type_name] = sqrt_m
        mean_w_by_type[type_name] = mean_w
        typed_x_by_type[type_name] = safe_div(sqrt_m, mean_w)

    for block in blocks:
        type_name = str(block.get("type", "unknown"))
        w_norm = as_float(block.get("w_norm"))
        r_measured = as_float(block.get("r_i_measured", block.get("r_i")))
        u_norm_ideal = ideal_muon_norm(block)
        r_ideal = safe_div(u_norm_ideal, w_norm)
        sqrt_m = typed_sqrt_m_by_type.get(type_name, float("nan"))
        x_typed = typed_x_by_type.get(type_name, float("nan"))
        block["r_i_measured"] = r_measured
        block["r_i_ideal"] = r_ideal
        block["r_i"] = r_ideal
        block["r_i_typed_ideal"] = r_ideal
        block["u_norm_ideal"] = u_norm_ideal
        block["vi_raw_norm_over_ideal"] = safe_div(block.get("vi_raw_norm"), u_norm_ideal)
        block["rbar_C"] = x_typed
        block["rbar_all"] = x_all
        block["x_plain"] = r_ideal
        block["x_typed"] = x_typed
        block["x_all"] = x_all
        block["typed_weight_mean"] = mean_w_by_type.get(type_name, float("nan"))

    groups = record.get("groups")
    if isinstance(groups, dict):
        for type_name, type_blocks in by_type.items():
            group = groups.get(type_name)
            if not isinstance(group, dict):
                continue
            ideal_type_rs = [as_float(block["r_i_ideal"]) for block in type_blocks]
            measured_type_rs = [as_float(block["r_i_measured"]) for block in type_blocks]
            group["rbar_C"] = typed_x_by_type[type_name]
            group["mean_r_i"] = sum(ideal_type_rs) / len(ideal_type_rs)
            group["mean_r_i_measured"] = sum(measured_type_rs) / len(measured_type_rs)
            group["typed_ideal_sqrt_m"] = typed_sqrt_m_by_type[type_name]
            group["typed_weight_mean"] = mean_w_by_type[type_name]

    record["rbar_all"] = x_all
    record["x_definitions"] = {
        "plain": "x_i = ideal r_i = sqrt(m_i) / ||W_i||_F",
        "all": "x_i = harmonic_mean_all_blocks(ideal r_i)",
        "typed": "x_i = sqrt(m_G) / mean_{j in G}(||W_j||_F)",
    }
    record["muon_norm_convention"] = (
        "criterion post-processing uses ideal Muon norm sqrt(m_i); "
        "vi_raw_norm is retained only as a finite-NS sanity check"
    )
    return record


def read_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of diagnostic records")
    return [apply_alignment_policy(record) for record in data]


def cross_matrix(record: dict[str, Any]) -> list[list[float]]:
    cross = record.get("cross_curvature")
    if not isinstance(cross, dict):
        raise ValueError("missing cross_curvature section")
    matrix = cross.get("matrix")
    if not matrix:
        raise ValueError(
            "missing cross_curvature.matrix; rerun Muon with "
            "MYOPT_LOCAL_QUADRATIC_CROSS_CURVATURE=1 and "
            "MYOPT_LOCAL_QUADRATIC_SAVE_CROSS_CURVATURE_MATRIX=1"
        )
    n = len(record.get("blocks", []))
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError(
            f"cross_curvature.matrix shape mismatch: expected {n}x{n}, "
            f"got {len(matrix)} rows"
        )
    return matrix


def candidate_x(block: dict[str, Any], candidate: str) -> float:
    return as_float(block.get(CANDIDATE_X_KEY[candidate]))


def candidate_r(block: dict[str, Any], candidate: str) -> float:
    return as_float(block.get(CANDIDATE_R_KEY[candidate]))


def score_x(
    block: dict[str, Any],
    candidate: str,
    score_lr_policy: str,
    explicit_lr_by_group: dict[str, float],
) -> float:
    x_value = candidate_x(block, candidate)
    if not is_finite(x_value):
        return float("nan")
    if score_lr_policy == "unit":
        return x_value
    if score_lr_policy == "base_lr":
        lr = as_float(block.get("base_lr"))
        return x_value * lr if is_finite(lr) else float("nan")
    if score_lr_policy == "explicit_split":
        group = TYPE_TO_LR_GROUP.get(block.get("type", ""))
        lr = explicit_lr_by_group.get(group)
        return x_value * lr if is_finite(lr) else float("nan")
    raise ValueError(f"unknown score_lr_policy {score_lr_policy!r}")


def grouped_indices(blocks: list[dict[str, Any]]) -> dict[str, list[int]]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for index, block in enumerate(blocks):
        by_type[block.get("type", "unknown")].append(index)
    return dict(sorted(by_type.items(), key=lambda item: diagnostic_type_sort_key(item[0])))


def cross_adjusted_ltilde(
    record: dict[str, Any],
    candidate: str,
    indices: list[int],
) -> tuple[dict[int, float], int]:
    blocks = record["blocks"]
    matrix = cross_matrix(record)
    output: dict[int, float] = {}
    skipped_terms = 0
    for i in indices:
        block_i = blocks[i]
        r_i = candidate_r(block_i, candidate)
        x_i = candidate_x(block_i, candidate)
        w_i = as_float(block_i.get("w_norm"))
        ltilde = as_float(matrix[i][i])
        for j in indices:
            if i == j:
                continue
            block_j = blocks[j]
            r_j = candidate_r(block_j, candidate)
            x_j = candidate_x(block_j, candidate)
            w_j = as_float(block_j.get("w_norm"))
            denominator = w_i * (r_i * x_j + r_j * x_i)
            if (
                not all(is_finite(value) for value in (r_i, x_i, w_i, r_j, x_j, w_j, denominator))
                or denominator == 0.0
            ):
                skipped_terms += 1
                continue
            ltilde += (2.0 * r_j * x_j * w_j / denominator) * as_float(matrix[i][j])
        output[i] = ltilde
    return output, skipped_terms


def score_terms(
    record: dict[str, Any],
    candidate: str,
    indices: list[int],
    score_lr_policy: str,
    explicit_lr_by_group: dict[str, float],
) -> dict[str, float | int]:
    blocks = record["blocks"]
    matrix = cross_matrix(record)
    valid_indices = []
    first_order = 0.0
    for i in indices:
        block = blocks[i]
        x_value = score_x(block, candidate, score_lr_policy, explicit_lr_by_group)
        weight_norm = as_float(block.get("w_norm"))
        descent_term = as_float(block.get("g_dot_v"))
        if all(is_finite(value) for value in (x_value, weight_norm, descent_term)):
            valid_indices.append(i)
            first_order += x_value * weight_norm * descent_term

    curvature = 0.0
    for i in valid_indices:
        x_i = score_x(blocks[i], candidate, score_lr_policy, explicit_lr_by_group)
        w_i = as_float(blocks[i].get("w_norm"))
        for j in valid_indices:
            x_j = score_x(blocks[j], candidate, score_lr_policy, explicit_lr_by_group)
            w_j = as_float(blocks[j].get("w_norm"))
            c_ij = as_float(matrix[i][j])
            if all(is_finite(value) for value in (x_i, w_i, x_j, w_j, c_ij)):
                curvature += x_i * x_j * w_i * w_j * c_ij

    return {
        "valid_block_count": len(valid_indices),
        "first_order_sum": first_order,
        "quadratic_curvature_sum": curvature,
        "score": safe_div(first_order * first_order, 2.0 * curvature),
        "eta_opt": safe_div(first_order, curvature),
    }


def criterion_terms(
    record: dict[str, Any],
    candidate: str,
    indices: list[int],
) -> dict[str, float | int | bool | None]:
    blocks = record["blocks"]
    ltilde, skipped_terms = cross_adjusted_ltilde(record, candidate, indices)
    weights: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    ls: list[float] = []
    valid = 0
    for i in indices:
        block = blocks[i]
        r_i = candidate_r(block, candidate)
        w_i = as_float(block.get("w_norm"))
        x_i = candidate_x(block, candidate)
        y_numerator = as_float(block.get("g_dot_v"))
        l_i = as_float(ltilde.get(i))
        if (
            not all(is_finite(value) for value in (r_i, w_i, x_i, y_numerator, l_i))
            or r_i == 0.0
            or w_i == 0.0
        ):
            continue
        weights.append((r_i * w_i) ** 2)
        xs.append(x_i / r_i)
        ys.append(y_numerator / (r_i * w_i))
        ls.append(l_i)
        valid += 1

    xbar = weighted_mean(xs, weights)
    ybar = weighted_mean(ys, weights)
    lbar = weighted_mean(ls, weights)
    if (
        not all(is_finite(value) for value in (xbar, ybar, lbar))
        or xbar == 0.0
        or ybar == 0.0
        or lbar == 0.0
    ):
        return {
            "valid_block_count": valid,
            "skipped_cross_terms": skipped_terms,
            "xbar": xbar,
            "ybar": ybar,
            "lbar": lbar,
            "c1": float("nan"),
            "c2": float("nan"),
            "rhs_variance": float("nan"),
            "lhs_sufficient": float("nan"),
            "lhs_exact": float("nan"),
            "ratio_rhs_over_lhs_sufficient": float("nan"),
            "ratio_rhs_over_lhs_exact": float("nan"),
            "condition_sufficient": None,
            "condition_exact": None,
        }

    x_norm = [x_value / xbar for x_value in xs]
    y_norm = [y_value / ybar for y_value in ys]
    x2_norm = [(x_value * x_value) / (xbar * xbar) for x_value in xs]
    l_norm = [l_value / lbar for l_value in ls]

    c1 = weighted_covariance(x_norm, y_norm, weights)
    c2 = weighted_covariance(x2_norm, l_norm, weights)
    variance = weighted_variance(x_norm, weights)
    lhs_sufficient = 2.0 * c1 - c2
    lhs_exact = 2.0 * c1 + c1 * c1 - c2
    condition_sufficient = bool(is_finite(lhs_sufficient) and is_finite(variance) and lhs_sufficient > variance)
    condition_exact = bool(is_finite(lhs_exact) and is_finite(variance) and lhs_exact > variance)
    return {
        "valid_block_count": valid,
        "skipped_cross_terms": skipped_terms,
        "xbar": xbar,
        "ybar": ybar,
        "lbar": lbar,
        "c1": c1,
        "c2": c2,
        "rhs_variance": variance,
        "lhs_sufficient": lhs_sufficient,
        "lhs_exact": lhs_exact,
        "ratio_rhs_over_lhs_sufficient": safe_div(variance, lhs_sufficient),
        "ratio_rhs_over_lhs_exact": safe_div(variance, lhs_exact),
        "condition_sufficient": condition_sufficient,
        "condition_exact": condition_exact,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_rows(
    records: list[dict[str, Any]],
    score_lr_policy: str = "unit",
    explicit_lr_by_group: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    explicit_lr_by_group = explicit_lr_by_group or {}
    records = [apply_alignment_policy(record) for record in records]
    block_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    criteria_rows: list[dict[str, Any]] = []

    for record in records:
        matrix = cross_matrix(record)
        blocks = record["blocks"]
        step = int(record["step"])
        diag_index = int(record.get("diag_index", -1))
        all_indices = list(range(len(blocks)))
        by_type = grouped_indices(blocks)

        ltyped_global, _ = cross_adjusted_ltilde(record, "typed", all_indices)
        lall_global, _ = cross_adjusted_ltilde(record, "all_layer", all_indices)
        ltyped_by_type: dict[int, float] = {}
        for _, indices in by_type.items():
            partial, _ = cross_adjusted_ltilde(record, "typed", indices)
            ltyped_by_type.update(partial)

        for index, block in enumerate(blocks):
            r_i = candidate_r(block, "plain")
            r_i_typed_ideal = candidate_r(block, "typed")
            ltilde_typed_by_type = as_float(ltyped_by_type.get(index))
            ltilde_typed_aggregate = as_float(ltyped_global.get(index))
            ltilde_all_layer = as_float(lall_global.get(index))
            self_curv = as_float(matrix[index][index])
            block_rows.append({
                "diag_index": diag_index,
                "step": step,
                "matrix_index": index,
                "block_name": block.get("block_name"),
                "type": block.get("type"),
                "block_idx": block.get("block_idx"),
                "w_norm": as_float(block.get("w_norm")),
                "r_i": r_i,
                "r_i_measured": as_float(block.get("r_i_measured")),
                "r_i_ideal": r_i,
                "r_i_typed_ideal": r_i_typed_ideal,
                "u_norm_ideal": as_float(block.get("u_norm_ideal")),
                "vi_raw_norm": as_float(block.get("vi_raw_norm")),
                "vi_raw_norm_over_ideal": as_float(block.get("vi_raw_norm_over_ideal")),
                "rbar_C": as_float(block.get("rbar_C")),
                "rbar_all": as_float(block.get("rbar_all")),
                "x_plain": as_float(block.get("x_plain")),
                "x_typed": as_float(block.get("x_typed")),
                "x_all": as_float(block.get("x_all")),
                "base_lr": as_float(block.get("base_lr")),
                "weight_decay_raw": as_float(block.get("weight_decay_raw")),
                "x_typed_over_r": safe_div(block.get("x_typed"), r_i_typed_ideal),
                "x_all_over_r": safe_div(block.get("x_all"), r_i),
                "g_dot_v": as_float(block.get("g_dot_v")),
                "v_h_v": as_float(block.get("v_h_v")),
                "cross_self_cii": self_curv,
                "cross_curvature_typed_sum": ltilde_typed_by_type - self_curv,
                "cross_curvature_all_sum": ltilde_all_layer - self_curv,
                "ltilde_typed_by_type": ltilde_typed_by_type,
                "ltilde_typed_aggregate": ltilde_typed_aggregate,
                "ltilde_all_layer": ltilde_all_layer,
                "t_i": as_float(block.get("t_i")),
                "t_i_valid": block.get("t_i_valid"),
                "diag_g_frob_norm": as_float(block.get("diag_g_frob_norm")),
                "train_grad_frob_norm": as_float(block.get("train_grad_frob_norm")),
                "hvp_frob_norm": as_float(block.get("hvp_frob_norm")),
            })

        for type_name, indices in by_type.items():
            typed_over_r = [
                safe_div(blocks[i].get("x_typed"), candidate_r(blocks[i], "typed"))
                for i in indices
            ]
            all_over_r = [safe_div(blocks[i].get("x_all"), blocks[i].get("r_i")) for i in indices]
            r_stats = basic_stats([blocks[i].get("r_i") for i in indices])
            r_measured_stats = basic_stats([blocks[i].get("r_i_measured") for i in indices])
            typed_stats = basic_stats(typed_over_r)
            all_stats = basic_stats(all_over_r)
            ratio_rows.append({
                "diag_index": diag_index,
                "step": step,
                "type": type_name,
                "block_count": len(indices),
                "r_mean": r_stats["mean"],
                "r_std": r_stats["std"],
                "r_min": r_stats["min"],
                "r_max": r_stats["max"],
                "r_max_over_min": r_stats["max_over_min"],
                "r_measured_mean": r_measured_stats["mean"],
                "r_measured_std": r_measured_stats["std"],
                "w_norm_mean": basic_stats([blocks[i].get("w_norm") for i in indices])["mean"],
                "x_typed_mean": basic_stats([blocks[i].get("x_typed") for i in indices])["mean"],
                "x_all_mean": basic_stats([blocks[i].get("x_all") for i in indices])["mean"],
                "typed_x_over_r_mean": typed_stats["mean"],
                "typed_x_over_r_std": typed_stats["std"],
                "typed_x_over_r_min": typed_stats["min"],
                "typed_x_over_r_max": typed_stats["max"],
                "typed_x_over_r_max_over_min": typed_stats["max_over_min"],
                "all_x_over_r_mean": all_stats["mean"],
                "all_x_over_r_std": all_stats["std"],
                "all_x_over_r_min": all_stats["min"],
                "all_x_over_r_max": all_stats["max"],
                "all_x_over_r_max_over_min": all_stats["max_over_min"],
            })

        score_scopes = [("global", all_indices)]
        score_scopes.extend((f"type:{type_name}", indices) for type_name, indices in by_type.items())
        for scope, indices in score_scopes:
            for candidate in ("plain", "typed", "all_layer"):
                terms = score_terms(
                    record,
                    candidate,
                    indices,
                    score_lr_policy,
                    explicit_lr_by_group,
                )
                score_rows.append({
                    "diag_index": diag_index,
                    "step": step,
                    "scope": scope,
                    "candidate": candidate,
                    "score_lr_policy": score_lr_policy,
                    **terms,
                })

        criteria_specs = [
            ("global", "plain_control", "plain", all_indices),
            ("global", "typed_aggregate", "typed", all_indices),
            ("global", "all_layer", "all_layer", all_indices),
        ]
        criteria_specs.extend(
            (f"type:{type_name}", "typed_by_type", "typed", indices)
            for type_name, indices in by_type.items()
        )
        for scope, label, candidate, indices in criteria_specs:
            terms = criterion_terms(record, candidate, indices)
            criteria_rows.append({
                "diag_index": diag_index,
                "step": step,
                "scope": scope,
                "candidate": label,
                "x_source": candidate,
                **terms,
            })

    return block_rows, ratio_rows, score_rows, criteria_rows


def summarize(
    records: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    criteria_rows: list[dict[str, Any]],
    output_paths: dict[str, Path],
    input_path: Path,
    score_lr_policy: str,
) -> dict[str, Any]:
    global_scores: dict[int, dict[str, float]] = defaultdict(dict)
    for row in score_rows:
        if row["scope"] == "global":
            global_scores[row["step"]][row["candidate"]] = row["score"]

    score_counts: Counter[str] = Counter()
    for _, scores in global_scores.items():
        plain = scores.get("plain")
        typed = scores.get("typed")
        all_layer = scores.get("all_layer")
        if is_finite(plain) and is_finite(typed) and typed > plain:
            score_counts["typed_gt_plain"] += 1
        if is_finite(plain) and is_finite(all_layer) and all_layer > plain:
            score_counts["all_layer_gt_plain"] += 1
        finite_scores = {key: value for key, value in scores.items() if is_finite(value)}
        if finite_scores:
            winner = max(finite_scores, key=finite_scores.get)
            score_counts[f"winner:{winner}"] += 1

    criterion_counts: Counter[str] = Counter()
    for row in criteria_rows:
        key = f"{row['scope']}|{row['candidate']}"
        if row.get("condition_sufficient") is True:
            criterion_counts[f"{key}|sufficient_true"] += 1
        if row.get("condition_exact") is True:
            criterion_counts[f"{key}|exact_true"] += 1
        criterion_counts[f"{key}|total"] += 1

    steps = [int(record["step"]) for record in records]
    first = records[0] if records else {}
    return {
        "input_diag": str(input_path),
        "record_count": len(records),
        "steps": steps,
        "seed": first.get("seed"),
        "optimizer": first.get("optimizer"),
        "diag_data_source": first.get("diag_data_source"),
        "hvp": first.get("hvp"),
        "formulas": {
            "score": "(sum_i x_i*w_i*<G_i,V_i>)^2 / (2 * sum_ij x_i*x_j*w_i*w_j*C_ij)",
            "score_lr_policy": score_lr_policy,
            "base_lr_score": "when score_lr_policy=base_lr, x_i is multiplied by each block's diagnostic base_lr, not scheduled lr_current",
            "C_ij": "<V_i,H_ij V_j>",
            "ltilde_i": "C_ii + sum_{j!=i} 2*r_j*x_j*w_j/(w_i*(r_i*x_j+r_j*x_i))*C_ij",
            "muon_norm_convention": "r_i = sqrt(m_i)/||W_i||_F; saved finite-NS vi_raw_norm is not used in criteria",
            "weights": "(r_i*w_i)^2 = m_i under the Muon-specific ideal norm convention",
            "sufficient_lhs": "2*c1 - c2",
            "exact_lhs": "2*c1 + c1^2 - c2",
            "condition": "lhs > Var(x_i/r_i normalized)",
        },
        "score_counts": dict(score_counts),
        "criterion_counts": dict(criterion_counts),
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }
