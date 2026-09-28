#!/usr/bin/env python3
"""Postprocess type-specific learning-rate diagnostics into paper tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
ICLR_DIR = SCRIPT.parents[2]
sys.path.insert(0, str(ICLR_DIR))

from src.diagnostics import build_rows, read_records, summarize, write_csv
from src.utils import json_clean


BLOCK_FIELDNAMES = [
    "diag_index", "step", "matrix_index", "block_name", "type", "block_idx",
    "w_norm", "r_i", "r_i_measured", "r_i_ideal", "r_i_typed_ideal",
    "u_norm_ideal", "vi_raw_norm", "vi_raw_norm_over_ideal", "rbar_C", "rbar_all",
    "x_plain", "x_typed", "x_all", "base_lr", "weight_decay_raw",
    "x_typed_over_r", "x_all_over_r",
    "g_dot_v", "v_h_v", "cross_self_cii",
    "cross_curvature_typed_sum", "cross_curvature_all_sum",
    "ltilde_typed_by_type", "ltilde_typed_aggregate", "ltilde_all_layer",
    "t_i", "t_i_valid", "diag_g_frob_norm", "train_grad_frob_norm",
    "hvp_frob_norm",
]

RATIO_FIELDNAMES = [
    "diag_index", "step", "type", "block_count", "r_mean", "r_std",
    "r_min", "r_max", "r_max_over_min", "r_measured_mean",
    "r_measured_std", "w_norm_mean", "x_typed_mean", "x_all_mean",
    "typed_x_over_r_mean", "typed_x_over_r_std",
    "typed_x_over_r_min", "typed_x_over_r_max",
    "typed_x_over_r_max_over_min", "all_x_over_r_mean",
    "all_x_over_r_std", "all_x_over_r_min", "all_x_over_r_max",
    "all_x_over_r_max_over_min",
]

SCORE_FIELDNAMES = [
    "diag_index", "step", "scope", "candidate", "score_lr_policy",
    "valid_block_count", "first_order_sum", "quadratic_curvature_sum",
    "score", "eta_opt",
]

CRITERIA_FIELDNAMES = [
    "diag_index", "step", "scope", "candidate", "x_source",
    "valid_block_count", "skipped_cross_terms", "xbar", "ybar", "lbar",
    "c1", "c2", "rhs_variance", "lhs_sufficient", "lhs_exact",
    "ratio_rhs_over_lhs_sufficient", "ratio_rhs_over_lhs_exact",
    "condition_sufficient", "condition_exact",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("diag_json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--score-lr-policy",
        choices=("unit", "base_lr", "explicit_split"),
        default="unit",
        help=(
            "unit preserves the original LR-free score. base_lr multiplies x_i "
            "by the block's diagnostic base_lr. explicit_split uses the four "
            "LR values below."
        ),
    )
    parser.add_argument("--qkv-lr", type=float, default=0.025)
    parser.add_argument("--attn-proj-lr", type=float, default=0.030)
    parser.add_argument("--mlp-fc-lr", type=float, default=0.035)
    parser.add_argument("--mlp-proj-lr", type=float, default=0.035)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records(args.diag_json)
    prefix = args.prefix or args.diag_json.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)

    explicit_lr_by_group = {
        "qkv": args.qkv_lr,
        "attn_proj": args.attn_proj_lr,
        "mlp_fc": args.mlp_fc_lr,
        "mlp_proj": args.mlp_proj_lr,
    }
    block_rows, ratio_rows, score_rows, criteria_rows = build_rows(
        records,
        score_lr_policy=args.score_lr_policy,
        explicit_lr_by_group=explicit_lr_by_group,
    )

    output_paths = {
        "block_theory_terms_csv": args.out_dir / f"{prefix}_block_theory_terms.csv",
        "uw_ratio_by_type_csv": args.out_dir / f"{prefix}_uw_ratio_by_type.csv",
        "scores_csv": args.out_dir / f"{prefix}_scores.csv",
        "sufficient_criteria_csv": args.out_dir / f"{prefix}_sufficient_criteria.csv",
        "summary_json": args.out_dir / f"{prefix}_summary.json",
    }

    write_csv(output_paths["block_theory_terms_csv"], block_rows, BLOCK_FIELDNAMES)
    write_csv(output_paths["uw_ratio_by_type_csv"], ratio_rows, RATIO_FIELDNAMES)
    write_csv(output_paths["scores_csv"], score_rows, SCORE_FIELDNAMES)
    write_csv(output_paths["sufficient_criteria_csv"], criteria_rows, CRITERIA_FIELDNAMES)

    summary = summarize(
        records,
        score_rows,
        criteria_rows,
        output_paths,
        args.diag_json,
        args.score_lr_policy,
    )
    output_paths["summary_json"].write_text(
        json.dumps(json_clean(summary), indent=2, sort_keys=True) + "\n"
    )

    print(f"wrote {output_paths['block_theory_terms_csv']}")
    print(f"wrote {output_paths['uw_ratio_by_type_csv']}")
    print(f"wrote {output_paths['scores_csv']}")
    print(f"wrote {output_paths['sufficient_criteria_csv']}")
    print(f"wrote {output_paths['summary_json']}")


if __name__ == "__main__":
    main()
