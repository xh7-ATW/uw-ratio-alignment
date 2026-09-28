#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICLR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ICLR_DIR}"
python3 "${ICLR_DIR}/scripts/run_paper_experiment.py" \
  --config "${ICLR_DIR}/configs/seed42_type_lr_muon_diag.json" \
  --execute

python3 "${ICLR_DIR}/scripts/run_paper_experiment.py" \
  --config "${ICLR_DIR}/configs/seed42_type_lr_tmuonl.json" \
  --execute

python3 "${ICLR_DIR}/scripts/plot_figures/type_lr_seed42.py" \
  --diag-json "${ICLR_DIR}/outputs/seed42_type_lr_muon_diag/raw/seed42_type_lr_muon_diag_diag.json" \
  --muon-val-jsonl "${ICLR_DIR}/outputs/seed42_type_lr_muon_diag/raw/seed42_type_lr_muon_diag_val_loss.jsonl" \
  --tmuonl-val-jsonl "${ICLR_DIR}/outputs/seed42_type_lr_tmuonl/raw/seed42_type_lr_tmuonl_val_loss.jsonl" \
  --out-dir "${ICLR_DIR}/figures"

python3 "${ICLR_DIR}/scripts/plot_figures/type_lr_assumptions.py" \
  --shared-diag "${ICLR_DIR}/experiment_results/shared_lr/seed42_diagnostics/raw/seed42_plain_muon_local_quadratic_diag.json" \
  --opt-diag "${ICLR_DIR}/outputs/seed42_type_lr_muon_diag/raw/seed42_type_lr_muon_diag_diag.json" \
  --out-dir "${ICLR_DIR}/figures"
