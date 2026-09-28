#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICLR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ICLR_DIR}"
python3 "${ICLR_DIR}/pipeline/run_paper_experiment.py" \
  --config "${ICLR_DIR}/configs/seed42_uw_floor_u_over_w.json" \
  --execute

python3 "${ICLR_DIR}/pipeline/plot_figures/uw_floor_u_over_w.py" \
  --input-json "${ICLR_DIR}/runs/uw_floor/seed42_c035_u_over_w/normuon_c035_u_over_w/raw/seed42_normuon_benchmark9_uwfloor_c0p35_u_over_w.json" \
  --output "${ICLR_DIR}/figures/uw_floor_u_over_w.png" \
  --ratio-field blocks \
  --kind attnproj \
  --xmax 3250 \
  --ymin 0.1 \
  --ymax 1.0
