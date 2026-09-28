#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICLR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ICLR_DIR}"
python3 "${ICLR_DIR}/pipeline/run_paper_experiment.py" \
  --config "${ICLR_DIR}/configs/random5_uw_floor_c035.json" \
  --execute

python3 "${ICLR_DIR}/pipeline/plot_figures/uw_floor_loss.py" \
  --shared-run-root "${ICLR_DIR}/runs/shared_lr/random5_main" \
  --uw-floor-run-root "${ICLR_DIR}/outputs/random5_uw_floor_c035" \
  --output "${ICLR_DIR}/figures/uw_floor_loss.png"
