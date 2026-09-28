#!/usr/bin/env bash
set -euo pipefail
trap '' HUP
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICLR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ICLR_DIR}"
for config in random5_shared_lr_plain_muon random5_shared_lr_tmuonl random5_shared_lr_muonl; do
  python3 scripts/run_paper_experiment.py --config "configs/${config}.json" --execute
done
python3 scripts/plot_figures/shared_lr_random5_loss.py \
  --run-root "${ICLR_DIR}/outputs/random5_shared_lr" \
  --output "${ICLR_DIR}/figures/shared_lr_random5_loss.png"
