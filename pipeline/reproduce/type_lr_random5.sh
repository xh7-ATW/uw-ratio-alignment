#!/usr/bin/env bash
set -euo pipefail
trap '' HUP
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICLR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ICLR_DIR}"
for config in random5_type_lr_plain_muon random5_type_lr_tmuonl; do
  python3 pipeline/run_paper_experiment.py --config "configs/${config}.json" --execute
done
python3 pipeline/plot_figures/type_lr_loss.py \
  --plain-root "${ICLR_DIR}/outputs/random5_type_lr/plain_muon" \
  --typed-root "${ICLR_DIR}/outputs/random5_type_lr/typed_muonl" \
  --output "${ICLR_DIR}/figures/type_lr_random5_loss.png"
