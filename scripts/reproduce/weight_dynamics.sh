#!/usr/bin/env bash
set -euo pipefail
trap '' HUP
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICLR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ICLR_DIR}"
for config in seed42_weight_dynamics_muonl seed42_weight_dynamics_decoupled_orscale seed42_weight_dynamics_orscale; do
  python3 scripts/run_paper_experiment.py --config "configs/${config}.json" --execute
done
python3 scripts/plot_figures/weight_dynamics.py \
  --muonl-run "${ICLR_DIR}/outputs/weight_dynamics/muonl/raw" \
  --decoupled-orscale-run "${ICLR_DIR}/outputs/weight_dynamics/decoupled_orscale/raw" \
  --orscale-run "${ICLR_DIR}/outputs/weight_dynamics/orscale/raw" \
  --output "${ICLR_DIR}/figures/weight_dynamics.png"
