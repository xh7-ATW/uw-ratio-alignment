# u-w Ratio Alignment Experiments

This directory contains the public-facing code and evidence map for the
MuonL/t-MuonL u-w ratio alignment experiments. It is organized so that each
paper result can be traced as:

```text
experiment config -> resolved run config -> raw output -> figure script -> figure
```

## Layout

- `configs/`: canonical experiment definitions used for paper claims.
- `experiment_lib/`: importable utilities plus data/model/optimizer and
  diagnostic components used by training.
- `pipeline/run_paper_experiment.py`: config-driven runner; prints commands by
  default and executes only with `--execute`.
- `pipeline/reproduce/`: one-command reproduction scripts for major experiments.
- `pipeline/train/train_alignment.py`: public training entry point used by the
  paper configs; it delegates to the modular implementation.
- `pipeline/analysis/`: diagnostic post-processing scripts.
- `pipeline/plot_figures/`: canonical scripts for paper figures.
- `runs/`: paper-named run directories for checking raw outputs.
- `reference_results/`: externally obtained benchmark outputs used only as
  comparison baselines; these are not runs produced by this artifact's
  training code.
- `runs/benchmark/`: our 15-trial MuonL benchmark run,
  including its exact standalone training script and raw logs.
- `figures/`: generated paper figures.

## Canonical Names

- `Muon`: the baseline Muon update.
- `NorMuon-lite`: benchmark #9's Muon update with per-row/column second-moment
  normalization and Frobenius renormalization.
- `MuonL`: all-layer u-w ratio alignment.
- `t-MuonL`: type-level u-w ratio alignment.
- `Muon + u-w floor`: Muon with a lower bound on pre-learning-rate
  `||U||_F / ||W||_F`.
- `shared learning-rate regime`: used for the shared-LR Muon, MuonL, and
  t-MuonL comparisons. All matrix groups use lr `0.035` and weight decay
  `0.025`.
- `type-specific learning-rate regime`: used for the type-specific Muon and
  t-MuonL comparisons. The matrix groups use:
  `q/k/v` lr `0.025`, wd `0.025`; `attn.proj` lr `0.030`, wd `0.025`;
  `mlp.fc` lr `0.035`, wd `0.0375`; and `mlp.proj` lr `0.035`, wd `0.025`.
- `Muon + u-w floor` ablations: use the shared matrix lr `0.035`, set all
  matrix weight decays to zero, and use the u-w floor as the stabilizing
  mechanism.
- Figure 1(b) reproduces benchmark #9 with `NorMuon-lite`, matrix lr `0.0375`,
  matrix weight decay `0`, floor `c=0.35`, and 3250 optimizer steps. Its only
  added instrumentation records every matrix block every 5 optimizer steps.

## Environment

The plotting and analysis scripts run from this `iclr_code` directory after
extracting the artifact; they do not require the enclosing `modded-nanogpt`
checkout. Install the package in editable mode from the artifact root:

```bash
cd iclr_code
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Verified analysis/plotting environment:

```text
Python 3.12.11
NumPy 1.26.4
Matplotlib 3.8.2
PyTorch 2.10.0+cu128
torch.version.cuda == 12.8
```

The saved figures can be regenerated on CPU from the included raw logs. New
training runs require CUDA, FineWeb token shards, and single-process
`torchrun`; the paper runs were generated on NVIDIA H100/H200-class GPUs. By
default training looks for data under `data/fineweb10B` relative to the
standalone artifact root; set `MYOPT_DATA_DIR=/path/to/fineweb10B` to use an
external data location.

## Print a Reproducible Command

By default the paper runner only prints the resolved command and does not start
training:

```bash
cd iclr_code
python3 pipeline/run_paper_experiment.py \
  --config configs/seed42_uw_floor_u_over_w.json
```

To run it:

```bash
python3 pipeline/run_paper_experiment.py \
  --config configs/seed42_uw_floor_u_over_w.json \
  --execute
```

The runner writes `resolved_config.json`, `hardware.json`, the exact command,
the environment, raw logs, and JSONL validation losses into the configured run
directory.

## Plot from Bundled Records

The following commands regenerate figures from the records included in this
artifact; they do not start new training:

```bash
python3 pipeline/plot_figures/shared_lr_seed42_loss.py
python3 pipeline/plot_figures/shared_lr_random5_loss.py
# Summarize our 15-trial MuonL benchmark run against the target loss.
python3 pipeline/analysis/benchmark_significance.py
python3 pipeline/plot_figures/benchmark_u_w_performance.py
python3 pipeline/plot_figures/weight_dynamics.py
```

These commands intentionally read bundled records: our results from `runs/`,
and Figure 1(a)'s external comparison data from `reference_results/`.

## Retrain and Then Plot

These scripts run training and pass the newly generated output paths explicitly
to their plotting commands:

```bash
# Seed-42 benchmark #9 NorMuon-lite reproduction with u-w floor c=0.35,
# recording every matrix block every 5 steps, then regenerating Figure 1(b).
bash pipeline/reproduce/u_over_w.sh

# Seed-42 type-specific Muon diagnostics and t-MuonL loss; then plot the new outputs.
bash pipeline/reproduce/type_lr_seed42.sh

# Five paired u/w-floor seeds; then plot them against bundled paired baselines.
bash pipeline/reproduce/uw_floor_random5.sh

# Five paired shared-LR seeds for Muon, t-MuonL, and MuonL; then plot new outputs.
bash pipeline/reproduce/shared_lr_random5.sh

# Five paired type-specific-LR seeds for Muon and t-MuonL; then plot new outputs.
bash pipeline/reproduce/type_lr_random5.sh

# Figure 8 MuonL, decoupled-OrScale, and OrScale; then plot new weight norms.
bash pipeline/reproduce/weight_dynamics.sh
```

## Paper Evidence

Use `configs/manifest.json` as the paper checklist. It lists, for each claim
group:

- raw run directories,
- diagnostic or figure scripts,
- generated figures,
- public configs when the run can be regenerated from the new config layer.

This manifest is the source of truth for checking that raw runs, scripts, and
figures still line up.
