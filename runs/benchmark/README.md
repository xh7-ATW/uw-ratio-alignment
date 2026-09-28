# MuonL All-Layer Benchmark Results

This folder contains the benchmark runner for the original all-layer MuonL
variant from `../muonl_draft`.

Optimizer hyperparameters are aligned to Track 3 result #12 Muon, except that
the hidden matrix optimizer is replaced by all-layer MuonL.

Default run settings:

- optimizer: `DistributedMuonLAllLayer`
- mode: `alllayer`
- trials: `15`
- train steps: `3300`
- schedule: stable then linear decay with `cooldown_frac=0.7`
- validation: every `125` steps before `3000`, every `25` steps from `3000`,
  every `10` steps from `3200`, plus final validation
- matrix groups: all transformer matrix weights use lr `0.035`, wd `0.025`
- auxiliary AdamW, matching result #12:
  - token embedding: lr `0.3`
  - output projection: lr `1/320`
  - 1D parameters: lr `0.01`
  - betas `(0.8, 0.95)`, eps `1e-10`, weight decay `0`

The `3300`-step horizon and denser tail validation are part of this confidence
experiment protocol, not optimizer hyperparameter changes.

All-layer MuonL scaling:

```text
r_i = ||W_i||_F / ||U_i||_F
x_i = EMA(r_i / mean_j r_j)
W_i <- (1 - eta * wd) W_i - eta * x_i U_i
```

where the mean is over all nonzero Muon matrix blocks. Zero-weight matrices use
`x_i = 1` until they become nonzero.

Run all 15 trials:

```bash
cd iclr_code
torchrun --standalone --nproc_per_node=1 \
  runs/benchmark/train_gpt_myopt_muonl_alllayer_random15.py 15
```
