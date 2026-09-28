#!/usr/bin/env python3
"""Run or print a named paper experiment from a resolved JSON config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
ICLR_DIR = SCRIPT.parents[1]
sys.path.insert(0, str(ICLR_DIR))

from experiment_lib.launch import execute_run, resolve_runs, shell_command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--torchrun-bin", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runs = resolve_runs(
        args.config,
        run_root=args.run_root,
        nproc_per_node=args.nproc_per_node,
        cuda_visible_devices=args.cuda_visible_devices,
        torchrun_bin=args.torchrun_bin,
    )

    if not args.execute:
        for run in runs:
            print(shell_command(run))
        return 0

    for run in runs:
        print(f"running {run.experiment_name} seed={run.seed} run_dir={run.run_dir}")
        status = execute_run(run, force=args.force)
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
