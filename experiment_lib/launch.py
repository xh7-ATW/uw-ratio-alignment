"""Config-driven launcher for the paper experiments.

The public experiment entry point is intentionally a thin orchestration layer
around the historical training implementation. This preserves numerical
behavior while giving every paper run a resolved configuration and clean name.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import iclr_code_dir, repository_root, resolve_inside_iclr


@dataclass(frozen=True)
class ResolvedRun:
    """A concrete run produced from a paper experiment config."""

    experiment_name: str
    seed: int
    run_dir: Path
    raw_dir: Path
    log_dir: Path
    log_stem: str
    command: list[str]
    environment: dict[str, str]
    config: dict[str, Any]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a JSON experiment configuration."""

    with config_path.open() as handle:
        config = json.load(handle)
    required = ["experiment_name", "default_run_dir", "training", "environment"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{config_path} is missing required keys: {missing}")
    return config


def _format_value(value: Any, variables: dict[str, Any]) -> str:
    if isinstance(value, str):
        return value.format(**variables)
    return str(value)


def _seed_list(config: dict[str, Any]) -> list[int]:
    if "seeds" in config:
        return [int(seed) for seed in config["seeds"]]
    return [int(config.get("seed", 42))]


def resolve_runs(
    config_path: Path,
    *,
    run_root: Path | None = None,
    nproc_per_node: int | None = None,
    cuda_visible_devices: str | None = None,
    torchrun_bin: str | None = None,
) -> list[ResolvedRun]:
    """Resolve a paper experiment config into one or more seed-specific runs."""

    config_path = config_path.resolve()
    config = load_config(config_path)
    iclr_dir = iclr_code_dir(config_path)
    repo_root = repository_root(iclr_dir)
    root_dir = (run_root or resolve_inside_iclr(config["default_run_dir"], base=iclr_dir)).resolve()

    training = config["training"]
    train_script = resolve_inside_iclr(training["script"], base=iclr_dir)
    nproc = int(nproc_per_node or training.get("nproc_per_node", 1))
    torchrun = torchrun_bin or training.get("torchrun", "torchrun")

    runs: list[ResolvedRun] = []
    seeds = _seed_list(config)
    for seed in seeds:
        is_multiseed = len(seeds) > 1
        run_dir = root_dir / f"seed_{seed}" if is_multiseed else root_dir
        raw_dir = run_dir / "raw"
        log_dir = run_dir / "logs"
        variables = {
            "seed": seed,
            "experiment_name": config["experiment_name"],
            "run_root": str(root_dir),
            "run_dir": str(run_dir),
            "raw_dir": str(raw_dir),
        }
        log_stem = _format_value(config.get("log_stem", config["experiment_name"]), variables)
        variables["log_stem"] = log_stem

        env = {key: _format_value(value, variables) for key, value in config["environment"].items()}
        env.update(
            {
                "MYOPT_SEED": str(seed),
                "MYOPT_OUTPUT_DIR": str(raw_dir),
                "MYOPT_LOG_STEM": log_stem,
            }
        )
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        elif "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

        command = [
            torchrun,
            "--standalone",
            f"--nproc_per_node={nproc}",
            str(train_script),
        ]
        resolved_config = {
            **config,
            "config_path": str(config_path),
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
            "iclr_code_dir": str(iclr_dir),
            "repository_root": str(repo_root),
            "run_root": str(root_dir),
            "seed": seed,
            "run_dir": str(run_dir),
            "raw_dir": str(raw_dir),
            "log_dir": str(log_dir),
            "log_stem": log_stem,
            "command": command,
            "environment": env,
        }
        runs.append(
            ResolvedRun(
                experiment_name=config["experiment_name"],
                seed=seed,
                run_dir=run_dir,
                raw_dir=raw_dir,
                log_dir=log_dir,
                log_stem=log_stem,
                command=command,
                environment=env,
                config=resolved_config,
            )
        )
    return runs


def shell_command(run: ResolvedRun) -> str:
    """Return a copy-pasteable command for one resolved run."""

    repo_root = run.config["repository_root"]
    env_parts = [
        f"{key}={shlex.quote(value)}"
        for key, value in sorted(run.environment.items())
    ]
    cmd = " ".join(shlex.quote(part) for part in run.command)
    return f"cd {shlex.quote(repo_root)} && " + " ".join(env_parts + [cmd])


def write_resolved_files(run: ResolvedRun) -> None:
    """Write resolved config, command, environment, and hardware metadata."""

    run.raw_dir.mkdir(parents=True, exist_ok=True)
    run.log_dir.mkdir(parents=True, exist_ok=True)
    (run.run_dir / "resolved_config.json").write_text(
        json.dumps(run.config, indent=2, sort_keys=True) + "\n"
    )
    (run.log_dir / f"{run.log_stem}_command.txt").write_text(shell_command(run) + "\n")
    (run.log_dir / f"{run.log_stem}_environment.json").write_text(
        json.dumps(run.environment, indent=2, sort_keys=True) + "\n"
    )
    hardware = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch

        hardware["torch"] = torch.__version__
        hardware["cuda_available"] = str(torch.cuda.is_available())
        hardware["cuda_version"] = str(torch.version.cuda)
    except Exception as exc:  # pragma: no cover - best-effort metadata only.
        hardware["torch_metadata_error"] = repr(exc)
    (run.run_dir / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")


def execute_run(run: ResolvedRun, *, force: bool = False) -> int:
    """Execute one resolved run and stream stdout/stderr to a log file."""

    raw_log = run.raw_dir / f"{run.log_stem}.txt"
    if raw_log.exists() and not force:
        text = raw_log.read_text(errors="replace")
        train_steps = run.environment.get("MYOPT_TRAIN_STEPS", "3300")
        if f"step:{train_steps}/{train_steps} val_loss:" in text:
            print(f"skip complete run: {run.experiment_name} seed={run.seed} raw_log={raw_log}")
            return 0

    write_resolved_files(run)
    stdout_path = run.log_dir / f"{run.log_stem}_stdout.log"
    env = os.environ.copy()
    env.update(run.environment)
    with stdout_path.open("w") as stdout:
        process = subprocess.run(
            run.command,
            cwd=run.config["repository_root"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            check=False,
        )
    return int(process.returncode)
