"""FineWeb token-shard data loading for the paper training runs."""

from __future__ import annotations

import glob
from pathlib import Path

import torch
import torch.distributed as dist


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520
    assert header[1] == 1
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens


def distributed_data_generator(
    filename_pattern: str,
    batch_size: int,
    seq_len: int = 1024,
    *,
    repo_root: Path | None = None,
):
    files = sorted(Path(p) for p in glob.glob(str(filename_pattern)))
    if not files and repo_root is not None:
        files = sorted(Path(p) for p in glob.glob(str(repo_root / filename_pattern)))
    assert files, f"no data shards match {filename_pattern!r}"
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)
