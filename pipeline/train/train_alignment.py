#!/usr/bin/env python3
"""Public training entry point for u-w ratio alignment experiments."""

from __future__ import annotations

import sys
from pathlib import Path


ICLR_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ICLR_DIR))

from experiment_lib.training_runner import main


if __name__ == "__main__":
    main()
