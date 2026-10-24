# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University
"""Source-checkout entry point for release validation."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "src"),
]
from validation.release import main

if __name__ == "__main__":
    raise SystemExit(main())
