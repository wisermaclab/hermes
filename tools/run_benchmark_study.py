#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Source-checkout entry point for validation's configuration study command."""

from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src"), str(root), str(root / "tools")]

from validation.study import main  # noqa: E402

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not any(a == "--config" or a.startswith("--config=") for a in arguments):
        arguments = ["--config", str(root / "configs/benchmark_6843aop.json"), *arguments]
    raise SystemExit(main(arguments))
