# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Source-checkout launcher for the installed mmwave-prepare-bundle command."""

from pathlib import Path
import sys

PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PUBLIC_ROOT / "src"), str(PUBLIC_ROOT / "tools")]

from bundle_prepare.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
