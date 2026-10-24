#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Run Dynamic Scenes using configs/human_room.json, or --config another.json."""

from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src"), str(root), str(root / "tools")]

from mmWaveRadar.gui.scene_cli import main  # noqa: E402

if __name__ == "__main__":
    configuration = root / "configs" / "human_room.json"
    raise SystemExit(main(default_config_path=configuration))
