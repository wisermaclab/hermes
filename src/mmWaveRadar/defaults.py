# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Portable source-checkout defaults and explicit environment overrides."""

import os
from pathlib import Path


def source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_smpl_model_dir() -> Path:
    configured = os.environ.get("MMWAVE_SMPL_MODEL_DIR")
    return (
        Path(configured).expanduser() if configured else source_root() / "models" / "smpl_models"
    )


def default_amass_npz(dataset_root: str | Path | None = None) -> Path:
    configured = os.environ.get("MMWAVE_AMASS_NPZ")
    if configured:
        return Path(configured).expanduser()
    root = dataset_root or os.environ.get("MMWAVE_DATASET_ROOT")
    root = Path(root).expanduser() if root else source_root() / "data"
    return root / "AMASS" / "walking_poses_cmu_105_02.npz"
