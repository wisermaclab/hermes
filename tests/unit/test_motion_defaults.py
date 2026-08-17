# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Checks source-checkout defaults for the bundled CMU walking motion."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MOTION_RELATIVE_PATH = Path("data/AMASS/walking_poses_cmu_105_02.npz")
MODEL_RELATIVE_PATH = Path("models/smpl_models")


def _load_script(relative_path: str, module_name: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def human_rt_module():
    return _load_script("benchmarks/human_rt/run.py", "test_human_rt_run")


@pytest.fixture(scope="module")
def human_po_module():
    return _load_script("benchmarks/human_po/run.py", "test_human_po_run")


@pytest.fixture(scope="module")
def human_only_module():
    return _load_script(
        "benchmarks/human_only_rt_po_range_profile/run.py",
        "test_human_only_rt_po_range_profile_run",
    )


@pytest.fixture(scope="module")
def hybrid_po_module():
    return _load_script(
        "tools/hybrid_po/run_calibrated_hybrid_po.py",
        "test_run_calibrated_hybrid_po",
    )


@pytest.fixture(scope="module")
def amass_converter_module():
    return _load_script(
        "tools/amass_to_mesh_sequence.py",
        "test_amass_to_mesh_sequence",
    )


@pytest.mark.parametrize("module_fixture", ["human_rt_module", "human_only_module"])
def test_benchmark_motion_default_uses_bundled_fixture(
    module_fixture,
    request,
    monkeypatch,
):
    module = request.getfixturevalue(module_fixture)
    monkeypatch.delenv("MMWAVE_AMASS_NPZ", raising=False)
    monkeypatch.delenv("MMWAVE_DATASET_ROOT", raising=False)

    expected = REPO_ROOT / MOTION_RELATIVE_PATH
    assert module._default_amass_npz() == expected
    assert expected.is_file()


@pytest.mark.parametrize("module_fixture", ["human_rt_module", "human_only_module"])
def test_benchmark_motion_environment_precedence(
    module_fixture,
    request,
    monkeypatch,
    tmp_path,
):
    module = request.getfixturevalue(module_fixture)
    dataset_root = tmp_path / "dataset"
    explicit_motion = tmp_path / "motion.npz"
    monkeypatch.setenv("MMWAVE_DATASET_ROOT", str(dataset_root))
    monkeypatch.delenv("MMWAVE_AMASS_NPZ", raising=False)
    assert module._default_amass_npz() == (
        dataset_root / "AMASS" / MOTION_RELATIVE_PATH.name
    )

    monkeypatch.setenv("MMWAVE_AMASS_NPZ", str(explicit_motion))
    assert module._default_amass_npz() == explicit_motion


def test_hybrid_motion_default_uses_dataset_root(
    hybrid_po_module,
    monkeypatch,
    tmp_path,
):
    dataset_root = tmp_path / "dataset"
    monkeypatch.delenv("MMWAVE_AMASS_NPZ", raising=False)

    assert hybrid_po_module._default_amass_npz(dataset_root) == (
        dataset_root / "AMASS" / MOTION_RELATIVE_PATH.name
    )


def test_hybrid_source_checkout_default_exists(hybrid_po_module, monkeypatch):
    monkeypatch.delenv("MMWAVE_AMASS_NPZ", raising=False)
    expected = REPO_ROOT / MOTION_RELATIVE_PATH

    assert hybrid_po_module._default_amass_npz(REPO_ROOT / "data") == expected
    assert expected.is_file()


def test_hybrid_motion_environment_override(
    hybrid_po_module,
    monkeypatch,
    tmp_path,
):
    explicit_motion = tmp_path / "motion.npz"
    monkeypatch.setenv("MMWAVE_AMASS_NPZ", str(explicit_motion))

    assert hybrid_po_module._default_amass_npz(tmp_path / "dataset") == (
        explicit_motion
    )


def test_human_po_benchmark_binds_fmcw_to_single_channel_hardware(
    human_po_module,
):
    mesh_sequence = human_po_module.MeshSequence(
        vertices=np.asarray(
            [[[1.0, -0.2, 0.0], [1.0, 0.2, 0.0], [1.0, 0.0, 1.0]]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
        times=np.asarray([0.0]),
    )
    args = SimpleNamespace(
        carrier_frequency=60e9,
        slope=68e12,
        chirp_duration=58e-6,
        chirp_repetition_time=65e-6,
        sampling_frequency=4.5e6,
        num_adc_samples=225,
        num_chirps_per_frame=64,
        frame_period=50e-3,
        radar_position=None,
        radar_clearance_m=0.5,
        radar_look_at=None,
        radar_orientation=None,
    )

    radar, _look_at = human_po_module._build_radar(args, mesh_sequence)

    assert radar.fmcw.num_tx == radar.hardware.num_tx == 1


@pytest.mark.parametrize(
    "module_fixture",
    [
        "amass_converter_module",
        "human_rt_module",
        "human_only_module",
        "hybrid_po_module",
    ],
)
def test_smpl_model_default_uses_repository_local_directory(
    module_fixture,
    request,
    monkeypatch,
):
    module = request.getfixturevalue(module_fixture)
    monkeypatch.delenv("MMWAVE_SMPL_MODEL_DIR", raising=False)

    assert module._default_smpl_model_dir() == REPO_ROOT / MODEL_RELATIVE_PATH


@pytest.mark.parametrize(
    "module_fixture",
    [
        "amass_converter_module",
        "human_rt_module",
        "human_only_module",
        "hybrid_po_module",
    ],
)
def test_smpl_model_environment_override_takes_precedence(
    module_fixture,
    request,
    monkeypatch,
    tmp_path,
):
    module = request.getfixturevalue(module_fixture)
    explicit_model_dir = tmp_path / "licensed-smpl-models"
    monkeypatch.setenv("MMWAVE_SMPL_MODEL_DIR", str(explicit_model_dir))

    assert module._default_smpl_model_dir() == explicit_model_dir
