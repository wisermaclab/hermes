# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest


PUBLIC_ROOT = Path(__file__).resolve().parents[2]


def _load_tool(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, PUBLIC_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_direct_inputs(root: Path, *, trial: str = "p1_an0_ac4_r0") -> tuple[Path, Path]:
    radar_path = root / "radar" / f"data_cube_parsed_{trial}.npz"
    radar_path.parent.mkdir(parents=True)
    cube = (
        np.arange(4, dtype=np.float32)[:, None, None, None, None] * 10000
        + np.arange(4, dtype=np.float32)[None, :, None, None, None] * 1000
        + np.arange(4, dtype=np.float32)[None, None, :, None, None] * 100
        + np.arange(64, dtype=np.float32)[None, None, None, :, None] * 10
        + np.arange(128, dtype=np.float32)[None, None, None, None, :]
    ).astype(np.complex64)
    cube[:, 0:2, 0:2, :, :] = 0
    np.savez(radar_path, radar_cube=cube)
    (root / "radar_to_smpl_alignment_inferred.json").write_text(
        json.dumps(
            {
                "status": "inferred_from_dataset",
                "all_angles": {
                    "radar_to_hss_like": {
                        "rotation_matrix": np.eye(3).tolist(),
                        "translation": [0.16, -2.85, 1.07],
                    },
                    "radar_to_smpl": {
                        "rotation_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, -1.0, 0.0],
                        ],
                        "translation": [0.16, 1.07, 2.85],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    smpl_path = root / "fits" / "fit.npz"
    smpl_path.parent.mkdir()
    np.savez(
        smpl_path,
        poses=np.zeros((4, 72), dtype=np.float32),
        trans=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
            dtype=np.float32,
        ),
        betas=np.zeros(10, dtype=np.float32),
        gender=np.asarray("neutral"),
        world_up_axis=np.asarray("z"),
        world_forward_axis=np.asarray("-y"),
        frame_ids=np.asarray(["000000", "000001", "000002", "000003"]),
        times=np.arange(4, dtype=np.float64) / 15.0,
        mocap_framerate=np.asarray(15.0),
        profile=np.asarray("mmradarpose"),
        source_sequence_id=np.asarray(trial),
        fitted_joints=np.zeros((4, 26, 3), dtype=np.float32),
        target_joints=np.ones((4, 26, 3), dtype=np.float32),
    )
    return radar_path, smpl_path


def _direct_args(module, dataset: Path, smpl: Path, output_root: Path, *extra: str):
    model_dir = dataset / "models"
    model_dir.mkdir(exist_ok=True)
    return module.build_arg_parser().parse_args(
        [
            "--dataset-dir",
            str(dataset),
            "--trial",
            "p1_an0_ac4_r0",
            "--radar-frame-id",
            "1",
            "--amass-npz",
            str(smpl),
            "--smpl-model-dir",
            str(model_dir),
            "--output-root",
            str(output_root),
            *extra,
        ]
    )


def _fake_center_mesh(**kwargs):
    del kwargs
    vertices = np.asarray(
        [[[0, 3, 0], [1, 3, 0], [0, 3, 1], [0, 4, 0]]],
        dtype=np.float32,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.uint32,
    )
    return vertices, faces


def test_default_output_root_is_outside_the_source_checkout():
    module = _load_tool(
        "tools/bundle_prepare/mmradarpose/prepare_mmradarpose_bundle.py",
        "prepare_mmradarpose_default_output_test",
    )

    expected = (
        Path(tempfile.gettempdir())
        / "hermes-validation-bundles/mmradarpose/p1_an0_ac4_r0/bundles"
    ).resolve()
    output = module.resolve_output_root("p1_an0_ac4_r0", None)

    assert output == expected
    assert not output.is_relative_to(PUBLIC_ROOT.resolve())


def test_direct_bundle_consumes_sparse_fit_and_raw_radar(tmp_path, monkeypatch):
    monkeypatch.delenv("MMRADARPOSE_ROOT", raising=False)
    module = _load_tool(
        "tools/bundle_prepare/mmradarpose/prepare_mmradarpose_bundle.py",
        "prepare_mmradarpose_direct_test",
    )
    dataset = tmp_path / "dataset"
    _, smpl = _write_direct_inputs(dataset)
    monkeypatch.setattr(module, "build_direct_center_mesh", _fake_center_mesh)
    output_root = tmp_path / "results"

    result = module.prepare_bundle(_direct_args(module, dataset, smpl, output_root))
    output = output_root / "frame0001"

    assert result["integrity_check"]["valid"] is True
    assert result["radar_frame_id"] == "000001"
    assert result["pose_frame_id"] == "000001"
    assert result["frame_mapping"]["pose_to_radar_time_offset_s"] == pytest.approx(0.0)
    assert result["frame_mapping"]["mapping_override"] is False
    assert result["amass_sequence_source_indices"] == [0, 1, 2]
    assert result["amass_sequence_relative_times_s"] == pytest.approx(
        [-1.0 / 15.0, 0.0, 1.0 / 15.0]
    )
    with np.load(output / "radar_adc.npz", allow_pickle=False) as data:
        adc = data["adc"]
        assert adc.shape == (1, 128, 64, 12)
        assert adc[0, 127, 63, 0] == pytest.approx(12857.0)
        assert adc[0, 127, 63, 11] == pytest.approx(13957.0)
    source_frame, _ = module.load_mmradarpose_frame(
        dataset / "radar" / "data_cube_parsed_p1_an0_ac4_r0.npz", 1
    )
    flattened_adc, explicit_adc = module.export_mmradarpose_adc(source_frame)
    assert explicit_adc.shape == (1, 128, 64, 3, 4)
    assert np.array_equal(flattened_adc, explicit_adc.reshape(1, 128, 64, 12))
    assert not (output / "mesh_sequence.npz").exists()
    with np.load(output / "amass_sequence.npz", allow_pickle=False) as data:
        assert data["bundle_times"].tolist() == pytest.approx(
            [-1.0 / 15.0, 0.0, 1.0 / 15.0]
        )
        assert data["times"].tolist() == pytest.approx(
            [0.0, 1.0 / 15.0, 2.0 / 15.0]
        )
        assert data["frame_ids"].tolist() == ["000000", "000001", "000002"]
        assert data["output_axes"].tolist() == [0, 1, 2]
        assert data["output_signs"].tolist() == pytest.approx([1.0, 1.0, 1.0])
        assert data["trans"][1].tolist() == pytest.approx([0.84, 4.85, 1.93])
        assert data["radar_to_smpl_translation"].tolist() == pytest.approx(
            [0.16, 1.07, 2.85]
        )
    frames = json.loads((output / "frames.json").read_text(encoding="utf-8"))
    assert frames["frames"][0]["dataset_frame_id"] == "p1_an0_ac4_r0:000001"
    sensor = json.loads((output / "sensor.json").read_text(encoding="utf-8"))
    assert sensor["fmcw"]["tdm_enabled"] is True
    assert sensor["board_model"] == "IWR6843AOPEVM"
    assert sensor["metadata"]["alignment"]["source"] == (
        "radar_to_smpl_alignment_inferred.json"
    )
    assert sensor["metadata"]["alignment"]["selected_alignment"] == "all_angles"
    assert sensor["metadata"]["alignment"]["status"] == "inferred_from_dataset"
    assert sensor["metadata"]["frame_mapping"] == result["frame_mapping"]
    assert sensor["metadata"]["environment"]["type"] == "empty"
    benchmark = json.loads(
        (output / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    assert benchmark["motion"]["file"] == "amass_sequence.npz"
    assert benchmark["motion"]["timing_fields"]["bundle_times"].endswith("t=0.")
    assert result["source_amass_npz"] == "fits/fit.npz"
    assert not (output / "pointcloud.npz").exists()
    assert not (output / "skeleton.npz").exists()
    assert str(tmp_path) not in json.dumps(result)


def test_adc_export_rejects_nonzero_serialized_grid_padding():
    module = _load_tool(
        "tools/bundle_prepare/mmradarpose/prepare_mmradarpose_bundle.py",
        "prepare_mmradarpose_padding_test",
    )
    source = np.zeros((4, 4, 64, 128), dtype=np.complex64)
    source[0, 0, 0, 0] = 1.0

    with pytest.raises(ValueError, match="padding must be identically zero"):
        module.export_mmradarpose_adc(source)


def test_direct_bundle_rejects_wrong_sparse_fit_profile(tmp_path, monkeypatch):
    module = _load_tool(
        "tools/bundle_prepare/mmradarpose/prepare_mmradarpose_bundle.py",
        "prepare_mmradarpose_direct_profile_test",
    )
    dataset = tmp_path / "dataset"
    _, smpl = _write_direct_inputs(dataset)
    with np.load(smpl, allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    arrays["profile"] = np.asarray("rtpose")
    np.savez(smpl, **arrays)
    monkeypatch.setattr(module, "build_direct_center_mesh", _fake_center_mesh)

    with pytest.raises(ValueError, match="profile must be mmradarpose"):
        module.prepare_bundle(_direct_args(module, dataset, smpl, tmp_path / "results"))


def test_direct_bundle_rejects_non_amass_world_axes(tmp_path, monkeypatch):
    module = _load_tool(
        "tools/bundle_prepare/mmradarpose/prepare_mmradarpose_bundle.py",
        "prepare_mmradarpose_direct_world_axes_test",
    )
    dataset = tmp_path / "dataset"
    _, smpl = _write_direct_inputs(dataset)
    with np.load(smpl, allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    arrays["world_up_axis"] = np.asarray("y")
    arrays["world_forward_axis"] = np.asarray("z")
    np.savez(smpl, **arrays)
    monkeypatch.setattr(module, "build_direct_center_mesh", _fake_center_mesh)

    with pytest.raises(ValueError, match="canonical AMASS world axes"):
        module.prepare_bundle(_direct_args(module, dataset, smpl, tmp_path / "results"))


def test_alignment_is_composed_into_amass_world(tmp_path):
    module = _load_tool(
        "tools/bundle_prepare/mmradarpose/prepare_mmradarpose_bundle.py",
        "prepare_mmradarpose_amass_alignment_test",
    )
    dataset = tmp_path / "dataset"
    _write_direct_inputs(dataset)

    translation, axes, signs, metadata = module.load_alignment(
        dataset / "radar_to_smpl_alignment_inferred.json",
        trial="p1_an0_ac4_r0",
    )

    assert translation.tolist() == pytest.approx([0.16, -2.85, 1.07])
    assert axes.tolist() == [0, 1, 2]
    assert signs.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert metadata["smpl_model_to_radar_output_axes"] == [0, 2, 1]
    assert metadata["smpl_model_to_radar_output_signs"] == pytest.approx(
        [1.0, -1.0, 1.0]
    )


def test_direct_bundle_scans_hidden_radar_members_without_pickle(tmp_path, monkeypatch):
    module = _load_tool(
        "tools/bundle_prepare/mmradarpose/prepare_mmradarpose_bundle.py",
        "prepare_mmradarpose_direct_pickle_test",
    )
    dataset = tmp_path / "dataset"
    radar, smpl = _write_direct_inputs(dataset)
    with np.load(radar, allow_pickle=False) as source:
        cube = np.array(source["radar_cube"], copy=True)
    np.savez(radar, radar_cube=cube, hidden=np.asarray({"unsafe": True}, dtype=object))
    monkeypatch.setattr(module, "build_direct_center_mesh", _fake_center_mesh)

    with pytest.raises(ValueError, match="pickle-free plain dtype"):
        module.prepare_bundle(_direct_args(module, dataset, smpl, tmp_path / "results"))


def test_laser_environment_assets_are_converted_back_to_mmradarpose_axes(tmp_path):
    module = _load_tool(
        "tools/bundle_prepare/mmradarpose/mmradarpose_environment.py",
        "mmradarpose_environment_axis_test",
    )
    fitted = tmp_path / "fit"
    fitted.mkdir()
    (fitted / "environment_surfaces.json").write_text(
        json.dumps(
            {
                "fit": {
                    "floor_z_m": 0.0,
                    "back_wall_x_m": 4.0,
                    "left_wall_y_m": -2.0,
                    "right_wall_y_m": 2.0,
                },
                "surfaces": [
                    {
                        "id": "wall_0",
                        "semantic": "wall",
                        "material": "wall",
                        "center": [3.0, 1.0, 1.0],
                        "size": [0.1, 4.0, 2.0],
                        "source_points": 100,
                        "confidence": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    outputs = module._convert_fitted_assets(
        fitted,
        tmp_path / "bundle" / "environment_geometry",
        source_name="laser.pcd",
        input_points=120,
        removed_body_points=20,
    )

    environment_dir = tmp_path / "bundle" / "environment_geometry"
    payload = json.loads(
        (environment_dir / "environment_surfaces.json").read_text(encoding="utf-8")
    )
    assert payload["coordinate_frame"].startswith("mmRadarPose/radar")
    assert payload["surfaces"][0]["center"] == pytest.approx([1.0, 3.0, 1.0])
    assert payload["surfaces"][0]["size"] == pytest.approx([4.0, 0.1, 2.0])
    assert payload["fit"]["back_wall_y_m"] == pytest.approx(4.0)
    assert payload["fit"]["left_wall_x_m"] == pytest.approx(-2.0)
    assert outputs["geometry_source"] == "laser_fit"
    with np.load(environment_dir / "environment_mesh_sequence.npz", allow_pickle=False) as archive:
        assert archive["vertices"].shape == (1, 8, 3)
        assert archive["faces"].shape == (12, 3)
    assert "RT-Pose" not in (environment_dir / "environment_surfaces.obj").read_text()


def test_environment_auto_discovery_uses_laser_without_camera_inputs(tmp_path, monkeypatch):
    module = _load_tool(
        "tools/bundle_prepare/mmradarpose/mmradarpose_environment.py",
        "mmradarpose_environment_discovery_test",
    )
    dataset = tmp_path / "dataset"
    laser = dataset / "laser" / "environment.npy"
    laser.parent.mkdir(parents=True)
    np.save(laser, np.zeros((10, 3), dtype=np.float32))
    observed = {}

    def fake_fit(source, bundle_dir, *, body_vertices):
        observed.update(source=source, bundle_dir=bundle_dir, body_vertices=body_vertices)
        return {"geometry_source": "laser_fit", "scene_path": "environment_geometry/scene.xml"}

    monkeypatch.setattr(module, "fit_laser_environment", fake_fit)
    body = np.ones((4, 3), dtype=np.float32)
    outputs = module.prepare_environment_geometry(
        dataset_dir=dataset,
        bundle_dir=tmp_path / "bundle",
        body_vertices=body,
        provided_geometry=None,
    )

    assert observed["source"] == laser.resolve()
    assert np.array_equal(observed["body_vertices"], body)
    assert outputs["geometry_source"] == "dataset_laser_fit"


def test_laser_environment_fitting_is_a_single_pickle_free_stage(tmp_path):
    module = _load_tool(
        "tools/bundle_prepare/mmradarpose/mmradarpose_environment.py",
        "mmradarpose_environment_fit_test",
    )
    floor = np.asarray(
        [
            (lateral, distance, 0.0)
            for distance in np.linspace(0.5, 3.5, 25)
            for lateral in np.linspace(-1.0, 1.0, 25)
        ],
        dtype=np.float32,
    )
    wall = np.asarray(
        [
            (lateral, 4.0, height)
            for lateral in np.linspace(-1.0, 1.0, 25)
            for height in np.linspace(0.1, 2.0, 25)
        ],
        dtype=np.float32,
    )
    laser = tmp_path / "laser.npy"
    np.save(laser, np.concatenate([floor, wall], axis=0))
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    outputs = module.fit_laser_environment(
        laser,
        bundle,
        body_vertices=np.asarray([[8.0, 8.0, 8.0], [9.0, 9.0, 9.0]]),
    )

    assert outputs["scene_path"] == "environment_geometry/environment_scene.xml"
    assert not any(path.name == "laser.npy" for path in bundle.rglob("*"))
    with np.load(
        bundle / outputs["environment_mesh_sequence"],
        allow_pickle=False,
    ) as archive:
        assert archive.files == ["vertices", "faces", "times"]
        assert archive["vertices"].shape[0] == 1
