# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool(relative_path: str, module_name: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_rtpose_bundle_cli_contains_only_core_and_operational_inputs(tmp_path):
    module = _load_tool(
        "tools/bundle_prepare/rtpose/prepare_rtpose_bundle.py",
        "prepare_rtpose_bundle_defaults_test",
    )
    args = module.build_arg_parser().parse_args(
        [
            "--sequence",
            "176",
            "--radar-frame-id",
            "20",
            "--amass-npz",
            str(tmp_path / "fit.npz"),
        ]
    )

    assert set(vars(args)) == {
        "amass_npz",
        "environment_geometry",
        "force",
        "output_root",
        "radar_frame_id",
        "rtpose_root",
        "sequence",
        "smpl_model_dir",
    }
    assert args.amass_npz == tmp_path / "fit.npz"
    assert module.rtpose_frame_name("20") == "frame0020"


def test_rtpose_adc_reader_preserves_measured_iq_layout(tmp_path):
    module = _load_tool(
        "tools/bundle_prepare/rtpose/rtpose_adc.py",
        "rtpose_adc_layout_test",
    )
    params = module.RadarParams(
        num_adc_samples=2,
        num_devices=1,
        num_rx_per_device=2,
        num_chirps_in_loop=3,
        num_loops=2,
        tx_to_enable=(3, 2, 1),
        rx_for_mimo_process=(1, 2),
    )
    shape = (2, 2, 2, 3)
    real = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    expected = real.astype(np.complex64) - 1j * real.astype(np.complex64)
    device_order = expected.transpose(2, 0, 3, 1).reshape(-1, order="F")
    words = np.empty(device_order.size * 2, dtype="<i2")
    words[0::2] = device_order.real.astype(np.int16)
    words[1::2] = device_order.imag.astype(np.int16)
    data_file = tmp_path / "master_0000_data.bin"
    data_file.write_bytes(words.tobytes())

    actual = module.read_device_frame(data_file, 1, params)

    assert actual.dtype == np.complex64
    assert np.array_equal(actual, expected)


def test_rtpose_background_reconstruction_uses_dataset_frame_and_baked_mesh(tmp_path):
    module = _load_tool(
        "tools/bundle_prepare/rtpose/reconstruct_rtpose_background.py",
        "rtpose_background_reconstruction_test",
    )
    dataset = tmp_path / "dataset"
    sequence_root = dataset / "Data/sequences/176"
    lidar = sequence_root / "lidar/000005.pcd"
    lidar.parent.mkdir(parents=True)
    points = np.asarray(
        [
            [2.0, -0.5, -0.2],
            [2.0, 0.0, 0.0],
            [2.0, 0.5, 0.2],
            [3.0, 0.0, 0.5],
        ]
    )
    lidar.write_text(
        "\n".join(
            [
                "VERSION .7",
                "FIELDS x y z intensity",
                "SIZE 4 4 4 4",
                "TYPE F F F F",
                "COUNT 1 1 1 1",
                f"WIDTH {len(points)}",
                "HEIGHT 1",
                f"POINTS {len(points)}",
                "DATA ascii",
                *(f"{x} {y} {z} 1" for x, y, z in points),
                "",
            ]
        ),
        encoding="utf-8",
    )
    calibration = {
        "intrinsic": [[20, 0, 16, 0], [0, 20, 16, 0], [0, 0, 1, 0]],
        "extrinsic": [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
    }
    calibration_dir = dataset / "Data/calib/camera"
    calibration_dir.mkdir(parents=True)
    for side, color in (("left", (200, 10, 10)), ("right", (10, 10, 200))):
        (calibration_dir / f"{side}.json").write_text(
            json.dumps(calibration),
            encoding="utf-8",
        )
        image_path = sequence_root / f"camera/{side}/000005.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (32, 32), color).save(image_path)

    result = module.reconstruct_background(
        dataset_dir=dataset,
        sequence="176",
        source_frame_id="000005",
        body_vertices=np.asarray([[8.0, 0.0, 0.0], [8.2, 0.2, 0.2]]),
        output_dir=tmp_path / "output",
    )

    assert result.background_ply.is_file()
    assert result.stats_path.is_file()
    assert result.stats["background_points"] == len(points)
    assert result.stats["source_lidar"] == (
        "Data/sequences/176/lidar/000005.pcd"
    )
    assert "/private/" not in result.stats_path.read_text(encoding="utf-8")


def test_rtpose_bundle_calls_bundled_environment_fitter(tmp_path, monkeypatch):
    module = _load_tool(
        "tools/bundle_prepare/rtpose/prepare_rtpose_bundle.py",
        "prepare_rtpose_bundle_environment_test",
    )
    dataset = tmp_path / "dataset"
    background = (
        dataset
        / "Data/sequences/176/map/background_reconstruction/"
        "000005_background_lidar_colored.ply"
    )
    background.parent.mkdir(parents=True)
    background.write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n",
        encoding="utf-8",
    )
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        assert check is True
        assert capture_output is True
        assert text is True
        calls.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "environment_surfaces.json").write_text("{}\n", encoding="utf-8")
        (output_dir / "environment_surfaces.obj").write_text("# test\n", encoding="utf-8")
        (output_dir / "environment_scene.xml").write_text(
            '<scene version="2.1.0"/>\n',
            encoding="utf-8",
        )
        np.savez_compressed(
            output_dir / "environment_mesh_sequence.npz",
            vertices=np.zeros((1, 3, 3), dtype=np.float32),
            faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
            times=np.asarray([0.0]),
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    outputs = module.prepare_environment_geometry(
        rtpose_root=dataset,
        sequence="176",
        bundle_dir=bundle_dir,
        source_frame_id="000005",
        body_vertices=np.zeros((3, 3), dtype=np.float32),
        provided_geometry=None,
    )

    assert len(calls) == 1
    assert Path(calls[0][1]) == (
        REPO_ROOT
        / "tools/bundle_prepare/rtpose/fit_rtpose_environment_geometry.py"
    )
    wall_mode_index = calls[0].index("--wall-mode")
    assert calls[0][wall_mode_index + 1] == "solid"
    assert outputs["scene_path"].endswith("environment_scene.xml")
    assert outputs["source_background_ply"] == background.name
    assert outputs["geometry_source"] == "prepared_background"


def test_rtpose_bundle_reconstructs_and_fits_missing_geometry(tmp_path, monkeypatch):
    module = _load_tool(
        "tools/bundle_prepare/rtpose/prepare_rtpose_bundle.py",
        "prepare_rtpose_bundle_environment_validation_test",
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    observed: dict[str, object] = {}

    def fake_reconstruct(**kwargs):
        observed.update(kwargs)
        output_dir = kwargs["output_dir"]
        background = output_dir / "000005_background_lidar_colored.ply"
        stats = output_dir / "reconstruction_stats.json"
        background.write_text("ply\n", encoding="utf-8")
        stats.write_text('{"source": "/private/source.pcd"}\n', encoding="utf-8")
        return SimpleNamespace(background_ply=background, stats_path=stats)

    def fake_fit(background_ply, bundle_dir):
        observed["fit_background"] = background_ply
        output_dir = bundle_dir / "environment_geometry"
        output_dir.mkdir()
        outputs = {}
        for key, filename in (
            ("environment_surfaces_json", "environment_surfaces.json"),
            ("environment_obj", "environment_surfaces.obj"),
            ("scene_path", "environment_scene.xml"),
        ):
            path = output_dir / filename
            path.write_text("{}\n", encoding="utf-8")
            outputs[key] = str(path)
        mesh = output_dir / "environment_mesh_sequence.npz"
        np.savez_compressed(
            mesh,
            vertices=np.zeros((1, 3, 3), dtype=np.float32),
            faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
            times=np.asarray([0.0]),
        )
        outputs["environment_mesh_sequence"] = str(mesh)
        return outputs

    monkeypatch.setattr(module, "reconstruct_background", fake_reconstruct)
    monkeypatch.setattr(module, "fit_background_environment", fake_fit)

    outputs = module.prepare_environment_geometry(
        rtpose_root=tmp_path / "dataset",
        sequence="176",
        bundle_dir=bundle,
        source_frame_id="000005",
        body_vertices=np.ones((3, 3), dtype=np.float32),
        provided_geometry=None,
    )

    assert observed["source_frame_id"] == "000005"
    assert np.array_equal(observed["body_vertices"], np.ones((3, 3)))
    assert observed["fit_background"].name.endswith("background_lidar_colored.ply")
    assert outputs["geometry_source"] == "reconstructed_lidar_stereo"
    assert outputs["reconstruction_stats"] == (
        "environment_geometry/reconstruction_stats.json"
    )
    assert (bundle / outputs["reconstruction_stats"]).is_file()


def test_rtpose_bundle_reuses_provided_environment_folder(tmp_path):
    module = _load_tool(
        "tools/bundle_prepare/rtpose/prepare_rtpose_bundle.py",
        "prepare_rtpose_bundle_provided_environment_test",
    )
    source = tmp_path / "shared_environment"
    source.mkdir()
    (source / "environment_scene.xml").write_text(
        '<scene version="3.0.0"/>\n',
        encoding="utf-8",
    )
    (source / "environment_surfaces.json").write_text("{}\n", encoding="utf-8")
    np.savez_compressed(
        source / "environment_mesh_sequence.npz",
        vertices=np.zeros((1, 3, 3), dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
        times=np.asarray([0.0]),
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    outputs = module.prepare_environment_geometry(
        rtpose_root=tmp_path / "unused_dataset",
        sequence="176",
        bundle_dir=bundle,
        source_frame_id="000005",
        body_vertices=np.zeros((3, 3), dtype=np.float32),
        provided_geometry=source,
    )

    assert outputs["geometry_source"] == "provided"
    assert outputs["scene_path"] == "environment_geometry/environment_scene.xml"
    assert (bundle / outputs["scene_path"]).is_file()
    assert (bundle / outputs["environment_mesh_sequence"]).is_file()


def test_rtpose_environment_fitter_writes_pickle_free_scene(tmp_path):
    module = _load_tool(
        "tools/bundle_prepare/rtpose/fit_rtpose_environment_geometry.py",
        "fit_rtpose_environment_geometry_test",
    )
    background = tmp_path / "background.ply"
    points = []
    for x in np.linspace(0.5, 3.5, 10):
        for y in np.linspace(-1.0, 1.0, 10):
            points.append((x, y, 0.0, 120, 120, 120))
    for y in np.linspace(-1.0, 1.0, 10):
        for z in np.linspace(0.1, 2.0, 10):
            points.append((4.0, y, z, 180, 180, 180))
    rows = [" ".join(str(value) for value in point) for point in points]
    background.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(points)}",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                *rows,
                "",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "environment"

    assert module.main(
        [
            str(background),
            "--output-dir",
            str(output_dir),
            "--min-plane-bin-count",
            "10",
            "--min-wall-bin-count",
            "10",
            "--min-plane-inliers",
            "30",
            "--min-wall-inliers",
            "30",
            "--min-wall-height",
            "0.5",
            "--min-wall-span",
            "0.5",
            "--wall-mode",
            "solid",
            "--no-furniture",
            "--no-preview",
        ]
    ) == 0

    with np.load(
        output_dir / "environment_mesh_sequence.npz",
        allow_pickle=False,
    ) as archive:
        assert archive.files == ["vertices", "faces", "times"]
        assert archive["vertices"].shape[0] == 1
        assert archive["faces"].shape[1] == 3
        assert archive["times"].tolist() == [0.0]
    assert (output_dir / "environment_scene.xml").is_file()
    assert (output_dir / "environment_surfaces.obj").is_file()


def test_mmradarpose_cli_contains_only_core_and_operational_inputs(tmp_path):
    module = _load_tool(
        "tools/bundle_prepare/mmradarpose/prepare_mmradarpose_bundle.py",
        "prepare_mmradarpose_bundle_parser_test",
    )

    assert module.trial_parts("p1_an0_ac4_r0") == {
        "participant": 1,
        "angle": 0,
        "activity": 4,
        "recording": 0,
    }
    for invalid in ("../p1_an0_ac4_r0", "p1_an0_ac4_r0_extra", "p-1_an0_ac4_r0"):
        with pytest.raises(ValueError, match="Invalid mmRadarPose trial"):
            module.trial_parts(invalid)

    args = module.build_arg_parser().parse_args(
        [
            "--dataset-dir",
            "/tmp/dataset",
            "--trial",
            "p1_an0_ac4_r0",
            "--radar-frame-id",
            "1",
            "--amass-npz",
            str(tmp_path / "fit.npz"),
        ]
    )
    assert set(vars(args)) == {
        "amass_npz",
        "dataset_dir",
        "environment_geometry",
        "force",
        "frame",
        "output_root",
        "smpl_model_dir",
        "trial",
    }
    assert args.amass_npz == tmp_path / "fit.npz"
    assert module.frame_name("1") == "frame0001"
