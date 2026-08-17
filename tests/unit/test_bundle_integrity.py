# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Tests for preparation-owned bundle loading and integrity checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from bundle_prepare.contract import Bundle, _safe_bundle_file, make_bundle_descriptor
from bundle_prepare.integrity import check_bundle_integrity
import bundle_prepare.integrity as integrity_module


def _write_bundle(root: Path) -> np.ndarray:
    root.mkdir(parents=True, exist_ok=True)
    (root / "bundle.json").write_text(
        json.dumps(
            make_bundle_descriptor(
                primary_origin="measurement",
                motion_parameters="amass_sequence.npz",
                environment="environment.json",
            )
        ),
        encoding="utf-8",
    )
    (root / "environment.json").write_text("{}", encoding="utf-8")
    (root / "sensor.json").write_text(
        json.dumps(
            {
                "board_model": "IWR6843ISK",
                "pattern_mode": "none",
                "radar": {
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0],
                },
                "fmcw": {
                    "carrier_frequency_hz": 60.0e9,
                    "slope_hz_per_s": 20.0e12,
                    "chirp_duration_s": 60.0e-6,
                    "chirp_repetition_time_s": 80.0e-6,
                    "sampling_frequency_hz": 2.0e6,
                    "num_adc_samples": 8,
                    "num_chirps_per_frame": 4,
                    "frame_period_s": 0.1,
                    "num_tx": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
    np.savez(
        root / "amass_sequence.npz",
        poses=np.zeros((2, 72), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        faces=faces,
        times=np.asarray([0.0, 0.1]),
    )
    adc = np.zeros((2, 4, 8, 12), dtype=np.complex64)
    np.savez(root / "radar_adc.npz", adc=adc)
    (root / "frames.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"benchmark_index": 0, "motion_time_s": 0.0},
                    {"benchmark_index": 1, "motion_time_s": 0.1},
                ]
            }
        ),
        encoding="utf-8",
    )
    return adc


def test_safe_bundle_file_rejects_intermediate_symlink_escape(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.npz").write_bytes(b"outside")
    (bundle_root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not traverse a symlink"):
        _safe_bundle_file(
            bundle_root,
            "linked/secret.npz",
            label="bundle.json target.mesh",
        )


def test_safe_bundle_file_returns_canonical_contained_file(tmp_path):
    bundle_root = tmp_path / "bundle"
    nested = bundle_root / "nested"
    nested.mkdir(parents=True)
    artifact = nested / "artifact.npz"
    artifact.write_bytes(b"contained")

    assert _safe_bundle_file(
        bundle_root,
        "nested/artifact.npz",
        label="artifact",
    ) == artifact.resolve()


def _replace_amass_arrays(root: Path, **replacements: np.ndarray) -> None:
    with np.load(root / "amass_sequence.npz", allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays.update(replacements)
    np.savez(root / "amass_sequence.npz", **arrays)


def test_integrity_is_independent_of_simulation(tmp_path):
    adc = _write_bundle(tmp_path)

    report = check_bundle_integrity(tmp_path)
    bundle = Bundle.load(tmp_path)

    assert report["valid"] is True
    assert report["check"] == "bundle_integrity"
    assert report["primary_adc_artifact"]["shape"] == list(adc.shape)
    assert report["profile"] == "hermes"
    assert report["primary_adc_artifact"]["origin"] == "measurement"
    assert report["capabilities"] == [
        "environment-metadata",
        "human-motion-resimulation",
        "primary-adc",
    ]
    assert bundle.num_frames == 2
    assert bundle.can_resimulate is True


def test_bundle_requires_unified_descriptor(tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / "bundle.json").unlink()

    with pytest.raises(FileNotFoundError, match="Bundle descriptor"):
        Bundle.load(tmp_path)
    with pytest.raises(FileNotFoundError, match="Bundle descriptor"):
        check_bundle_integrity(tmp_path)


def test_integrity_rejects_nonfinite_adc(tmp_path):
    adc = _write_bundle(tmp_path)
    adc[0, 0, 0, 0] = np.nan
    np.savez(tmp_path / "radar_adc.npz", adc=adc)

    with pytest.raises(ValueError, match="only finite values"):
        check_bundle_integrity(tmp_path)


def test_integrity_rejects_frame_time_outside_motion_timeline(tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / "frames.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"benchmark_index": 0, "motion_time_s": 0.0},
                    {"benchmark_index": 1, "motion_time_s": 0.2},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="within the amass_sequence.npz timeline"):
        check_bundle_integrity(tmp_path)


def test_integrity_requires_explicit_motion_time(tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / "frames.json").write_text(
        json.dumps({"frames": [{"benchmark_index": 0}, {"benchmark_index": 1}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must include motion_time_s"):
        check_bundle_integrity(tmp_path)


def test_integrity_rejects_invalid_amass_timing_and_faces(tmp_path):
    _write_bundle(tmp_path)
    np.savez(
        tmp_path / "amass_sequence.npz",
        poses=np.zeros((2, 72), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        times=np.asarray([0.0, 0.0]),
        faces=np.asarray([[0, 1, 3]], dtype=np.uint32),
    )

    with pytest.raises(ValueError, match="times must be strictly increasing"):
        check_bundle_integrity(tmp_path)

    np.savez(
        tmp_path / "amass_sequence.npz",
        poses=np.zeros((2, 72), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        times=np.asarray([0.0, 0.1]),
        faces=np.asarray([[0, 1, 6890]], dtype=np.uint32),
    )
    with pytest.raises(ValueError, match="outside base-SMPL topology"):
        check_bundle_integrity(tmp_path)


def test_integrity_and_bundle_prefer_canonical_amass_sequence(tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / "frames.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"benchmark_index": 0, "motion_time_s": -0.1},
                    {"benchmark_index": 1, "motion_time_s": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        tmp_path / "amass_sequence.npz",
        poses=np.zeros((2, 72), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        times=np.asarray([7.2, 7.3]),
        bundle_times=np.asarray([-0.1, 0.0]),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
    )

    report = check_bundle_integrity(tmp_path)
    bundle = Bundle.load(tmp_path)

    assert report["amass_sequence"]["present"] is True
    assert report["amass_sequence"]["timing_key"] == "bundle_times"
    assert bundle.motion_path == tmp_path / "amass_sequence.npz"


def test_integrity_validates_source_times_even_when_bundle_times_are_present(
    tmp_path,
):
    _write_bundle(tmp_path)
    _replace_amass_arrays(
        tmp_path,
        times=np.asarray([7.2, 7.2]),
        bundle_times=np.asarray([0.0, 0.1]),
    )

    with pytest.raises(
        ValueError,
        match=r"amass_sequence\.npz times must be strictly increasing",
    ):
        check_bundle_integrity(tmp_path)


@pytest.mark.parametrize(
    ("key", "values", "message"),
    (
        (
            "bundle_times",
            np.asarray([0.0, 0.0]),
            r"bundle_times must be strictly increasing",
        ),
        (
            "source_capture_times",
            np.asarray([7.2, np.nan]),
            r"source_capture_times must contain finite plain numeric values",
        ),
        (
            "radar_capture_times",
            np.asarray([2.0]),
            r"radar_capture_times must have shape \[2\]",
        ),
    ),
)
def test_integrity_validates_each_optional_motion_timeline(
    tmp_path,
    key,
    values,
    message,
):
    _write_bundle(tmp_path)
    _replace_amass_arrays(tmp_path, **{key: values})

    with pytest.raises(ValueError, match=message):
        check_bundle_integrity(tmp_path)


def test_integrity_reports_each_validated_motion_timeline(tmp_path):
    _write_bundle(tmp_path)
    _replace_amass_arrays(
        tmp_path,
        bundle_times=np.asarray([0.0, 0.1]),
        source_capture_times=np.asarray([7.2, 7.3]),
        radar_capture_times=np.asarray([2.0, 2.1]),
    )

    report = check_bundle_integrity(tmp_path)

    assert report["amass_sequence"]["timeline_shapes"] == {
        "times": [2],
        "bundle_times": [2],
        "source_capture_times": [2],
        "radar_capture_times": [2],
    }


def test_bundle_rejects_undeclared_body_motion_filenames(tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / "amass_sequence.npz").unlink()
    np.savez(
        tmp_path / "smpl_params.npz",
        poses=np.zeros((2, 72), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        times=np.asarray([0.0, 0.1]),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
    )
    np.savez(
        tmp_path / "mesh_sequence.npz",
        vertices=np.zeros((2, 3, 3), dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
        times=np.asarray([0.0, 0.1]),
    )

    with pytest.raises(FileNotFoundError, match="amass_sequence.npz"):
        Bundle.load(tmp_path)
    with pytest.raises(FileNotFoundError, match="amass_sequence.npz"):
        check_bundle_integrity(tmp_path)


def test_integrity_checks_environment_references(tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / "environment.json").write_text(
        json.dumps({"scene_path": "environment_geometry/scene.xml"}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="environment asset"):
        check_bundle_integrity(tmp_path)

    scene = tmp_path / "environment_geometry" / "scene.xml"
    scene.parent.mkdir()
    scene.write_text('<scene version="2.1.0"/>\n', encoding="utf-8")
    report = check_bundle_integrity(tmp_path)
    assert report["environment"]["referenced_assets"] == {
        "scene_path": "environment_geometry/scene.xml"
    }


def test_integrity_validates_optional_environment_mesh_sequence(tmp_path):
    _write_bundle(tmp_path)
    mesh_path = tmp_path / "environment_geometry/environment_mesh_sequence.npz"
    mesh_path.parent.mkdir()
    np.savez(
        mesh_path,
        vertices=np.asarray(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
        times=np.asarray([0.0]),
    )
    (tmp_path / "environment.json").write_text(
        json.dumps(
            {
                "environment_mesh_sequence": (
                    "environment_geometry/environment_mesh_sequence.npz"
                )
            }
        ),
        encoding="utf-8",
    )

    report = check_bundle_integrity(tmp_path)

    assert report["environment"]["referenced_assets"] == {
        "environment_mesh_sequence": (
            "environment_geometry/environment_mesh_sequence.npz"
        )
    }


def test_integrity_rejects_absolute_environment_reference(tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / "environment.json").write_text(
        json.dumps({"scene_path": str(tmp_path / "scene.xml")}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bundle-relative"):
        check_bundle_integrity(tmp_path)


def test_integrity_checks_benchmark_camera_references(tmp_path):
    _write_bundle(tmp_path)
    images = {
        "left": "camera/left/000038.png",
        "right": "camera/right/000038.png",
    }
    (tmp_path / "benchmark_metadata.json").write_text(
        json.dumps({"bundle_contents": {"camera_images": images}}),
        encoding="utf-8",
    )
    for relative in images.values():
        image = tmp_path / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 1), color=(10, 20, 30)).save(image, format="PNG")

    report = check_bundle_integrity(tmp_path)
    assert report["camera_images"] == {
        "present": True,
        "referenced_images": images,
    }

    (tmp_path / images["right"]).unlink()
    with pytest.raises(FileNotFoundError, match="right camera image"):
        check_bundle_integrity(tmp_path)


def test_integrity_rejects_camera_file_with_only_a_png_signature(tmp_path):
    _write_bundle(tmp_path)
    images = {
        "left": "camera/left/000038.png",
        "right": "camera/right/000038.png",
    }
    (tmp_path / "benchmark_metadata.json").write_text(
        json.dumps({"bundle_contents": {"camera_images": images}}),
        encoding="utf-8",
    )
    for relative in images.values():
        image = tmp_path / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 1), color=(10, 20, 30)).save(image, format="PNG")
    (tmp_path / images["left"]).write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    with pytest.raises(ValueError, match="left camera image could not be decoded"):
        check_bundle_integrity(tmp_path)


def test_integrity_rejects_camera_that_exceeds_decoded_pixel_cap(
    tmp_path,
    monkeypatch,
):
    _write_bundle(tmp_path)
    images = {"left": "left.png", "right": "right.png"}
    (tmp_path / "benchmark_metadata.json").write_text(
        json.dumps({"bundle_contents": {"camera_images": images}}),
        encoding="utf-8",
    )
    for relative in images.values():
        Image.new("RGB", (2, 1)).save(tmp_path / relative, format="PNG")
    monkeypatch.setattr(integrity_module, "_MAX_CAMERA_PIXELS", 1)

    with pytest.raises(ValueError, match="pixel limit"):
        check_bundle_integrity(tmp_path)


def test_integrity_rejects_mitsuba_include_in_declared_scene(tmp_path):
    _write_bundle(tmp_path)
    descriptor_path = tmp_path / "bundle.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["scene"] = {"file": "scene.xml", "assets": []}
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    (tmp_path / "scene.xml").write_text(
        '<scene><include filename="../outside.xml"/></scene>',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="include.*not permitted"):
        check_bundle_integrity(tmp_path)


def test_integrity_rejects_mitsuba_filename_escape(tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / "environment.json").write_text(
        json.dumps({"scene_path": "scene.xml"}),
        encoding="utf-8",
    )
    (tmp_path / "scene.xml").write_text(
        (
            '<scene><shape type="obj"><string name="filename" '
            'value="../outside.obj"/></shape></scene>'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="filename must be bundle-relative"):
        check_bundle_integrity(tmp_path)


def test_integrity_caps_obj_fan_triangulation(tmp_path, monkeypatch):
    _write_bundle(tmp_path)
    descriptor_path = tmp_path / "bundle.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["target"] = {"obj": "target.obj"}
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    (tmp_path / "target.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(integrity_module, "_MAX_OBJ_TRIANGLES", 1)

    with pytest.raises(ValueError, match="triangle limit"):
        check_bundle_integrity(tmp_path)
