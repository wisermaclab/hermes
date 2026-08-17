# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    path = REPO_ROOT / "tools/bundle_prepare/rtpose/prepare_rtpose_bundle.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_rtpose_bundle_sparse_fit_contract_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fit_archive(module, *, frame_ids=("000002", "000005", "000008"), **timing):
    frame_count = len(frame_ids)
    return module.ArrayArchive(
        poses=np.zeros((frame_count, 72), dtype=np.float32),
        trans=np.zeros((frame_count, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        gender=np.asarray("neutral"),
        world_up_axis=np.asarray("z"),
        world_forward_axis=np.asarray("-y"),
        frame_ids=np.asarray(frame_ids),
        **{name: np.asarray(value) for name, value in timing.items()},
    )


def _write_labels(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "176": {
                    "000002": [
                        {"Radar_frameID": "000010", "LiDAR val": 1},
                        {"Radar_frameID": "000110", "LiDAR val": 0},
                    ],
                    "000005": [
                        {"Radar_frameID": "000011", "LiDAR val": 1},
                        {"Radar_frameID": "000111", "LiDAR val": 1},
                    ],
                    "000008": [
                        {"Radar_frameID": "000012", "LiDAR val": 0},
                        {"Radar_frameID": "000112", "LiDAR val": 1},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_filemeta(path: Path, sequence: str = "176") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                sequence: {
                    "Activity": "Walk and Wave Hand",
                    "Location": "Indoor",
                    "Senarios": "Clean",
                    "Occlusion": "Nothing",
                    "Subject type": "Single-Person",
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_camera_frame(dataset: Path, sequence: str, frame: str) -> None:
    for side in ("left", "right"):
        path = dataset / f"Data/sequences/{sequence}/camera/{side}/{frame}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + side.encode("ascii"))


def _write_frame_archive(path: Path, frame_ids: tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frame_ids=np.asarray(frame_ids))
    return path


def test_default_output_root_is_outside_the_source_checkout():
    module = _load_tool()

    expected = (
        Path(tempfile.gettempdir())
        / "hermes-validation-bundles/rtpose/sequence_176/bundles"
    ).resolve()
    output = module.resolve_output_root("176", None)

    assert output == expected
    assert not output.is_relative_to(REPO_ROOT.resolve())


def test_dataset_data_directory_is_normalized_to_checkout_root(tmp_path):
    module = _load_tool()
    dataset = tmp_path / "RT-Pose"
    data = dataset / "Data"
    (data / "sequences").mkdir(parents=True)

    assert module.resolve_rtpose_root(dataset) == dataset.resolve()
    assert module.resolve_rtpose_root(data) == dataset.resolve()


def test_amass_archive_is_discovered_below_sequence_smpl_directory(tmp_path):
    module = _load_tool()
    dataset = tmp_path / "RT-Pose"
    labels = _write_labels(dataset / "RT-POSE/Train.json")
    expected = (
        dataset
        / "Data/sequences/176/smpl/rtpose_best_gender_smpl_amass_like.npz"
    )
    _write_frame_archive(expected, ("000002", "000005", "000008"))
    # A preferred top-level release artifact wins over nested checkpoints.
    checkpoint = expected.parent / "checkpoints/old.npz"
    _write_frame_archive(checkpoint, ("000084", "000086", "000088"))

    assert module.resolve_amass_npz(
        None,
        rtpose_root=dataset,
        sequence="176",
        labels_path=labels,
        radar_frame_id="000111",
    ) == expected.resolve()


def test_ambiguous_discovered_amass_archives_require_explicit_path(tmp_path):
    module = _load_tool()
    dataset = tmp_path / "RT-Pose"
    labels = _write_labels(dataset / "RT-POSE/Train.json")
    smpl_dir = dataset / "Data/sequences/176/smpl"
    _write_frame_archive(smpl_dir / "candidate_a.npz", ("000002", "000005"))
    _write_frame_archive(smpl_dir / "candidate_b.npz", ("000005", "000008"))

    with pytest.raises(ValueError, match="Multiple fitted SMPL"):
        module.resolve_amass_npz(
            None,
            rtpose_root=dataset,
            sequence="176",
            labels_path=labels,
            radar_frame_id="000111",
        )


def test_segment_manifest_selects_archive_containing_synchronized_pose(tmp_path):
    module = _load_tool()
    dataset = tmp_path / "RT-Pose"
    labels = _write_labels(dataset / "RT-POSE/Train.json")
    smpl_dir = dataset / "Data/sequences/176/smpl"
    first = _write_frame_archive(
        smpl_dir / "segment_000/fit.npz",
        ("000002", "000005", "000008"),
    )
    _write_frame_archive(
        smpl_dir / "segment_001/fit.npz",
        ("000084", "000086", "000088"),
    )
    (smpl_dir / "segments.json").write_text(
        json.dumps(
            {
                "schema": "sparse2smpl.segmented_pose_manifest.v1",
                "parent_sequence_id": "176",
                "segments": [
                    {"output_path": "segment_000/fit.npz"},
                    {"output_path": "segment_001/fit.npz"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert module.resolve_amass_npz(
        None,
        rtpose_root=dataset,
        sequence="176",
        labels_path=labels,
        radar_frame_id="000111",
    ) == first.resolve()


def test_sequence_metadata_uses_rtpose_filemeta_fields(tmp_path):
    module = _load_tool()
    path = _write_filemeta(tmp_path / "Data/filemeta.txt")

    assert module.load_rtpose_sequence_metadata(path, "176") == {
        "activity": "Walk and Wave Hand",
        "location": "Indoor",
        "scenario": "Clean",
        "occlusion": "Nothing",
        "subject_type": "Single-Person",
    }


def test_camera_copy_requires_both_synchronized_pngs(tmp_path):
    module = _load_tool()
    dataset = tmp_path / "RT-Pose"
    left = dataset / "Data/sequences/176/camera/left/000005.png"
    left.parent.mkdir(parents=True)
    left.write_bytes(b"\x89PNG\r\n\x1a\nleft")
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    with pytest.raises(FileNotFoundError, match="right camera frame"):
        module.copy_rtpose_camera_frame(
            rtpose_root=dataset,
            sequence="176",
            camera_frame_id="5",
            bundle_dir=bundle,
        )


def test_mapping_comes_from_train_json_for_selected_person(tmp_path):
    module = _load_tool()
    labels = _write_labels(tmp_path / "Train.json")

    first = module.load_rtpose_frame_mapping(labels, "176", 0)
    second = module.load_rtpose_frame_mapping(labels, "176", 1)

    assert list(second) == ["000002", "000005", "000008"]
    assert first["000005"]["radar_frame_id"] == "000011"
    assert second["000005"] == {
        "pose_frame_id": "000005",
        "camera_frame_id": "000005",
        "radar_frame_id": "000111",
        "lidar_valid": 1,
    }
    with pytest.raises(IndexError, match="object index 2"):
        module.load_rtpose_frame_mapping(labels, "176", 2)


def test_pose_ids_never_use_radar_fields_from_sparse_fit():
    module = _load_tool()
    canonical = _fit_archive(module)
    canonical["source_frame_ids"] = np.asarray(["000901", "000902", "000903"])
    canonical["source_radar_frame_ids"] = np.asarray(["000010", "000011", "000012"])

    assert module.pose_frame_ids(canonical).tolist() == [
        "000002",
        "000005",
        "000008",
    ]

    legacy = module.ArrayArchive(canonical)
    del legacy["frame_ids"]
    legacy["source_frame_ids"] = np.asarray(["000002", "000005", "000008"])
    with pytest.raises(ValueError, match="canonical pose frame_ids"):
        module.pose_frame_ids(legacy)


def test_pose_timing_prefers_explicit_times_and_falls_back_to_framerate():
    module = _load_tool()
    explicit = _fit_archive(
        module,
        times=[5.0, 5.04, 5.15],
        mocap_framerate=0.0,
    )
    indices = np.asarray([0, 1, 2], dtype=np.int64)

    assert np.allclose(module.source_pose_times(explicit), [5.0, 5.04, 5.15])
    assert np.allclose(
        module.motion_relative_times(
            explicit,
            window_indices=indices,
            center_index=1,
        ),
        [-0.04, 0.0, 0.11],
    )

    fixed_rate = _fit_archive(module, mocap_framerate=20.0)
    assert np.allclose(module.source_pose_times(fixed_rate), [0.0, 0.05, 0.1])
    assert np.allclose(
        module.motion_relative_times(
            fixed_rate,
            window_indices=indices,
            center_index=1,
        ),
        [-0.05, 0.0, 0.05],
    )

    segmented = _fit_archive(
        module,
        times=[0.0, 0.1, 0.2],
        source_capture_times=[7.1, 7.2, 7.3],
    )
    assert np.allclose(module.segment_pose_times(segmented), [0.0, 0.1, 0.2])
    assert np.allclose(module.source_pose_times(segmented), [7.1, 7.2, 7.3])


@pytest.mark.parametrize(
    ("timing", "message"),
    [
        ({}, "times or mocap_framerate"),
        ({"mocap_framerate": 0.0}, "mocap_framerate"),
        ({"mocap_framerate": 30.0 + 1.0j}, "real numeric"),
        ({"times": [0.0, 0.1]}, "shape"),
        ({"times": [0.0 + 0.0j, 0.1 + 0.0j, 0.2 + 0.0j]}, "real numeric"),
        ({"times": [0.0, 0.2, 0.1]}, "increasing"),
    ],
)
def test_pose_timing_rejects_incomplete_or_ambiguous_archives(timing, message):
    module = _load_tool()
    data = _fit_archive(module, **timing)

    with pytest.raises(ValueError, match=message):
        module.source_pose_times(data)


def test_pose_archive_provenance_must_match_rtpose_sequence():
    module = _load_tool()
    wrong_profile = _fit_archive(module, mocap_framerate=10.0)
    wrong_profile["profile"] = np.asarray("mmradarpose")
    with pytest.raises(ValueError, match="profile must be rtpose"):
        module.validate_smpl_sequence(wrong_profile, expected_sequence="176")

    wrong_sequence = _fit_archive(module, mocap_framerate=10.0)
    wrong_sequence["profile"] = np.asarray("rtpose")
    wrong_sequence["source_sequence_id"] = np.asarray("177")
    with pytest.raises(ValueError, match="does not match requested sequence"):
        module.validate_smpl_sequence(wrong_sequence, expected_sequence="176")

    wrong_model = _fit_archive(module, mocap_framerate=10.0)
    wrong_model["model_type"] = np.asarray("smplx")
    with pytest.raises(ValueError, match="model_type must be 'smpl'"):
        module.validate_smpl_sequence(wrong_model, expected_sequence="176")

    complex_poses = _fit_archive(module, mocap_framerate=10.0)
    complex_poses["poses"] = np.zeros((3, 72), dtype=np.complex64)
    with pytest.raises(ValueError, match="finite real values"):
        module.validate_smpl_sequence(complex_poses, expected_sequence="176")

    overflowing_poses = _fit_archive(module, mocap_framerate=10.0)
    overflowing_poses["poses"] = np.zeros((3, 72), dtype=np.float64)
    overflowing_poses["poses"][0, 0] = float(np.finfo(np.float32).max) * 2.0
    with pytest.raises(ValueError, match="representable as float32"):
        module.validate_smpl_sequence(overflowing_poses, expected_sequence="176")

    native_smpl_world = _fit_archive(module, mocap_framerate=10.0)
    native_smpl_world["world_up_axis"] = np.asarray("y")
    native_smpl_world["world_forward_axis"] = np.asarray("z")
    with pytest.raises(ValueError, match="canonical AMASS world axes"):
        module.validate_smpl_sequence(native_smpl_world, expected_sequence="176")


def test_rtpose_output_transform_converts_amass_world_to_radar_axes():
    module = _load_tool()
    # AMASS [left, -forward, up] -> RT-Pose [range, lateral, up].
    amass_points = np.asarray([[1.0, -2.0, 3.0]], dtype=np.float32)
    radar_points = (
        amass_points[..., module.AMASS_TO_RTPOSE_AXES]
        * module.AMASS_TO_RTPOSE_SIGNS
    )

    assert np.allclose(radar_points, [[2.0, 1.0, 3.0]])


def test_pose_window_extends_to_cover_full_tdm_acquisition():
    module = _load_tool()
    times = np.arange(5, dtype=np.float64) / 30.0
    duration = module.radar_acquisition_duration_s(
        np.zeros((256, 64, 16, 12), dtype=np.complex64),
        {},
    )

    assert duration == pytest.approx((64 * 12 - 1) * 65.0e-6)
    assert module.select_window_indices(
        total=5,
        center_index=1,
        radius=1,
        source_times=times,
        required_after_s=duration,
    ).tolist() == [0, 1, 2, 3]

    with pytest.raises(ValueError, match="does not extend far enough"):
        module.select_window_indices(
            total=5,
            center_index=4,
            radius=1,
            source_times=times,
            required_after_s=duration,
        )

    with pytest.raises(ValueError, match="num_tx=.*does not match"):
        module.radar_acquisition_duration_s(
            np.zeros((256, 64, 16, 12), dtype=np.complex64),
            {"params": {"num_tx": 11}},
        )
    with pytest.raises(ValueError, match="num_tx must be an integer"):
        module.radar_acquisition_duration_s(
            np.zeros((256, 64, 16, 12), dtype=np.complex64),
            {"params": {"num_tx": 12.5}},
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"sequence": "177", "radar_frame_id": "111"},
        {"sequence": "176", "radar_frame_id": "112"},
    ],
)
def test_existing_adc_cache_must_match_requested_identity(tmp_path, metadata):
    module = _load_tool()
    dataset = tmp_path / "dataset"
    cache = dataset / "Data/sequences/176/radar/npy_raw/frame0111_adc.npz"
    cache.parent.mkdir(parents=True)
    np.savez_compressed(
        cache,
        adc_cube=np.zeros((4, 2, 1, 1), dtype=np.complex64),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    with pytest.raises(ValueError, match="Existing ADC cache is invalid"):
        module.load_or_build_measured_adc(
            dataset,
            "176",
            "000111",
        )


def test_missing_adc_cache_is_decoded_from_measured_capture(tmp_path, monkeypatch):
    module = _load_tool()
    dataset = tmp_path / "dataset"
    expected = np.zeros((4, 2, 1, 1), dtype=np.complex64)
    observed: dict[str, object] = {}

    def fake_decode(root, sequence, radar_frame_id):
        observed["request"] = (root, sequence, radar_frame_id)
        return expected, {
            "sequence": sequence,
            "radar_frame_id": radar_frame_id,
            "rx_order": [1],
        }

    monkeypatch.setattr(module, "build_measured_adc_cube", fake_decode)
    adc, metadata, source = module.load_or_build_measured_adc(
        dataset,
        "176",
        "000111",
    )

    assert observed["request"] == (dataset, "176", "000111")
    assert np.array_equal(adc, expected)
    assert metadata["radar_frame_id"] == "000111"
    assert metadata["source_rx_order"] == [1]
    assert metadata["exported_rx_order"] == [1]
    assert source == dataset / "Data/sequences/176/radar/bin"


def test_rtpose_adc_rx_axis_is_normalized_to_physical_label_order():
    module = _load_tool()
    source_order = [13, 14, 15, 16, 1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8]
    adc = np.zeros((1, 1, 16, 1), dtype=np.complex64)
    adc[0, 0, :, 0] = np.asarray(source_order, dtype=np.float32)

    normalized, metadata = module.normalize_adc_rx_order(
        adc,
        {
            "layout": ["adc_sample", "chirp_loop", "rx_mimo_order", "tx_slot"],
            "rx_order": source_order,
        },
    )

    assert normalized[0, 0, :, 0].real.tolist() == list(range(1, 17))
    assert metadata["source_rx_order"] == source_order
    assert metadata["exported_rx_order"] == list(range(1, 17))
    assert metadata["rx_order"] == list(range(1, 17))
    assert metadata["layout"][2] == "rx_physical_label_order"


def test_rtpose_adc_rx_axis_accepts_explicit_legacy_mimo_layout():
    module = _load_tool()
    source_order = list(module.RTPOSE_MIMO_RX_ORDER)
    adc = np.zeros((1, 1, 16, 1), dtype=np.complex64)
    adc[0, 0, :, 0] = np.asarray(source_order, dtype=np.float32)

    normalized, metadata = module.normalize_adc_rx_order(
        adc,
        {"layout": ["adc_sample", "chirp_loop", "rx_mimo_order", "tx_slot"]},
    )

    assert normalized[0, 0, :, 0].real.tolist() == list(range(1, 17))
    assert metadata["source_rx_order"] == source_order


def test_rtpose_adc_cache_rejects_ambiguous_receiver_order(tmp_path):
    module = _load_tool()
    dataset = tmp_path / "dataset"
    cache = dataset / "Data/sequences/176/radar/npy_raw/frame0111_adc.npz"
    cache.parent.mkdir(parents=True)
    np.savez_compressed(
        cache,
        adc_cube=np.zeros((4, 2, 16, 12), dtype=np.complex64),
        metadata_json=np.asarray(
            json.dumps({"sequence": "176", "radar_frame_id": "000111"})
        ),
    )

    with pytest.raises(ValueError, match="receiver ordering cannot be inferred"):
        module.load_or_build_measured_adc(dataset, "176", "000111")


def test_adc_cache_rejects_values_that_overflow_complex64(tmp_path):
    module = _load_tool()
    cache = tmp_path / "adc.npz"
    adc = np.zeros((4, 2, 1, 1), dtype=np.complex128)
    adc[0, 0, 0, 0] = float(np.finfo(np.float32).max) * 2.0
    np.savez_compressed(cache, adc_cube=adc)

    with pytest.raises(ValueError, match="representable as complex64"):
        module.load_adc_cube(cache)


def test_center_selection_joins_fit_pose_ids_to_dataset_radar_ids(tmp_path):
    module = _load_tool()
    labels = _write_labels(tmp_path / "Train.json")
    data = _fit_archive(module, mocap_framerate=30.0)
    aligned, object_index = module.select_rtpose_frame_mapping(
        labels,
        "176",
        pose_ids=module.pose_frame_ids(data),
        radar_frame_id="000111",
    )

    assert object_index == 1
    assert module.select_center_smpl_index(
        frame_mapping=aligned,
        radar_frame_id="000111",
    ) == 1

    legacy = module.ArrayArchive(data)
    del legacy["frame_ids"]
    legacy["source_frame_ids"] = np.asarray(["000110", "000111", "000112"])
    with pytest.raises(ValueError, match="pose frame|SMPL archive"):
        module.align_pose_frames_to_rtpose(
            module.pose_frame_ids(legacy),
            mapping,
        )


def test_alignment_rejects_missing_pose_and_ambiguous_radar_matches(tmp_path):
    module = _load_tool()
    mapping = module.load_rtpose_frame_mapping(
        _write_labels(tmp_path / "Train.json"),
        "176",
        0,
    )
    missing_pose = _fit_archive(
        module,
        frame_ids=("000002", "000005", "000099"),
        mocap_framerate=30.0,
    )
    with pytest.raises(ValueError, match="missing from RT-Pose labels.*000099"):
        module.align_pose_frames_to_rtpose(
            module.pose_frame_ids(missing_pose),
            mapping,
        )

    data = _fit_archive(module, mocap_framerate=30.0)
    aligned = module.align_pose_frames_to_rtpose(module.pose_frame_ids(data), mapping)
    aligned[1]["radar_frame_id"] = aligned[0]["radar_frame_id"]
    with pytest.raises(ValueError, match="multiple fitted poses"):
        module.select_center_smpl_index(
            frame_mapping=aligned,
            radar_frame_id="000010",
        )


def test_bundle_accepts_pose_only_fit_and_writes_aligned_timing(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_tool()
    dataset = tmp_path / "rtpose"
    _write_labels(dataset / "RT-POSE" / "Train.json")
    _write_filemeta(dataset / "Data" / "filemeta.txt")
    _write_camera_frame(dataset, "176", "000005")
    models = tmp_path / "models"
    models.mkdir()
    fit_npz = tmp_path / "fit.npz"
    np.savez_compressed(
        fit_npz,
        poses=np.zeros((3, 72), dtype=np.float32),
        trans=np.zeros((3, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        gender=np.asarray("neutral"),
        world_up_axis=np.asarray("z"),
        world_forward_axis=np.asarray("-y"),
        frame_ids=np.asarray(["000002", "000005", "000008"]),
        times=np.asarray([5.0, 5.04, 5.15]),
        mocap_framerate=np.asarray(30.0),
    )
    observed: dict[str, object] = {}

    def fake_load_or_build_adc(_root, sequence, radar_frame_id):
        observed["adc_identity"] = (sequence, radar_frame_id)
        return (
            np.zeros((4, 2, 1, 1), dtype=np.complex64),
            {
                "source_rx_order": [1],
                "exported_rx_order": [1],
                "rx_order": [1],
            },
            tmp_path / "radar/bin",
        )

    def fake_build_center_mesh(**kwargs):
        observed["center_index"] = kwargs["center_index"]
        vertices = np.asarray(
            [[[0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 2]]],
            dtype=np.float32,
        )
        faces = np.asarray(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            dtype=np.uint32,
        )
        return vertices, faces

    monkeypatch.setattr(
        module,
        "load_or_build_measured_adc",
        fake_load_or_build_adc,
    )
    monkeypatch.setattr(module, "build_center_mesh", fake_build_center_mesh)
    monkeypatch.setattr(module, "prepare_environment_geometry", lambda **_kwargs: {})
    monkeypatch.setattr(
        module,
        "check_bundle_output_integrity",
        lambda _path: {"valid": True},
    )

    output_root = tmp_path / "bundles"
    args = module.build_arg_parser().parse_args(
        [
            "--dataset-dir",
            str(dataset),
            "--sequence",
            "176",
            "--radar-frame-id",
            "111",
            "--amass-npz",
            str(fit_npz),
            "--smpl-model-dir",
            str(models),
            "--output-root",
            str(output_root),
        ]
    )

    summary = module.prepare_bundle(args)
    bundle = output_root / "frame0111"
    captured = capsys.readouterr()

    assert observed["center_index"] == 1
    assert "Selected SMPL archive for radar frame 000111" in captured.err
    assert summary["source_smpl_segment"] is None
    assert observed["adc_identity"] == ("176", "000111")
    assert summary["pose_frame_id"] == "000005"
    assert summary["source_radar_frame_id"] == "000111"
    assert summary["source_smpl_time_s"] == pytest.approx(5.04)
    assert summary["action"] == "Walk and Wave Hand"
    assert summary["environment"] == "clean"
    assert summary["subject_count"] == "single"
    assert summary["source_sequence_metadata"]["location"] == "Indoor"
    assert summary["camera_images"] == {
        "left": "camera/left/000005.png",
        "right": "camera/right/000005.png",
    }
    assert (bundle / "camera/left/000005.png").read_bytes().endswith(b"left")
    assert (bundle / "camera/right/000005.png").read_bytes().endswith(b"right")
    benchmark = json.loads(
        (bundle / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    sensor = json.loads((bundle / "sensor.json").read_text(encoding="utf-8"))
    assert sensor["fmcw"]["tdm_enabled"] is True
    assert sensor["metadata"]["source_rx_order"] == [1]
    assert sensor["metadata"]["exported_rx_order"] == [1]
    assert sensor["metadata"]["rx_order"] == [1]
    assert "physical 1-based" in sensor["metadata"]["rx_order_note"]
    assert sensor["metadata"]["velocity_resolution_mps"] > 0.0
    assert benchmark["camera_privacy"]["faces_obfuscated"] is False
    assert benchmark["bundle_contents"]["camera_images"] == summary["camera_images"]
    assert benchmark["motion"]["file"] == "amass_sequence.npz"
    assert "without refitting" in benchmark["motion"]["parameter_source"]
    assert benchmark["motion"]["timing_fields"]["bundle_times"].endswith("t=0.")
    assert summary["amass_sequence_relative_times_s"] == pytest.approx(
        [-0.04, 0.0, 0.11]
    )
    assert summary["mesh_sequence_included"] is False
    assert benchmark["motion"]["mesh_sequence_included"] is False
    assert not (bundle / "mesh_sequence.npz").exists()
    with np.load(bundle / "amass_sequence.npz", allow_pickle=False) as amass:
        assert amass["output_axes"].tolist() == [1, 0, 2]
        assert amass["output_signs"].tolist() == pytest.approx([-1.0, 1.0, 1.0])
        assert amass["frame_ids"].tolist() == [
            "000002",
            "000005",
            "000008",
        ]
        assert amass["radar_frame_ids"].tolist() == [
            "000110",
            "000111",
            "000112",
        ]
        assert np.allclose(amass["times"], [5.0, 5.04, 5.15])
        assert np.allclose(amass["source_capture_times"], [5.0, 5.04, 5.15])
        assert np.allclose(amass["bundle_times"], [-0.04, 0.0, 0.11])
        assert amass["betas"].shape == (10,)
        assert amass["faces"].shape == (4, 3)
        assert amass["selected_window_index"].item() == 1
        assert "source segment-local" in amass["timing_reference"].item()


def test_rtpose_publish_does_not_replace_late_foreign_destination(tmp_path):
    module = _load_tool()
    output_root = tmp_path / "results"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    (staged / "new.txt").write_text("new", encoding="utf-8")
    target = output_root / "frame0111"
    target.mkdir()
    (target / "foreign.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="appeared while preparing"):
        module.publish_staged_bundle(
            staged,
            target,
            force=False,
            sequence="176",
            radar_frame_id="000111",
        )

    assert (target / "foreign.txt").read_text(encoding="utf-8") == "keep"
    assert (staged / "new.txt").read_text(encoding="utf-8") == "new"


def test_rtpose_publish_restores_owned_bundle_on_failure(tmp_path, monkeypatch):
    module = _load_tool()
    output_root = tmp_path / "results"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    (staged / "new.txt").write_text("new", encoding="utf-8")
    target = output_root / "frame0111"
    target.mkdir()
    (target / "old.txt").write_text("keep old", encoding="utf-8")
    (target / "bundle_summary.json").write_text(
        json.dumps(
            {
                "tool": "prepare_rtpose_bundle",
                "dataset": "RT-Pose",
                "sequence": "176",
                "radar_frame_id": "000111",
            }
        ),
        encoding="utf-8",
    )
    real_replace = module.os.replace

    def fail_staged_publish(source, destination):
        if Path(source) == staged and Path(destination) == target:
            raise OSError("injected publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_staged_publish)
    with pytest.raises(OSError, match="injected publication failure"):
        module.publish_staged_bundle(
            staged,
            target,
            force=True,
            sequence="176",
            radar_frame_id="000111",
        )

    assert (target / "old.txt").read_text(encoding="utf-8") == "keep old"
    assert not list(output_root.glob(".frame0111.backup.*"))


def test_rtpose_ownership_marker_is_sequence_specific(tmp_path):
    module = _load_tool()
    target = tmp_path / "frame0111"
    target.mkdir()
    (target / "bundle_summary.json").write_text(
        json.dumps(
            {
                "dataset": "RT-Pose",
                "sequence": "177",
                "radar_frame_id": "000111",
            }
        ),
        encoding="utf-8",
    )

    assert not module.replaceable_bundle(
        target,
        sequence="176",
        radar_frame_id="000111",
    )


def test_automatically_prepared_scene_assets_are_staged_portably(tmp_path):
    module = _load_tool()
    bundle = tmp_path / "bundle"
    source_dir = bundle / "environment_geometry"
    source_dir.mkdir(parents=True)
    (source_dir / "person.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    scene = source_dir / "scene.xml"
    scene.write_text(
        '<scene version="3.0.0"><shape type="obj">'
        '<string name="filename" value="person.obj"/>'
        "</shape></scene>\n",
        encoding="utf-8",
    )
    outputs = {"scene_path": "environment_geometry/scene.xml"}

    staged = module.stage_scene_path(
        geometry_outputs=outputs,
        bundle_dir=bundle,
    )

    assert staged == scene.resolve()
    assert staged.is_file()
    assert (staged.parent / "person.obj").is_file()
    assert outputs["scene_path"] == str(staged)


def test_scene_rejects_absolute_asset_reference(tmp_path):
    module = _load_tool()
    bundle = tmp_path / "bundle"
    scene = bundle / "environment_geometry/scene.xml"
    scene.parent.mkdir(parents=True)
    absolute_asset = "/" + "private/local/person.obj"
    scene.write_text(
        '<scene version="3.0.0"><shape type="obj">'
        f'<string name="filename" value="{absolute_asset}"/>'
        "</shape></scene>\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-portable asset reference"):
        module.stage_scene_path(
            geometry_outputs={"scene_path": "environment_geometry/scene.xml"},
            bundle_dir=bundle,
        )


def test_environment_copy_sanitizes_json_and_drops_unneeded_object_arrays(tmp_path):
    module = _load_tool()
    dataset = tmp_path / "dataset"
    source_dir = dataset / "Data/sequences/176/map/environment_geometry"
    source_dir.mkdir(parents=True)
    local_source = "/" + "private/local/source.ply"
    (source_dir / "environment_surfaces.json").write_text(
        json.dumps({"metadata": {"source": local_source}}),
        encoding="utf-8",
    )
    np.savez_compressed(
        source_dir / "environment_mesh_sequence.npz",
        vertices=np.zeros((1, 3, 3), dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
        times=np.asarray([0.0]),
        local_paths=np.asarray([local_source], dtype=object),
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    outputs = module.copy_sequence_environment_geometry(
        rtpose_root=dataset,
        sequence="176",
        bundle_dir=bundle,
    )

    sanitized = json.loads(
        (bundle / "environment_geometry/environment_surfaces.json").read_text(
            encoding="utf-8"
        )
    )
    assert sanitized["metadata"]["source"] == "source.ply"
    assert outputs["environment_mesh_sequence"].endswith(
        "environment_mesh_sequence.npz"
    )
    assert module.portable_geometry_outputs(
        outputs,
        roots=(bundle, dataset),
    )["environment_mesh_sequence"] == (
        "environment_geometry/environment_mesh_sequence.npz"
    )
    with np.load(
        bundle / "environment_geometry/environment_mesh_sequence.npz",
        allow_pickle=False,
    ) as repacked:
        assert repacked.files == ["vertices", "faces", "times"]
