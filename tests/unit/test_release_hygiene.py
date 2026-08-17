# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Small checks that keep public documentation and demos portable."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

import numpy as np

from mmWaveRadar.radar import FMCWConfig, RadarHardware


REPO_ROOT = Path(__file__).resolve().parents[2]
AMASS_DATA_ROOT = REPO_ROOT / "data" / "AMASS"
VALIDATION_DATA_ROOT = REPO_ROOT / "data" / "validation_bundles"
PINNED_MOTION_FILES = {
    "walking_poses_cmu_105_02.npz": {
        "size": 807_203,
        "sha256": "a978f1b6dcd349e358cce922c767a07bf7bf86d044bcbf10a071945fe55d2c12",
    },
}
PINNED_VALIDATION_BINARY_FILES = {
    "mmradarpose_p1_an0_ac4_r0_frame0240/amass_sequence.npz": {
        "size": 66_133,
        "sha256": "12adb7141396c4d0a1e872a50fe88cd2b8fe8c2550349115ada5cb9e240667f7",
    },
    "mmradarpose_p1_an0_ac4_r0_frame0240/radar_adc.npz": {
        "size": 406_843,
        "sha256": "eefd8c82ffc87a9f2338ad77c049f8f347ce8e63256132d620d4ce4953044e0c",
    },
    "rtpose_seq10_frame0020/amass_sequence.npz": {
        "size": 67_985,
        "sha256": "f593dbf12f2013f2c7b4dcda9605788149ba052b388afef887dd39b6e64fcc92",
    },
    "rtpose_seq10_frame0020/camera/left/000056_face_occluded.png": {
        "size": 1_511_831,
        "sha256": "aa7edd6849e7f660cb707cbc17262ec47909424a858b0e3ec725ae5c3ce0bb8a",
    },
    "rtpose_seq10_frame0020/camera/right/000056_face_occluded.png": {
        "size": 1_469_732,
        "sha256": "d9bb4ad251ec427fcb61a4507a141c48f4a4005821ec847c7e241da3dcb2d0a6",
    },
    "rtpose_seq10_frame0020/radar_adc.npz": {
        "size": 23_351_922,
        "sha256": "7d2600a8401e58f8fbb062eedd973f1fcad0fc5709fe8e173a2ff89abc1835dd",
    },
}
VALIDATION_TEXT_FILES = {
    "README.md",
    "mmradarpose_p1_an0_ac4_r0_frame0240/NOTICE.md",
    "mmradarpose_p1_an0_ac4_r0_frame0240/README.md",
    "mmradarpose_p1_an0_ac4_r0_frame0240/benchmark_metadata.json",
    "mmradarpose_p1_an0_ac4_r0_frame0240/bundle.json",
    "mmradarpose_p1_an0_ac4_r0_frame0240/environment.json",
    "mmradarpose_p1_an0_ac4_r0_frame0240/frames.json",
    "mmradarpose_p1_an0_ac4_r0_frame0240/sensor.json",
    "rtpose_seq10_frame0020/NOTICE.md",
    "rtpose_seq10_frame0020/README.md",
    "rtpose_seq10_frame0020/benchmark_metadata.json",
    "rtpose_seq10_frame0020/bundle.json",
    "rtpose_seq10_frame0020/environment.json",
    "rtpose_seq10_frame0020/environment_geometry/environment_scene.xml",
    "rtpose_seq10_frame0020/frames.json",
    "rtpose_seq10_frame0020/sensor.json",
}

PROJECT_CODE_ROOTS = ("src", "validation", "tools", "benchmarks", "tests")
PROJECT_CODE_SPDX = "SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0"
MIXED_TI_SOURCE = Path("tools/bundle_prepare/rtpose/rtpose_adc.py")
MIXED_TI_SPDX = (
    "SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 AND BSD-3-Clause"
)


class _SourceLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = attributes.get("class", "").split()
        if tag == "a" and "source-link" in classes:
            self.links.append(attributes["href"])


def test_project_license_files_metadata_and_source_headers_are_consistent():
    code_license = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    docs_data_license = (REPO_ROOT / "LICENSE-DOCS-DATA").read_text(
        encoding="utf-8"
    )
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    commercial = (REPO_ROOT / "COMMERCIAL-LICENSING.md").read_text(
        encoding="utf-8"
    )
    package_metadata = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert code_license.startswith("# PolyForm Noncommercial License 1.0.0\n")
    assert "polyformproject.org/licenses/noncommercial/1.0.0" in code_license
    assert docs_data_license.startswith(
        "Attribution-NonCommercial 4.0 International\n"
    )
    assert 'license = "PolyForm-Noncommercial-1.0.0"' in package_metadata
    assert "LICENSE-DOCS-DATA" in package_metadata
    assert notice.startswith("Required Notice: Copyright (c) 2026 ")
    assert "Rong Zheng (rzheng@mcmaster.ca)" in notice
    assert "Rong Zheng" in commercial
    assert "rzheng@mcmaster.ca" in commercial

    source_files = sorted(
        path
        for root in PROJECT_CODE_ROOTS
        for path in (REPO_ROOT / root).rglob("*.py")
    )
    assert source_files
    for path in source_files:
        relative = path.relative_to(REPO_ROOT)
        first_lines = "\n".join(path.read_text(encoding="utf-8").splitlines()[:4])
        expected = MIXED_TI_SPDX if relative == MIXED_TI_SOURCE else PROJECT_CODE_SPDX
        assert expected in first_lines, f"incorrect SPDX header: {relative}"
        assert "GPL-3.0" not in first_lines


def test_third_party_fixtures_are_explicitly_excluded_from_project_licenses():
    third_party_notice = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(
        encoding="utf-8"
    )
    excluded_paths = (
        "data/AMASS/walking_poses_cmu_105_02.npz",
        "data/validation_bundles/rtpose_seq10_frame0020/",
        "data/validation_bundles/mmradarpose_p1_an0_ac4_r0_frame0240/",
        "tools/bundle_prepare/rtpose/rtpose_adc.py",
    )
    assert "Explicit exclusions from the HERMES project licenses" in third_party_notice
    assert all(path in third_party_notice for path in excluded_paths)


def test_api_documentation_source_links_resolve():
    api_index = REPO_ROOT / "docs" / "api" / "index.html"
    parser = _SourceLinkParser()
    parser.feed(api_index.read_text(encoding="utf-8"))

    assert parser.links
    missing = [
        link
        for link in parser.links
        if not (api_index.parent / link.split("#", 1)[0]).resolve().is_file()
    ]
    assert not missing, f"Broken API source links: {missing}"


def test_demo_readme_lists_every_notebook():
    notebook_names = {
        path.name for path in (REPO_ROOT / "demo").glob("*.ipynb")
    }
    demo_readme = (REPO_ROOT / "demo" / "README.md").read_text(
        encoding="utf-8"
    )

    missing = sorted(name for name in notebook_names if name not in demo_readme)
    assert not missing, f"Notebooks missing from demo/README.md: {missing}"


def test_human_only_po_notebook_binds_fmcw_to_its_hardware():
    notebook_path = REPO_ROOT / "demo" / "Human-Only-PO.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "single-virtual-channel" in "".join(cell.get("source", []))
        and "FMCWConfig(" in "".join(cell.get("source", []))
    )
    namespace = {
        "FMCWConfig": FMCWConfig,
        "RadarHardware": RadarHardware,
    }

    exec(compile(source, str(notebook_path), "exec"), namespace)  # noqa: S102

    fmcw = namespace["fmcw"]
    hardware = namespace["hardware"]
    assert fmcw.num_tx == hardware.num_tx == 1


def test_human_only_po_notebook_is_a_focused_api_tutorial():
    notebook_path = REPO_ROOT / "demo" / "Human-Only-PO.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert 'mobility_mode="human_only_po"' in source
    assert '"face_centroid"' in source
    assert '"parent_face_far_field_analytic"' in source
    assert "incremental_update=incremental" in source
    assert "mesh_preview_times_s = np.linspace" in source
    assert "plot_mesh_projection" in source
    assert "plot_bedroom_scene_projection" in source
    assert "range_profile_from_cube" in source
    assert "range_doppler_map" in source
    assert "db_relative" in source
    assert "run_rt_reference = False" in source
    assert all(
        not cell.get("outputs") and cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    research_diagnostics = (
        "Visibility and Phase Consistency",
        "importlib.reload",
        "runtime_profile",
        "relative_norm_error",
        "doppler_time_spectrum",
        "range_time_map",
        "phase_step",
    )
    assert not [name for name in research_diagnostics if name in source]


def test_public_demo_inventory_contains_only_focused_tutorials():
    expected = {
        "README.md",
        "Point-Targets.ipynb",
        "Human-Only-PO.ipynb",
        "Full-RT.ipynb",
        "Hybrid-PO.ipynb",
    }
    actual = {
        path.name
        for path in (REPO_ROOT / "demo").iterdir()
        if path.is_file()
    }

    assert actual == expected


def test_public_demo_notebooks_are_output_free():
    for notebook_path in (REPO_ROOT / "demo").glob("*.ipynb"):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
        assert kernelspec.get("name") == "python3"
        assert ".venv" not in kernelspec.get("display_name", "")
        saved_cells = [
            index
            for index, cell in enumerate(notebook["cells"])
            if cell["cell_type"] == "code"
            and (cell.get("outputs") or cell.get("execution_count") is not None)
        ]
        assert not saved_cells, (
            f"{notebook_path.name} contains saved output in cells {saved_cells}"
        )


def test_hybrid_po_notebook_is_a_focused_api_tutorial():
    notebook_path = REPO_ROOT / "demo" / "Hybrid-PO.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "run_calibrated_hybrid_po" in source
    assert 'coupling_enabled = True' in source
    assert '"human_po"' in source
    assert '"static_environment_blocked"' in source
    assert '"human_env"' in source
    assert '"env_human"' in source
    assert "mesh_preview_times_s = np.linspace" in source
    assert "plot_mesh_projection" in source
    assert "plot_bedroom_scene_projection" in source
    assert "BEDROOM_SCENE_XML" in source
    assert "BEDROOM_SCENE_BOXES" in source
    assert "range_profile_from_cube" in source
    assert "range_doppler_map" in source
    assert "db_relative" in source
    assert "progress=False" in source
    assert all(
        not cell.get("outputs") and cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    research_diagnostics = (
        "Blocking and Coupling Diagnostics",
        "Component Cost Profile",
        "Optional Full RT Comparison",
        "importlib.reload",
        "runtime_profile",
        "runtime_summary",
        "relative_norm_error",
        "range_time_map",
        "calibration_gain_array",
        "run_full_scene_rt_comparison",
        "phase-continuity",
    )
    assert not [name for name in research_diagnostics if name in source]


def test_full_rt_notebook_is_a_focused_api_tutorial():
    notebook_path = REPO_ROOT / "demo" / "Full-RT.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert '"mobility_mode": "rt_retrace"' in source
    assert '"mobility_mode": "rt_coherent_bank"' in source
    assert "coherent_depths = (1, 2, 4)" in source
    assert '"retrace_period_chirps": 1' in source
    assert "plot_mesh_projection" in source
    assert "plot_bedroom_scene_projection" in source
    assert "BEDROOM_SCENE_XML" in source
    assert "BEDROOM_SCENE_BOXES" in source
    assert "range_profile_from_cube" in source
    assert "range_doppler_map" in source
    assert "db_relative" in source
    assert "display_floor_db = -80.0" in source
    assert "def to_db" not in source
    assert "cmu_asf_amc_base_smpl" not in source

    research_diagnostics = (
        "_hybrid_pipeline",
        "importlib.reload",
        "cube_matches_run",
        "path_depth_histogram",
        "human_touch_path_power",
        "range_time_map",
        "RTCoherentTransitionConfig",
    )
    assert not [name for name in research_diagnostics if name in source]


def test_public_data_contains_only_documented_pinned_motion_files():
    expected = {"README.md", *PINNED_MOTION_FILES}
    actual = {
        path.relative_to(AMASS_DATA_ROOT).as_posix()
        for path in AMASS_DATA_ROOT.rglob("*")
        if path.is_file()
    }

    assert actual == expected
    for name, pin in PINNED_MOTION_FILES.items():
        raw = (AMASS_DATA_ROOT / name).read_bytes()
        assert len(raw) == pin["size"]
        assert hashlib.sha256(raw).hexdigest() == pin["sha256"]


def test_cmu_walking_fixture_is_pickle_free_and_self_identifying():
    fixture = AMASS_DATA_ROOT / "walking_poses_cmu_105_02.npz"
    with np.load(fixture, allow_pickle=False) as archive:
        assert archive["poses"].shape == (1748, 72)
        assert archive["trans"].shape == (1748, 3)
        assert archive["betas"].shape == (10,)
        assert archive["source_dataset"].item() == "CMU"
        assert archive["source_subject"].item() == "105"
        assert archive["source_sequence"].item() == "105_02"
        assert archive["pose_source"].item() == "direct_profile_source_rotations"
        assert archive["gender"].item() == "neutral"
        assert archive["shape_method"].item() == "neutral_zero_public_fixture"
        assert np.count_nonzero(archive["betas"]) == 0
        assert (
            archive["public_fixture_note"].item()
            == "subject-specific SMPL shape and gender removed for public distribution"
        )
        assert archive["conversion_tool"].item() == "SparseSMPLFit"
        assert archive["conversion_tool_version"].item() == "0.1.0"
        assert archive["source_shape_method"].item() == (
            "asf_lengths+c3d_surface_markers"
        )
        assert archive["source_skeleton"].item() == "105.asf"
        assert archive["source_motion"].item() == "105_02.amc.txt"
        assert archive["source_sync_metadata"].item() == "105_02.c3d"
        assert json.loads(archive["retarget_metrics"].item())["frames"] == 1748.0
        assert archive["source_retarget_frame_count"].item() == 1748
        assert archive["output_frame_count"].item() == 1748
        assert (
            json.loads(archive["source_retarget_metrics"].item())["frames"]
            == 1748.0
        )
        assert (
            archive["source_asf_sha256"].item()
            == "de06a1ee5d917e4bd23461e22e49b43591ed8d9bd84a92d6b6927f423a972b30"
        )
        assert (
            archive["source_amc_sha256"].item()
            == "e9fb90488a44dcd2d2480022e320bd75e5cc71edcb7dc48cf316aec10e6c9074"
        )
        assert (
            archive["source_c3d_sha256"].item()
            == "cd9490c1408b99dd6838fe7bb3fe12c18d375372bbf81581ae61f3531fe6f206"
        )
        assert archive["mocap_framerate"].item() == 120.0
        assert archive["source_mocap_framerate"].item() == 120.0
        assert archive["resample_method"].item() == "none_source_rate_preserved"
        assert np.isclose(
            np.median(np.diff(archive["times"])),
            1.0 / 120.0,
        )
        assert np.array_equal(
            archive["coordinate_transform"],
            np.eye(3),
        )


def test_public_validation_bundles_have_exact_reviewed_inventory_and_hashes():
    expected = VALIDATION_TEXT_FILES | set(PINNED_VALIDATION_BINARY_FILES)
    actual = {
        path.relative_to(VALIDATION_DATA_ROOT).as_posix()
        for path in VALIDATION_DATA_ROOT.rglob("*")
        if path.is_file()
    }

    assert actual == expected
    for relative, pin in PINNED_VALIDATION_BINARY_FILES.items():
        raw = (VALIDATION_DATA_ROOT / relative).read_bytes()
        assert len(raw) == pin["size"]
        assert hashlib.sha256(raw).hexdigest() == pin["sha256"]


def test_public_validation_arrays_are_pickle_free_and_minimal():
    expected = {
        "mmradarpose_p1_an0_ac4_r0_frame0240": (1, 128, 64, 12),
        "rtpose_seq10_frame0020": (1, 64, 256, 192),
    }
    for bundle_name, adc_shape in expected.items():
        bundle = VALIDATION_DATA_ROOT / bundle_name
        with np.load(bundle / "radar_adc.npz", allow_pickle=False) as archive:
            assert set(archive.files) == {"adc"}
            assert archive["adc"].shape == adc_shape
            assert archive["adc"].dtype == np.complex64
            assert np.isfinite(archive["adc"]).all()

        with np.load(bundle / "amass_sequence.npz", allow_pickle=False) as archive:
            required = {
                "poses",
                "trans",
                "betas",
                "times",
                "source_capture_times",
                "bundle_times",
                "faces",
                "gender",
                "model_type",
                "frame_ids",
                "radar_frame_ids",
                "output_axes",
                "output_signs",
                "coordinate_frame",
                "profile",
                "source_sequence_id",
            }
            assert required <= set(archive.files)
            assert all(archive[name].dtype.kind != "O" for name in archive.files)
            assert archive["poses"].shape == (3, 72)
            assert archive["trans"].shape == (3, 3)
            assert archive["betas"].shape == (10,)
            assert archive["faces"].shape == (13776, 3)
            assert archive["times"].shape == (3,)
            assert archive["bundle_times"].shape == (3,)
            assert np.isfinite(archive["poses"]).all()
            assert np.isfinite(archive["trans"]).all()
            assert np.all(archive["betas"] == 0)
            assert archive["gender"].item() == "neutral"
            assert archive["model_type"].item() == "smpl"
            assert np.all(np.diff(archive["times"]) > 0)
            assert np.all(np.diff(archive["bundle_times"]) > 0)
            assert archive["bundle_times"][1] == 0.0

            if bundle_name.startswith("rtpose"):
                assert archive["source_sequence_id"].item() == "10"
                assert archive["selected_frame_id"].item() == "000056"
                assert archive["selected_radar_frame_id"].item() == "000020"
                assert archive["shape_method"].item() == "neutral_zero_public_fixture"
                assert "subject_id" not in archive.files
                assert "smpl_root_joint_offset" not in archive.files
            else:
                assert archive["source_sequence_id"].item() == "p1_an0_ac4_r0"
                assert archive["frame_ids"].tolist() == [
                    "000239",
                    "000240",
                    "000241",
                ]


def test_public_validation_metadata_is_portable_and_discloses_neutralization():
    for path in VALIDATION_DATA_ROOT.rglob("*.json"):
        raw = path.read_text(encoding="utf-8")
        assert "/" + "Users/" not in raw
        assert "file://" not in raw
        json.loads(raw)

    for sensor_path in VALIDATION_DATA_ROOT.glob("*/sensor.json"):
        sensor = json.loads(sensor_path.read_text(encoding="utf-8"))
        metadata = sensor["metadata"]
        assert metadata["motion_source"] == "amass_sequence.npz"
        assert metadata["neutral_shape"] is True
        assert metadata["amass_sequence_included"] is True
        assert metadata["mesh_sequence_included"] is False

    rt_metadata = json.loads(
        (VALIDATION_DATA_ROOT / "rtpose_seq10_frame0020/benchmark_metadata.json")
        .read_text(encoding="utf-8")
    )
    privacy = rt_metadata["camera_privacy"]
    assert privacy["method"] == "opaque solid face mask"
    assert privacy["irreversible_in_fixture"] is True
    assert privacy["original_images_included"] is False
    assert privacy["image_size_px"] == [1440, 1080]

    rt_sensor = json.loads(
        (VALIDATION_DATA_ROOT / "rtpose_seq10_frame0020/sensor.json").read_text(
            encoding="utf-8"
        )
    )
    rt_channel_metadata = rt_sensor["metadata"]
    assert rt_channel_metadata["source_rx_order"] == [
        13, 14, 15, 16, 1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8
    ]
    assert rt_channel_metadata["exported_rx_order"] == list(range(1, 17))
    assert rt_channel_metadata["rx_order"] == list(range(1, 17))
    assert rt_channel_metadata["velocity_resolution_mps"] > 0.0
    assert rt_channel_metadata["scene_path"] == (
        "environment_geometry/environment_scene.xml"
    )


def test_public_rtpose_images_are_face_occluded_pngs_at_source_resolution():
    camera_root = VALIDATION_DATA_ROOT / "rtpose_seq10_frame0020/camera"
    for side in ("left", "right"):
        path = camera_root / side / "000056_face_occluded.png"
        raw = path.read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        assert int.from_bytes(raw[16:20], "big") == 1440
        assert int.from_bytes(raw[20:24], "big") == 1080
        assert not (camera_root / side / "000056.png").exists()
