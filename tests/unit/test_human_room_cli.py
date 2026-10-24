# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""GUI-equivalent headless scene settings, overrides, and safe exports."""

from __future__ import annotations

from functools import wraps
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from mmWaveRadar.gui import scene_cli


def resolve(*arguments):
    return scene_cli.resolve_config(scene_cli.build_parser().parse_args(arguments))


def write_motion(path):
    tetrahedron = np.asarray(
        [[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.4]],
        dtype=np.float32,
    )
    np.savez(
        path,
        vertices=np.stack([tetrahedron + [i * 0.1, 0, 0] for i in range(6)]),
        faces=np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32),
        times=np.arange(6) * 0.05,
    )


@pytest.fixture
def execution(monkeypatch, tmp_path):
    motion = tmp_path / "motion.npz"
    write_motion(motion)
    monkeypatch.setattr(scene_cli, "_default_amass_motion_path", lambda: motion)
    calls, loads, exported = [], [], {}
    real_load = scene_cli.load_amass_motion

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return real_load(*args, **kwargs)

    @wraps(scene_cli.run_human_room_experiment)
    def run(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            solver=kwargs["solver"],
            adc=np.zeros((1, 1, 1, 1)),
        )

    def export(result, *, solver_results):
        exported.update(primary=result, results=solver_results)
        return BytesIO(b"bundle")

    monkeypatch.setattr(scene_cli, "load_amass_motion", load)
    monkeypatch.setattr(scene_cli, "run_human_room_experiment", run)
    monkeypatch.setattr(scene_cli, "_human_room_export", export)
    return SimpleNamespace(
        motion=motion,
        calls=calls,
        loads=loads,
        exported=exported,
        output=tmp_path / "output.zip",
    )


def test_current_gui_defaults_are_explicit_and_json_serializable():
    config = resolve()
    assert config["board_model"] == "IWR6843AOPEVM"
    assert config["selected_tx_indices"] == [0, 1, 2]
    assert config["selected_rx_indices"] == [0, 1, 2, 3]
    assert config["tx_schedule"] == "simultaneous"
    assert config["array_model"] == "phase_center"
    assert config["carrier_frequency_hz"] == 60e9
    assert config["slope_hz_per_s"] == 68e12
    assert config["num_adc_samples"] == 225
    assert config["num_chirps_per_frame"] == 16
    assert config["frame_period_s"] == 0.05
    assert config["antenna_pattern_mode"] == "cosine"
    assert config["human_position_m"] == [1.6, 0, 0]
    assert config["human_yaw_deg"] == 0
    assert config["radar_orientation_deg"] == [0, 0, 0]
    assert config["motion_start_s"] == 0
    assert config["motion_end_s"] is None and config["num_frames"] is None
    assert config["schema_version"] == 3
    assert config["solvers"] == ["hybrid_po"]
    assert config["hybrid_calibration"] == "rt_first_pose"
    assert config["rt_human_bounces"] == "single"
    assert config["rt_refresh_frames"] == 1 and config["po_refresh_frames"] == 1
    assert config["compute_backend"] == "auto"
    assert config["motion_source_up_axis"] is None
    assert all(config[key] == value for key, value in scene_cli._SOLVER_SETTINGS_DEFAULTS.items())
    assert json.loads(json.dumps(config)) == config


def test_radar_flags_use_gui_units_and_zero_based_antennas():
    config = resolve(
        "--board",
        "AWRL6844EVM",
        "--tx-indices",
        "0",
        "3",
        "--rx-indices",
        "1",
        "2",
        "--tx-schedule",
        "tdm",
        "--carrier-ghz",
        "60.5",
        "--slope-mhz-us",
        "50",
        "--chirp-duration-us",
        "60",
        "--chirp-repetition-us",
        "70",
        "--sampling-msps",
        "5",
        "--adc-mode",
        "complex2x",
        "--if-bandwidth-mhz",
        "2",
        "--adc-samples",
        "200",
        "--chirps",
        "32",
        "--frame-period-ms",
        "100",
        "--antenna-pattern",
        "none",
        "--beamwidth-deg",
        "70",
        "--radar-orientation",
        "10",
        "20",
        "30",
    )
    assert config["selected_tx_indices"] == [0, 3]
    assert config["selected_rx_indices"] == [1, 2]
    assert config["tx_schedule"] == "tdm"
    assert config["carrier_frequency_hz"] == 60.5e9
    assert config["slope_hz_per_s"] == 50e12
    assert config["chirp_duration_s"] == pytest.approx(60e-6)
    assert config["chirp_repetition_time_s"] == pytest.approx(70e-6)
    assert config["sampling_frequency_hz"] == 5e6
    assert config["adc_mode"] == "complex2x"
    assert config["if_bandwidth_hz"] == 2e6
    assert config["num_adc_samples"] == 200
    assert config["num_chirps_per_frame"] == 32
    assert config["frame_period_s"] == 0.1
    assert config["radar_orientation_deg"] == [10, 20, 30]
    assert resolve("--board", "AWRL6844EVM")["carrier_frequency_hz"] == 59e9
    assert resolve("--board", "MMWCAS_RF_EVM")["selected_tx_indices"] == [0, 1, 2, 9]


def test_phase_trigger_default_is_shared_and_json_overrides_it(execution, tmp_path):
    import inspect
    from mmWaveRadar.simulation import POConfig
    from validation.runner import BenchmarkRunConfig

    default = POConfig().po_refresh_on_phase_error_rad
    assert default == 0.1
    assert scene_cli._SOLVER_SETTINGS_DEFAULTS["po_refresh_on_phase_error_rad"] == default
    assert BenchmarkRunConfig().po_refresh_on_phase_error_rad == default
    assert (
        inspect.signature(scene_cli.run_human_room_experiment)
        .parameters["po_refresh_on_phase_error_rad"]
        .default
        == default
    )
    assert resolve()["po_refresh_on_phase_error_rad"] == default
    example = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/human_room.json").read_text()
    )
    assert example["po_refresh_on_phase_error_rad"] == default

    config_file = tmp_path / "phase-trigger.json"
    config_file.write_text(
        json.dumps(
            {
                "human_motion": str(execution.motion),
                "output": str(execution.output),
                "po_refresh_on_phase_error_rad": 0.037,
            }
        )
    )
    assert scene_cli.main(["--config", str(config_file)]) == 0
    assert execution.calls[0]["po_refresh_on_phase_error_rad"] == 0.037


def test_solver_and_coordinate_overrides():
    config = resolve(
        "--po-integration-mode",
        "parent_face_quadrature",
        "--po-curvature-error-rad",
        "0.01",
        "--po-quadrature-max-refinement-depth",
        "2",
        "--po-quadrature-max-subfaces-per-parent",
        "16",
        "--po-visibility-samples-per-face",
        "4",
        "--po-refresh-frames",
        "0",
        "--po-visibility-refresh-frames",
        "2",
        "--po-refresh-on-phase-error-rad",
        "0.037",
        "--po-visibility-fade-frames",
        "1",
        "--rt-samples-per-source",
        "200",
        "--rt-max-paths-per-source",
        "300",
        "--rt-max-depth",
        "2",
        "--rt-human-bounces",
        "any",
        "--rt-refresh-frames",
        "3",
        "--rt-visibility-refresh-frames",
        "0",
        "--rt-visibility-fade-frames",
        "2",
        "--hybrid-reflector-cutoff-db",
        "-30",
        "--hybrid-max-reflectors",
        "2",
        "--no-hybrid-coupling",
        "--hybrid-calibration",
        "none",
        "--source-up",
        "+y",
        "--source-forward=-z",
        "--source-handedness",
        "left",
        "--source-units",
        "cm",
        "--rt-human-diffuse-coefficient",
        "0.2",
        "--backend",
        "numpy",
    )
    assert config["po_quadrature_phase_span_scale_rad"] == 1
    assert config["po_curvature_error_rad"] == 0.01
    assert config["po_refresh_on_phase_error_rad"] == 0.037
    assert config["po_quadrature_max_refinement_depth"] == 2
    assert config["po_quadrature_max_subfaces_per_parent"] == 16
    assert config["po_visibility_samples_per_face"] == 4
    assert config["rt_samples_per_source"] == 200
    assert config["rt_max_paths_per_source"] == 300
    assert config["rt_max_depth"] == 2
    assert config["rt_human_bounces"] == "any"
    assert config["rt_refresh_frames"] == 3
    assert config["rt_visibility_refresh_frames"] == 0
    assert config["rt_visibility_fade_frames"] == 2
    assert config["po_refresh_frames"] == 0
    assert config["po_visibility_refresh_frames"] == 2
    assert config["po_visibility_fade_frames"] == 1
    assert config["hybrid_reflector_cutoff_db"] == -30
    assert config["hybrid_max_reflectors"] == 2
    assert config["hybrid_coupling"] is False
    assert config["hybrid_calibration"] == "none"
    assert config["motion_source_forward_axis"] == "-z"
    assert config["motion_source_length_unit"] == "cm"
    assert config["rt_human_diffuse_coefficient"] == 0.2
    assert resolve("--if-bandwidth-mhz", "0")["if_bandwidth_hz"] is None


def test_config_precedence_relative_paths_and_window_override(tmp_path):
    config_file = tmp_path / "run.json"
    config_file.write_text(
        json.dumps(
            {
                "human_motion": "inputs/motion.npz",
                "scene_xml": "room.xml",
                "smpl_model_dir": "models",
                "human_position_m": [2, 1, 0],
                "carrier_frequency_hz": 61e9,
                "tx_schedule": "tdm",
                "selected_tx_indices": [1],
                "motion_start_s": 1.0,
                "motion_end_s": 2.0,
            }
        )
    )
    config = resolve(
        "--config",
        str(config_file),
        "--position",
        "3",
        "0",
        "0.1",
        "--carrier-ghz",
        "62",
        "--tx-schedule",
        "simultaneous",
        "--start-s",
        "0.5",
    )
    assert config["human_motion"] == str(tmp_path / "inputs/motion.npz")
    assert config["scene_xml"] == str(tmp_path / "room.xml")
    assert config["smpl_model_dir"] == str(tmp_path / "models")
    assert config["human_position_m"] == [3, 0, 0.1]
    assert config["carrier_frequency_hz"] == 62e9
    assert config["tx_schedule"] == "simultaneous"
    assert config["selected_tx_indices"] == [1]
    assert config["motion_start_s"] == 0.5
    assert config["motion_end_s"] == 2.0 and config["num_frames"] is None
    # A command-line end replaces the file's end in either form.
    frames = resolve("--config", str(config_file), "--num-frames", "4")
    assert frames["num_frames"] == 4 and frames["motion_end_s"] is None
    assert resolve("--start-s", "3", "--end-s", "4")["motion_end_s"] == 4.0


def test_dry_run_writes_reusable_config_without_loading_motion(
    execution, tmp_path, monkeypatch, capsys
):
    def fail(*_args, **_kwargs):
        pytest.fail("dry run must not load motion")

    monkeypatch.setattr(scene_cli, "load_amass_motion", fail)
    path = tmp_path / "settings.json"
    assert scene_cli.main(["--dry-run", "--write-config", str(path)]) == 0
    config = json.loads(capsys.readouterr().out)
    assert json.loads(path.read_text()) == config
    assert resolve("--config", str(path)) == config
    assert execution.calls == []


def test_complete_example_matches_gui_defaults(monkeypatch):
    monkeypatch.delenv("MMWAVE_SMPL_MODEL_DIR", raising=False)
    public_root = Path(__file__).resolve().parents[2]
    path = public_root / "configs" / "human_room.json"
    raw = json.loads(path.read_text())
    assert set(raw) == set(scene_cli.default_config())
    config = resolve("--config", str(path))
    expected = resolve()
    assert config["output"] == str(public_root.parent / "results/tutorial-bedroom-bundle.zip")
    config["output"] = expected["output"]
    assert config == expected


def test_source_script_automatically_reads_config_from_any_directory(tmp_path):
    public_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(public_root / "tools/run_human_room.py"), "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    config = json.loads(completed.stdout)
    assert config["board_model"] == "IWR6843AOPEVM"
    assert config["human_motion"] == str(public_root / "data/AMASS/walking_poses_cmu_105_02.npz")
    assert config["output"] == str(public_root.parent / "results/tutorial-bedroom-bundle.zip")


def test_entire_run_including_output_comes_from_one_file(execution, tmp_path):
    path = tmp_path / "single-file.json"
    path.write_text(
        json.dumps(
            {
                "human_motion": "motion.npz",
                "board_model": "AWRL6844EVM",
                "carrier_frequency_hz": 59e9,
                "human_position_m": [2.2, 0.3, 0.1],
                "human_yaw_deg": 20,
                "motion_start_s": 0.1,
                "num_frames": 2,
                "solvers": ["po"],
                "output": "from-config/run.zip",
                "quiet": True,
                "force": False,
            }
        )
    )
    assert scene_cli.main(default_config_path=path, argv=[]) == 0
    call = execution.calls[0]
    assert call["board_model"] == "AWRL6844EVM"
    assert call["selected_tx_indices"] == [0, 1, 2, 3]
    assert call["carrier_frequency_hz"] == 59e9
    assert call["human_position_m"] == [2.2, 0.3, 0.1]
    assert call["motion_start_s"] == 0.1 and call["num_frames"] == 2
    assert call["motion_end_s"] is None
    assert (tmp_path / "from-config/run.zip").read_bytes() == b"bundle"
    assert all(key not in call for key in ("output", "quiet", "force"))


def test_explicit_config_and_flags_override_default_file(execution, tmp_path):
    default = tmp_path / "default.json"
    alternate = tmp_path / "alternate.json"
    default.write_text(json.dumps({"human_position_m": [1, 0, 0]}))
    alternate.write_text(
        json.dumps(
            {
                "human_motion": "motion.npz",
                "human_position_m": [2, 0, 0],
                "output": "alternate.zip",
                "quiet": True,
            }
        )
    )
    assert (
        scene_cli.main(
            ["--config", str(alternate), "--yaw", "15", "--output", str(execution.output)],
            default_config_path=default,
        )
        == 0
    )
    assert execution.calls[0]["human_position_m"] == [2, 0, 0]
    assert execution.calls[0]["human_yaw_deg"] == 15
    assert execution.output.read_bytes() == b"bundle"
    assert not (tmp_path / "alternate.zip").exists()


def test_config_overwrite_policy_can_be_disabled_by_flag(execution, tmp_path):
    config = tmp_path / "overwrite.json"
    config.write_text(
        json.dumps(
            {
                "human_motion": "motion.npz",
                "output": "output.zip",
                "force": True,
                "quiet": True,
            }
        )
    )
    execution.output.write_bytes(b"previous")
    with pytest.raises(SystemExit):
        scene_cli.main(["--config", str(config), "--no-force"])
    assert execution.loads == []
    assert execution.output.read_bytes() == b"previous"
    assert scene_cli.main(["--config", str(config)]) == 0
    assert execution.output.read_bytes() == b"bundle"


def test_all_selected_solvers_share_motion_and_export_together(execution):
    assert (
        scene_cli.main(
            [
                "--position",
                "2",
                "0.3",
                "0.1",
                "--yaw",
                "30",
                "--start-s",
                "0.1",
                "--solvers",
                "rt",
                "po",
                "hybrid_po",
                "po",
                "--output",
                str(execution.output),
                "--quiet",
            ]
        )
        == 0
    )
    assert len(execution.loads) == 1
    assert [call["solver"] for call in execution.calls] == ["rt", "po", "hybrid_po"]
    for call in execution.calls:
        assert call["human_motion"] is execution.calls[0]["human_motion"]
        assert call["human_position_m"] == [2, 0.3, 0.1]
        assert call["human_yaw_deg"] == 30
        assert call["motion_start_s"] == 0.1
        assert call["selected_tx_indices"] == [0, 1, 2]
        assert call["compute_backend"] == "auto"
    assert execution.exported["primary"] is execution.exported["results"]["hybrid_po"]
    assert execution.output.read_bytes() == b"bundle"


def test_scene_override_keeps_radar_and_uses_validated_xml(execution, tmp_path):
    scene = tmp_path / "different-room.xml"
    payload = b'<scene version="2.1.0"/>'
    scene.write_bytes(payload)
    assert (
        scene_cli.main(
            [
                "--scene",
                str(scene),
                "--start-s",
                "0.05",
                "--end-s",
                "0.2",
                "--output",
                str(execution.output),
                "--quiet",
            ]
        )
        == 0
    )
    call = execution.calls[0]
    assert call["scene_xml"] == payload
    assert call["scene_xml_filename"] == scene.name
    assert call["motion_start_s"] == 0.05
    assert call["motion_end_s"] == 0.2
    assert call["board_model"] == "IWR6843AOPEVM"
    assert call["carrier_frequency_hz"] == 60e9


@pytest.mark.parametrize(
    "arguments",
    [
        ["--start-s", "-1"],
        ["--start-s", "nan"],
        ["--end-s", "0.1", "--num-frames", "2"],
        ["--start-s", "3", "--end-s", "2"],
        ["--start-s", "0", "--end-s", "0.0005"],
        ["--num-frames", "0"],
        ["--position", "nan", "0", "0"],
        ["--source-up", "+z"],
        ["--source-up", "+z", "--source-forward=-z"],
        ["--source-units", "cm"],
        ["--tx-indices", "-1"],
        ["--rx-indices", "4"],
        ["--tx-indices", "0", "0"],
        ["--po-quadrature-max-refinement-depth", "-1"],
        ["--hybrid-max-reflectors", "5"],
        ["--rt-human-diffuse-coefficient", "2"],
        ["--yaw", "inf"],
        ["--rt-refresh-frames", "-1"],
        ["--po-visibility-fade-frames", "101"],
        ["--array-model", "element"],
    ],
)
def test_invalid_flags_fail_before_loading(execution, arguments):
    with pytest.raises(SystemExit) as error:
        scene_cli.main([*arguments, "--output", str(execution.output)])
    assert error.value.code == 2
    assert execution.calls == []
    assert execution.loads == []


@pytest.mark.parametrize(
    "contents",
    [
        [],
        {"unknown_setting": 2},
        {"schema_version": 4},
        {"schema_version": True},
        {"schema_version": 1},
        {"schema_version": 2},
        {"begin_frame_index": 0},
        {"motion_end_s": "1"},
        {"num_frames": 0},
        {"motion_end_s": 1.0, "num_frames": 2},
        {"tx_schedule": False},
        {"tx_schedule": "round_robin"},
        {"num_adc_samples": 225.5},
        {"selected_tx_indices": [True]},
        {"human_position_m": [1, 2]},
        {"if_bandwidth_hz": False},
        {"solvers": []},
        {"solvers": ["full_rt"]},
        {"scene_xml": 5},
        {"rt_human_bounces": "one_target_bounce"},
        {"hybrid_calibration": "first_pose"},
        {"array_model": "element"},
        {"rt_refresh_frames": 1.5},
        {"output": None},
        {"quiet": "false"},
        {"force": 1},
    ],
)
def test_invalid_json_configuration_is_rejected(tmp_path, contents):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(contents))
    with pytest.raises(SystemExit) as error:
        scene_cli.main(["--config", str(path), "--dry-run"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("removed", "replacement"),
    [
        (["--modes", "full_rt"], "--solvers"),
        (["--tdm"], "--tx-schedule tdm"),
        (["--no-coupling"], "--no-hybrid-coupling"),
        (["--human-diffuse", "0.2"], "--rt-human-diffuse-coefficient"),
        (["--rt-samples", "200"], "--rt-samples-per-source"),
        (["--coupling-cutoff-db", "-30"], "--hybrid-reflector-cutoff-db"),
        (["--po-integration", "face_centroid"], "--po-integration-mode"),
        (["--tx", "1"], "--tx-indices, which is zero-based (TI's TX1 is 0)"),
        (["--rx", "1"], "--rx-indices, which is zero-based (TI's RX1 is 0)"),
    ],
)
def test_removed_flags_name_their_replacement(execution, capsys, removed, replacement):
    with pytest.raises(SystemExit) as error:
        scene_cli.main([*removed, "--output", str(execution.output)])
    assert error.value.code == 2
    assert f"{removed[0]} was removed; use {replacement}" in capsys.readouterr().err
    assert execution.calls == []


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            {"schema_version": 2, "simulation_modes": ["hybrid_po"], "tdm_enabled": False},
            "simulation_modes was renamed to solvers; tdm_enabled was renamed to tx_schedule",
        ),
        ({"coupling_enabled": True}, "coupling_enabled was renamed to hybrid_coupling"),
        (
            {"diffuse_reflection_coefficient": 0.3},
            "diffuse_reflection_coefficient was renamed to rt_human_diffuse_coefficient",
        ),
        ({"schema_version": 2}, "set schema_version to 3"),
    ],
)
def test_old_configuration_keys_name_their_replacement(tmp_path, capsys, contents, message):
    path = tmp_path / "old.json"
    path.write_text(json.dumps(contents))
    with pytest.raises(SystemExit):
        scene_cli.main(["--config", str(path), "--dry-run"])
    assert message in capsys.readouterr().err


def test_output_guard_force_and_failed_run_preserve_existing_file(execution, monkeypatch):
    execution.output.write_bytes(b"previous bundle")
    with pytest.raises(SystemExit):
        scene_cli.main(["--output", str(execution.output)])
    assert execution.loads == []

    @wraps(scene_cli.run_human_room_experiment)
    def fail(**_kwargs):
        raise RuntimeError("solver failure")

    original = scene_cli.run_human_room_experiment
    monkeypatch.setattr(scene_cli, "run_human_room_experiment", fail)
    arguments = ["--output", str(execution.output), "--force", "--quiet"]
    assert scene_cli.main(arguments) == 1
    assert execution.output.read_bytes() == b"previous bundle"
    monkeypatch.setattr(scene_cli, "run_human_room_experiment", original)
    assert scene_cli.main(arguments) == 0
    assert execution.output.read_bytes() == b"bundle"
    assert not list(execution.output.parent.glob(".hermes-scene-*"))


def test_input_and_frame_bounds_are_protected(execution):
    with pytest.raises(SystemExit):
        scene_cli.main(["--output", str(execution.motion), "--force"])
    # The fixture motion lasts 0.25 s; two frames from 0.2 s would end at 0.251 s.
    arguments = ["--start-s", "0.2", "--num-frames", "2", "--output", str(execution.output)]
    assert scene_cli.main(arguments) == 1
    assert execution.calls == []
    assert not execution.output.exists()


@pytest.mark.parametrize("custom_scene", [False, True])
def test_real_mesh_po_run_exports_selected_frame_and_radar_metadata(tmp_path, custom_scene):
    pytest.importorskip("sionna.rt")
    motion = tmp_path / "mesh.npz"
    output = tmp_path / "real.zip"
    write_motion(motion)
    arguments = [
        "--motion",
        str(motion),
        "--smpl-model-dir",
        str(tmp_path / "no-model"),
        "--position",
        "2",
        "0.3",
        "0.1",
        "--yaw",
        "30",
        "--start-s",
        "0.1",
        "--solvers",
        "po",
        "--backend",
        "numpy",
        "--tx-indices",
        "1",
        "--rx-indices",
        "0",
        "3",
        "--output",
        str(output),
        "--quiet",
    ]
    payload = b"""<scene version="2.1.0">
  <bsdf type="itu-radio-material" id="mat-concrete"><string name="type" value="concrete"/></bsdf>
  <shape type="cube" id="floor">
    <transform name="to_world"><scale x="3" y="3" z="0.04"/><translate x="2" y="0" z="-1.04"/></transform>
    <ref name="bsdf" id="mat-concrete"/>
  </shape>
</scene>"""
    if custom_scene:
        scene = tmp_path / "custom.xml"
        scene.write_bytes(payload)
        # Exercise a real, file-only custom-scene run (including output settings).
        settings = resolve(*arguments, "--scene", str(scene))
        configuration = tmp_path / "run.json"
        configuration.write_text(json.dumps(settings))
        arguments = ["--config", str(configuration)]
    assert scene_cli.main(arguments) == 0
    with zipfile.ZipFile(output) as bundle:
        settings = json.loads(bundle.read("environment.json"))
        sensor = json.loads(bundle.read("sensor.json"))
        descriptor = json.loads(bundle.read("bundle.json"))
        frames = json.loads(bundle.read("frames.json"))["frames"]
        assert descriptor["primary_solver"] == "po"
        assert descriptor["adcs"]["po"]["path"] == "simulated_adc_po.npz"
        assert sensor["tx_schedule"] == "simultaneous"
        assert settings["human_position_m"] == [2, 0.3, 0.1]
        assert settings["human_yaw_deg"] == 30
        assert settings["motion_time_window"]["start_s"] == pytest.approx(0.1)
        assert len(frames) == 1
        assert sensor["tx_indices_zero_based"] == [1]
        assert sensor["rx_indices_zero_based"] == [0, 3]
        if custom_scene:
            assert bundle.read("scene.xml") == payload
            assert settings["prepared_scene"] is None
            assert settings["scene_source"] == "custom.xml"
        with np.load(BytesIO(bundle.read("radar_adc.npz"))) as data:
            assert data["adc"].shape == (1, 16, 225, 2)
            assert np.isfinite(data["adc"]).all()


def test_po_geometry_and_visibility_settings_accept_json_and_cli_overrides(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "po_refresh_on_area_fraction": 0.3,
                "po_refresh_on_normal_deg": 20.0,
                "po_visibility_samples_per_face": 16,
                "po_adaptive_visibility_sampling": False,
                "po_adaptive_visibility_edge_margin": 0.2,
            }
        )
    )
    config = resolve(
        "--config",
        str(path),
        "--po-refresh-on-normal-deg",
        "5",
        "--po-adaptive-visibility-sampling",
    )
    assert config["po_refresh_on_area_fraction"] == 0.3
    assert config["po_refresh_on_normal_deg"] == 5.0
    assert config["po_visibility_samples_per_face"] == 16
    assert config["po_adaptive_visibility_sampling"] is True
    assert config["po_adaptive_visibility_edge_margin"] == 0.2
