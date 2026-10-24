# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Preparation CLI, portability, feature gates, and capture conventions."""

from dataclasses import replace
from importlib import resources
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from validation import simulation as _validation_simulation

import numpy as np
import pytest

from bundle_prepare import Bundle, BundleSensorConfig, BundleWriter
from bundle_prepare.cli import main, prepare_from_config
from bundle_prepare.contract import make_bundle_descriptor
from bundle_prepare.integrity import check_bundle_integrity, collect_bundle_errors
from mmWaveRadar.measurements import load_bundle


def example(root):
    root.mkdir()
    for item in resources.files("bundle_prepare").joinpath("data", "static_target").iterdir():
        if item.is_file():
            (root / item.name).write_bytes(item.read_bytes())
    return root / "prepare.json"


@pytest.mark.parametrize("suffix", ["", ".zip"])
def test_config_prepares_relocatable_bundle_from_another_cwd(tmp_path, monkeypatch, suffix):
    config = example(tmp_path / "inputs")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / f"bundle{suffix}"
    report = prepare_from_config(config, output=output)
    shutil.rmtree(config.parent)
    moved = tmp_path / "relocated" / output.name
    moved.parent.mkdir()
    output.rename(moved)
    loaded = load_bundle(moved)
    assert report["valid"] and loaded.data_origin == "simulation"
    assert loaded.can_resimulate
    assert check_bundle_integrity(moved)["valid"]
    assert loaded.bundle.scene_path.is_file()
    assert loaded.bundle.motion_path is None


def test_cli_packaged_example_and_channel_table(tmp_path, capsys):
    root = tmp_path / "example"
    assert main(["--example", str(root)]) == 0
    assert main(["--config", str(root / "prepare.json")]) == 0
    assert (root / "static-target.zip").is_file()
    capsys.readouterr()
    assert main(["--channel-table", "IWR6843AOP"]) == 0
    table = json.loads(capsys.readouterr().out)
    assert len(table["channels"]) == 12
    assert {c["phase_sign"] for c in table["channels"]} == {-1, 1}


@pytest.mark.parametrize("suffix", ["", ".zip"])
def test_direct_cli_matches_config_and_is_portable_from_another_cwd(
    tmp_path, monkeypatch, capsys, suffix
):
    config = example(tmp_path / "capture inputs")
    settings = json.loads(config.read_text())
    expected = tmp_path / "from-config"
    prepare_from_config(config, output=expected)
    monkeypatch.chdir(tmp_path)
    output = Path(f"from-cli{suffix}")
    assert (
        main(
            [
                "--adc",
                "capture inputs/radar_adc.npz",
                "--sensor",
                "capture inputs/sensor.json",
                "--scene",
                "capture inputs/scene.xml",
                "--target-mesh",
                "capture inputs/target_mesh.npz",
                "--target-material",
                "capture inputs/material.json",
                "--frame-times-s",
                "0",
                "--origin",
                settings["origin"],
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["path"] == str(tmp_path / output)
    shutil.rmtree(config.parent)
    actual = load_bundle(output)
    reference = Bundle.load(expected)
    assert actual.can_resimulate
    assert actual.bundle.descriptor == reference.descriptor
    assert actual.bundle.sensor_config.to_json() == reference.sensor_config.to_json()
    np.testing.assert_array_equal(actual.bundle.real_adc, reference.real_adc)
    np.testing.assert_array_equal(actual.bundle.times, reference.times)
    assert actual.bundle.scene_path.read_bytes() == reference.scene_path.read_bytes()


def test_direct_cli_defaults_to_measurement_and_normalizes_capture_and_background(
    tmp_path, capsys
):
    config = example(tmp_path / "inputs")
    sensor = BundleSensorConfig.from_json(config.parent / "sensor.json")
    signs = sensor.to_sensor().hardware.virtual_channel_phase_signs
    with np.load(config.parent / "radar_adc.npz") as archive:
        canonical = archive["adc"].copy()
    decoded = (canonical / signs).conj()
    np.savez(config.parent / "decoded.npz", adc=decoded)
    np.savez(config.parent / "background.npz", adc=decoded[0])
    output = tmp_path / "normalized"
    assert (
        main(
            [
                "--adc",
                str(config.parent / "decoded.npz"),
                "--sensor",
                str(config.parent / "sensor.json"),
                "--background-adc",
                str(config.parent / "background.npz"),
                "--iq-transform",
                "conjugate",
                "--phase-signs-removed",
                "--adc-units",
                "V",
                "--timestamp-tolerance-s",
                "0.000001",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    bundle = Bundle.load(output)
    assert bundle.data_origin == "measurement"
    assert bundle.sensor_config.adc_units == "V"
    assert bundle.sensor_config.timestamp_tolerance_s == 1e-6
    np.testing.assert_allclose(bundle.real_adc, canonical)
    np.testing.assert_allclose(bundle.background_adc, canonical[0])


@pytest.mark.parametrize("mode", ["frames", "frame-times", "period"])
def test_direct_cli_frame_timing_modes(tmp_path, mode):
    config = example(tmp_path / "inputs")
    with np.load(config.parent / "radar_adc.npz") as archive:
        np.savez(config.parent / "two-frames.npz", adc=np.repeat(archive["adc"], 2, axis=0))
    records = {
        "frames": [{"bundle_index": i, "motion_time_s": t} for i, t in enumerate([2.0, 2.1])]
    }
    (config.parent / "frames.json").write_text(json.dumps(records))
    timing = {
        "frames": ["--frames", str(config.parent / "frames.json")],
        "frame-times": ["--frame-times-s", "2", "2.1"],
        "period": ["--start-time-s", "2", "--frame-period-s", "0.1"],
    }[mode]
    output = tmp_path / "bundle"
    assert (
        main(
            [
                "--adc",
                str(config.parent / "two-frames.npz"),
                "--sensor",
                str(config.parent / "sensor.json"),
                "--output",
                str(output),
                *timing,
            ]
        )
        == 0
    )
    bundle = Bundle.load(output)
    assert [frame.motion_time_s for frame in bundle.frames] == pytest.approx([2, 2.1])
    np.testing.assert_allclose(bundle.times[:, 0], [0, 0.1])


def test_direct_cli_imports_ti_config_without_a_sensor_json(tmp_path):
    ti = tmp_path / "radar.cfg"
    ti.write_text("""channelCfg 5 5 0
adcCfg 2 1
profileCfg 0 60 7 4 40 0 0 20 1 64 2000 0 0 30
chirpCfg 0 0 0 0 0 0 0 4
chirpCfg 1 1 0 0 0 0 0 1
frameCfg 0 1 4 0 50 1 0
""")
    np.savez(tmp_path / "adc.npz", adc=np.ones((1, 4, 64, 4), dtype=complex))
    output = tmp_path / "bundle"
    assert (
        main(
            [
                "--adc",
                str(tmp_path / "adc.npz"),
                "--ti-config",
                str(ti),
                "--board-model",
                "IWR6843AOP",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    sensor = Bundle.load(output).sensor_config
    assert sensor.fmcw.carrier_frequency == pytest.approx(60.08e9)
    assert sensor.tdm_tx_order == (2, 0)
    assert sensor.to_sensor().hardware.num_virtual_channels == 4


def test_direct_motion_requires_world_frame_confirmation(tmp_path, capsys):
    config = example(tmp_path / "inputs")
    with np.load(config.parent / "target_mesh.npz") as target:
        np.savez(
            config.parent / "motion.npz",
            vertices=np.repeat(target["vertices"][None], 2, axis=0),
            faces=target["faces"],
            times=[0, 0.1],
        )
    output = tmp_path / "bundle"
    args = [
        "--adc",
        str(config.parent / "radar_adc.npz"),
        "--sensor",
        str(config.parent / "sensor.json"),
        "--evaluated-mesh",
        str(config.parent / "motion.npz"),
        "--frame-times-s",
        "0",
        "--output",
        str(output),
    ]
    assert main(args) == 1
    assert "motion_coordinate_frame" in capsys.readouterr().err
    assert not output.exists()
    assert main([*args, "--motion-coordinate-frame", "world"]) == 0
    assert Bundle.load(output).descriptor["motion"]["coordinate_frame"] == "world"


@pytest.mark.parametrize(
    "args, message",
    [
        (["--adc", "adc.npz", "--sensor", "sensor.json"], "--output is required"),
        (["--adc", "adc.npz", "--output", "bundle.zip"], "requires --sensor or --ti-config"),
        (
            ["--adc", "adc.npz", "--ti-config", "radar.cfg", "--output", "bundle"],
            "requires --board-model",
        ),
        (
            ["--adc", "a", "--sensor", "s", "--board-model", "IWR6843AOP", "--output", "b"],
            "only used with --ti-config",
        ),
        (["--config", "prepare.json", "--scene", "room.xml"], "cannot be combined with --config"),
        (
            ["--example", "example", "--origin", "measurement"],
            "Direct input arguments require --adc",
        ),
        (["--config", "prepare.json", "--adc", "adc.npz"], "not allowed with argument --config"),
        (
            ["--adc", "a", "--sensor", "s", "--ti-config", "cfg", "--output", "b"],
            "not allowed with argument --sensor",
        ),
        (
            [
                "--adc",
                "a",
                "--sensor",
                "s",
                "--output",
                "b",
                "--frames",
                "f",
                "--frame-times-s",
                "0",
            ],
            "not allowed with argument --frames",
        ),
        (
            [
                "--adc",
                "a",
                "--sensor",
                "s",
                "--output",
                "b",
                "--frame-times-s",
                "0",
                "--frame-period-s",
                ".1",
            ],
            "--frame-period-s cannot be combined",
        ),
    ],
)
def test_direct_cli_rejects_incomplete_or_conflicting_modes(args, message, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_direct_cli_force_preserves_existing_bundle_when_validation_fails(tmp_path):
    config = example(tmp_path / "inputs")
    output = tmp_path / "bundle.zip"
    args = [
        "--adc",
        str(config.parent / "radar_adc.npz"),
        "--sensor",
        str(config.parent / "sensor.json"),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    before = output.read_bytes()
    np.savez(config.parent / "radar_adc.npz", adc=np.ones((1, 4, 64, 11), dtype=complex))
    assert main([*args, "--force"]) == 1
    assert output.read_bytes() == before


def test_direct_cli_refuses_to_replace_its_input_directory(tmp_path):
    config = example(tmp_path / "inputs")
    before = (config.parent / "radar_adc.npz").read_bytes()
    assert (
        main(
            [
                "--adc",
                str(config.parent / "radar_adc.npz"),
                "--sensor",
                str(config.parent / "sensor.json"),
                "--output",
                str(config.parent),
                "--force",
            ]
        )
        == 1
    )
    assert (config.parent / "radar_adc.npz").read_bytes() == before


def test_failed_force_preparation_preserves_existing_output(tmp_path):
    config = example(tmp_path / "inputs")
    output = tmp_path / "bundle.zip"
    prepare_from_config(config, output=output)
    before = output.read_bytes()
    np.savez(config.parent / "radar_adc.npz", adc=np.ones((1, 4, 64, 11), dtype=complex))
    with pytest.raises(ValueError, match="channel"):
        prepare_from_config(config, output=output, force=True)
    assert output.read_bytes() == before


def test_writer_copies_and_rewrites_xml_obj_material_texture_graph(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.obj").write_text("mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (source / "material.mtl").write_text("newmtl sample\nmap_Kd texture.png\n")
    (source / "texture.png").write_bytes(b"test-texture")
    (source / "scene.xml").write_text(
        '<scene version="2.1.0"><shape type="obj"><string name="filename" value="a.obj"/></shape></scene>'
    )
    with (
        pytest.raises(RuntimeError, match="staging inspection complete"),
        BundleWriter() as writer,
    ):
        scene = writer.add_scene(source / "scene.xml")
        root = writer.root
        obj_name = ET.parse(root / scene["file"]).find(".//string").attrib["value"]
        obj = root / "scene" / obj_name
        mtl = obj.parent / obj.read_text().splitlines()[0].split()[1]
        texture = mtl.parent / mtl.read_text().splitlines()[1].split()[1]
        assert texture.read_bytes() == b"test-texture"
        assert len(scene["assets"]) == 3
        # Aborting a staging-only operation must not attempt publication.
        raise RuntimeError("staging inspection complete")


def test_custom_array_and_tdm_subset_preserve_physical_labels_and_slot_timing(
    tmp_path, monkeypatch
):
    config = example(tmp_path / "inputs")
    base = BundleSensorConfig.from_json(config.parent / "sensor.json")
    sensor = replace(
        base,
        board_model=None,
        array={
            "tx_positions_m": [[0, 0, 0], [0, 0.0025, 0], [0, 0.005, 0]],
            "rx_positions_m": [[0, 0, 0], [0, 0.0025, 0]],
            "phase_signs": [1, -1, 1, -1, 1, -1],
        },
        fmcw=replace(base.fmcw, tx_schedule="tdm"),
        tx_indices_zero_based=(0, 2),
        tdm_tx_order=(2, 1, 0),
        channel_map=((2, 1), (2, 0), (0, 1), (0, 0)),
    )
    output = tmp_path / "custom"
    with BundleWriter(output) as writer:
        writer.write_json("sensor.json", sensor.to_json())
        writer.write_json("frames.json", [{"motion_time_s": 0.0}])
        writer.write_npz("radar_adc.npz", adc=np.ones((1, 4, 64, 4), dtype=complex))
        writer.copy_file(config.parent / "target_mesh.npz", "target.npz")
        descriptor = make_bundle_descriptor(primary_origin="measurement")
        descriptor["target"] = {"mesh": "target.npz"}
        writer.write_json("bundle.json", descriptor)
    clip = Bundle.load(output)
    assert clip.descriptor["schema_version"] == 2
    assert clip.sensor_config.fmcw.slow_time_interval == pytest.approx(150e-6)
    assert clip.sensor().fmcw.slow_time_interval == pytest.approx(150e-6)
    assert clip.sensor().hardware.num_tx == 2
    from validation import runner

    selection = runner._subarray_selection(clip, runner.BenchmarkRunConfig(), full_channels=4)
    assert selection.channel_tx_indices_1based.tolist() == [3, 3, 1, 1]
    assert selection.channel_rx_indices_1based.tolist() == [2, 1, 2, 1]
    simultaneous, order = runner._simultaneous_tdm_sensor(clip, selection)
    assert simultaneous.fmcw.slow_time_interval == pytest.approx(150e-6)
    assert simultaneous.fmcw.tx_schedule == "simultaneous"
    import sionna.rt

    monkeypatch.setattr(sionna.rt, "load_scene", lambda *args: object())
    monkeypatch.setattr(_validation_simulation, "validation_simulator", lambda *args: object())
    from mmWaveRadar.targets import MeshSequence
    from bundle_prepare.contract import load_static_target_mesh

    vertices, faces = load_static_target_mesh(clip.target_mesh_path)
    moving = MeshSequence(np.stack([vertices, vertices + [0, 0.1, 0]]), faces, np.array([0, 0.1]))
    monkeypatch.setattr(_validation_simulation, "simulation_mesh_sequence", lambda *args: moving)
    observed = []

    def run(simulator, scene, runtime, target, count, settings, solver):
        observed.append(
            [target.mesh_sequence.vertices_at(t)[0, 1] - vertices[0, 1] for t in (0.0, 1.0)]
        )
        label = int(runtime.name.rsplit("tx", 1)[1])
        return SimpleNamespace(adc=np.full((1, 4, 64, 2), label, dtype=complex))

    monkeypatch.setattr(_validation_simulation, "run_observed_simulator", run)
    adc = runner._simulate_adc(clip, runner.BenchmarkRunConfig(), solver="po")
    assert adc[0, 0, 0].tolist() == [3, 3, 1, 1]
    assert observed[0][0] == 0
    assert observed[1][0] == pytest.approx(100e-6, abs=1e-7)
    assert observed[0][1] == pytest.approx(150e-6, abs=1e-7)


@pytest.mark.parametrize("feature", ["future_feature", "custom_array"])
def test_unknown_and_undeclared_features_fail_before_arrays_are_loaded(tmp_path, feature):
    config = example(tmp_path / "inputs")
    output = tmp_path / "bundle"
    prepare_from_config(config, output=output)
    descriptor = json.loads((output / "bundle.json").read_text())
    descriptor.update(schema_version=2, required_features=[feature])
    (output / "bundle.json").write_text(json.dumps(descriptor))
    if feature == "custom_array":
        sensor = json.loads((output / "sensor.json").read_text())
        sensor["fmcw"]["tdm_tx_order"] = [0, 1, 2]
        sensor["fmcw"]["tx_schedule"] = "tdm"
        (output / "sensor.json").write_text(json.dumps(sensor))
    with pytest.raises(ValueError, match="required_features"):
        Bundle.load(output)


@pytest.mark.parametrize("version", [1, 2])
def test_evaluated_motion_coverage_is_checked_without_smpl(tmp_path, version):
    config = example(tmp_path / "inputs")
    with np.load(config.parent / "target_mesh.npz") as target:
        vertices, faces = target["vertices"], target["faces"]
        np.savez(
            config.parent / "motion.npz",
            vertices=np.repeat(vertices[None], 2, axis=0),
            faces=faces,
            times=[0.0, 0.1],
        )
    settings = json.loads(config.read_text())
    settings["evaluated_mesh"] = "motion.npz"
    settings["motion_coordinate_frame"] = "world"
    config.write_text(json.dumps(settings))
    output = tmp_path / "bundle"
    prepare_from_config(config, output=output)
    # Emulate an older export; new writers reject this incomplete timeline.
    np.savez(output / "motion.npz", vertices=vertices[None], faces=faces, times=[0.0])
    if version == 1:
        descriptor = json.loads((output / "bundle.json").read_text())
        descriptor.update(schema_version=1)
        descriptor.pop("required_features", None)
        descriptor["target"].pop("material_parameters", None)
        descriptor["motion"].pop("coordinate_frame", None)
        (output / "bundle.json").write_text(json.dumps(descriptor))
    loaded = load_bundle(output)
    assert not loaded.can_resimulate
    assert "full radar acquisition" in loaded.bundle.resimulation_issues[0]
    assert loaded.primary_products().adc.shape == (1, 4, 64, 12)


def test_conflicting_adc_and_frame_timestamps_are_rejected(tmp_path):
    config = example(tmp_path / "inputs")
    path = config.parent / "radar_adc.npz"
    with np.load(path) as data:
        adc = data["adc"]
    np.savez(path, adc=adc, times=np.arange(4)[None] * 0.001)
    with pytest.raises(ValueError, match="timestamps disagree"):
        prepare_from_config(config, output=tmp_path / "bundle.zip")


def test_ti_import_uses_sampled_carrier_rx_mask_and_physical_tdm_order(tmp_path):
    source = tmp_path / "radar.cfg"
    source.write_text("""channelCfg 5 5 0
adcCfg 2 1
profileCfg 0 60 7 4 40 0 0 20 1 64 2000 0 0 30
chirpCfg 0 0 0 0 0 0 0 4
chirpCfg 1 1 0 0 0 0 0 1
frameCfg 0 1 4 0 50 1 0
""")
    sensor = BundleSensorConfig.from_ti_cli_config(source, board_model="IWR6843AOP")
    assert sensor.fmcw.carrier_frequency == pytest.approx(60.08e9)
    assert sensor.rx_indices_zero_based == (0, 2)
    assert sensor.tdm_tx_order == (2, 0)
    assert sensor.to_sensor().hardware.num_virtual_channels == 4
    assert sensor.fmcw.slow_time_interval == pytest.approx(94e-6)


def test_iq_and_phase_normalization_restore_canonical_capture_and_background(tmp_path):
    config = example(tmp_path / "inputs")
    sensor = BundleSensorConfig.from_json(config.parent / "sensor.json")
    signs = sensor.to_sensor().hardware.virtual_channel_phase_signs
    with np.load(config.parent / "radar_adc.npz") as source:
        expected = source["adc"].copy()
    decoded = (expected / signs).conj()
    np.savez(config.parent / "radar_adc.npz", adc=decoded)
    np.savez(config.parent / "background.npz", adc=decoded[0])
    settings = json.loads(config.read_text())
    settings.update(
        iq_transform="conjugate", phase_signs_removed=True, background_adc="background.npz"
    )
    config.write_text(json.dumps(settings))
    output = tmp_path / "normalized"
    prepare_from_config(config, output=output)
    bundle = Bundle.load(output)
    np.testing.assert_allclose(bundle.real_adc, expected)
    np.testing.assert_allclose(bundle.background_adc, expected[0])


def test_all_errors_collects_independent_adc_and_scene_failures(tmp_path):
    config = example(tmp_path / "inputs")
    output = tmp_path / "bundle"
    prepare_from_config(config, output=output)
    adc_path = output / "radar_adc.npz"
    with np.load(adc_path) as data:
        adc = data["adc"].copy()
    adc[0, 0, 0, 0] = np.nan
    np.savez(adc_path, adc=adc)
    descriptor = json.loads((output / "bundle.json").read_text())
    (output / descriptor["scene"]["file"]).write_text(
        '<scene><include filename="outside.xml"/></scene>'
    )
    errors = collect_bundle_errors(output)
    assert {e["file"] for e in errors} == {"radar_adc.npz", descriptor["scene"]["file"]}


def test_schema_checks_declared_environment_filename(tmp_path):
    config = example(tmp_path / "inputs")
    output = tmp_path / "bundle"
    prepare_from_config(config, output=output)
    descriptor = json.loads((output / "bundle.json").read_text())
    descriptor["environment"] = "placement.json"
    (output / "bundle.json").write_text(json.dumps(descriptor))
    (output / "placement.json").write_text('{"human_position_m":[0,0]}')
    errors = collect_bundle_errors(output)
    assert errors[0]["file"] == "placement.json"
    assert errors[0]["field"] == "human_position_m"


def test_subresolution_static_target_gate_uses_overlapping_range_cell():
    from validation.metrics import MeshRangeSummary, range_gate_mask

    summary = MeshRangeSummary(
        min_m=np.array([2.08]), median_m=np.array([2.083]), max_m=np.array([2.085])
    )
    ranges = np.array([2.0, 2.05, 2.10, 2.15])
    assert range_gate_mask(ranges, summary).tolist() == [False, False, True, False]
    assert not range_gate_mask(ranges - 2, summary).any()


def test_new_tdm_input_requires_explicit_slot_order(tmp_path):
    config = example(tmp_path / "inputs")
    sensor_path = config.parent / "sensor.json"
    sensor = json.loads(sensor_path.read_text())
    sensor["fmcw"]["tx_schedule"] = "tdm"
    sensor_path.write_text(json.dumps(sensor))
    with pytest.raises(ValueError, match="explicit fmcw.tdm_tx_order"):
        prepare_from_config(config, output=tmp_path / "bundle")


def test_all_errors_reports_independent_sidecar_fields(tmp_path):
    config = example(tmp_path / "inputs")
    output = tmp_path / "bundle"
    prepare_from_config(config, output=output)
    sensor = json.loads((output / "sensor.json").read_text())
    sensor["fmcw"]["num_adc_samples"] = -1
    sensor["position"] = [0, 0]
    (output / "sensor.json").write_text(json.dumps(sensor))
    (output / "frames.json").write_text('[{"motion_time_s":"bad"}]')
    errors = collect_bundle_errors(output)
    assert {e["file"] for e in errors} == {"sensor.json", "frames.json"}
    assert len(errors) >= 3
    assert all({"file", "field", "code", "message"} <= set(e) for e in errors)


def test_diagnostics_constrain_known_target_and_warn_on_iq_conjugation(tmp_path):
    from bundle_prepare.diagnostics import diagnose_bundle

    config = example(tmp_path / "inputs")
    output = tmp_path / "bundle"
    prepare_from_config(config, output=output)
    good = diagnose_bundle(output, tmp_path / "good", reference_position=[2, 0.5, 0.3])
    assert not any(w["code"].startswith("reference.") for w in good["warnings"])
    assert good["peak_u"] > 0 and good["peak_v"] > 0
    path = output / "radar_adc.npz"
    with np.load(path) as data:
        adc, times = data["adc"].copy(), data["times"].copy()
    np.savez(path, adc=adc.conj(), times=times)
    bad = diagnose_bundle(output, tmp_path / "bad", reference_position=[2, 0.5, 0.3])
    assert "adc.possible_iq_mirror" in {w["code"] for w in bad["warnings"]}
    assert "reference.range_mismatch" in {w["code"] for w in bad["warnings"]}
    assert check_bundle_integrity(output)["valid"]


@pytest.mark.parametrize("slots", [(0, 1, 2), (2, 0)])
def test_stored_tx_subset_keeps_acquisition_pri_in_sensor_and_ignore_tdm_path(
    tmp_path, monkeypatch, slots
):
    from validation import runner

    config = example(tmp_path / "inputs")
    sensor = BundleSensorConfig.from_json(config.parent / "sensor.json")
    sensor = replace(
        sensor,
        fmcw=replace(sensor.fmcw, tx_schedule="tdm", num_tx=len(slots)),
        tx_indices_zero_based=(0, 2),
        tdm_tx_order=slots,
    )
    (config.parent / "sensor.json").write_text(json.dumps(sensor.to_json()))
    with np.load(config.parent / "radar_adc.npz") as data:
        np.savez(config.parent / "subset.npz", adc=data["adc"][..., [0, 1, 2, 3, 8, 9, 10, 11]])
    settings = json.loads(config.read_text())
    settings["adc"] = "subset.npz"
    config.write_text(json.dumps(settings))
    output = tmp_path / "bundle"
    prepare_from_config(config, output=output)
    clip = Bundle.load(output)
    interval = len(slots) * 50e-6
    assert clip.sensor().hardware.num_tx == 2
    assert clip.sensor().fmcw.slow_time_interval == pytest.approx(interval)
    selection = runner._subarray_selection(clip, runner.BenchmarkRunConfig(), full_channels=8)
    simultaneous, order = runner._simultaneous_tdm_sensor(clip, selection)
    assert simultaneous.fmcw.slow_time_interval == pytest.approx(interval)
    observed = []
    monkeypatch.setattr(_validation_simulation, "load_bundle_scene", lambda *a: object())
    monkeypatch.setattr(_validation_simulation, "validation_simulator", lambda *a: object())

    def run(simulator, scene, runtime, target, count, settings, solver):
        observed.append(runtime.fmcw.slow_time_interval)
        return SimpleNamespace(adc=np.zeros((1, 4, 64, 8), dtype=complex))

    monkeypatch.setattr(_validation_simulation, "run_observed_simulator", run)
    result = runner._simulate_adc(
        clip, runner.BenchmarkRunConfig(ignore_tdm_timing=True), solver="po"
    )
    assert result.shape == (1, 4, 64, 8)
    assert observed == pytest.approx([interval])


@pytest.mark.parametrize("force", [False, True])
def test_existing_empty_output_directory_survives_writer_failure(tmp_path, force):
    config = example(tmp_path / "inputs")
    output = tmp_path / "already-exists"
    output.mkdir()
    before = output.stat().st_ino
    with pytest.raises((FileExistsError, ValueError)):
        prepare_from_config(config, output=output, force=force)
    assert output.is_dir() and output.stat().st_ino == before


def test_writer_keeps_source_waveform_provenance_and_obj_free_text_quotes(tmp_path):
    config = example(tmp_path / "inputs")
    path = config.parent / "sensor.json"
    sensor = json.loads(path.read_text())
    source = {
        "start_frequency_hz": 59e9,
        "adc_start_time_s": 7e-6,
        "source_profile": {"adcBits": 16, "vendor_note": "unchanged"},
    }
    sensor["fmcw"].update(source)
    sensor["fmcw"]["carrier_frequency_hz"] = (
        source["start_frequency_hz"]
        + sensor["fmcw"]["slope_hz_per_s"] * source["adc_start_time_s"]
    )
    path.write_text(json.dumps(sensor))
    output = tmp_path / "bundle"
    prepare_from_config(config, output=output)
    written = json.loads((output / "sensor.json").read_text())["fmcw"]
    assert all(written[k] == v for k, v in source.items())
    source_obj = tmp_path / "quoted.obj"
    source_obj.write_text(
        "o Bob's plate\ng \"unmatched name\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    )
    with pytest.raises(RuntimeError), BundleWriter() as writer:
        scene = writer.add_scene(source_obj)
        assert (writer.root / scene["file"]).read_bytes() == source_obj.read_bytes()
        raise RuntimeError("abort staging")


@pytest.mark.parametrize("version", [1, 2])
def test_epoch_and_jitter_times_are_checked_in_every_bundle_version(tmp_path, version):
    config = example(tmp_path / "inputs")
    output = tmp_path / "bundle"
    prepare_from_config(config, output=output)
    descriptor = json.loads((output / "bundle.json").read_text())
    descriptor["schema_version"] = version
    if version == 1:
        descriptor.pop("required_features", None)
        descriptor["target"].pop("material_parameters", None)
    (output / "bundle.json").write_text(json.dumps(descriptor))
    with np.load(output / "radar_adc.npz") as data:
        adc = data["adc"].copy()
    times = 1_780_000_000.0 + np.arange(4)[None] * 50e-6
    times[0, 2] += 0.25e-6
    np.savez(output / "radar_adc.npz", adc=adc, times=times)
    loaded = load_bundle(output)
    np.testing.assert_allclose(loaded.bundle.times, times - times[0, 0], rtol=0, atol=0)
    times[0, 2] += 0.01
    np.savez(output / "radar_adc.npz", adc=adc, times=times)
    errors = collect_bundle_errors(output)
    assert any(e["code"] == "timing.inconsistent" and e["field"] == "times" for e in errors)


def test_null_target_is_absent_and_public_load_decodes_contract_once(tmp_path, monkeypatch):
    from bundle_prepare.integrity import main as check_main

    config = example(tmp_path / "inputs")
    output = tmp_path / "bundle"
    prepare_from_config(config, output=output)
    descriptor = json.loads((output / "bundle.json").read_text())
    descriptor["target"] = None
    (output / "bundle.json").write_text(json.dumps(descriptor))
    count = []
    original = Bundle.load.__func__

    def load(cls, path):
        count.append(path)
        return original(cls, path)

    monkeypatch.setattr(Bundle, "load", classmethod(load))
    loaded = load_bundle(output)
    assert not loaded.can_resimulate
    assert len(count) == 1
    count.clear()
    assert (
        check_main(
            [str(output), "--all-errors", "--json", "--diagnose", str(tmp_path / "diagnostics")]
        )
        == 0
    )
    assert len(count) == 1
    count.clear()
    assert check_main([str(output), "--all-errors", "--json"]) == 0
    assert len(count) == 1


def test_legacy_scene_target_is_removed_before_target_injection(tmp_path, monkeypatch):
    import sionna.rt
    from validation.runner import _load_bundle_scene, BenchmarkRunConfig

    config = example(tmp_path / "inputs")
    output = tmp_path / "bundle"
    prepare_from_config(config, output=output)
    (output / "target.obj").write_text("v 0 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n")
    (output / "room.obj").write_text("v 1 0 0\nv 1 1 0\nv 1 0 1\nf 1 2 3\n")
    (output / "legacy.xml").write_text(
        '<scene><shape type="obj"><string name="filename" value="target.obj"/></shape><shape type="obj" id="room"><string name="filename" value="room.obj"/></shape></scene>'
    )
    descriptor = json.loads((output / "bundle.json").read_text())
    descriptor["target"]["obj"] = "target.obj"
    descriptor["schema_version"] = 1
    descriptor.pop("required_features", None)
    descriptor["target"].pop("material_parameters", None)
    descriptor["scene"] = {"file": "legacy.xml"}
    (output / "bundle.json").write_text(json.dumps(descriptor))
    clip = Bundle.load(output)
    seen = []

    def load(path):
        tree = ET.parse(path)
        seen.extend(node.get("value") for node in tree.findall(".//string[@name='filename']"))
        return object()

    monkeypatch.setattr(sionna.rt, "load_scene", load)
    original = (output / "legacy.xml").read_bytes()
    _load_bundle_scene(clip, BenchmarkRunConfig())
    assert seen == [str(output / "room.obj")]
    assert (output / "legacy.xml").read_bytes() == original


def test_motion_cli_requires_explicit_world_coordinate_confirmation(tmp_path):
    config = example(tmp_path / "inputs")
    settings = json.loads(config.read_text())
    settings["evaluated_mesh"] = "not-yet-normalized.npz"
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="motion_coordinate_frame"):
        prepare_from_config(config, output=tmp_path / "bundle")


def test_legacy_slot_packed_rx_subset_has_physical_rx_labels(tmp_path):
    from validation import runner

    config = example(tmp_path / "inputs")
    output = tmp_path / "legacy"
    prepare_from_config(config, output=output)
    s = json.loads((output / "sensor.json").read_text())
    s["rx_indices_zero_based"] = [1, 3]
    s["fmcw"].update(tx_schedule="tdm", num_tx=3)
    s["metadata"] = {
        "source_adc_axes": ["adc_sample", "chirp_loop", "rx", "tx"],
        "tdm_tx_order": [3, 2, 1],
    }
    (output / "sensor.json").write_text(json.dumps(s))
    descriptor = json.loads((output / "bundle.json").read_text())
    descriptor["schema_version"] = 1
    descriptor.pop("required_features", None)
    descriptor["target"].pop("material_parameters", None)
    (output / "bundle.json").write_text(json.dumps(descriptor))
    np.savez(
        output / "radar_adc.npz",
        adc=np.ones((1, 4, 64, 6), dtype=complex),
        times=np.arange(4)[None] * 150e-6,
    )
    clip = Bundle.load(output)
    selection = runner._subarray_selection(clip, runner.BenchmarkRunConfig(), full_channels=6)
    assert selection.channel_tx_indices_1based.tolist() == [3, 3, 2, 2, 1, 1]
    assert selection.channel_rx_indices_1based.tolist() == [2, 4, 2, 4, 2, 4]
    assert selection.virtual_positions_lambda.shape == (6, 2)


def test_waveform_start_and_sampled_centre_import_to_same_first_sample(tmp_path):
    config = example(tmp_path / "inputs")
    raw = json.loads((config.parent / "sensor.json").read_text())
    f = raw["fmcw"]
    f.pop("carrier_frequency_hz")
    f.update(start_frequency_hz=60e9, adc_start_time_s=4e-6, bandwidth_hz=99e9)
    first = BundleSensorConfig.from_mapping(raw)
    expected = 60e9 + f["slope_hz_per_s"] * 4e-6
    assert first.fmcw.carrier_frequency == pytest.approx(expected)
    f.pop("start_frequency_hz")
    f["center_frequency_hz"] = expected + f["slope_hz_per_s"] * f["num_adc_samples"] / (
        2 * f["sampling_frequency_hz"]
    )
    centre = BundleSensorConfig.from_mapping(raw)
    assert centre.fmcw.carrier_frequency == pytest.approx(expected)


def test_cli_retains_epoch_precision_and_jitter_after_rebasing(tmp_path):
    config = example(tmp_path / "inputs")
    settings = json.loads(config.read_text())
    settings["timestamp_tolerance_s"] = 0
    config.write_text(json.dumps(settings))
    path = config.parent / "radar_adc.npz"
    with np.load(path) as data:
        adc = data["adc"].copy()
    times = 1_780_000_000.0 + np.arange(4)[None] * 50e-6
    times[0, 2] += 0.25e-6
    np.savez(path, adc=adc, times=times)
    output = tmp_path / "bundle"
    prepare_from_config(config, output=output)
    loaded = load_bundle(output)
    np.testing.assert_array_equal(loaded.bundle.times, times - times[0, 0])
    assert loaded.bundle.sensor_config.timestamp_tolerance_s >= 4 * np.spacing(times[0, 0])
