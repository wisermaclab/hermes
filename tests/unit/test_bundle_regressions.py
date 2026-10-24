# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Legacy carrier compatibility and strict publication of new bundles."""

from importlib import resources
import json

import numpy as np
import pytest

from bundle_prepare import Bundle, BundleSensorConfig
from bundle_prepare.cli import main, prepare_from_config
from bundle_prepare.contract import BundleValidationError


def inputs(root):
    root.mkdir()
    for item in resources.files("bundle_prepare").joinpath("data/static_target").iterdir():
        if item.is_file():
            (root / item.name).write_bytes(item.read_bytes())
    return root / "prepare.json"


@pytest.mark.parametrize("version", [1, 2])
def test_ambiguous_legacy_carrier_keeps_playback_but_blocks_resimulation_and_reexport(
    tmp_path, version
):
    config = inputs(tmp_path / "inputs")
    output = tmp_path / "legacy"
    prepare_from_config(config, output=output)
    sensor_path = output / "sensor.json"
    sensor = json.loads(sensor_path.read_text())
    sensor["fmcw"].pop("carrier_reference")
    sensor_path.write_text(json.dumps(sensor))
    if version == 1:
        descriptor_path = output / "bundle.json"
        descriptor = json.loads(descriptor_path.read_text())
        descriptor["schema_version"] = 1
        descriptor.pop("required_features", None)
        descriptor["target"].pop("material_parameters")
        descriptor_path.write_text(json.dumps(descriptor))

    with pytest.warns(RuntimeWarning, match="Carrier reference is unspecified"):
        bundle = Bundle.load(output)
    assert bundle.real_adc.shape == (1, 4, 64, 12)
    assert not bundle.can_resimulate
    assert "Carrier reference" in bundle.resimulation_issues[0]
    with pytest.raises(BundleValidationError, match="unverified carrier"):
        bundle.sensor_config.to_json()


def test_legacy_carrier_is_derived_from_waveform_and_original_value_is_retained(tmp_path):
    config = inputs(tmp_path / "inputs")
    raw = json.loads((config.parent / "sensor.json").read_text())
    fmcw = raw["fmcw"]
    fmcw.pop("carrier_reference")
    fmcw.update(carrier_frequency_hz=61e9, start_frequency_hz=60e9, adc_start_time_s=4e-6)
    with pytest.warns(RuntimeWarning, match="Legacy carrier converted"):
        sensor = BundleSensorConfig.from_mapping(raw)
    expected = 60e9 + fmcw["slope_hz_per_s"] * 4e-6
    assert sensor.fmcw.carrier_frequency == pytest.approx(expected)
    canonical = sensor.to_json()
    assert canonical["fmcw"]["carrier_reference"] == "first_adc_sample"
    assert canonical["fmcw"]["carrier_conversion"]["source_frequency_hz"] == 61e9
    assert not BundleSensorConfig.from_mapping(canonical).resimulation_issues
    # An explicit first-sample claim must not override contradictory provenance.
    fmcw["carrier_reference"] = "first_adc_sample"
    with pytest.raises(BundleValidationError) as error:
        BundleSensorConfig.from_mapping(raw)
    assert error.value.code == "carrier.inconsistent"


def test_new_writer_refuses_to_attest_an_ambiguous_input_carrier(tmp_path):
    config = inputs(tmp_path / "inputs")
    sensor_path = config.parent / "sensor.json"
    raw = json.loads(sensor_path.read_text())
    raw["fmcw"].pop("carrier_reference")
    sensor_path.write_text(json.dumps(raw))
    output = tmp_path / "bundle.zip"
    with pytest.warns(RuntimeWarning, match="Carrier reference is unspecified"):
        with pytest.raises(BundleValidationError, match="unverified carrier"):
            prepare_from_config(config, output=output)
    assert not output.exists()


def test_failed_motion_publication_retains_existing_bundle(tmp_path, capsys):
    config = inputs(tmp_path / "inputs")
    output = tmp_path / "bundle.zip"
    prepare_from_config(config, output=output)
    original = output.read_bytes()
    with np.load(config.parent / "target_mesh.npz") as target:
        np.savez(
            config.parent / "motion.npz",
            vertices=target["vertices"][None],
            faces=target["faces"],
            times=[0.0],
        )
    settings = json.loads(config.read_text())
    settings.update(evaluated_mesh="motion.npz", motion_coordinate_frame="world")
    config.write_text(json.dumps(settings))
    assert main(["--config", str(config), "--output", str(output), "--force"]) == 1
    assert "full radar acquisition" in capsys.readouterr().err
    assert output.read_bytes() == original


def test_epoch_motion_requires_explicit_frame_timing_and_remains_aligned(tmp_path):
    config = inputs(tmp_path / "inputs")
    epoch = 1792169773.427746
    with np.load(config.parent / "target_mesh.npz") as target:
        np.savez(
            config.parent / "motion.npz",
            vertices=np.repeat(target["vertices"][None], 2, axis=0),
            faces=target["faces"],
            times=[epoch, epoch + 0.1],
        )
    adc_path = config.parent / "radar_adc.npz"
    with np.load(adc_path) as data:
        adc = data["adc"].copy()
    np.savez(adc_path, adc=adc, times=epoch + np.arange(4)[None, :] * 5e-5)
    settings = json.loads(config.read_text())
    settings.pop("frame_times_s")
    settings.update(evaluated_mesh="motion.npz", motion_coordinate_frame="world")
    config.write_text(json.dumps(settings))
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="explicit frame timing"):
        prepare_from_config(config, output=output)
    assert not output.exists()
    settings["frame_times_s"] = [epoch]
    config.write_text(json.dumps(settings))
    prepare_from_config(config, output=output)
    loaded = Bundle.load(output)
    assert loaded.can_resimulate
    assert loaded.motion_times_s.tolist() == [epoch]
    np.testing.assert_allclose(loaded.times, np.arange(4)[None, :] * 5e-5, atol=1e-6)


@pytest.mark.parametrize(
    "mode,complex_samples,valid",
    [
        ("complex1x", False, False),
        ("complex2x", False, False),
        ("real", True, False),
        ("real", False, True),
        ("complex1x", True, True),
    ],
)
def test_adc_sampling_mode_matches_representation(tmp_path, mode, complex_samples, valid):
    config = inputs(tmp_path / "inputs")
    path = config.parent / "sensor.json"
    sensor = json.loads(path.read_text())
    sensor["fmcw"]["adc_mode"] = mode
    path.write_text(json.dumps(sensor))
    adc_path = config.parent / "radar_adc.npz"
    with np.load(adc_path) as data:
        adc = data["adc"].copy()
    np.savez(adc_path, adc=adc if complex_samples else adc.real)
    if valid:
        prepare_from_config(config, output=tmp_path / "bundle")
    else:
        with pytest.raises(BundleValidationError) as error:
            prepare_from_config(config, output=tmp_path / "bundle")
        assert error.value.code == "adc.mode_dtype"
