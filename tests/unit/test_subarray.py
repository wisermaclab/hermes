# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Tests for array_model="subarray": one phase centre per Tx/Rx group pair."""

from __future__ import annotations

from dataclasses import fields
import warnings

import numpy as np
import pytest
from sionna.rt import load_scene

from mmWaveRadar.materials import PhysicalOpticsMaterial, human_skin_material
from mmWaveRadar.naming import (
    array_group_partition,
    parse_index_spec,
    validate_array_groups,
    validate_array_model,
)
from mmWaveRadar.radar import FMCWConfig, RadarHardware, RadarSensor
from mmWaveRadar.simulation import MmWaveRadarSimulator, POConfig
from mmWaveRadar.simulation.simulator import PhaseCenterApproximationWarning
from mmWaveRadar.simulation.subarray import (
    MERGED_ARRAY_FIELDS,
    group_pairs,
    local_array_groups,
    merge_subarray_cubes,
    subarray_radar,
)
from mmWaveRadar.simulation.types import RadarCube, SensingMetadata
from mmWaveRadar.targets import MeshSequence, MeshTarget


def _radar(*, tx_schedule: str = "simultaneous", spacing_m: float = 0.0025) -> RadarSensor:
    """Two Tx and four Rx in a line along y, offset from the radar origin."""

    tx = [[0.0, 0.01 + i * 2 * spacing_m, 0.0] for i in range(2)]
    rx = [[0.0, 0.02 + i * spacing_m, 0.0] for i in range(4)]
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=68e12,
        chirp_duration=58e-6,
        chirp_repetition_time=65e-6,
        sampling_frequency=4.5e6,
        num_adc_samples=32,
        num_chirps_per_frame=2,
        frame_period=5e-3,
        num_tx=2,
        tx_schedule=tx_schedule,
    )
    return RadarSensor(
        name="subarray-test",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware.from_positions(tx, rx),
        fmcw=fmcw,
    )


def _plate(distance_m: float = 1.5, size_m: float = 0.4, *, material=None) -> MeshTarget:
    half = 0.5 * size_m
    vertices = np.array(
        [
            [distance_m, -half, -half],
            [distance_m, half, -half],
            [distance_m, half, half],
            [distance_m, -half, half],
        ],
        dtype=np.float32,
    )
    # Wound so the normal faces the radar at the origin.
    faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)
    return MeshTarget(
        "plate",
        MeshSequence(
            vertices=np.stack([vertices, vertices]), faces=faces, times=np.array([0.0, 1.0])
        ),
        material=PhysicalOpticsMaterial(1.0, 0.0, model="pec") if material is None else material,
    )


def _po(**kwargs) -> MmWaveRadarSimulator:
    return MmWaveRadarSimulator(
        solver="po",
        po_config=POConfig(po_integration_mode="face_centroid"),
        compute_backend="numpy",
        compute_precision="float64",
        **kwargs,
    )


def _run(simulator, radar, scene=None, *, target=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PhaseCenterApproximationWarning)
        return simulator.run(scene, radar, [_plate() if target is None else target], num_frames=1)


def test_naming_replaces_element_and_parses_groups():
    with pytest.raises(ValueError, match="replaced by 'subarray'"):
        validate_array_model("element")
    assert validate_array_model("subarray") == "subarray"
    assert parse_index_spec("0-2,9") == (0, 1, 2, 9)
    assert parse_index_spec("3-1") == (3, 2, 1)
    assert validate_array_groups([[0, 1], [2]], name="g") == ((0, 1), (2,))
    assert validate_array_groups(None, name="g") is None
    for bad, message in (
        ([[0], [0]], "more than once"),
        ([[]], "empty group"),
        ("0-1", "list of index lists"),
        ([[-1]], "non-negative"),
        ([], "at least one group"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_array_groups(bad, name="g")
    assert array_group_partition(None, 3, name="g") == ((0, 1, 2),)
    with pytest.raises(ValueError, match=r"leave out \[2\]"):
        array_group_partition(((0, 1),), 3, name="g")


def test_every_metadata_array_has_a_merge_rule():
    array_fields = {item.name for item in fields(SensingMetadata) if "ndarray" in str(item.type)}
    assert array_fields <= MERGED_ARRAY_FIELDS, sorted(array_fields - MERGED_ARRAY_FIELDS)


def test_group_pairs_own_each_channel_once_in_full_array_order():
    radar = _radar()
    pairs = group_pairs(radar, ((0,), (1,)), ((0, 1), (2, 3)))
    owned = np.concatenate([pair.channels for pair in pairs])
    assert sorted(owned.tolist()) == list(range(radar.hardware.num_virtual_channels))
    for pair in pairs:
        sub = subarray_radar(radar, pair).hardware
        tx_group, rx_group = np.asarray(pair.tx_group), np.asarray(pair.rx_group)
        np.testing.assert_array_equal(
            radar.hardware.virtual_tx_indices()[pair.channels],
            tx_group[sub.virtual_tx_indices()],
        )
        np.testing.assert_array_equal(
            radar.hardware.virtual_rx_indices()[pair.channels],
            rx_group[sub.virtual_rx_indices()],
        )


def test_one_group_is_the_phase_center_model():
    radar = _radar()
    base = _run(_po(), radar)
    one = _run(_po(array_model="subarray"), radar)
    # Equivalent arithmetic can differ by roundoff with different array alignment.
    np.testing.assert_allclose(one.adc, base.adc, rtol=2e-14, atol=1e-18)
    assert one.metadata.events[0]["reason"] == "subarray_partition"


@pytest.mark.parametrize("tx_schedule", ["simultaneous", "tdm"])
def test_split_equals_running_each_group_pair(tx_schedule):
    radar = _radar(tx_schedule=tx_schedule)
    groups = dict(array_tx_groups=[[0], [1]], array_rx_groups=[[0, 1], [2, 3]])
    split = _run(_po(array_model="subarray", **groups), radar)
    base = _run(_po(), radar)
    assert split.adc.shape == base.adc.shape
    if tx_schedule == "tdm":
        np.testing.assert_array_equal(
            split.metadata.virtual_channel_times_s,
            base.metadata.virtual_channel_times_s,
        )
        return
    manual = np.zeros_like(split.adc)
    for pair in group_pairs(radar, ((0,), (1,)), ((0, 1), (2, 3))):
        manual[..., pair.channels] = _run(_po(), subarray_radar(radar, pair)).adc
    np.testing.assert_allclose(split.adc, manual, rtol=1e-12, atol=0.0)
    # Group phase centres differ from the array centre, so the result moves a little.
    assert 0.0 < np.linalg.norm(split.adc - base.adc) / np.linalg.norm(base.adc) < 0.1
    # Per-pair counts add up; refresh flags combine.
    np.testing.assert_array_equal(
        split.metadata.path_counts,
        sum(
            _run(_po(), subarray_radar(radar, pair)).metadata.path_counts
            for pair in group_pairs(radar, ((0,), (1,)), ((0, 1), (2, 3)))
        ),
    )


def test_rt_split_equals_running_each_group_pair():
    radar = _radar()

    def rt(**kwargs):
        return MmWaveRadarSimulator(
            solver="rt",
            rt_samples_per_source=2000,
            rt_max_paths_per_source=2000,
            rt_max_depth=1,
            rt_human_bounces="any",
            rt_human_specular_reflection=True,
            compute_backend="numpy",
            **kwargs,
        )

    def run(simulator, run_radar):
        target = _plate(material=human_skin_material(f"plate-skin-{id(simulator)}"))
        return _run(simulator, run_radar, load_scene(), target=target)

    base = run(rt(), radar)
    one = run(rt(array_model="subarray"), radar)
    np.testing.assert_array_equal(one.adc, base.adc)
    split = run(rt(array_model="subarray", array_tx_groups=[[0], [1]]), radar)
    manual = np.zeros_like(split.adc)
    for pair in group_pairs(radar, ((0,), (1,)), None):
        manual[..., pair.channels] = run(rt(), subarray_radar(radar, pair)).adc
    np.testing.assert_allclose(split.adc, manual, rtol=1e-6, atol=0.0)
    assert np.any(np.abs(split.adc) > 0.0)


def test_group_settings_are_checked():
    with pytest.raises(ValueError, match="need array_model='subarray'"):
        MmWaveRadarSimulator(array_tx_groups=[[0], [1]])
    with pytest.raises(ValueError, match="array_rx_groups must split the 4 active"):
        _run(_po(array_model="subarray", array_rx_groups=[[0, 1]]), _radar())


def _cube(gain: float, rt_power: float, po_power: float, channels: int) -> RadarCube:
    """A Hybrid-like cube whose human parts were scaled by its own gain."""

    shape = (1, 2, 4, channels)
    human = np.full(shape, gain * 1.0, dtype=np.complex128)
    static = np.full(shape, 10.0, dtype=np.complex128)
    metadata = SensingMetadata(
        events=[],
        path_counts=np.ones((1, 2), dtype=np.int64),
        max_displacements=np.zeros((1, 2)),
        virtual_channel_order="tx_major",
        solver="hybrid_po",
        po_calibration={
            "mode": "rt_first_pose",
            "human_po_amplitude_gain": gain,
            "rt_reference_rms_power": rt_power,
            "raw_po_reference_rms_power": po_power,
            "reference_power_floor": 0.0,
        },
    )
    return RadarCube(
        adc=static + human,
        times=np.zeros((1, 2)),
        metadata=metadata,
        components={"static_environment_blocked": static, "human_po": human},
    )


def test_pairs_share_one_first_pose_gain_fitted_on_all_channels():
    radar = _radar()
    pairs = group_pairs(radar, ((0,), (1,)), None)
    cubes = [
        _cube(2.0, rt_power=4.0, po_power=1.0, channels=4),
        _cube(1.0, rt_power=1.0, po_power=1.0, channels=4),
    ]
    merged = merge_subarray_cubes(cubes, pairs, 8, solver="hybrid_po")
    calibration = merged.metadata.po_calibration
    # Mean powers over all eight channels: RT 2.5, PO 1.0.
    assert calibration["human_po_amplitude_gain"] == pytest.approx(np.sqrt(2.5))
    assert calibration["fitted_on_subarrays"] == "all"
    assert calibration["subarray_fitted_gains"] == [2.0, 1.0]
    np.testing.assert_allclose(merged.components["human_po"], np.sqrt(2.5))
    np.testing.assert_allclose(
        merged.adc,
        merged.components["static_environment_blocked"] + merged.components["human_po"],
    )
    np.testing.assert_array_equal(merged.metadata.path_counts, 2 * np.ones((1, 2)))


def test_sequence_calibration_shares_the_first_pairs_gain():
    radar = _radar()
    pairs = group_pairs(radar, ((0,), (1,)), None)
    cubes = [_cube(2.0, 4.0, 1.0, 4), _cube(1.0, 1.0, 1.0, 4)]
    for cube in cubes:
        cube.metadata.po_calibration["mode"] = "rt_sequence"
    merged = merge_subarray_cubes(cubes, pairs, 8, solver="hybrid_po")
    assert merged.metadata.po_calibration["fitted_on_subarrays"] == [0]
    np.testing.assert_allclose(merged.components["human_po"], 2.0)


def test_warning_suggests_groups_and_subarrays_shrink_the_error():
    radar = _radar(spacing_m=0.05)
    target = [_plate(distance_m=0.3, size_m=0.05)]
    with pytest.warns(PhaseCenterApproximationWarning, match=r"about \d+ Tx and \d+ Rx groups"):
        event = _po()._phase_center_approximation(radar, target)
    assert event["suggested_array_rx_group_count"] > 1
    single = _po(
        array_model="subarray", array_tx_groups=[[0], [1]], array_rx_groups=[[0], [1], [2], [3]]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", PhaseCenterApproximationWarning)
        assert single._phase_center_approximation(radar, target) is None


def test_rt_devices_sit_at_the_array_centres():
    radar = _radar()
    scene = load_scene()
    tx, rx = radar.configure_scene(scene)
    np.testing.assert_allclose(
        np.asarray(tx.position).ravel(), radar.world_tx_positions().mean(axis=0), atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(rx.position).ravel(), radar.world_rx_positions().mean(axis=0), atol=1e-7
    )


def test_board_groups_map_to_active_antennas():
    assert local_array_groups([[9], [1, 2]], [1, 2, 9], name="g") == ((2,), (0, 1))
    assert local_array_groups(None, [0, 1], name="g") is None
    with pytest.raises(ValueError, match=r"not active"):
        local_array_groups([[0], [1, 2, 3]], [0, 1, 2], name="g")


def test_scene_cli_reads_groups(tmp_path):
    from mmWaveRadar.gui import scene_cli

    def resolve(*arguments):
        return scene_cli.resolve_config(
            scene_cli.build_parser().parse_args(
                [*arguments, "--output", str(tmp_path / "out.zip")]
            )
        )

    config = resolve(
        "--array-model",
        "subarray",
        "--array-tx-groups",
        "0",
        "1-2",
        "--array-rx-groups",
        "0-1",
        "2-3",
    )
    assert config["array_tx_groups"] == [[0], [1, 2]]
    assert config["array_rx_groups"] == [[0, 1], [2, 3]]
    assert resolve()["array_tx_groups"] is None
    with pytest.raises(ValueError, match="needs array_model 'subarray'"):
        resolve("--array-tx-groups", "0-2")
    with pytest.raises(ValueError, match=r"leave out \[2\]"):
        resolve("--array-model", "subarray", "--array-tx-groups", "0", "1")
