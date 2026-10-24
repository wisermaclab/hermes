# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Canonical solver names, the MIMO array model, and RT path visibility."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
import warnings

from sensing_test_helpers import *  # noqa: F401,F403
from mmWaveRadar import naming
from mmWaveRadar.simulation import PhaseCenterApproximationWarning


def test_removed_solver_names_point_to_their_replacement():
    assert naming.validate_solver("hybrid_po") == "hybrid_po"
    for removed, replacement in (
        ("full_rt", "'rt' with rt_refresh_frames=0"),
        ("coherent_rt", "'rt' with rt_refresh_frames=1"),
        ("human_only_po", "'po'"),
        ("hybrid_static_env_po", "'hybrid_po'"),
    ):
        with pytest.raises(ValueError, match=f"replaced by {replacement}"):
            naming.validate_solver(removed)
    with pytest.raises(ValueError, match="solver must be one of: po, rt, hybrid_po"):
        naming.validate_solver("ray_tracing")


def test_removed_keyword_arguments_name_their_replacement():
    with pytest.raises(TypeError, match="mobility_mode was renamed to solver"):
        MmWaveRadarSimulator(mobility_mode="rt_coherent_bank")
    with pytest.raises(TypeError, match="coupling_mode was renamed to rt_human_bounces"):
        MmWaveRadarSimulator(coupling_mode="unrestricted")
    with pytest.raises(TypeError, match="unexpected argument 'colour'"):
        MmWaveRadarSimulator(colour="red")


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"tx_schedule": "tdm"}, "tdm"),
        ({"tx_schedule": "simultaneous", "tdm_enabled": True}, "simultaneous"),
        ({"tdm_enabled": True}, "tdm"),
        ({"tdm_enabled": False}, "simultaneous"),
        ({}, "simultaneous"),
    ],
)
def test_saved_tx_schedule_reads_the_earlier_tdm_flag(record, expected):
    assert naming.saved_tx_schedule(record) == expected


def test_tx_schedule_rejects_the_earlier_boolean():
    with pytest.raises(ValueError, match="replaced the tdm_enabled flag"):
        naming.validate_tx_schedule(True)


def test_removed_command_line_options_fail_with_replacement(capsys):
    parser = argparse.ArgumentParser(prog="hermes")
    parser.add_argument("--solvers", nargs="+")
    naming.add_removed_options(parser, {"--modes": "--solvers"})

    assert parser.parse_args(["--solvers", "rt"]).solvers == ["rt"]
    with pytest.raises(SystemExit):
        parser.parse_args(["--modes", "full_rt"])
    assert "--modes was removed; use --solvers" in capsys.readouterr().err
    assert "--modes" not in parser.format_help()


def test_element_array_model_became_subarray():
    with pytest.raises(ValueError, match="'element' was replaced by 'subarray'"):
        MmWaveRadarSimulator(array_model="element")
    with pytest.raises(ValueError, match="array_model must be one of"):
        MmWaveRadarSimulator(array_model="element_wise")
    assert MmWaveRadarSimulator(array_model="subarray").array_model == "subarray"


def _array_radar(tx_y_m, rx_y_m=(0.0,)):
    # A slow ramp keeps metre-scale beat tones inside the 1 kHz IF band.
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=1e10,
        chirp_duration=1e-3,
        chirp_repetition_time=0.5,
        sampling_frequency=1e3,
        num_adc_samples=4,
        num_chirps_per_frame=1,
        frame_period=1.0,
        num_tx=len(tx_y_m),
    )
    return RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware.from_positions(
            [[0.0, y, 0.0] for y in tx_y_m],
            [[0.0, y, 0.0] for y in rx_y_m],
        ),
        fmcw=fmcw,
    )


@pytest.mark.parametrize("solver", ["rt", "po", "hybrid_po"])
def test_phase_center_warns_only_when_the_aperture_is_too_large(solver):
    target = MeshTarget(
        "target",
        _single_face_mesh_sequence(x0=1.0),
        material=human_skin_material("aperture-target"),
    )
    simulator = MmWaveRadarSimulator(solver=solver)

    with pytest.warns(PhaseCenterApproximationWarning, match="phase-center"):
        event = simulator._phase_center_approximation(
            _array_radar((-0.2, 0.2)),
            [target],
        )
    assert event["reason"] == "phase_center_approximation"
    assert event["warning"] is True
    assert event["max_tx_offset_m"] == pytest.approx(0.2)

    quarter_wavelength = 0.25 * 299792458.0 / 60e9
    far_target = MeshTarget(
        "far-target",
        MeshSequence(
            vertices=np.stack([_front_facing_triangle(50.0, scale=0.001)] * 2),
            faces=np.array([[0, 1, 2]], dtype=np.uint32),
            times=np.array([0.0, 1.0]),
        ),
        material=human_skin_material("aperture-far-target"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", PhaseCenterApproximationWarning)
        event = simulator._phase_center_approximation(
            _array_radar((-quarter_wavelength, quarter_wavelength)),
            [far_target],
        )
    assert event["warning"] is False


def test_single_element_array_needs_no_phase_center_check():
    target = MeshTarget(
        "target",
        _single_face_mesh_sequence(x0=1.0),
        material=human_skin_material("single-element-target"),
    )
    simulator = MmWaveRadarSimulator(solver="rt")

    assert simulator._phase_center_approximation(_array_radar((0.0,)), [target]) is None


def _single_hit_paths(mesh_target):
    """One specular return from (1, 0, 0) on the target's only face."""

    object_id = int(np.asarray(mesh_target.ensure_scene_object().object_id).reshape(-1)[0])
    hit = np.array([1.0, 0.0, 0.0])
    a = np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex128)
    tau = np.array([[[2.0 * np.linalg.norm(hit) / c]]], dtype=float)
    return SimpleNamespace(
        tau=tau,
        cir=lambda normalize_delays=False, out_type="numpy": (a, tau),
        vertices=hit.reshape(1, 1, 1, 1, 3),
        interactions=np.array([[[[InteractionType.SPECULAR]]]], dtype=np.uint32),
        objects=np.array([[[[object_id]]]], dtype=np.uint32),
        primitives=np.zeros((1, 1, 1, 1), dtype=np.uint32),
    )


@pytest.mark.parametrize("entry", ["solver", "baseline"])
@pytest.mark.parametrize("fade_frames", [0, 1])
@pytest.mark.parametrize(
    ("visibility_frames", "refreshed", "blocked"),
    [
        (1, [[False, False], [True, False], [True, False]], [[0, 0], [1, 1], [1, 1]]),
        (2, [[False, False], [False, False], [True, False]], [[0, 0], [0, 0], [1, 1]]),
    ],
)
def test_rt_visibility_refresh_blocks_paths_between_retraces(
    visibility_frames,
    refreshed,
    blocked,
    entry,
    fade_frames,
):
    reflector = MeshTarget(
        "reflector",
        _single_face_mesh_sequence(x0=1.0),
        material=human_skin_material("visibility-reflector"),
    )
    # A second body swings into the line of sight just before frame 1.
    away = _front_facing_triangle(0.5, scale=2.0, y_shift=10.0, z_shift=-0.5)
    across = _front_facing_triangle(0.5, scale=2.0, z_shift=-0.5)
    blocker = MeshTarget(
        "blocker",
        MeshSequence(
            vertices=np.stack([away, away, across]),
            faces=np.array([[0, 1, 2]], dtype=np.uint32),
            times=np.array([0.0, 1.9, 2.0]),
        ),
        material=human_skin_material("visibility-blocker"),
    )

    class TestSimulator(MmWaveRadarSimulator):
        trace_count = 0

        def _trace(self, scene):
            del scene
            self.trace_count += 1
            return _single_hit_paths(reflector)

    radar = _array_radar((0.0,))
    radar = replace(radar, fmcw=replace(radar.fmcw, num_chirps_per_frame=2, frame_period=2.0))
    simulator = TestSimulator(
        solver="rt",
        rt_samples_per_source=1,
        rt_human_specular_reflection=True,
        rt_human_bounces="any",
        rt_refresh_frames=3,
        rt_visibility_refresh_frames=visibility_frames,
        rt_visibility_fade_frames=fade_frames,
    )

    if entry == "baseline":
        from mmWaveRadar.simulation.hybrid_pipeline import run_rt_baseline

        baseline = run_rt_baseline(
            scene=load_scene(),
            radar=radar,
            targets=[reflector, blocker],
            simulator=simulator,
            num_frames=3,
            calibration_reference_component="single_human_only",
        )
        cube = baseline.cube
    else:
        cube = simulator.run(load_scene(), radar, [reflector, blocker], num_frames=3)

    assert simulator.trace_count == 1
    np.testing.assert_array_equal(cube.metadata.rt_visibility_refresh, refreshed)
    weights = np.ones((3, 2))
    for frame in range(1, 3):
        delta = (-1.0 if blocked[frame][0] else 1.0) / (fade_frames + 1.0)
        weights[frame] = np.clip(weights[frame - 1] + delta, 0.0, 1.0)
    np.testing.assert_array_equal(cube.metadata.rt_blocked_path_counts, weights == 0.0)
    np.testing.assert_allclose(cube.adc, cube.adc[0, 0][None, None] * weights[..., None, None])
    if entry == "baseline":
        power = baseline.reference_adc_power["single_human_only"]
        np.testing.assert_allclose(power, power[0, 0] * weights[..., None] ** 2)
    blocked_chirps = weights == 0.0
    assert np.all(np.abs(cube.adc[~blocked_chirps]) > 0.0)
    assert np.all(cube.adc[blocked_chirps] == 0.0)
