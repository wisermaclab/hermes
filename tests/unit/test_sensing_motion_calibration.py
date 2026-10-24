# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Motion diagnostic and calibration tests."""

from __future__ import annotations

from sensing_test_helpers import *  # noqa: F401,F403


def test_equivalent_bistatic_radial_velocity_monostatic_factor():
    points = np.array([[1.0, 0.0, 0.0]])
    velocities = np.array([[2.0, 0.0, 0.0]])
    tx = np.array([[0.0, 0.0, 0.0]])
    rx = np.array([[0.0, 0.0, 0.0]])

    radial = equivalent_bistatic_radial_velocity(points, velocities, tx, rx)

    assert radial.shape == (1, 1)
    assert np.isclose(radial[0, 0], 2.0)


def test_equivalent_bistatic_radial_velocity_uses_board_channel_map():
    points = np.array([[1.0, 0.2, 0.0]])
    velocities = np.array([[1.0, 0.5, 0.0]])
    tx = np.array([[0.0, 0.0, 0.0], [0.0, 0.3, 0.0]])
    rx = np.array([[0.0, -0.2, 0.0], [0.0, 0.4, 0.0]])

    canonical = equivalent_bistatic_radial_velocity(
        points,
        velocities,
        tx,
        rx,
    )
    permuted = equivalent_bistatic_radial_velocity(
        points,
        velocities,
        tx,
        rx,
        virtual_channel_tx_indices=np.array([1, 0, 1, 0]),
        virtual_channel_rx_indices=np.array([1, 0, 0, 1]),
    )

    assert np.allclose(permuted, canonical[:, [3, 0, 2, 1]])


def test_mesh_sequence_radial_velocity_samples_face_centroid():
    class DummySequence:
        vertices = np.array(
            [
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
                [[1.1, 0.0, 0.0], [1.1, 1.0, 0.0], [1.1, 0.0, 1.0]],
            ]
        )
        faces = np.array([[0, 1, 2]])
        times = np.array([0.0, 0.1])

    samples = mesh_sequence_radial_velocity_samples(
        DummySequence(),
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        point_kind="face_centroid",
    )
    summary = radial_velocity_summary(samples)

    assert samples.radial_velocity_mps.shape == (1, 1, 1)
    assert summary["sample_count"] == 1
    assert summary["signed_max_mps"] > 0.0


@pytest.mark.parametrize("coherent", [False, True])
def test_rt_sequence_calibration_uses_retrace_chirps_and_last_retrace_alignment(coherent):
    times = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=float)
    rt_power = np.array([[1.0, 99.0, 4.0, 99.0]], dtype=float)
    metadata = SensingMetadata(
        events=[
            {"frame": 0, "chirp": 0, "time": 0.0, "reason": "initial"},
            {"frame": 0, "chirp": 2, "time": 2.0, "reason": "displacement"},
        ],
        path_counts=np.ones(times.shape, dtype=np.int64),
        max_displacements=np.zeros(times.shape, dtype=float),
        virtual_channel_order="tx_major",
        solver="rt",
        single_human_only_path_power=rt_power,
    )
    rt_cube = RadarCube(
        adc=np.zeros((1, 4, 2, 1), dtype=np.complex128),
        times=times,
        metadata=metadata,
    )
    config = POCalibrationConfig(
        mode="rt_sequence",
        sample_mode="retraced_chirps",
        reference_component="single_human_only",
        alignment="last_retrace",
    )

    fit_function = (
        fit_sequence_adc_calibration if coherent else fit_sequence_path_power_calibration
    )
    power_argument = (
        {"po_adc": np.ones((1, 4, 2, 1), dtype=np.complex128)}
        if coherent
        else {"po_path_power": np.ones((1, 4), dtype=float)}
    )
    gain, fit = fit_function(
        rt_baseline={
            "cube": rt_cube,
            "reference_adc_power": {"single_human_only": rt_power[..., None]},
        },
        po_times=np.array([[0.2, 1.2, 2.2, 3.2]], dtype=float),
        config=config,
        **power_argument,
    )

    assert np.isclose(gain, np.sqrt(2.5))
    assert np.isclose(fit["rt_reference_rms_power"], 2.5)
    assert np.isclose(fit["rt_reference_integrated_power"], 5.0)
    assert fit["reference_chirps"] == 2
    assert fit["alignment"] == "last_retrace"
    assert fit["sample_mode"] == "retrace_chirps"
    assert fit["reference_power_kind"] == (
        "adc_rms_power" if coherent else "path_coefficient_power"
    )
    assert fit["calibration_gain_strategy"] == "time_varying_last_retrace"
    gain_map = calibration_gain_array(fit, times.shape)
    np.testing.assert_allclose(gain_map, [[1.0, 1.0, 2.0, 2.0]])


def test_po_amplitude_calibration_matches_rt_reference_power():
    config = POCalibrationConfig(mode="rt_first_pose")

    gain, metadata = MmWaveRadarSimulator._po_amplitude_calibration(
        rt_reference_adc=np.full((2, 4, 1), 6.0 + 0.0j),
        po_reference_adc=np.full((2, 4, 1), 2.0 + 0.0j),
        config=config,
    )

    assert np.isclose(gain, 3.0)
    assert metadata["applied"] is True
    assert np.isclose(metadata["rt_reference_rms_power"], 36.0)
    assert np.isclose(metadata["calibrated_po_reference_rms_power"], 36.0)
    assert metadata["warnings"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference_power_floor", np.nan),
        ("reference_power_floor", np.inf),
        ("keep_rt_baseline", "yes"),
        ("rt_refresh_frames", True),
        ("rt_refresh_frames", 1.5),
        ("rt_refresh_frames", -1),
    ],
)
def test_po_calibration_config_rejects_ambiguous_values(field, value):
    with pytest.raises(ValueError, match=field):
        POCalibrationConfig(**{field: value})


def test_po_calibration_config_normalizes_numpy_scalars():
    config = POCalibrationConfig(
        reference_power_floor=np.float32(1e-12),
        keep_rt_baseline=np.bool_(False),
        rt_refresh_frames=np.int64(2),
    )

    assert config.reference_power_floor == float(np.float32(1e-12))
    assert config.keep_rt_baseline is False
    assert config.rt_refresh_frames == 2
    assert type(config.rt_refresh_frames) is int
