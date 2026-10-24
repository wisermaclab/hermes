# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Radar-frame conversion for post-trace antenna patterns."""

from types import SimpleNamespace

from sensing_test_helpers import *  # noqa: F401,F403
from sionna.rt.utils import rotation_matrix

from mmWaveRadar.radar.ti import SimpleCosinePattern
from mmWaveRadar.simulation.channel_gain import (
    apply_path_hardware_gain,
    path_azimuth_elevation_deg,
)
from mmWaveRadar.simulation.static_blocking import _path_arrays


def _scene_rotation(orientation):
    # Use Sionna's own rotation as an independent check of our pose convention.
    return np.asarray(rotation_matrix(mi.Point3f(*orientation))).reshape(3, 3)


@pytest.mark.parametrize(
    "orientation",
    [
        (0.0, 0.0, 0.0),
        (np.pi / 2, 0.0, 0.0),
        (0.0, np.pi / 6, 0.0),
        (0.0, 0.0, np.pi / 3),
        (0.7, -0.4, 0.2),
    ],
)
def test_rt_path_angles_are_transformed_from_scene_to_radar_frame(orientation):
    azimuth = np.array([0.0, 25.0, -35.0])
    elevation = np.array([0.0, 10.0, -20.0])
    az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
    local = np.stack(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        axis=-1,
    )
    world = local @ _scene_rotation(orientation).T
    paths = SimpleNamespace(
        theta_t=np.arccos(np.clip(world[:, 2], -1.0, 1.0))[None, None, :],
        phi_t=np.arctan2(world[:, 1], world[:, 0])[None, None, :],
    )

    actual_az, actual_el = path_azimuth_elevation_deg(
        paths,
        radar_orientation=orientation,
    )

    # Sionna's reference rotation uses single-precision device angles.
    np.testing.assert_allclose(actual_az, azimuth, atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(actual_el, elevation, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize(
    "orientation",
    [
        (0.0, 0.0, 0.0),
        (np.pi / 2, 0.0, 0.0),
        (0.7, -0.4, 0.2),
    ],
)
@pytest.mark.parametrize("bank_kind", ["static", "coherent"])
def test_real_sionna_boresight_gain_is_independent_of_scene_orientation(
    orientation,
    bank_kind,
):
    radar = _phase_one_radar(num_chirps=1)
    radar = replace(
        radar,
        position=(2.0, -3.0, 0.5),
        orientation=orientation,
        hardware=replace(radar.hardware, antenna_pattern=SimpleCosinePattern(1)),
    )
    scene = load_scene()
    _, receiver = radar.configure_scene(scene)
    # A separate receiver lets a real LOS ray probe the transmitter boresight.
    receiver.position = np.asarray(radar.position) + _scene_rotation(orientation)[:, 0]
    simulator = MmWaveRadarSimulator(
        rt_samples_per_source=100,
        rt_max_paths_per_source=100,
        rt_max_depth=1,
        compute_backend="numpy",
    )
    paths = simulator.trace_scene(scene)
    raw, _, valid = _path_arrays(paths, virtual_channel_order="tx_major")
    if bank_kind == "static":
        bank = extract_static_path_bank(paths, radar)
    else:
        bank = extract_coherent_rt_path_bank(paths, radar, [])

    # The Tx boresight ray arrives at the back of the identically oriented Rx.
    # The combined-pattern floor is -80 dB, split equally between endpoints.
    np.testing.assert_allclose(
        bank.coefficients[:, valid], 0.01 * raw[:, valid], rtol=1e-6, atol=0.0
    )
    azimuth, elevation = path_azimuth_elevation_deg(
        paths,
        radar_orientation=radar.orientation,
    )
    np.testing.assert_allclose(azimuth[valid], 0.0, atol=2e-5)
    np.testing.assert_allclose(elevation[valid], 0.0, atol=2e-5)


def test_missing_path_angles_preserve_scalar_and_phase_gain_for_rotated_radar():
    hardware = replace(
        _phase_one_radar(num_chirps=1).hardware,
        antenna_pattern=SimpleCosinePattern(1),
        scalar_antenna_gain_dbi=6.0,
        virtual_channel_phase_signs=np.array([-1.0]),
    )
    coefficients = np.array([[0.3 + 0.2j]])
    actual = apply_path_hardware_gain(
        hardware,
        coefficients,
        paths=SimpleNamespace(),
        radar_orientation=(0.7, -0.4, 0.2),
    )
    np.testing.assert_allclose(actual, -(10.0 ** (6.0 / 20.0)) * coefficients)
