# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Static path blocking tests."""

from __future__ import annotations

from sensing_test_helpers import *  # noqa: F401,F403


def test_static_path_blocking_detects_crossing_segment():
    vertices = _triangle_from_yz(0.5, (-0.5, -0.5), (0.5, -0.5), (0.0, 0.5))
    scene = HumanVisibilityScene(
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        vertices=vertices,
        name="blocking-test-human",
    )
    bank = StaticPathBank(
        coefficients=np.ones((1, 2), dtype=np.complex128),
        delays_s=np.array([1e-9, 2e-9], dtype=float),
        valid=np.array([True, True]),
        segment_starts=np.array([[0.0, 0.0, 0.0], [0.0, 0.75, 0.0]]),
        segment_ends=np.array([[1.0, 0.0, 0.0], [1.0, 0.75, 0.0]]),
        segment_path_indices=np.array([0, 1], dtype=np.int64),
    )

    result = blocked_path_visibility(
        bank,
        scene,
        fade_chirps=1,
        ray_epsilon_m=1e-5,
    )

    assert np.array_equal(result.blocked_paths, [True, False])
    assert np.array_equal(result.raw_visibility, [0.0, 1.0])
    assert np.array_equal(result.visibility_weights, [0.0, 1.0])


def test_static_path_blocking_aabb_culls_far_segments():
    vertices = _triangle_from_yz(0.5, (-0.5, -0.5), (0.5, -0.5), (0.0, 0.5))
    scene = HumanVisibilityScene(
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        vertices=vertices,
        name="blocking-aabb-human",
    )
    bank = StaticPathBank(
        coefficients=np.ones((1, 3), dtype=np.complex128),
        delays_s=np.array([1e-9, 2e-9, 3e-9], dtype=float),
        valid=np.array([True, True, True]),
        segment_starts=np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.75, 0.0],
            [0.0, 10.0, 0.0],
        ]),
        segment_ends=np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.75, 0.0],
            [1.0, 10.0, 0.0],
        ]),
        segment_path_indices=np.array([0, 1, 2], dtype=np.int64),
    )

    candidates = segment_indices_intersecting_aabb(
        bank,
        vertices,
        margin_m=0.0,
    )
    result = blocked_path_visibility(
        bank,
        scene,
        blocker_vertices=vertices,
        aabb_culling=True,
        aabb_margin_m=0.0,
        fade_chirps=1,
        ray_epsilon_m=1e-5,
    )

    assert np.array_equal(candidates, [0])
    assert np.array_equal(result.blocked_paths, [True, False, False])
    assert np.array_equal(result.raw_visibility, [0.0, 1.0, 1.0])


def test_static_path_blocking_ignores_zero_length_segment():
    vertices = _triangle_from_yz(0.5, (-0.5, -0.5), (0.5, -0.5), (0.0, 0.5))
    scene = HumanVisibilityScene(
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        vertices=vertices,
        name="blocking-zero-segment-human",
    )
    bank = StaticPathBank(
        coefficients=np.ones((1, 1), dtype=np.complex128),
        delays_s=np.array([1e-9], dtype=float),
        valid=np.array([True]),
        segment_starts=np.array([[0.0, 0.0, 0.0]]),
        segment_ends=np.array([[0.0, 0.0, 0.0]]),
        segment_path_indices=np.array([0], dtype=np.int64),
    )

    result = blocked_path_visibility(bank, scene, fade_chirps=1)

    assert np.array_equal(result.blocked_paths, [False])
    assert np.array_equal(result.visibility_weights, [1.0])


def test_path_visibility_state_fades_transitions():
    fade, latched = update_path_visibility_state(
        np.array([1.0]),
        state=None,
        fade_chirps=4,
    )
    state = PathVisibilityState(
        was_visible=np.array([True]),
        fade_weights=fade,
        latched_visibility=latched,
    )

    fade, latched = update_path_visibility_state(
        np.array([0.0]),
        state=state,
        fade_chirps=4,
    )

    assert np.allclose(fade, [0.75])
    assert np.allclose(latched, [1.0])
    state = PathVisibilityState(
        was_visible=np.array([False]),
        fade_weights=fade,
        latched_visibility=latched,
    )
    fade, latched = update_path_visibility_state(
        np.array([1.0]),
        state=state,
        fade_chirps=4,
    )
    assert np.allclose(fade, [1.0])
    assert np.allclose(latched, [1.0])


def test_extract_static_path_bank_builds_direct_and_reflected_segments():
    fmcw = FMCWConfig(carrier_frequency=60e9, slope=1e12,
                      chirp_duration=1e-3, chirp_repetition_time=0.5,
                      sampling_frequency=2e3, num_adc_samples=4,
                      num_chirps_per_frame=2, frame_period=1.0)
    radar = RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware.from_positions(
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
        ),
        fmcw=fmcw,
    )
    a = np.zeros((1, 1, 1, 1, 3, 1), dtype=np.complex128)
    a[0, 0, 0, 0, :, 0] = [1.0, 2.0, 3.0]
    tau = np.array([[[1e-9, 2e-9, -1.0]]], dtype=float)
    vertices = np.zeros((2, 1, 1, 3, 3), dtype=float)
    vertices[0, 0, 0, 1] = [0.5, 0.25, 0.0]
    vertices[1, 0, 0, 1] = [0.75, 0.25, 0.0]
    interactions = np.zeros((2, 1, 1, 3), dtype=np.uint32)
    interactions[:, 0, 0, 1] = 1

    class FakePaths:
        def cir(self, normalize_delays=False, out_type="numpy"):
            del normalize_delays, out_type
            return a, tau

        @property
        def vertices(self):
            return vertices

        @property
        def interactions(self):
            return interactions

    bank = extract_static_path_bank(FakePaths(), radar)

    assert np.array_equal(bank.valid, [True, True, False])
    assert np.array_equal(bank.coefficients, [[1.0, 2.0, 3.0]])
    assert bank.segment_starts.shape == (4, 3)
    assert np.array_equal(bank.segment_path_indices, [0, 1, 1, 1])
    assert np.allclose(bank.segment_starts[0], [0.0, 0.0, 0.0])
    assert np.allclose(bank.segment_ends[0], [1.0, 0.0, 0.0])


def test_extract_static_path_bank_uses_board_channel_map():
    hardware = RadarHardware(
        name="permuted-static-rt",
        tx_positions=np.zeros((2, 3), dtype=float),
        rx_positions=np.ones((2, 3), dtype=float),
        virtual_channel_tx_indices=np.array([1, 0, 1, 0]),
        virtual_channel_rx_indices=np.array([1, 0, 0, 1]),
    )
    radar = RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=hardware,
        fmcw=FMCWConfig(
            carrier_frequency=60e9,
            slope=1e12,
            chirp_duration=1e-3,
            chirp_repetition_time=0.5,
            sampling_frequency=2e3,
            num_adc_samples=4,
            num_chirps_per_frame=2,
            frame_period=1.0,
            num_tx=2,
        ),
    )
    a = np.zeros((1, 2, 1, 2, 1, 1), dtype=np.complex128)
    a[0, :, 0, :, 0, 0] = np.array([[10.0, 30.0], [20.0, 40.0]])
    tau = np.array([[[1e-9]]], dtype=float)

    class FakePaths:
        vertices = np.zeros((1, 1, 1, 1, 3), dtype=float)
        interactions = np.zeros((1, 1, 1, 1), dtype=np.uint32)

        def cir(self, normalize_delays=False, out_type="numpy"):
            del normalize_delays, out_type
            return a, tau

    bank = extract_static_path_bank(FakePaths(), radar)

    assert np.array_equal(bank.coefficients[:, 0], [40.0, 10.0, 30.0, 20.0])
