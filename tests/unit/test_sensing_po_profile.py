# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""PO profile cache and sweep tests."""

from __future__ import annotations

from sensing_test_helpers import *  # noqa: F401,F403
from mmWaveRadar.simulation import relative_norm_error


def test_po_profile_cache_round_trip_and_match(tmp_path):
    fmcw = FMCWConfig(carrier_frequency=60e9, slope=1e12,
                      chirp_duration=1e-3, chirp_repetition_time=0.5,
                      sampling_frequency=2e3, num_adc_samples=4,
                      num_chirps_per_frame=2, frame_period=1.0)
    adc = np.ones((2, 2, 4, 1), dtype=np.complex128)
    cube = _profile_test_cube(
        adc,
        fmcw,
        wall_scale=0.25,
        recomputed=np.array([[1, 0], [0, 1]], dtype=np.int64),
    )
    human_touch_counts = np.arange(4, dtype=np.int64).reshape(2, 2)
    depth_histogram = np.ones((2, 2, 3), dtype=np.int64)
    path_power = np.arange(1, 5, dtype=np.float64).reshape(2, 2)
    cube = RadarCube(
        adc=cube.adc,
        times=cube.times,
        metadata=replace(
            cube.metadata,
            events=[
                {"frame": 0, "chirp": 0, "reason": "initial"},
                {"frame": 1, "chirp": 1, "reason": "displacement"},
            ],
            po_calibration={
                "mode": "rt_first_pose",
                "applied": True,
                "human_po_amplitude_gain": 2.0,
                "warnings": [],
            },
            human_touch_path_counts=human_touch_counts,
            one_human_touch_path_counts=human_touch_counts + 1,
            single_human_only_path_counts=human_touch_counts + 2,
            human_env_coupled_path_counts=human_touch_counts + 3,
            human_multi_touch_path_counts=human_touch_counts + 4,
            path_depth_histogram=depth_histogram,
            total_path_power=path_power,
            human_touch_path_power=path_power + 1.0,
            one_human_touch_path_power=path_power + 2.0,
            single_human_only_path_power=path_power + 3.0,
            human_env_coupled_path_power=path_power + 4.0,
            human_multi_touch_path_power=path_power + 5.0,
        ),
        components=cube.components,
    )
    path = tmp_path / "cube.npz"

    save_radar_cube_npz(path, cube, wall_time_s=3.0, num_chirps=4,
                        label="profile")
    loaded, wall_time_s, num_chirps = load_radar_cube_npz(path)

    assert wall_time_s == 3.0
    assert num_chirps == 4
    assert np.array_equal(loaded.adc, cube.adc)
    assert loaded.metadata.metadata_schema_version == SENSING_METADATA_SCHEMA_VERSION
    assert np.array_equal(loaded.metadata.incremental_recomputed_face_counts,
                          cube.metadata.incremental_recomputed_face_counts)
    assert loaded.metadata.runtime_profile_s["po_incremental_update"] == 0.25
    assert loaded.metadata.events == [
        {"frame": 0, "chirp": 0, "reason": "initial"},
        {"frame": 1, "chirp": 1, "reason": "displacement"},
    ]
    assert loaded.metadata.po_calibration["mode"] == "rt_first_pose"
    assert loaded.metadata.po_calibration["human_po_amplitude_gain"] == 2.0
    assert np.array_equal(
        loaded.metadata.human_touch_path_counts,
        human_touch_counts,
    )
    assert np.array_equal(
        loaded.metadata.one_human_touch_path_counts,
        human_touch_counts + 1,
    )
    assert np.array_equal(
        loaded.metadata.single_human_only_path_counts,
        human_touch_counts + 2,
    )
    assert np.array_equal(
        loaded.metadata.human_env_coupled_path_counts,
        human_touch_counts + 3,
    )
    assert np.array_equal(
        loaded.metadata.human_multi_touch_path_counts,
        human_touch_counts + 4,
    )
    assert np.array_equal(
        loaded.metadata.path_depth_histogram,
        depth_histogram,
    )
    assert np.array_equal(loaded.metadata.total_path_power, path_power)
    assert np.array_equal(
        loaded.metadata.human_touch_path_power,
        path_power + 1.0,
    )
    assert np.array_equal(
        loaded.metadata.one_human_touch_path_power,
        path_power + 2.0,
    )
    assert np.array_equal(
        loaded.metadata.single_human_only_path_power,
        path_power + 3.0,
    )
    assert np.array_equal(
        loaded.metadata.human_env_coupled_path_power,
        path_power + 4.0,
    )
    assert np.array_equal(
        loaded.metadata.human_multi_touch_path_power,
        path_power + 5.0,
    )
    assert cube_matches_run(loaded, num_frames=2, fmcw=fmcw,
                            last_chirp_time_s=fmcw.chirp_time(1, 1))
    assert not cube_matches_run(loaded, num_frames=3, fmcw=fmcw)


def test_run_incremental_po_sweep_summarizes_and_reuses_cache(tmp_path):
    fmcw = FMCWConfig(carrier_frequency=60e9, slope=1e12,
                      chirp_duration=1e-3, chirp_repetition_time=0.5,
                      sampling_frequency=2e3, num_adc_samples=4,
                      num_chirps_per_frame=2, frame_period=1.0)
    baseline_adc = np.ones((1, 2, 4, 1), dtype=np.complex128)
    baseline = _profile_test_cube(baseline_adc, fmcw)
    setting = POIncrementalSweepSetting(
        label="k64 loose",
        visibility_refresh_chirps=64,
        normal_threshold_deg=20.0,
        centroid_displacement_threshold_m=0.02,
        area_relative_threshold=0.30,
    )
    calls = []

    def run_cube(label, config):
        calls.append((label, config))
        adc = baseline_adc * 1.1
        cube = _profile_test_cube(
            adc,
            fmcw,
            recomputed=np.array([[2, 4]], dtype=np.int64),
        )
        return cube, 2.0, 2

    first = run_incremental_po_sweep(
        settings=[setting],
        baseline_cube=baseline,
        baseline_wall_time_s=10.0,
        fmcw=fmcw,
        visibility_samples_per_face=4,
        run_cube=run_cube,
        output_dir=tmp_path,
        reuse_saved=True,
    )
    second = run_incremental_po_sweep(
        settings=[setting],
        baseline_cube=baseline,
        baseline_wall_time_s=10.0,
        fmcw=fmcw,
        visibility_samples_per_face=4,
        run_cube=run_cube,
        output_dir=tmp_path,
        reuse_saved=True,
        cache_match=lambda cube: cube_matches_run(cube, num_frames=1, fmcw=fmcw),
    )

    assert len(calls) == 1
    config = calls[0][1]
    assert isinstance(config, HumanPOMobilityConfig)
    assert config.incremental_visibility_refresh_chirps == 64
    assert config.incremental_full_refresh_chirps == 0
    assert config.incremental_normal_threshold_deg == 20.0
    assert first[0].speedup == 5.0
    assert first[0].recomputed_mean == 3.0
    assert first[0].adc_relative_error > 0.0
    assert second[0].speedup == 5.0


def test_make_incremental_po_config_uses_setting_values():
    setting = POIncrementalSweepSetting(
        label="candidate",
        visibility_refresh_chirps=128,
        normal_threshold_deg=30.0,
        centroid_displacement_threshold_m=0.05,
        area_relative_threshold=0.5,
    )

    config = make_incremental_po_config(setting, visibility_samples_per_face=4)

    assert config.visibility_samples_per_face == 4
    assert config.adaptive_visibility_sampling is True
    assert config.incremental_update is True
    assert config.incremental_visibility_refresh_chirps == 128
    assert config.incremental_full_refresh_chirps == 0
    assert config.incremental_normal_threshold_deg == 30.0
    assert config.incremental_centroid_displacement_threshold_m == 0.05
    assert config.incremental_area_relative_threshold == 0.5


def test_relative_norm_error_rejects_broadcastable_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        relative_norm_error(np.ones((2, 1)), np.ones((2,)))


def test_load_legacy_single_frame_cube_counts_chirps_not_samples(tmp_path):
    path = tmp_path / "legacy-single-frame.npz"
    np.savez_compressed(
        path,
        adc=np.ones((3, 8, 1), dtype=np.complex128),
        times=np.arange(3, dtype=float),
        path_counts=np.ones(3, dtype=np.int64),
    )

    cube, _, num_chirps = load_radar_cube_npz(path)

    assert num_chirps == 3
    assert cube_matches_run(
        cube,
        num_frames=1,
        fmcw=FMCWConfig(
            carrier_frequency=60e9,
            slope=1e12,
            chirp_duration=1e-3,
            chirp_repetition_time=0.5,
            sampling_frequency=2e3,
            num_adc_samples=8,
            num_chirps_per_frame=3,
            frame_period=2.0,
        ),
        last_chirp_time_s=2.0,
    )


def test_save_radar_cube_rejects_colliding_component_names(tmp_path):
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=1e12,
        chirp_duration=1e-3,
        chirp_repetition_time=0.5,
        sampling_frequency=2e3,
        num_adc_samples=4,
        num_chirps_per_frame=2,
        frame_period=1.0,
    )
    base = _profile_test_cube(np.ones((1, 2, 4, 1)), fmcw)
    cube = RadarCube(
        adc=base.adc,
        times=base.times,
        metadata=base.metadata,
        components={"human env": base.adc, "human-env": base.adc},
    )

    with pytest.raises(ValueError, match="duplicate slug"):
        save_radar_cube_npz(
            tmp_path / "cube.npz",
            cube,
            wall_time_s=1.0,
            num_chirps=2,
        )
