# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Core sensing, target, path-array, and ADC synthesis tests."""

from __future__ import annotations

from sensing_test_helpers import *  # noqa: F401,F403
import mmWaveRadar.simulation.adc_synthesis as adc_synthesis_module


def test_ti_cli_config_parser_xwr68xx():
    cfg = """
profileCfg 0 60 7 7 58 0 0 68 1 225 4500 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
chirpCfg 2 2 0 0 0 0 0 4
frameCfg 0 2 32 0 50 1 0
"""
    path = join(tempfile.gettempdir(), "sionna_rt_test_xwr68xx.cfg")
    with open(path, "w", encoding="utf-8") as f:
        f.write(cfg)

    radar = RadarSensor.from_ti_cli_config(path)

    assert radar.fmcw.carrier_frequency == 60e9
    assert radar.fmcw.slope == 68e12
    assert radar.fmcw.num_adc_samples == 225
    assert radar.fmcw.sampling_frequency == 4.5e6
    assert radar.fmcw.num_chirps_per_frame == 32
    assert radar.fmcw.tdm_enabled is True
    assert radar.hardware.num_virtual_channels == 12


def test_mesh_sequence_interpolates_vertices():
    seq = _mesh_sequence(displacement=0.2)

    mid = seq.vertices_at(0.5)

    assert seq.vertex_count == 3
    assert seq.face_count == 1
    assert np.allclose(mid[:, 0], [1.1, 1.1, 1.1])
    assert np.isclose(seq.max_vertex_displacement(1.0, seq.vertices[0]), 0.2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("vertices", np.array([[[np.nan, 0.0, 0.0]]]), "finite"),
        ("faces", np.array([[-1, 0, 0]]), "outside vertices"),
        ("faces", np.array([[0.0, 0.0, 0.0]]), "integer"),
        ("times", np.array([np.inf]), "finite"),
    ],
)
def test_mesh_sequence_rejects_invalid_geometry(field, value, message):
    values = {
        "vertices": np.zeros((1, 1, 3), dtype=np.float32),
        "faces": np.empty((0, 3), dtype=np.uint32),
        "times": np.array([0.0]),
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        MeshSequence(**values)


def test_mesh_sequence_rejects_nonfinite_interpolation_time():
    seq = _mesh_sequence()

    with pytest.raises(ValueError, match="time must be finite"):
        seq.vertices_at(np.nan)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"visibility_samples_per_face": 0},
        {"visibility_fade_chirps": 1.5},
        {"visibility_use_phase_center": 1},
        {"adaptive_visibility_sampling": "false"},
        {"incremental_update": None},
        {"adaptive_visibility_edge_margin": np.nan},
        {"compute_backend": "invalid"},
        {"po_integration_mode": "invalid"},
        {"calibration": {}},
        {"precomputed_raw_fractional_visibility": np.array([1.0 + 0.0j])},
        {"precomputed_raw_fractional_visibility": np.array([1.1])},
        {"precomputed_visible_sample_counts": np.array([1])},
    ],
)
def test_human_po_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        HumanPOMobilityConfig(**kwargs)


def test_human_po_config_validates_precomputed_visibility_counts():
    config = HumanPOMobilityConfig(
        visibility_samples_per_face=4,
        precomputed_raw_fractional_visibility=np.array([0.25, 1.0]),
        precomputed_visible_sample_counts=np.array([1, 4]),
    )

    assert "precomputed_raw_fractional_visibility" not in repr(config)
    assert config == HumanPOMobilityConfig(
        visibility_samples_per_face=4,
        precomputed_raw_fractional_visibility=np.array([0.5]),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": 1},
        {"crossfade": "true"},
        {"match_delay_tolerance_fraction": np.inf},
    ],
)
def test_rt_transition_config_rejects_ambiguous_values(kwargs):
    with pytest.raises(ValueError):
        RTCoherentTransitionConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blocking_enabled": 1},
        {"blocking_aabb_culling": "true"},
        {"coupling_enabled": None},
        {"coupling_incremental_update": 0},
    ],
)
def test_hybrid_po_config_rejects_ambiguous_booleans(kwargs):
    with pytest.raises(ValueError):
        HybridStaticEnvPOConfig(**kwargs)


def test_amass_pose_interpolation_slerps_joint_rotations():
    poses = np.zeros((2, 72), dtype=np.float32)
    poses[1, 2] = np.pi
    translations = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]],
                            dtype=np.float32)
    times = np.array([0.0, 2.0], dtype=float)

    pose, trans = interpolate_amass_pose_params(poses, translations, times, 1.0)

    assert np.isclose(np.linalg.norm(pose[:3]), 0.5 * np.pi, atol=1e-6)
    assert np.allclose(pose[:2], 0.0, atol=1e-6)
    assert np.allclose(pose[3:], 0.0, atol=1e-6)
    assert np.allclose(trans, [1.0, 2.0, 3.0])


def test_amass_pose_interpolation_uses_shortest_rotation_arc():
    poses = np.zeros((2, 72), dtype=np.float32)
    poses[0, 2] = np.deg2rad(170.0)
    poses[1, 2] = np.deg2rad(-170.0)
    translations = np.zeros((2, 3), dtype=np.float32)
    times = np.array([0.0, 1.0], dtype=float)

    pose, _ = interpolate_amass_pose_params(poses, translations, times, 0.5)

    assert np.isclose(abs(pose[2]), np.pi, atol=1e-6)


def test_human_skin_material_has_diffuse_scattering():
    material = human_skin_material("test-tissue")

    assert np.isclose(float(material.scattering_coefficient[0]), 0.35)


def test_mesh_target_updates_scene_mesh_vertices():
    scene = load_scene()
    target = MeshTarget("target", _mesh_sequence(displacement=0.1),
                        material=human_skin_material("target-material"))
    obj = target.add_to_scene(scene)

    target.update_to_time(1.0)

    vertices = obj.mi_mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
    assert np.allclose(vertices[:, 0], [1.1, 1.1, 1.1])


def test_mesh_target_update_refreshes_rt_visibility():
    scene = load_scene()
    vertices0 = np.array([
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
    ], dtype=np.float32)
    vertices1 = vertices0.copy()
    vertices1[:, 1] += 2.0
    sequence = MeshSequence(
        vertices=np.stack([vertices0, vertices1], axis=0),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32),
        times=np.array([0.0, 1.0]),
    )
    target = MeshTarget(
        "target-blocker",
        sequence,
        material=human_skin_material("target-blocker-material"),
    )
    target.add_to_scene(scene)

    ray = mi.Ray3f(mi.Point3f(0.0, 0.0, 0.0), mi.Vector3f(1.0, 0.0, 0.0))
    ray.maxt = mi.Float(1.0)
    assert bool(scene.mi_scene.ray_test(ray)[0])

    target.update_to_time(1.0)

    assert not bool(scene.mi_scene.ray_test(ray)[0])


def test_rt_target_face_hit_counts_preserve_original_mesh_indices():
    from mmWaveRadar.simulation.simulator import _target_face_hit_counts

    vertices = np.asarray(
        [
            [
                [1.0, -0.5, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.5, 0.0],
                [1.2, 0.0, 0.5],
            ]
        ],
        dtype=np.float32,
    )
    sequence = MeshSequence(
        vertices=vertices,
        faces=np.asarray([(0, 1, 2), (0, 3, 1)], dtype=np.uint32),
        times=np.asarray([0.0]),
    )
    target = MeshTarget(
        "hit-target",
        sequence,
        material=human_skin_material("hit-target-material"),
    )
    object_id = int(
        np.asarray(target.ensure_scene_object().object_id).reshape(-1)[0]
    )
    invalid = np.uint32(0xFFFFFFFF)

    class FakePaths:
        valid = np.asarray([[[True, True, False]]])
        objects = np.asarray(
            [
                [[[object_id, object_id, object_id]]],
                [[[object_id, invalid, object_id]]],
            ],
            dtype=np.uint32,
        )
        primitives = np.asarray(
            [
                [[[0, 1, 0]]],
                [[[1, invalid, 1]]],
            ],
            dtype=np.uint32,
        )

    counts = _target_face_hit_counts(FakePaths(), [target])

    np.testing.assert_array_equal(counts["hit-target"], [1, 2])


def test_one_target_bounce_filter():
    target = MeshTarget("target", _mesh_sequence(),
                        material=human_skin_material("target-material"))
    tid = int(np.asarray(target.ensure_scene_object().object_id).reshape(-1)[0])

    class FakePaths:
        tau = np.zeros((1, 1, 3))
        objects = np.array([
            [[[tid, 7, tid]]],
            [[[9, tid, tid]]],
        ], dtype=np.uint32)

    mask = MmWaveRadarSimulator.one_target_bounce_mask(FakePaths(), [target])

    assert np.array_equal(mask, [True, True, False])


def test_human_specular_reflection_flag_preserves_legacy_specular_paths():
    target = MeshTarget("target", _mesh_sequence(),
                        material=human_skin_material("target-spec-material"))
    tid = int(np.asarray(target.ensure_scene_object().object_id).reshape(-1)[0])
    a = np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex128)
    tau = np.array([[[1e-9]]], dtype=float)
    interactions = np.array([[[[InteractionType.SPECULAR]]]], dtype=np.uint32)
    objects = np.array([[[[tid]]]], dtype=np.uint32)

    class FakePaths:
        def cir(self, normalize_delays=False, out_type="numpy"):
            del normalize_delays, out_type
            return a, tau

        @property
        def interactions(self):
            return interactions

        @property
        def objects(self):
            return objects

    sim = MmWaveRadarSimulator(
        coupling_mode="unrestricted",
        human_specular_reflection=True,
    )

    _, _, valid = sim._path_arrays(
        FakePaths(), [target], virtual_channel_order="tx_major")

    assert np.array_equal(valid, [True])


def test_human_specular_reflection_false_drops_only_human_specular_paths():
    target = MeshTarget("target", _mesh_sequence(),
                        material=human_skin_material("target-nospec-material"))
    tid = int(np.asarray(target.ensure_scene_object().object_id).reshape(-1)[0])
    env_id = 99
    a = np.ones((1, 1, 1, 1, 4, 1), dtype=np.complex128)
    tau = np.array([[[1e-9, 2e-9, 3e-9, 4e-9]]], dtype=float)
    interactions = np.array([[[[
        InteractionType.SPECULAR,
        InteractionType.DIFFUSE,
        InteractionType.SPECULAR,
        InteractionType.SPECULAR,
    ]]]], dtype=np.uint32)
    objects = np.array([[[[tid, tid, env_id, env_id]]]], dtype=np.uint32)

    class FakePaths:
        def cir(self, normalize_delays=False, out_type="numpy"):
            del normalize_delays, out_type
            return a, tau

        @property
        def interactions(self):
            return interactions

        @property
        def objects(self):
            return objects

    sim = MmWaveRadarSimulator(coupling_mode="unrestricted")

    _, _, valid = sim._path_arrays(
        FakePaths(), [target], virtual_channel_order="tx_major")

    assert np.array_equal(valid, [False, True, True, True])


def test_path_arrays_respect_virtual_channel_order():
    a = np.zeros((1, 2, 1, 3, 1, 1), dtype=np.complex128)
    a[0, 0, 0, 0, 0, 0] = 10.0
    a[0, 1, 0, 0, 0, 0] = 20.0
    a[0, 0, 0, 1, 0, 0] = 30.0
    a[0, 1, 0, 1, 0, 0] = 40.0
    a[0, 0, 0, 2, 0, 0] = 50.0
    a[0, 1, 0, 2, 0, 0] = 60.0
    tau = np.array([[[7e-9]]], dtype=float)

    class FakePaths:
        def cir(self, normalize_delays=False, out_type="numpy"):
            del normalize_delays, out_type
            return a, tau

    sim = MmWaveRadarSimulator(coupling_mode="unrestricted")

    tx_major, tau_tx, valid_tx = sim._path_arrays(
        FakePaths(), [], virtual_channel_order="tx_major")
    rx_major, tau_rx, valid_rx = sim._path_arrays(
        FakePaths(), [], virtual_channel_order="rx_major")

    assert np.array_equal(tx_major[:, 0], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    assert np.array_equal(rx_major[:, 0], [10.0, 30.0, 50.0, 20.0, 40.0, 60.0])
    assert np.array_equal(tau_tx, [7e-9])
    assert np.array_equal(tau_rx, [7e-9])
    assert np.array_equal(valid_tx, [True])
    assert np.array_equal(valid_rx, [True])


def test_path_arrays_respect_explicit_board_channel_map():
    a = np.zeros((1, 2, 1, 3, 1, 1), dtype=np.complex128)
    a[0, :, 0, :, 0, 0] = np.array([
        [10.0, 30.0, 50.0],
        [20.0, 40.0, 60.0],
    ])
    tau = np.array([[[7e-9]]], dtype=float)

    class FakePaths:
        def cir(self, normalize_delays=False, out_type="numpy"):
            del normalize_delays, out_type
            return a, tau

    sim = MmWaveRadarSimulator(coupling_mode="unrestricted")
    coefficients, _, _ = sim._path_arrays(
        FakePaths(),
        [],
        virtual_channel_order="tx_major",
        virtual_channel_tx_indices=np.array([2, 0, 1, 2, 0, 1]),
        virtual_channel_rx_indices=np.array([1, 0, 1, 0, 1, 0]),
    )

    assert np.array_equal(
        coefficients[:, 0],
        [60.0, 10.0, 40.0, 50.0, 20.0, 30.0],
    )


def test_path_arrays_handle_empty_paths():
    a = np.zeros((1, 2, 1, 3, 0, 1), dtype=np.complex128)
    tau = np.zeros((1, 1, 0), dtype=float)

    class FakePaths:
        def cir(self, normalize_delays=False, out_type="numpy"):
            del normalize_delays, out_type
            return a, tau

    sim = MmWaveRadarSimulator(coupling_mode="unrestricted")

    coeffs, delays, valid = sim._path_arrays(
        FakePaths(), [], virtual_channel_order="tx_major")

    assert coeffs.shape == (6, 0)
    assert delays.shape == (0,)
    assert valid.shape == (0,)


def test_adc_synthesis_matches_single_path_beat():
    fmcw = FMCWConfig(carrier_frequency=60e9, slope=2e12,
                      chirp_duration=1e-3, chirp_repetition_time=1e-3,
                      sampling_frequency=10e3, num_adc_samples=8,
                      num_chirps_per_frame=1, frame_period=1e-3)
    tau = np.array([2e-9])
    a = np.array([[1.0 + 0.0j]])

    y = MmWaveRadarSimulator.synthesize_adc_from_paths(a, tau, fmcw)

    t = np.arange(fmcw.num_adc_samples) / fmcw.sampling_frequency
    expected = np.exp(1j * 2.0 * np.pi * fmcw.slope * tau[0] * t)
    assert np.allclose(y[:, 0], expected)


def test_torch_fast_time_reuses_immutable_sampling_tensor():
    torch = pytest.importorskip("torch")
    adc_synthesis_module._torch_fast_time.cache_clear()

    first = adc_synthesis_module._torch_fast_time(
        225,
        4.5e6,
        torch.device("cpu"),
        torch.float32,
    )
    second = adc_synthesis_module._torch_fast_time(
        225,
        4.5e6,
        torch.device("cpu"),
        torch.float32,
    )
    different = adc_synthesis_module._torch_fast_time(
        225,
        5.0e6,
        torch.device("cpu"),
        torch.float32,
    )

    assert first is second
    assert different is not first
    np.testing.assert_array_equal(
        first.numpy(),
        torch.arange(225, dtype=torch.float32).numpy() / 4.5e6,
    )


def test_adc_synthesis_is_public_simulation_api():
    from mmWaveRadar.simulation import synthesize_adc_from_paths

    assert synthesize_adc_from_paths is MmWaveRadarSimulator.synthesize_adc_from_paths


def test_adc_synthesis_accepts_channel_path_delay_matrix():
    fmcw = FMCWConfig(carrier_frequency=60e9, slope=2e12,
                      chirp_duration=1e-3, chirp_repetition_time=1e-3,
                      sampling_frequency=10e3, num_adc_samples=8,
                      num_chirps_per_frame=1, frame_period=1e-3)
    tau = np.array([[2e-9], [3e-9]])
    a = np.ones((2, 1), dtype=np.complex128)

    y = MmWaveRadarSimulator.synthesize_adc_from_paths(a, tau, fmcw,
                                                       backend="numpy")

    t = np.arange(fmcw.num_adc_samples) / fmcw.sampling_frequency
    expected = np.exp(1j * 2.0 * np.pi * fmcw.slope * tau * t[None, :])
    assert np.allclose(y[:, 0], expected[0])
    assert np.allclose(y[:, 1], expected[1])


@pytest.mark.parametrize(
    ("coefficients", "delays", "message"),
    [
        (np.ones((1, 1)), np.array([np.nan]), "finite delays"),
        (np.ones((1, 1)), np.array([1.0 + 0.0j]), "real-valued delays"),
        (np.array([[np.nan + 0.0j]]), np.array([1e-9]), "finite coefficients"),
    ],
)
def test_adc_synthesis_rejects_nonfinite_or_complex_path_data(
    coefficients,
    delays,
    message,
):
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=2e12,
        chirp_duration=1e-3,
        chirp_repetition_time=1e-3,
        sampling_frequency=10e3,
        num_adc_samples=8,
        num_chirps_per_frame=1,
        frame_period=1e-3,
    )

    with pytest.raises(ValueError, match=message):
        MmWaveRadarSimulator.synthesize_adc_from_paths(
            coefficients,
            delays,
            fmcw,
        )


def test_adc_synthesis_validates_backend_for_an_empty_path_bank():
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=2e12,
        chirp_duration=1e-3,
        chirp_repetition_time=1e-3,
        sampling_frequency=10e3,
        num_adc_samples=8,
        num_chirps_per_frame=1,
        frame_period=1e-3,
    )

    with pytest.raises(ValueError, match="backend"):
        MmWaveRadarSimulator.synthesize_adc_from_paths(
            np.empty((1, 0), dtype=np.complex128),
            np.empty((0,), dtype=float),
            fmcw,
            backend="invalid",
        )
