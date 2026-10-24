# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""RT retrace and coherent-path-bank tests."""

from __future__ import annotations

from types import SimpleNamespace

from sensing_test_helpers import *  # noqa: F401,F403
from mmWaveRadar.tutorial_support import BEDROOM_SCENE_BOXES
from mmWaveRadar.simulation.compute import resolve_torch_runtime


def test_barycentric_coordinates_reconstruct_point():
    triangle = np.array(
        [
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 0.0, 2.0],
        ]
    )
    point = np.mean(triangle, axis=0)

    bary = barycentric_coordinates(point, triangle)

    assert np.allclose(bary, [1.0 / 3.0] * 3)
    assert np.allclose(bary @ triangle, point)


def test_coherent_rt_path_bank_updates_dynamic_hit_phase_and_delay():
    radar = _phase_one_radar(num_chirps=2)
    vertices0 = np.array(
        [
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    vertices1 = vertices0.copy()
    vertices1[:, 0] += 0.1
    mesh_sequence = MeshSequence(
        vertices=np.stack([vertices0, vertices1], axis=0),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        times=np.array([0.0, 1.0]),
    )

    class DummySceneObject:
        object_id = np.array([7], dtype=np.uint32)

    class DummyTarget:
        def __init__(self, sequence):
            self.mesh_sequence = sequence

        def ensure_scene_object(self):
            return DummySceneObject()

    a = np.zeros((1, 1, 1, 1, 2, 1), dtype=np.complex128)
    a[0, 0, 0, 0, :, 0] = [2.0 + 0.0j, 3.0 + 0.0j]
    tau = np.array([[[2.0 / c, 2.0 / c]]], dtype=float)
    path_vertices = np.zeros((1, 1, 1, 2, 3), dtype=float)
    path_vertices[0, 0, 0, 0] = np.mean(vertices0, axis=0)
    path_vertices[0, 0, 0, 1] = [0.0, 1.0, 0.0]
    interactions = np.full((1, 1, 1, 2), InteractionType.SPECULAR, dtype=np.uint32)
    objects = np.array([[[[7, 99]]]], dtype=np.uint32)
    primitives = np.array([[[[0, 0]]]], dtype=np.uint32)

    class FakePaths:
        def cir(self, normalize_delays=False, out_type="numpy"):
            del normalize_delays, out_type
            return a, tau

        @property
        def vertices(self):
            return path_vertices

        @property
        def interactions(self):
            return interactions

        @property
        def objects(self):
            return objects

        @property
        def primitives(self):
            return primitives

    bank = extract_coherent_rt_path_bank(FakePaths(), radar, [DummyTarget(mesh_sequence)])
    update = update_coherent_rt_path_bank(bank, [vertices1])

    assert isinstance(bank, CoherentRTPathBank)
    assert isinstance(update, CoherentRTPathUpdate)
    assert bank.path_signatures == (
        ((InteractionType.SPECULAR, 7, 0),),
        ((InteractionType.SPECULAR, 99, 0),),
    )
    assert np.array_equal(bank.dynamic_target_indices, [[0, -1]])
    assert np.allclose(bank.dynamic_barycentrics[0, 0], [1.0 / 3.0] * 3)
    assert np.allclose(update.delay_lengths_m, [2.2, 2.0])

    k0 = 2.0 * pi * radar.fmcw.carrier_frequency / c
    length_delta = update.path_lengths_m[0, 0] - bank.anchor_path_lengths_m[0, 0]
    spread_ratio = bank.anchor_spread_products_m[0, 0] / update.spread_products_m[0, 0]
    expected_dynamic = 2.0 * np.exp(-1j * k0 * length_delta) * spread_ratio
    assert np.allclose(update.coefficients[0, 0], expected_dynamic)
    assert np.allclose(update.coefficients[0, 1], 3.0 + 0.0j)
    assert np.allclose(update.delays_s, [2.2 / c, 2.0 / c])


def test_single_vc_path_lengths_match_scalar_reference():
    path_vertices = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [np.nan, np.nan, np.nan]],
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [np.nan, np.nan, np.nan]],
        ],
        dtype=float,
    )
    interactions = np.array(
        [
            [InteractionType.SPECULAR, InteractionType.SPECULAR, InteractionType.NONE],
            [InteractionType.SPECULAR, InteractionType.NONE, InteractionType.NONE],
        ],
        dtype=np.uint32,
    )
    primitives = np.array(
        [
            [0, 0, INVALID_PRIMITIVE],
            [0, INVALID_PRIMITIVE, INVALID_PRIMITIVE],
        ],
        dtype=np.uint32,
    )
    dynamic_target_indices = np.array(
        [
            [0, -1, -1],
            [-1, -1, -1],
        ],
        dtype=np.int64,
    )
    dynamic_barycentrics = np.full((2, 3, 3), np.nan, dtype=float)
    dynamic_barycentrics[0, 0] = [0.2, 0.3, 0.5]
    target_faces = (np.array([[0, 1, 2]], dtype=np.int64),)
    target_vertices = (
        np.array(
            [
                [1.2, -1.0, 0.0],
                [1.2, 1.0, 0.0],
                [1.2, 0.0, 2.0],
            ],
            dtype=float,
        ),
    )
    tx_position = np.array([[0.0, 0.0, 0.0]], dtype=float)
    rx_position = np.array([[0.0, 0.0, 0.0]], dtype=float)

    lengths, spreads = rt_coherence._path_lengths_and_spreads(
        path_vertices=path_vertices,
        interactions=interactions,
        primitives=primitives,
        dynamic_target_indices=dynamic_target_indices,
        dynamic_barycentrics=dynamic_barycentrics,
        target_faces=target_faces,
        target_vertices=target_vertices,
        tx_positions=tx_position,
        rx_positions=rx_position,
    )

    expected_lengths = []
    expected_spreads = []
    for path_index in range(path_vertices.shape[1]):
        points = [tx_position[0]]
        for depth_index in np.flatnonzero(interactions[:, path_index] != InteractionType.NONE):
            points.append(
                rt_coherence._interaction_point(
                    depth_index=depth_index,
                    path_index=path_index,
                    path_vertices=path_vertices,
                    primitives=primitives,
                    dynamic_target_indices=dynamic_target_indices,
                    dynamic_barycentrics=dynamic_barycentrics,
                    target_faces=target_faces,
                    target_vertices=target_vertices,
                )
            )
        points.append(rx_position[0])
        segment_lengths = np.array(
            [
                np.linalg.norm(np.asarray(end) - np.asarray(start))
                for start, end in zip(points[:-1], points[1:])
            ]
        )
        expected_lengths.append(np.sum(segment_lengths))
        expected_spreads.append(np.prod(np.maximum(segment_lengths, 1e-18)))

    assert np.allclose(lengths[0], expected_lengths)
    assert np.allclose(spreads[0], expected_spreads)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_coherent_rt_path_update_matches_multichannel_reference_geometry(precision):
    geometry_tolerance = 2e-14 if precision == "float64" else 2e-6
    field_tolerance = 2e-10 if precision == "float64" else 2e-3
    dtype = np.float64 if precision == "float64" else np.float32
    num_paths = 4
    path_vertices = np.zeros((3, num_paths, 3), dtype=np.float64)
    interactions = np.full(
        (3, num_paths),
        InteractionType.NONE,
        dtype=np.uint32,
    )
    primitives = np.full(
        (3, num_paths),
        INVALID_PRIMITIVE,
        dtype=np.uint32,
    )
    dynamic_indices = np.full((3, num_paths), -1, dtype=np.int64)
    barycentrics = np.full((3, num_paths, 3), np.nan, dtype=np.float64)
    anchor_vertices = np.array(
        [
            [1.0, -0.2, 0.8],
            [1.0, 0.2, 0.8],
            [1.0, 0.0, 1.2],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    interactions[0, 1] = InteractionType.DIFFUSE
    primitives[0, 1] = 0
    dynamic_indices[0, 1] = 0
    barycentrics[0, 1] = [0.2, 0.3, 0.5]
    path_vertices[0, 1] = barycentrics[0, 1] @ anchor_vertices[faces[0]]

    interactions[0, 2] = InteractionType.SPECULAR
    interactions[2, 2] = InteractionType.DIFFUSE
    primitives[0, 2] = 0
    primitives[2, 2] = 0
    path_vertices[0, 2] = [0.45, -0.35, 0.65]
    dynamic_indices[2, 2] = 0
    barycentrics[2, 2] = [0.3, 0.4, 0.3]
    path_vertices[2, 2] = barycentrics[2, 2] @ anchor_vertices[faces[0]]

    interactions[:2, 3] = InteractionType.SPECULAR
    primitives[:2, 3] = 0
    path_vertices[0, 3] = [0.6, -0.45, 0.76]
    path_vertices[1, 3] = [1.55, -0.3, 1.1]

    tx_positions = np.array(
        [
            [0.0, -0.002, 0.0],
            [0.0, 0.002, 0.0],
        ],
        dtype=np.float64,
    )
    rx_positions = np.array(
        [
            [0.0, -0.003, 0.0],
            [0.0, 0.003, 0.0],
        ],
        dtype=np.float64,
    )
    vc_tx = np.array([1, 0, 1, 0], dtype=np.int64)
    vc_rx = np.array([1, 0, 0, 1], dtype=np.int64)
    geometry_kwargs = {
        "path_vertices": path_vertices,
        "interactions": interactions,
        "primitives": primitives,
        "dynamic_target_indices": dynamic_indices,
        "dynamic_barycentrics": barycentrics,
        "target_faces": (faces,),
    }
    anchor_lengths, anchor_spreads = rt_coherence._path_lengths_and_spreads(
        **geometry_kwargs,
        target_vertices=(anchor_vertices,),
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        virtual_channel_tx_indices=vc_tx,
        virtual_channel_rx_indices=vc_rx,
    )
    tx_center = np.mean(tx_positions, axis=0)
    rx_center = np.mean(rx_positions, axis=0)
    anchor_center_lengths, _ = rt_coherence._path_lengths_and_spreads(
        **geometry_kwargs,
        target_vertices=(anchor_vertices,),
        tx_positions=tx_center[None, :],
        rx_positions=rx_center[None, :],
    )
    coefficients = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    coefficients = coefficients + 1j * coefficients[::-1]
    bank = CoherentRTPathBank(
        coefficients=coefficients.astype(np.complex128),
        delays_s=anchor_center_lengths[0] / c,
        valid=np.array([True, True, True, False]),
        anchor_path_lengths_m=anchor_lengths,
        anchor_delay_lengths_m=anchor_center_lengths[0],
        anchor_spread_products_m=anchor_spreads,
        path_vertices=path_vertices,
        path_interactions=interactions,
        path_objects=np.full_like(interactions, INVALID_SHAPE),
        path_primitives=primitives,
        dynamic_target_indices=dynamic_indices,
        dynamic_barycentrics=barycentrics,
        target_faces=(faces,),
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        tx_center=tx_center,
        rx_center=rx_center,
        virtual_channel_order="tx_major",
        frequency_hz=60e9,
        path_signatures=tuple(() for _ in range(num_paths)),
        virtual_channel_tx_indices=vc_tx,
        virtual_channel_rx_indices=vc_rx,
    )
    moved_vertices = anchor_vertices.copy()
    moved_vertices[:, 0] += [0.002, 0.006, 0.011]
    moved_vertices[:, 1] += 0.004

    expected_lengths, expected_spreads = rt_coherence._path_lengths_and_spreads(
        **geometry_kwargs,
        target_vertices=(moved_vertices,),
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        virtual_channel_tx_indices=vc_tx,
        virtual_channel_rx_indices=vc_rx,
    )
    expected_center_lengths, _ = rt_coherence._path_lengths_and_spreads(
        **geometry_kwargs,
        target_vertices=(moved_vertices,),
        tx_positions=tx_center[None, :],
        rx_positions=rx_center[None, :],
    )
    update = update_coherent_rt_path_bank(bank, (moved_vertices,), precision=precision)

    expected_phase = np.exp(
        -1j * 2.0 * pi * bank.frequency_hz / c * (expected_lengths - anchor_lengths)
    )
    expected_coefficients = (
        bank.coefficients * expected_phase * anchor_spreads / np.maximum(expected_spreads, 1e-18)
    )
    fmcw = _phase_one_radar(num_chirps=1).fmcw
    expected_adc = MmWaveRadarSimulator.synthesize_adc_from_paths(
        expected_coefficients,
        np.where(bank.valid, expected_center_lengths[0] / c, -1.0),
        fmcw,
        backend="numpy",
        precision="float64",
    )

    def check_adc(candidate):
        actual_adc = MmWaveRadarSimulator.synthesize_adc_from_paths(
            candidate.coefficients,
            np.where(candidate.valid, candidate.delays_s, -1.0),
            fmcw,
            backend="numpy",
            precision="float64",
        )
        relative_error = np.linalg.norm(actual_adc - expected_adc) / np.linalg.norm(expected_adc)
        assert relative_error < field_tolerance

    check_adc(update)
    assert update.path_lengths_m.dtype == dtype
    assert np.allclose(
        update.path_lengths_m, expected_lengths, rtol=geometry_tolerance, atol=geometry_tolerance
    )
    assert np.allclose(
        update.delay_lengths_m,
        expected_center_lengths[0],
        rtol=geometry_tolerance,
        atol=geometry_tolerance,
    )
    assert np.allclose(
        update.spread_products_m,
        expected_spreads,
        rtol=geometry_tolerance,
        atol=geometry_tolerance,
    )
    assert np.allclose(
        update.coefficients, expected_coefficients, rtol=field_tolerance, atol=field_tolerance
    )
    assert np.array_equal(
        update.path_lengths_m[:, [0, 3]], anchor_lengths[:, [0, 3]].astype(dtype)
    )
    assert np.array_equal(
        update.spread_products_m[:, [0, 3]], anchor_spreads[:, [0, 3]].astype(dtype)
    )

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            torch_update = rt_coherence._update_coherent_rt_path_bank_torch(
                bank,
                (moved_vertices,),
                frequency_hz=None,
                runtime=replace(resolve_torch_runtime("torch", precision), device=device),
            )
            assert np.allclose(
                torch_update.path_lengths_m,
                expected_lengths,
                rtol=geometry_tolerance,
                atol=geometry_tolerance,
            )
            assert np.allclose(
                torch_update.delay_lengths_m,
                expected_center_lengths[0],
                rtol=geometry_tolerance,
                atol=geometry_tolerance,
            )
            assert np.allclose(
                torch_update.spread_products_m,
                expected_spreads,
                rtol=geometry_tolerance,
                atol=geometry_tolerance,
            )
            assert np.allclose(
                torch_update.coefficients,
                expected_coefficients,
                rtol=field_tolerance,
                atol=field_tolerance,
            )
            phase_error = np.max(abs(np.angle(torch_update.coefficients / expected_coefficients)))
            assert phase_error < field_tolerance
            check_adc(torch_update)
        if torch.cuda.is_available():
            cuda_update = update_coherent_rt_path_bank(
                bank,
                (moved_vertices,),
                backend="auto",
                precision=precision,
            )
            assert np.allclose(
                cuda_update.coefficients,
                expected_coefficients,
                rtol=field_tolerance,
                atol=field_tolerance,
            )


def test_coherent_rt_path_update_rejects_unknown_backend():
    bank = _coherent_test_bank([2.0], [((1, 2, 3),)])

    with pytest.raises(ValueError, match="backend"):
        update_coherent_rt_path_bank(bank, [], backend="invalid")


def test_coherent_rt_torch_honors_cpu_override(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")
    bank = _coherent_test_bank([2.0], [((1, 2, 3),)])
    expected = update_coherent_rt_path_bank(bank, [], backend="numpy")
    actual = update_coherent_rt_path_bank(bank, [], backend="torch")
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, rtol=1e-12)
    np.testing.assert_allclose(actual.delays_s, expected.delays_s, rtol=1e-12)
    cached_devices = [
        key[1] for key in rt_coherence._TORCH_COHERENT_BANK_CACHE if key[0] == id(bank)
    ]
    assert cached_devices == ["cpu"]


def test_coherent_rt_cache_separates_float32_and_float64(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")
    bank = _coherent_test_bank([2.0], [((1, 2, 3),)])
    runtime32 = resolve_torch_runtime("torch", "float32")
    runtime64 = resolve_torch_runtime("torch", "float64")
    data32 = rt_coherence._torch_coherent_bank_data(bank, runtime=runtime32)
    data64 = rt_coherence._torch_coherent_bank_data(bank, runtime=runtime64)
    assert data32["anchor_path_lengths"].dtype == torch.float32
    assert data32["coefficients"].dtype == torch.complex64
    assert data64["anchor_path_lengths"].dtype == torch.float64
    assert data64["coefficients"].dtype == torch.complex128
    assert rt_coherence._torch_coherent_bank_data(bank, runtime=runtime32) is data32
    assert rt_coherence._torch_coherent_bank_data(bank, runtime=runtime64) is data64


def test_path_lengths_follow_explicit_board_channel_map():
    tx_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    rx_positions = np.array(
        [
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    vc_tx = np.array([1, 0, 1, 0])
    vc_rx = np.array([1, 0, 0, 1])

    lengths, _ = rt_coherence._path_lengths_and_spreads(
        path_vertices=np.empty((0, 1, 3), dtype=float),
        interactions=np.empty((0, 1), dtype=np.uint32),
        primitives=np.empty((0, 1), dtype=np.uint32),
        dynamic_target_indices=np.empty((0, 1), dtype=np.int64),
        dynamic_barycentrics=np.empty((0, 1, 3), dtype=float),
        target_faces=(),
        target_vertices=None,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        virtual_channel_tx_indices=vc_tx,
        virtual_channel_rx_indices=vc_rx,
    )

    expected = np.array(
        [np.linalg.norm(rx_positions[rx] - tx_positions[tx]) for tx, rx in zip(vc_tx, vc_rx)]
    )
    assert np.allclose(lengths[:, 0], expected)


def test_extract_rt_reflector_bank_uses_static_path_metadata():
    tri = _triangle_from_yz(
        -0.5,
        (-1.0, -1.0),
        (1.0, -1.0),
        (0.0, 1.0),
    )
    mesh = mi.Mesh(
        name="rt-reflector-test",
        vertex_count=3,
        face_count=1,
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = mi.Float(tri.astype(np.float32).reshape(-1))
    mesh_params["faces"] = mi.UInt(np.array([[0, 1, 2]], dtype=np.uint32).reshape(-1))
    mesh_params.update()

    class FakeObject:
        object_id = 7
        mi_mesh = mesh
        radio_material = SimpleNamespace(
            relative_permittivity=4.0,
            conductivity=0.1,
            thickness=0.05,
            scattering_coefficient=0.6,
        )

    class FakeScene:
        objects = {"reflector": FakeObject()}

    bank = StaticPathBank(
        coefficients=np.ones((1, 2), dtype=np.complex128),
        delays_s=np.array([1e-9, 2e-9], dtype=float),
        valid=np.array([True, True], dtype=bool),
        segment_starts=np.zeros((0, 3), dtype=float),
        segment_ends=np.zeros((0, 3), dtype=float),
        segment_path_indices=np.zeros((0,), dtype=np.int64),
        path_vertices=np.array(
            [
                [[-0.5, 0.0, -0.25], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        path_interactions=np.array(
            [[InteractionType.SPECULAR, InteractionType.NONE]], dtype=np.uint32
        ),
        path_objects=np.array([[7, INVALID_SHAPE]], dtype=np.uint32),
        path_primitives=np.array([[0, INVALID_PRIMITIVE]], dtype=np.uint32),
    )

    reflectors = extract_rt_reflector_bank(FakeScene(), bank, max_reflectors=0)

    assert reflectors.vertices.shape == (1, 3, 3)
    assert np.allclose(reflectors.centroids[0], [-0.5, 0.0, -0.25])
    assert np.array_equal(reflectors.path_indices, [0])
    assert np.array_equal(reflectors.face_indices, [0])
    assert np.allclose(reflectors.reflection_gains, [1.0 + 0.0j])
    np.testing.assert_allclose(reflectors.specular_amplitude_scales, [0.8])
    assert reflectors.object_names == ("reflector",)


def test_free_space_deembedding_matches_real_sionna_cir():
    from mmWaveRadar.simulation.env_human_coupling import _rt_calibrated_reflection_gain

    radar = _phase_one_radar(num_chirps=1)
    simulator = MmWaveRadarSimulator(
        rt_samples_per_source=100,
        rt_max_paths_per_source=100,
        rt_max_depth=1,
        compute_backend="numpy",
    )
    from sionna.rt import PathSolver

    for distance in (1.0, 1.0 + radar.fmcw.wavelength / 8):
        scene = load_scene()
        _, rx = radar.configure_scene(scene)
        rx.position = (distance, 0.0, 0.0)
        # The direct path is the known reference here; radar traces omit it.
        paths = PathSolver()(
            scene,
            max_depth=1,
            samples_per_src=100,
            max_num_paths_per_src=100,
            synthetic_array=True,
            los=True,
            specular_reflection=False,
            diffuse_reflection=False,
            refraction=False,
            seed=simulator.rt_seed,
        )
        bank = extract_static_path_bank(paths, radar)
        path_index = int(np.flatnonzero(bank.valid)[0])
        gain = _rt_calibrated_reflection_gain(
            bank,
            path_index,
            frequency_hz=radar.fmcw.carrier_frequency,
        )
        # The unity LOS factor must not acquire a distance-dependent phase.
        np.testing.assert_allclose(gain, 1.0 + 0.0j, atol=2e-3, rtol=0.0)


@pytest.mark.parametrize("hardware_gain", [None, 1e-4, 2.0j])
def test_extract_rt_reflector_bank_derives_reflection_gain_from_static_rt(hardware_gain):
    tri = _triangle_from_yz(
        -0.5,
        (-1.0, -1.0),
        (1.0, -1.0),
        (0.0, 1.0),
    )
    mesh = mi.Mesh(
        name="rt-gain-test",
        vertex_count=3,
        face_count=1,
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = mi.Float(tri.astype(np.float32).reshape(-1))
    mesh_params["faces"] = mi.UInt(np.array([[0, 1, 2]], dtype=np.uint32).reshape(-1))
    mesh_params.update()

    class FakeObject:
        object_id = 7
        mi_mesh = mesh
        radio_material = PhysicalOpticsMaterial(1.0, 0.0, model="pec")

    class FakeScene:
        objects = {"reflector": FakeObject()}

    frequency_hz = 60e9
    path_length_m = 2.0
    wavelength_m = 299792458.0 / frequency_hz
    free_space = wavelength_m / (4.0 * np.pi * path_length_m)
    free_space *= np.exp(-1j * 2.0 * np.pi * path_length_m / wavelength_m)
    expected_gain = 0.25 - 0.1j
    propagation = np.array([[expected_gain * free_space]], dtype=np.complex128)
    bank = StaticPathBank(
        coefficients=propagation * (1.0 if hardware_gain is None else hardware_gain),
        delays_s=np.array([path_length_m / 299792458.0], dtype=float),
        valid=np.array([True], dtype=bool),
        segment_starts=np.zeros((0, 3), dtype=float),
        segment_ends=np.zeros((0, 3), dtype=float),
        segment_path_indices=np.zeros((0,), dtype=np.int64),
        path_vertices=np.array([[[-0.5, 0.0, -0.25]]], dtype=float),
        path_interactions=np.array([[InteractionType.SPECULAR]], dtype=np.uint32),
        path_objects=np.array([[7]], dtype=np.uint32),
        path_primitives=np.array([[0]], dtype=np.uint32),
        propagation_coefficients=None if hardware_gain is None else propagation,
    )

    reflectors = extract_rt_reflector_bank(
        FakeScene(),
        bank,
        max_reflectors=0,
        frequency_hz=frequency_hz,
    )

    assert np.allclose(reflectors.reflection_gains, [expected_gain])


def test_extract_rt_reflector_bank_uses_world_space_scene_triangles():
    scene = load_bedroom_scene(merge_shapes=False)
    obj = scene.objects["right_wall"]
    object_id = int(np.asarray(obj.object_id).reshape(-1)[0])
    right_wall = next(box for box in BEDROOM_SCENE_BOXES if box["id"] == "right_wall")
    inner_wall_y = right_wall["translate"][1] - right_wall["scale"][1]
    bank = StaticPathBank(
        coefficients=np.ones((1, 1), dtype=np.complex128),
        delays_s=np.array([1e-9], dtype=float),
        valid=np.array([True], dtype=bool),
        segment_starts=np.zeros((0, 3), dtype=float),
        segment_ends=np.zeros((0, 3), dtype=float),
        segment_path_indices=np.zeros((0,), dtype=np.int64),
        path_vertices=np.array([[[0.0, inner_wall_y, 1.0]]], dtype=float),
        path_interactions=np.array([[InteractionType.SPECULAR]], dtype=np.uint32),
        path_objects=np.array([[object_id]], dtype=np.uint32),
        path_primitives=np.array([[1]], dtype=np.uint32),
    )

    reflectors = extract_rt_reflector_bank(scene, bank, max_reflectors=0)

    assert reflectors.vertices.shape == (1, 3, 3)
    assert np.allclose(reflectors.centroids[0], [0.0, inner_wall_y, 1.0])
    assert np.allclose(reflectors.vertices[0, :, 1], inner_wall_y)
    assert np.allclose(np.abs(reflectors.normals[0]), [0.0, 1.0, 0.0])


def test_trace_forwards_interaction_flags():
    class FakePathSolver:
        def __init__(self):
            self.kwargs = None

        def __call__(self, scene, **kwargs):
            del scene
            self.kwargs = kwargs
            return object()

    solver = FakePathSolver()
    sim = MmWaveRadarSimulator(path_solver=solver, rt_diffuse_reflection=True)

    sim._trace(scene=object())

    assert solver.kwargs["diffuse_reflection"] is True
    assert solver.kwargs["refraction"] is False

    solver = FakePathSolver()
    sim = MmWaveRadarSimulator(path_solver=solver, rt_refraction=True)

    sim._trace(scene=object())

    assert solver.kwargs["refraction"] is True


def _single_hit_paths(mesh_target):
    """Fake Sionna paths with one specular hit on the target's first face."""

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


class _CountingRTSimulator(MmWaveRadarSimulator):
    """Counts traces of a single-hit fake scene."""

    def __init__(self, mesh_target, **kwargs):
        super().__init__(
            rt_samples_per_source=1, solver="rt", rt_human_specular_reflection=True, **kwargs
        )
        self.mesh_target = mesh_target
        self.trace_count = 0

    def _trace(self, scene):
        del scene
        self.trace_count += 1
        return _single_hit_paths(self.mesh_target)


def _single_element_radar(*, num_chirps, frame_period):
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=1e12,
        chirp_duration=1e-3,
        chirp_repetition_time=0.5,
        sampling_frequency=1e3,
        num_adc_samples=4,
        num_chirps_per_frame=num_chirps,
        frame_period=frame_period,
    )
    return RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware.from_positions([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
        fmcw=fmcw,
    )


def test_rt_refresh_on_displacement_adds_early_retraces():
    radar = _single_element_radar(num_chirps=3, frame_period=2.0)
    target = MeshTarget(
        "target",
        _mesh_sequence(displacement=0.02),
        material=human_skin_material("target-material"),
    )
    sim = _CountingRTSimulator(target, rt_refresh_on_displacement=0.005)

    cube = sim.run(load_scene(), radar, [target], num_frames=1)

    assert cube.adc.shape == (1, 3, 4, 1)
    assert sim.trace_count >= 2
    assert [event["reason"] for event in cube.metadata.events][1:] == ["displacement"] * (
        len(cube.metadata.events) - 1
    )


@pytest.mark.parametrize("baseline", [False, True])
def test_rt_motion_threshold_is_metres_and_adds_to_fixed_frame_cadence(baseline):
    radar = _single_element_radar(num_chirps=2, frame_period=2.0)
    first = _front_facing_triangle(1.0, scale=0.03)
    target = MeshTarget(
        "cadence-and-drift",
        MeshSequence(
            vertices=np.stack(
                [
                    first,
                    first + [0.005, 0.0, 0.0],
                    first + [0.02, 0.0, 0.0],
                    first + [0.02, 0.0, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2]], np.uint32),
            times=np.array([0.0, 0.5, 2.0, 4.5]),
        ),
        material=human_skin_material("cadence-and-drift-skin"),
    )
    sim = _CountingRTSimulator(target, rt_refresh_frames=2, rt_refresh_on_displacement=0.01)
    if baseline:
        cube = run_rt_baseline(
            scene=load_scene(), radar=radar, targets=[target], simulator=sim, num_frames=3
        ).cube
    else:
        cube = sim.run(load_scene(), radar, [target], num_frames=3)
    assert [(e["frame"], e["chirp"], e["reason"]) for e in cube.metadata.events] == [
        (0, 0, "initial"),
        (1, 0, "displacement"),
        (2, 0, "frame"),
    ]
    assert all(
        e["threshold"] == 0.01 and e["retrace_period_chirps"] == 4 for e in cube.metadata.events
    )
    assert sim.trace_count == 3


def test_rt_refresh_frames_one_traces_frame_starts_only():
    radar = _single_element_radar(num_chirps=3, frame_period=2.0)
    target = MeshTarget(
        "target",
        _mesh_sequence(displacement=0.0),
        material=human_skin_material("frame-policy-material"),
    )
    sim = _CountingRTSimulator(target, rt_refresh_frames=1)

    cube = sim.run(load_scene(), radar, [target], num_frames=2)

    assert cube.adc.shape == (2, 3, 4, 1)
    assert sim.trace_count == 2
    assert [event["reason"] for event in cube.metadata.events] == ["initial", "frame"]
    assert all(event["coherent_path_update"] for event in cube.metadata.events)
    assert all("threshold" not in event for event in cube.metadata.events)


def test_rt_refresh_frames_zero_traces_every_chirp():
    class FakePaths:
        pass

    class TestSimulator(MmWaveRadarSimulator):
        def __init__(self):
            super().__init__(rt_samples_per_source=1, rt_refresh_frames=0)
            self.trace_count = 0

        def _trace(self, scene):
            del scene
            self.trace_count += 1
            return FakePaths()

        def _path_arrays(self, paths, targets, *, virtual_channel_order):
            del paths, targets, virtual_channel_order
            return (np.ones((1, 1), dtype=np.complex128), np.array([1e-9]), np.array([True]))

    radar = _single_element_radar(num_chirps=3, frame_period=2.0)
    target = MeshTarget(
        "target",
        _mesh_sequence(displacement=0.0),
        material=human_skin_material("periodic-k1-material"),
    )

    sim = TestSimulator()
    cube = sim.run(load_scene(), radar, [target], num_frames=2)

    assert cube.adc.shape == (2, 3, 4, 1)
    assert sim.trace_count == 6
    assert [(event["frame"], event["chirp"]) for event in cube.metadata.events] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert [event["retrace_period_chirps"] for event in cube.metadata.events] == [1] * 6


def test_rt_refresh_frames_retraces_every_n_frames():
    radar = _single_element_radar(num_chirps=4, frame_period=2.0)
    target = MeshTarget(
        "target",
        _mesh_sequence(displacement=0.0),
        material=human_skin_material("periodic-frame-material"),
    )
    sim = _CountingRTSimulator(target, rt_refresh_frames=2)

    cube = sim.run(load_scene(), radar, [target], num_frames=4)

    assert cube.adc.shape == (4, 4, 4, 1)
    assert sim.trace_count == 2
    assert [(event["frame"], event["chirp"]) for event in cube.metadata.events] == [
        (0, 0),
        (2, 0),
    ]
    assert [event["reason"] for event in cube.metadata.events] == ["initial", "frame"]
    assert [event["retrace_period_chirps"] for event in cube.metadata.events] == [8, 8]


def test_rt_refresh_frames_sets_whole_frame_retrace_periods():
    fmcw = _single_element_radar(num_chirps=4, frame_period=2.0).fmcw

    assert MmWaveRadarSimulator(rt_refresh_frames=3).rt_retrace_period_chirps(fmcw) == 12
    assert MmWaveRadarSimulator(rt_refresh_frames=0).rt_retrace_period_chirps(fmcw) == 1
    with pytest.raises(ValueError, match="rt_refresh_frames"):
        MmWaveRadarSimulator(rt_refresh_frames=-1)
    with pytest.raises(ValueError, match="rt_refresh_frames"):
        MmWaveRadarSimulator(rt_refresh_frames=1.5)
    with pytest.raises(TypeError, match="periodic_retrace_period_chirps was renamed"):
        MmWaveRadarSimulator(periodic_retrace_period_chirps=2)


def test_coherent_transition_power_weights_preserve_disjoint_power():
    old_bank = _coherent_test_bank([2.0], [((1, 10, 0),)])
    new_bank = _coherent_test_bank([3.0], [((1, 10, 1),)])

    transition = rt_coherence.match_coherent_rt_path_banks(
        old_bank,
        new_bank,
        [],
        wavelength_m=0.005,
    )
    old_update = rt_coherence.update_coherent_rt_path_bank(old_bank, [])
    new_update = rt_coherence.update_coherent_rt_path_bank(new_bank, [])
    synth = rt_coherence.synthesize_coherent_rt_transition(
        old_update,
        new_update,
        transition,
        interval_chirps=3,
        interval_offset=1,
    )

    assert transition.matched_old_indices.size == 0
    assert transition.old_only_indices.tolist() == [0]
    assert transition.new_only_indices.tolist() == [0]
    assert np.isclose(np.sum(np.abs(synth.coefficients) ** 2), 1.0)


def test_coherent_transition_matched_paths_are_not_double_counted():
    signature = ((1, 10, 4),)
    old_bank = _coherent_test_bank([2.0], [signature])
    new_bank = _coherent_test_bank([2.0001], [signature])

    transition = rt_coherence.match_coherent_rt_path_banks(
        old_bank,
        new_bank,
        [],
        wavelength_m=0.005,
    )
    old_update = rt_coherence.update_coherent_rt_path_bank(old_bank, [])
    new_update = rt_coherence.update_coherent_rt_path_bank(new_bank, [])
    synth = rt_coherence.synthesize_coherent_rt_transition(
        old_update,
        new_update,
        transition,
        interval_chirps=3,
        interval_offset=1,
    )

    assert transition.matched_old_indices.tolist() == [0]
    assert synth.coefficients.shape == (1, 1)
    assert np.array_equal(synth.roles, [rt_coherence.TRANSITION_ROLE_PERSISTENT])
    assert np.isclose(np.sum(np.abs(synth.coefficients) ** 2), 1.0)


def test_rt_path_bank_frame_retrace_uses_active_bank_without_crossfade():
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=1e12,
        chirp_duration=1e-3,
        chirp_repetition_time=0.5,
        sampling_frequency=1e3,
        num_adc_samples=1,
        num_chirps_per_frame=3,
        frame_period=2.0,
    )
    radar = RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware.from_positions([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
        fmcw=fmcw,
    )
    target = MeshTarget(
        "target",
        _single_face_mesh_sequence(x0=1.0),
        material=human_skin_material("coherent-transition-material"),
    )

    class TestSimulator(MmWaveRadarSimulator):
        def __init__(self, mesh_target):
            super().__init__(
                rt_samples_per_source=1,
                rt_refresh_frames=1,
                solver="rt",
                rt_human_bounces="single",
                rt_human_specular_reflection=True,
            )
            self.mesh_target = mesh_target
            self.trace_count = 0

        def _trace(self, scene):
            del scene
            self.trace_count += 1
            object_id = int(
                np.asarray(self.mesh_target.ensure_scene_object().object_id).reshape(-1)[0]
            )
            if self.trace_count == 1:
                primitives_used = np.array([0, 1], dtype=np.uint32)
            else:
                primitives_used = np.array([1, 2, 3], dtype=np.uint32)
            num_paths = primitives_used.size
            hits = np.stack(
                [np.array([1.0 + 0.1 * int(p), 0.0, 0.0], dtype=float) for p in primitives_used],
                axis=0,
            )
            a = np.ones((1, 1, 1, 1, num_paths, 1), dtype=np.complex128)
            tau = (2.0 * np.linalg.norm(hits, axis=1) / c).reshape(1, 1, num_paths)
            path_vertices = hits.reshape(1, 1, 1, num_paths, 3)
            interactions = np.full(
                (1, 1, 1, num_paths),
                InteractionType.SPECULAR,
                dtype=np.uint32,
            )
            objects = np.full((1, 1, 1, num_paths), object_id, dtype=np.uint32)
            primitives = primitives_used.reshape(1, 1, 1, num_paths)

            class FakePaths:
                @property
                def tau(self):
                    return tau

                def cir(self, normalize_delays=False, out_type="numpy"):
                    del normalize_delays, out_type
                    return a, tau

                @property
                def vertices(self):
                    return path_vertices

                @property
                def interactions(self):
                    return interactions

                @property
                def objects(self):
                    return objects

                @property
                def primitives(self):
                    return primitives

            return FakePaths()

    sim = TestSimulator(target)
    cube = sim.run(load_scene(), radar, [target], num_frames=2)

    assert sim.trace_count == 2
    assert cube.metadata.events[0]["transition_diagnostic_valid"] is False
    assert cube.metadata.events[0]["transition_same_frame"] is False
    assert "persistent_path_count" not in cube.metadata.events[0]
    assert "death_path_count" not in cube.metadata.events[0]
    assert "birth_path_count" not in cube.metadata.events[0]
    assert cube.metadata.events[0]["transition_crossfade"] is False
    assert np.array_equal(cube.metadata.rt_transition_persistent_path_counts[0], [2, 2, 2])
    assert np.array_equal(cube.metadata.rt_transition_death_path_counts[0], [0, 0, 0])
    assert np.array_equal(cube.metadata.rt_transition_birth_path_counts[0], [0, 0, 0])
    assert np.array_equal(cube.metadata.path_counts[0], [2, 2, 2])
    assert np.array_equal(cube.metadata.path_counts[1], [3, 3, 3])
    assert np.all(cube.metadata.rt_transition_alpha == 0.0)
    assert np.all(cube.metadata.rt_transition_old_weights == 1.0)
    assert np.all(cube.metadata.rt_transition_new_weights == 0.0)


def test_displacement_retrace_crossfade_can_be_enabled():
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=1e12,
        chirp_duration=1e-3,
        chirp_repetition_time=0.5,
        sampling_frequency=1e3,
        num_adc_samples=1,
        num_chirps_per_frame=6,
        frame_period=4.0,
    )
    radar = RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware.from_positions([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
        fmcw=fmcw,
    )
    # 1 mm/s against a 1.25 mm threshold retraces only at
    # chirp 3 (t = 1.5 s), so the transition stays inside the frame.
    target = MeshTarget(
        "target",
        MeshSequence(
            vertices=np.stack([_front_facing_triangle(1.0), _front_facing_triangle(1.005)]),
            faces=np.array([[0, 1, 2]], dtype=np.uint32),
            times=np.array([0.0, 5.0]),
        ),
        material=human_skin_material("periodic-xfade-material"),
    )

    class TestSimulator(MmWaveRadarSimulator):
        def __init__(self, mesh_target):
            super().__init__(
                rt_samples_per_source=1,
                rt_refresh_on_displacement=0.00125,
                solver="rt",
                rt_human_bounces="any",
                rt_human_specular_reflection=True,
                rt_coherent_transition_config=(RTCoherentTransitionConfig(crossfade=True)),
            )
            self.mesh_target = mesh_target
            self.trace_count = 0

        def _trace(self, scene):
            del scene
            self.trace_count += 1
            object_id = int(
                np.asarray(self.mesh_target.ensure_scene_object().object_id).reshape(-1)[0]
            )
            primitive = self.trace_count - 1
            hit = np.array([1.0 + 0.1 * primitive, 0.0, 0.0], dtype=float)
            a = np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex128)
            tau = np.array([[[2.0 * np.linalg.norm(hit) / c]]], dtype=float)
            path_vertices = hit.reshape(1, 1, 1, 1, 3)
            interactions = np.array([[[[InteractionType.SPECULAR]]]], dtype=np.uint32)
            objects = np.array([[[[object_id]]]], dtype=np.uint32)
            primitives = np.array([[[[primitive]]]], dtype=np.uint32)

            class FakePaths:
                @property
                def tau(self):
                    return tau

                def cir(self, normalize_delays=False, out_type="numpy"):
                    del normalize_delays, out_type
                    return a, tau

                @property
                def vertices(self):
                    return path_vertices

                @property
                def interactions(self):
                    return interactions

                @property
                def objects(self):
                    return objects

                @property
                def primitives(self):
                    return primitives

            return FakePaths()

    sim = TestSimulator(target)
    cube = sim.run(load_scene(), radar, [target], num_frames=1)

    assert sim.trace_count == 2
    assert cube.metadata.events[0]["transition_crossfade"] is True
    assert np.allclose(cube.metadata.rt_transition_alpha[0, :3], [0.0, 0.5, 1.0])
    assert np.allclose(
        cube.metadata.rt_transition_old_weights[0, :3],
        [1.0, np.sqrt(0.5), 0.0],
    )
    assert np.allclose(
        cube.metadata.rt_transition_new_weights[0, :3],
        [0.0, np.sqrt(0.5), 1.0],
    )
    assert np.array_equal(cube.metadata.path_counts[0, :4], [1, 2, 1, 1])


def test_displacement_retrace_transition_counts_only_same_frame_pairs():
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=1e12,
        chirp_duration=1e-3,
        chirp_repetition_time=0.5,
        sampling_frequency=1e3,
        num_adc_samples=1,
        num_chirps_per_frame=4,
        frame_period=3.0,
    )
    radar = RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware.from_positions([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
        fmcw=fmcw,
    )
    # 2 mm/s against a 1.5 mm threshold retraces at chirps
    # 0 and 2 of both frames (t = 0, 1, 3, 4 s).
    target = MeshTarget(
        "target",
        MeshSequence(
            vertices=np.stack([_front_facing_triangle(1.0), _front_facing_triangle(1.01)]),
            faces=np.array([[0, 1, 2]], dtype=np.uint32),
            times=np.array([0.0, 5.0]),
        ),
        material=human_skin_material("periodic-transition-material"),
    )

    class TestSimulator(MmWaveRadarSimulator):
        def __init__(self, mesh_target):
            super().__init__(
                rt_samples_per_source=1,
                rt_refresh_on_displacement=0.0015,
                solver="rt",
                rt_human_bounces="single",
                rt_human_specular_reflection=True,
                rt_coherent_transition_config=(RTCoherentTransitionConfig(crossfade=False)),
            )
            self.mesh_target = mesh_target
            self.trace_count = 0

        def _trace(self, scene):
            del scene
            self.trace_count += 1
            object_id = int(
                np.asarray(self.mesh_target.ensure_scene_object().object_id).reshape(-1)[0]
            )
            primitive_sequences = [
                np.array([0, 1], dtype=np.uint32),
                np.array([1, 2, 3], dtype=np.uint32),
                np.array([10], dtype=np.uint32),
                np.array([10, 11], dtype=np.uint32),
            ]
            primitives_used = primitive_sequences[self.trace_count - 1]
            num_paths = primitives_used.size
            hits = np.stack(
                [np.array([1.0 + 0.1 * int(p), 0.0, 0.0], dtype=float) for p in primitives_used],
                axis=0,
            )
            a = np.ones((1, 1, 1, 1, num_paths, 1), dtype=np.complex128)
            tau = (2.0 * np.linalg.norm(hits, axis=1) / c).reshape(1, 1, num_paths)
            path_vertices = hits.reshape(1, 1, 1, num_paths, 3)
            interactions = np.full(
                (1, 1, 1, num_paths),
                InteractionType.SPECULAR,
                dtype=np.uint32,
            )
            objects = np.full((1, 1, 1, num_paths), object_id, dtype=np.uint32)
            primitives = primitives_used.reshape(1, 1, 1, num_paths)

            class FakePaths:
                @property
                def tau(self):
                    return tau

                def cir(self, normalize_delays=False, out_type="numpy"):
                    del normalize_delays, out_type
                    return a, tau

                @property
                def vertices(self):
                    return path_vertices

                @property
                def interactions(self):
                    return interactions

                @property
                def objects(self):
                    return objects

                @property
                def primitives(self):
                    return primitives

            return FakePaths()

    sim = TestSimulator(target)
    cube = sim.run(load_scene(), radar, [target], num_frames=2)

    assert sim.trace_count == 4
    assert [(event["frame"], event["chirp"]) for event in cube.metadata.events] == [
        (0, 0),
        (0, 2),
        (1, 0),
        (1, 2),
    ]
    assert cube.metadata.events[0]["transition_diagnostic_valid"] is True
    assert cube.metadata.events[0]["transition_same_frame"] is True
    assert cube.metadata.events[0]["transition_interval_chirps"] == 2
    assert cube.metadata.events[0]["persistent_path_count"] == 1
    assert cube.metadata.events[0]["death_path_count"] == 1
    assert cube.metadata.events[0]["birth_path_count"] == 2

    assert cube.metadata.events[1]["transition_diagnostic_valid"] is False
    assert cube.metadata.events[1]["transition_same_frame"] is False
    assert "death_path_count" not in cube.metadata.events[1]
    assert cube.metadata.events[2]["transition_diagnostic_valid"] is True
    assert cube.metadata.events[2]["transition_same_frame"] is True
    assert cube.metadata.events[2]["persistent_path_count"] == 1
    assert cube.metadata.events[2]["death_path_count"] == 0
    assert cube.metadata.events[2]["birth_path_count"] == 1

    assert np.array_equal(cube.metadata.rt_transition_persistent_path_counts[0], [1, 1, 3, 3])
    assert np.array_equal(cube.metadata.rt_transition_death_path_counts[0], [1, 1, 0, 0])
    assert np.array_equal(cube.metadata.rt_transition_birth_path_counts[0], [2, 2, 0, 0])
    assert np.array_equal(cube.metadata.rt_transition_persistent_path_counts[1], [1, 1, 2, 2])
    assert np.array_equal(cube.metadata.rt_transition_death_path_counts[1], [0, 0, 0, 0])
    assert np.array_equal(cube.metadata.rt_transition_birth_path_counts[1], [1, 1, 0, 0])


def test_rt_path_bank_updates_adc_between_frame_retraces():
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=1e10,
        chirp_duration=1e-3,
        chirp_repetition_time=0.5,
        sampling_frequency=1e3,
        num_adc_samples=1,
        num_chirps_per_frame=3,
        frame_period=2.0,
    )
    radar = RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware.from_positions([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
        fmcw=fmcw,
    )

    class CountingMeshSequence:
        def __init__(self, base):
            self.base = base
            self.vertices_at_calls = []

        def vertices_at(self, time_s):
            self.vertices_at_calls.append(float(time_s))
            return self.base.vertices_at(time_s)

        def __getattr__(self, name):
            return getattr(self.base, name)

    sequence = CountingMeshSequence(_single_face_mesh_sequence(x0=1.0, x1=1.1))
    target = MeshTarget("target", sequence, material=human_skin_material("coherent-bank-material"))

    class TestSimulator(MmWaveRadarSimulator):
        def __init__(self, mesh_target):
            super().__init__(
                rt_samples_per_source=1,
                rt_refresh_frames=1,
                solver="rt",
                rt_human_bounces="any",
                rt_human_specular_reflection=True,
            )
            self.mesh_target = mesh_target
            self.trace_count = 0

        def _trace(self, scene):
            del scene
            self.trace_count += 1
            object_id = int(
                np.asarray(self.mesh_target.ensure_scene_object().object_id).reshape(-1)[0]
            )
            current_vertices = (
                self.mesh_target.ensure_scene_object()
                .mi_mesh.vertex_positions_buffer()
                .numpy()
                .reshape(-1, 3)
            )
            hit = np.mean(current_vertices[self.mesh_target.mesh_sequence.faces[0]], axis=0)
            a = np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex128)
            tau = np.array([[[2.0 * np.linalg.norm(hit) / c]]], dtype=float)
            path_vertices = hit.reshape(1, 1, 1, 1, 3)
            interactions = np.array([[[[InteractionType.SPECULAR]]]], dtype=np.uint32)
            objects = np.array([[[[object_id]]]], dtype=np.uint32)
            primitives = np.array([[[[0]]]], dtype=np.uint32)

            class FakePaths:
                @property
                def tau(self):
                    return tau

                def cir(self, normalize_delays=False, out_type="numpy"):
                    del normalize_delays, out_type
                    return a, tau

                @property
                def vertices(self):
                    return path_vertices

                @property
                def interactions(self):
                    return interactions

                @property
                def objects(self):
                    return objects

                @property
                def primitives(self):
                    return primitives

            return FakePaths()

    sim = TestSimulator(target)
    cube = sim.run(load_scene(), radar, [target], num_frames=1)

    slow_adc = cube.adc[0, :, 0, 0]
    assert sim.trace_count == 1
    assert cube.metadata.solver == "rt"
    assert cube.metadata.events[0]["coherent_path_update"] is True
    assert np.linalg.norm(slow_adc - slow_adc[0]) > 1e-6
    assert np.max(cube.metadata.max_displacements[0]) > 0.0
    assert np.allclose(sequence.vertices_at_calls, [0.0, 0.0, 0.5, 1.0])


def test_rt_baseline_helper_uses_path_bank():
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=1e10,
        chirp_duration=1e-3,
        chirp_repetition_time=0.5,
        sampling_frequency=1e3,
        num_adc_samples=1,
        num_chirps_per_frame=3,
        frame_period=2.0,
    )
    radar = RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware.from_positions([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
        fmcw=fmcw,
    )
    target = MeshTarget(
        "target",
        _single_face_mesh_sequence(x0=1.0, x1=1.1),
        material=human_skin_material("coherent-helper-material"),
    )

    class TestSimulator(MmWaveRadarSimulator):
        def __init__(self, mesh_target):
            super().__init__(
                rt_samples_per_source=1,
                rt_refresh_frames=1,
                solver="rt",
                rt_human_bounces="any",
                rt_human_specular_reflection=True,
            )
            self.mesh_target = mesh_target
            self.trace_count = 0

        def _trace(self, scene):
            del scene
            self.trace_count += 1
            object_id = int(
                np.asarray(self.mesh_target.ensure_scene_object().object_id).reshape(-1)[0]
            )
            current_vertices = (
                self.mesh_target.ensure_scene_object()
                .mi_mesh.vertex_positions_buffer()
                .numpy()
                .reshape(-1, 3)
            )
            hit = np.mean(current_vertices[self.mesh_target.mesh_sequence.faces[0]], axis=0)
            a = np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex128)
            tau = np.array([[[2.0 * np.linalg.norm(hit) / c]]], dtype=float)
            path_vertices = hit.reshape(1, 1, 1, 1, 3)
            interactions = np.array([[[[InteractionType.SPECULAR]]]], dtype=np.uint32)
            objects = np.array([[[[object_id]]]], dtype=np.uint32)
            primitives = np.array([[[[0]]]], dtype=np.uint32)

            class FakePaths:
                @property
                def tau(self):
                    return tau

                def cir(self, normalize_delays=False, out_type="numpy"):
                    del normalize_delays, out_type
                    return a, tau

                @property
                def vertices(self):
                    return path_vertices

                @property
                def interactions(self):
                    return interactions

                @property
                def objects(self):
                    return objects

                @property
                def primitives(self):
                    return primitives

            return FakePaths()

    sim = TestSimulator(target)
    result = run_rt_baseline(
        scene=load_scene(),
        radar=radar,
        targets=[target],
        simulator=sim,
        num_frames=1,
        calibration_reference_component="single_human_only",
    )

    slow_adc = result.cube.adc[0, :, 0, 0]
    assert sim.trace_count == 1
    assert result.cube.metadata.solver == "rt"
    assert result.cube.metadata.events[0]["coherent_path_update"] is True
    np.testing.assert_allclose(
        result.reference_adc_power["single_human_only"],
        np.mean(np.abs(result.cube.adc) ** 2, axis=2),
    )
    assert np.linalg.norm(slow_adc - slow_adc[0]) > 1e-6


def test_rt_baseline_calibration_forces_path_bank_from_hybrid_solver():
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=1e10,
        chirp_duration=1e-3,
        chirp_repetition_time=0.5,
        sampling_frequency=1e3,
        num_adc_samples=1,
        num_chirps_per_frame=3,
        frame_period=2.0,
    )
    radar = RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware.from_positions([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
        fmcw=fmcw,
    )
    target = MeshTarget(
        "target",
        _single_face_mesh_sequence(x0=1.0, x1=1.1),
        material=human_skin_material("calibration-helper-material"),
    )

    class TestSimulator(MmWaveRadarSimulator):
        def __init__(self, mesh_target):
            super().__init__(
                rt_samples_per_source=1,
                rt_refresh_frames=1,
                solver="hybrid_po",
                rt_human_bounces="any",
                rt_human_specular_reflection=True,
            )
            self.mesh_target = mesh_target
            self.trace_count = 0

        def _trace(self, scene):
            del scene
            self.trace_count += 1
            object_id = int(
                np.asarray(self.mesh_target.ensure_scene_object().object_id).reshape(-1)[0]
            )
            current_vertices = (
                self.mesh_target.ensure_scene_object()
                .mi_mesh.vertex_positions_buffer()
                .numpy()
                .reshape(-1, 3)
            )
            hit = np.mean(
                current_vertices[self.mesh_target.mesh_sequence.faces[0]],
                axis=0,
            )
            a = np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex128)
            tau = np.array([[[2.0 * np.linalg.norm(hit) / c]]], dtype=float)
            path_vertices = hit.reshape(1, 1, 1, 1, 3)
            interactions = np.array([[[[InteractionType.SPECULAR]]]], dtype=np.uint32)
            objects = np.array([[[[object_id]]]], dtype=np.uint32)
            primitives = np.array([[[[0]]]], dtype=np.uint32)

            class FakePaths:
                @property
                def tau(self):
                    return tau

                def cir(self, normalize_delays=False, out_type="numpy"):
                    del normalize_delays, out_type
                    return a, tau

                @property
                def vertices(self):
                    return path_vertices

                @property
                def interactions(self):
                    return interactions

                @property
                def objects(self):
                    return objects

                @property
                def primitives(self):
                    return primitives

            return FakePaths()

    sim = TestSimulator(target)
    result = sim._run_rt_baseline_for_calibration(  # pylint: disable=protected-access
        scene=load_scene(),
        radar=radar,
        targets=[target],
        num_frames=1,
        rt_refresh_frames=1,
    )

    slow_adc = result["cube"].adc[0, :, 0, 0]
    assert sim.solver == "hybrid_po"
    assert sim.trace_count == 1
    assert result["coherent_path_update"] is True
    assert result["cube"].metadata.events[0]["coherent_path_update"] is True
    assert "rt_baseline_path_bank_update" in result["cube"].metadata.runtime_profile_s
    assert np.linalg.norm(slow_adc - slow_adc[0]) > 1e-6


def test_rt_baseline_helper_drops_human_specular_contribution():
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=1e12,
        chirp_duration=1e-3,
        chirp_repetition_time=0.5,
        sampling_frequency=1e3,
        num_adc_samples=1,
        num_chirps_per_frame=1,
        frame_period=1.0,
    )
    radar = RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware.from_positions([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
        fmcw=fmcw,
    )
    target = MeshTarget(
        "target",
        _single_face_mesh_sequence(x0=1.0),
        material=human_skin_material("rt-spec-as-diff-material"),
    )

    class TestSimulator(MmWaveRadarSimulator):
        def __init__(self, mesh_target):
            super().__init__(
                rt_samples_per_source=1, rt_refresh_frames=1, solver="rt", rt_human_bounces="any"
            )
            self.mesh_target = mesh_target

        def _trace(self, scene):
            del scene
            object_id = int(
                np.asarray(self.mesh_target.ensure_scene_object().object_id).reshape(-1)[0]
            )
            current_vertices = (
                self.mesh_target.ensure_scene_object()
                .mi_mesh.vertex_positions_buffer()
                .numpy()
                .reshape(-1, 3)
            )
            hit = np.mean(
                current_vertices[self.mesh_target.mesh_sequence.faces[0]],
                axis=0,
            )
            a = np.ones((1, 1, 1, 1, 2, 1), dtype=np.complex128)
            tau = np.array([[[2.0 * np.linalg.norm(hit) / c] * 2]], dtype=float)
            path_vertices = np.repeat(hit.reshape(1, 1, 1, 1, 3), 2, axis=3)
            interactions = np.array(
                [[[[InteractionType.SPECULAR, InteractionType.DIFFUSE]]]], dtype=np.uint32
            )
            objects = np.full((1, 1, 1, 2), object_id, dtype=np.uint32)
            primitives = np.zeros((1, 1, 1, 2), dtype=np.uint32)

            class FakePaths:
                @property
                def tau(self):
                    return tau

                def cir(self, normalize_delays=False, out_type="numpy"):
                    del normalize_delays, out_type
                    return a, tau

                @property
                def vertices(self):
                    return path_vertices

                @property
                def interactions(self):
                    return interactions

                @property
                def objects(self):
                    return objects

                @property
                def primitives(self):
                    return primitives

            return FakePaths()

    result = run_rt_baseline(
        scene=load_scene(),
        radar=radar,
        targets=[target],
        simulator=TestSimulator(target),
        num_frames=1,
    )

    assert result.cube.metadata.path_counts[0, 0] == 1
    assert result.cube.metadata.human_touch_path_counts[0, 0] == 1
    assert result.cube.metadata.single_human_only_path_counts[0, 0] == 1
