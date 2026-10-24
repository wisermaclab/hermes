# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Physical optics visibility and coupling tests."""

from __future__ import annotations

from sensing_test_helpers import *  # noqa: F401,F403
import mmWaveRadar.simulation.env_human_coupling as env_human_coupling
import mmWaveRadar.simulation.physical_optics as physical_optics_module
from mmWaveRadar.radar.ti import SimpleCosinePattern
from mmWaveRadar.simulation.physical_optics import (
    _visibility_adjacent_face_pairs,
    _visibility_edge_faces,
    _triangle_phase_span_rad,
    face_geometry,
    parent_face_far_field_analytic_geometry,
    parent_face_quadrature_geometry,
    vector_facet_face_contributions,
    vector_facet_face_contributions_for_pairs,
)
from mmWaveRadar.targets.mesh import (
    ensure_outward_face_winding,
    face_winding_signed_volume,
)


def _cube_mesh():
    vertices = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.uint32,
    )
    return vertices, faces


def test_cached_visibility_adjacency_matches_legacy_edge_detection():
    _, faces = _cube_mesh()
    adjacent_pairs = _visibility_adjacent_face_pairs(faces)
    for visible_bits in range(1 << faces.shape[0]):
        centroid_visible = np.asarray(
            [bool(visible_bits & (1 << index)) for index in range(faces.shape[0])]
        )
        legacy = _visibility_edge_faces(faces, centroid_visible)
        cached = _visibility_edge_faces(
            faces,
            centroid_visible,
            adjacent_face_pairs=adjacent_pairs,
        )
        np.testing.assert_array_equal(cached, legacy)


def test_cached_rx_phase_span_matches_legacy_loop_bitwise():
    rng = np.random.default_rng(42)
    triangles = rng.normal(size=(257, 3, 3))
    tx_positions = rng.normal(size=(3, 3))
    rx_positions = rng.normal(size=(4, 3))
    samples = np.einsum(
        "bs,tsc->tbc",
        physical_optics_module.PHASE_SPAN_SAMPLE_BARYCENTRICS,
        triangles,
    )
    max_delta_m = np.zeros((triangles.shape[0],), dtype=np.float64)
    for tx_position in tx_positions:
        tx_dist = np.linalg.norm(samples - tx_position[None, None, :], axis=2)
        for rx_position in rx_positions:
            rx_dist = np.linalg.norm(
                samples - rx_position[None, None, :],
                axis=2,
            )
            path = tx_dist + rx_dist
            delta_m = np.max(path, axis=1) - np.min(path, axis=1)
            max_delta_m = np.maximum(max_delta_m, delta_m)
    expected = 2.0 * np.pi * 60e9 / physical_optics_module.c * max_delta_m

    actual = _triangle_phase_span_rad(
        triangles,
        tx_positions,
        rx_positions,
        60e9,
    )

    np.testing.assert_array_equal(actual, expected)


def test_ensure_outward_face_winding_flips_inward_closed_mesh():
    vertices, outward = _cube_mesh()
    inward = outward[:, [0, 2, 1]]

    assert face_winding_signed_volume(vertices, outward) > 0.0
    assert face_winding_signed_volume(vertices, inward) < 0.0
    np.testing.assert_array_equal(
        ensure_outward_face_winding(vertices, outward),
        outward,
    )
    np.testing.assert_array_equal(
        ensure_outward_face_winding(vertices, inward),
        outward,
    )


def test_face_geometry_orient_outward_uses_closed_mesh_winding():
    vertices, outward = _cube_mesh()
    inward = outward[:, [0, 2, 1]]

    _, outward_normals, _ = face_geometry(vertices, outward, orient_outward=True)
    _, corrected_normals, _ = face_geometry(vertices, inward, orient_outward=True)

    np.testing.assert_allclose(corrected_normals, outward_normals)


def test_po_visibility_blocks_fully_occluded_back_face():
    vertices = np.vstack(
        [
            _front_facing_triangle(1.0, scale=2.0),
            _front_facing_triangle(2.0, scale=2.0),
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)

    result = po_channel(
        vertices=vertices,
        faces=faces,
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        material=human_skin_material("po-vis"),
    )

    assert np.isclose(result.raw_fractional_visibility[0], 1.0)
    assert np.isclose(result.raw_fractional_visibility[1], 0.0)
    assert np.array_equal(result.face_indices, [0])


def test_po_visibility_static_scene_blocks_direct_human_return():
    human_vertices = _front_facing_triangle(1.0)
    human_faces = np.array([[0, 1, 2]], dtype=np.uint32)
    static_blocker = HumanVisibilityScene(
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        vertices=_triangle_from_yz(
            0.5,
            (-0.6, -0.6),
            (0.6, -0.6),
            (0.0, 0.6),
        ),
        name="po-static-blocker",
    )

    clear = po_channel(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        material=human_skin_material("po-static-block-clear"),
        po_visibility_samples_per_face=1,
    )
    blocked = po_channel(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        material=human_skin_material("po-static-blocked"),
        static_visibility_scene=static_blocker.scene,
        po_visibility_samples_per_face=1,
    )

    assert np.isclose(clear.raw_fractional_visibility[0], 1.0)
    assert np.isclose(blocked.raw_fractional_visibility[0], 0.0)
    assert blocked.face_indices.size == 0


def test_batched_po_face_contributions_match_scalar_pairs():
    vertices = np.vstack(
        [
            _front_facing_triangle(1.0),
            _front_facing_triangle(1.3, y_shift=0.2),
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    centroids, normals, areas = face_geometry(vertices, faces)
    tx_positions = np.array(
        [
            [0.0, -0.02, 0.0],
            [0.0, 0.03, 0.0],
        ]
    )
    rx_positions = np.array(
        [
            [0.0, 0.01, 0.0],
            [0.0, -0.04, 0.0],
        ]
    )
    material = PhysicalOpticsMaterial(
        relative_permittivity=38.0,
        conductivity_s_per_m=1.5,
        thickness_m=0.01,
        model="slab",
        surface_attenuation_thickness_m=0.0,
    )

    batched_coefficients, batched_tau = vector_facet_face_contributions_for_pairs(
        face_indices=np.array([0, 1], dtype=np.int64),
        centroids=centroids,
        normals=normals,
        areas=areas,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        frequency_hz=60e9,
        material=material,
    )

    for pair_index, (tx_position, rx_position) in enumerate(zip(tx_positions, rx_positions)):
        scalar_coefficients, scalar_tau = vector_facet_face_contributions(
            face_indices=np.array([0, 1], dtype=np.int64),
            centroids=centroids,
            normals=normals,
            areas=areas,
            tx_position=tx_position,
            rx_position=rx_position,
            frequency_hz=60e9,
            material=material,
        )
        assert np.allclose(batched_coefficients[pair_index], scalar_coefficients)
        assert np.allclose(batched_tau[pair_index], scalar_tau)


@pytest.mark.parametrize("solver", ["numpy", "numpy_pairs", "torch"])
def test_po_channel_has_negative_propagation_phase(solver, monkeypatch):
    if solver == "torch":
        pytest.importorskip("torch")
        monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")
    frequency_hz = 60e9
    wavelength = c / frequency_hz
    distances = [1.0, 1.0 + wavelength / 16]
    coefficients = []
    for distance in distances:
        kwargs = dict(
            face_indices=np.array([0]),
            centroids=np.array([[distance, 0.0, 0.0]]),
            normals=np.array([[-1.0, 0.0, 0.0]]),
            areas=np.array([1e-4]),
            frequency_hz=frequency_hz,
            material=PhysicalOpticsMaterial(1.0, 0.0, model="pec"),
        )
        if solver == "numpy_pairs":
            value, _ = vector_facet_face_contributions_for_pairs(
                **kwargs,
                tx_positions=np.zeros((1, 3)),
                rx_positions=np.zeros((1, 3)),
            )
        else:
            value, _ = vector_facet_face_contributions(
                **kwargs,
                tx_position=np.zeros(3),
                rx_position=np.zeros(3),
                backend=solver,
                precision="float64",
            )
        coefficients.append(value.item())
    # Only the propagation phase and two-leg spreading change: the PEC
    # material, incidence, polarization, and face area are identical.
    ratio = coefficients[1] / coefficients[0] * (distances[1] / distances[0]) ** 2
    np.testing.assert_allclose(ratio, np.exp(-1j * np.pi / 4), rtol=1e-10)


@pytest.mark.parametrize(
    "integration_mode",
    [
        "face_centroid",
        "parent_face_quadrature",
        "parent_face_far_field_analytic",
        "parent_face_adaptive_analytic",
    ],
)
def test_po_integration_backends_agree_for_complex_material(integration_mode, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")
    kwargs = dict(
        vertices=_front_facing_triangle(2.0, scale=0.03, y_shift=0.01),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        tx_positions=np.zeros((1, 3)),
        rx_positions=np.array([[0.1, 0.2, -0.2]]),
        frequency_hz=60e9,
        material=PhysicalOpticsMaterial(4.0, 0.1, model="interface"),
        visibility="none",
        po_integration_mode=integration_mode,
        po_quadrature_phase_span_scale_rad=np.pi / 8,
        po_quadrature_max_refinement_depth=3,
        po_quadrature_max_subfaces_per_parent=64,
        compute_precision="float64",
    )
    numpy_result = po_channel(**kwargs, compute_backend="numpy")
    torch_result = po_channel(**kwargs, compute_backend="torch")
    np.testing.assert_allclose(
        torch_result.coefficients, numpy_result.coefficients, rtol=1e-9, atol=1e-15
    )


def test_parent_face_quadrature_po_keeps_parent_face_visibility_id():
    human_vertices = _front_facing_triangle(1.0, scale=2.0)
    human_faces = np.array([[0, 1, 2]], dtype=np.uint32)

    quad_centroids, _, quad_areas, parent_indices = parent_face_quadrature_geometry(
        vertices=human_vertices,
        faces=human_faces,
        selected_parent_indices=np.array([0], dtype=np.int64),
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        phase_span_scale_rad=1.0,
        max_refinement_depth=1,
    )
    assert quad_centroids.shape == (4, 3)
    assert np.array_equal(parent_indices, np.zeros(4, dtype=np.int64))
    assert np.isclose(
        quad_areas.sum(),
        face_geometry(human_vertices, human_faces)[2][0],
    )

    result = po_channel(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        material=human_skin_material("po-parent-face-quadrature"),
        visibility="none",
        po_integration_mode="parent_face_quadrature",
        po_quadrature_phase_span_scale_rad=1.0,
        po_quadrature_max_refinement_depth=1,
    )

    expected_material = PhysicalOpticsMaterial(
        relative_permittivity=38.0,
        conductivity_s_per_m=1.5,
        thickness_m=0.01,
        model="slab",
    )
    expected_subface_contrib, _ = vector_facet_face_contributions(
        face_indices=np.arange(quad_centroids.shape[0], dtype=np.int64),
        centroids=quad_centroids,
        normals=face_geometry(human_vertices, human_faces)[1][parent_indices],
        areas=quad_areas,
        tx_position=np.array([0.0, 0.0, 0.0]),
        rx_position=np.array([0.0, 0.0, 0.0]),
        frequency_hz=60e9,
        material=expected_material,
    )

    assert result.raw_fractional_visibility.shape == (1,)
    assert np.isclose(result.raw_fractional_visibility[0], 1.0)
    assert result.face_indices.size == 1
    assert np.array_equal(result.face_indices, np.array([0], dtype=np.int64))
    assert result.coefficients.shape == (1, 1)
    assert result.delays_s.shape == (1,)
    assert np.isclose(
        result.phase_center_contributions[0],
        np.sum(expected_subface_contrib),
    )
    assert np.isclose(
        result.face_areas[0],
        face_geometry(human_vertices, human_faces)[2][0],
    )


def test_parent_face_far_field_analytic_uses_global_mesh_distance_edge_rule():
    human_vertices = _front_facing_triangle(1.0, scale=0.2)
    human_faces = np.array([[0, 1, 2]], dtype=np.uint32)
    frequency_hz = 60e9
    wavelength_m = c / frequency_hz

    (
        centroids,
        _,
        areas,
        parent_indices,
        triangles,
        max_edge_m,
        nearest_distance_m,
    ) = parent_face_far_field_analytic_geometry(
        vertices=human_vertices,
        faces=human_faces,
        selected_parent_indices=np.array([0], dtype=np.int64),
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=frequency_hz,
        max_refinement_depth=4,
    )
    edge_lengths = np.stack(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ],
        axis=1,
    )

    assert np.isclose(nearest_distance_m, 1.0)
    assert np.isclose(max_edge_m, np.sqrt(wavelength_m * nearest_distance_m / 2.0))
    assert np.max(edge_lengths) <= max_edge_m * (1.0 + 1e-12)
    assert np.array_equal(parent_indices, np.zeros(parent_indices.size, dtype=np.int64))
    assert centroids.shape[0] == triangles.shape[0]
    assert np.isclose(areas.sum(), face_geometry(human_vertices, human_faces)[2][0])

    result = po_channel(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=frequency_hz,
        material=human_skin_material("po-parent-face-far-field"),
        visibility="none",
        po_integration_mode="parent_face_far_field_analytic",
        po_quadrature_max_refinement_depth=4,
    )

    assert np.array_equal(result.face_indices, np.array([0], dtype=np.int64))
    assert result.coefficients.shape == (1, 1)
    assert result.phase_center_contributions.shape == (1,)


def test_po_channel_projects_phase_center_paths_to_virtual_channel_delays():
    human_vertices = _front_facing_triangle(1.0)
    human_faces = np.array([[0, 1, 2]], dtype=np.uint32)
    tx_positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]])
    rx_positions = np.array([[0.0, 0.0, 0.0]])

    result = po_channel(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        frequency_hz=60e9,
        material=human_skin_material("po-vc-delay-projection"),
        visibility="none",
    )

    centroid = human_vertices.mean(axis=0).astype(float)
    expected_lengths = np.array(
        [
            np.linalg.norm(tx_position - centroid) + np.linalg.norm(rx_positions[0] - centroid)
            for tx_position in tx_positions
        ]
    )
    assert result.coefficients.shape == (2, 1)
    assert result.delays_s.shape == (2, 1)
    assert np.allclose(result.delays_s[:, 0] * c, expected_lengths)
    valid = np.any(np.abs(result.coefficients) > 0.0, axis=0)
    adc = MmWaveRadarSimulator().synthesize_adc(
        result.coefficients,
        result.delays_s,
        valid,
        _phase_one_radar(num_chirps=1).fmcw,
    )
    assert valid.shape == (1,)
    assert adc.shape == (4, 2)

    permuted = po_channel(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        virtual_channel_tx_indices=np.array([1, 0]),
        virtual_channel_rx_indices=np.array([0, 0]),
        frequency_hz=60e9,
        material=human_skin_material("po-vc-delay-projection-permuted"),
        visibility="none",
    )
    assert np.allclose(permuted.delays_s[:, 0] * c, expected_lengths[::-1])
    assert np.allclose(permuted.coefficients, result.coefficients[::-1])


def test_env_human_coupling_returns_single_reflection_terms():
    human_vertices = _front_facing_triangle(1.0)
    human_faces = np.array([[0, 1, 2]], dtype=np.uint32)
    reflector_vertices = _triangle_from_yz(
        -0.5,
        (-3.0, -3.0),
        (3.0, -3.0),
        (0.0, 3.0),
    )
    edge_1 = reflector_vertices[1] - reflector_vertices[0]
    edge_2 = reflector_vertices[2] - reflector_vertices[0]
    normal = np.cross(edge_1, edge_2)
    area = 0.5 * np.linalg.norm(normal)
    normal = normal / np.linalg.norm(normal)
    reflectors = EnvironmentReflectorBank(
        vertices=reflector_vertices[None, :, :],
        centroids=reflector_vertices.mean(axis=0, keepdims=True),
        normals=normal[None, :],
        areas=np.array([area], dtype=float),
        materials=(PhysicalOpticsMaterial(1.0, 0.0, model="pec"),),
        object_names=("reflector",),
        face_indices=np.array([0], dtype=np.int64),
    )
    human_scene = HumanVisibilityScene(
        faces=human_faces,
        vertices=human_vertices,
        name="po-coupling-human",
    )

    result = compute_env_human_coupling_channels(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        human_material=human_skin_material("po-coupled-human"),
        reflectors=reflectors,
        human_visibility_scene=human_scene,
        reflection_scale=1.0,
    )

    assert result.human_env_coefficients.shape == (1, 1)
    assert result.env_human_coefficients.shape == (1, 1)
    assert np.array_equal(result.human_env_valid, [True])
    assert np.array_equal(result.env_human_valid, [True])
    assert np.abs(result.human_env_coefficients[0, 0]) > 0.0
    assert np.abs(result.env_human_coefficients[0, 0]) > 0.0
    assert result.runtime_profile_s is not None
    assert result.runtime_profile_counts is not None
    for key in (
        "coupling_setup",
        "human_env_channel",
        "env_human_channel",
        "coupling_assembly",
        "coupling_total",
    ):
        assert key in result.runtime_profile_s
        assert result.runtime_profile_s[key] >= 0.0
    assert result.runtime_profile_counts["human_env_channel"] == 1
    assert result.runtime_profile_counts["env_human_channel"] == 1

    direct = po_channel(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        material=human_skin_material("po-coupled-direct"),
        po_visibility_samples_per_face=1,
    )
    centroid = np.mean(human_vertices, axis=0)
    expected_coupled_length_m = np.linalg.norm(
        centroid - np.array([0.0, 0.0, 0.0])
    ) + np.linalg.norm(centroid - np.array([-1.0, 0.0, 0.0]))
    assert np.isclose(
        result.human_env_delays_s[0] * 299792458.0,
        expected_coupled_length_m,
    )
    assert np.isclose(
        result.env_human_delays_s[0] * 299792458.0,
        expected_coupled_length_m,
    )
    assert result.human_env_delays_s[0] > direct.delays_s[0]
    assert result.env_human_delays_s[0] > direct.delays_s[0]


@pytest.mark.parametrize("num_tx", [1, 2])
@pytest.mark.parametrize(
    "orientation",
    [
        (0.0, 0.0, 0.0),
        (np.pi / 3, 0.0, 0.0),
        (0.7, -0.4, 0.2),
    ],
)
def test_coupling_hardware_gain_uses_real_legs_and_applies_scalar_phase_once(
    num_tx,
    orientation,
):
    # Human at (2, 0, 0), floor at z=-1: the reflected leg is at 45 degrees
    # for a centered antenna, while the direct human leg is on boresight.
    human = _front_facing_triangle(2.0, scale=0.03, z_shift=-0.01)
    floor = np.array([[-5.0, -5.0, -1.0], [5.0, -5.0, -1.0], [0.0, 5.0, -1.0]])
    from sionna.rt.utils import rotation_matrix

    rotation = np.asarray(rotation_matrix(mi.Point3f(*orientation))).reshape(3, 3)
    position = np.array([3.0, -2.0, 0.5])
    human = human @ rotation.T + position
    floor = floor @ rotation.T + position
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    reflectors = EnvironmentReflectorBank(
        vertices=floor[None],
        centroids=floor.mean(axis=0, keepdims=True),
        normals=np.array([[0.0, 0.0, 1.0]]) @ rotation.T,
        areas=np.array([50.0]),
        materials=(PhysicalOpticsMaterial(1.0, 0.0, model="pec"),),
        object_names=("floor",),
        face_indices=np.array([0]),
        reflection_gains=np.array([0.8 + 0.2j]),
    )
    tx_positions = np.zeros((num_tx, 3))
    if num_tx == 2:
        tx_positions[:, 2] = [-0.1, 0.1]
    signs = np.array([-1.0, 1.0])[:num_tx]
    hardware = replace(
        RadarHardware.from_positions(tx_positions, [[0.0, 0.0, 0.0]]),
        scalar_antenna_gain_dbi=6.0,
        virtual_channel_phase_signs=signs,
        antenna_pattern=SimpleCosinePattern(num_tx, half_power_angle_deg=45.0),
    )
    radar = RadarSensor(
        name="coupling-gain",
        position=tuple(position),
        orientation=orientation,
        hardware=hardware,
        fmcw=replace(_phase_one_radar().fmcw, num_tx=num_tx),
    )
    kwargs = dict(
        vertices=human,
        faces=faces,
        tx_positions=radar.world_tx_positions(),
        rx_positions=radar.world_rx_positions(),
        frequency_hz=60e9,
        human_material=human_skin_material("coupling-gain-human"),
        reflectors=reflectors,
        human_visibility_scene=HumanVisibilityScene(faces, human, name="coupling-gain"),
        polarization_axis=rotation[:, 2],
    )
    raw = compute_env_human_coupling_channels(**kwargs)
    weighted = compute_env_human_coupling_channels(**kwargs, radar=radar)
    cached = compute_env_human_coupling_channels(
        **kwargs,
        radar=radar,
        human_env_active_sets=weighted.human_env_active_sets,
        env_human_active_sets=weighted.env_human_active_sets,
    )
    height = tx_positions[:, 2]
    expected_cosines = {
        "human_env": (2.0 / np.sqrt(4.0 + height**2), 1.0 / np.sqrt(2.0)),
        "env_human": (2.0 / np.sqrt(4.0 + (2.0 + height) ** 2), 1.0),
    }
    for order, (cos_tx, cos_rx) in expected_cosines.items():
        expected = signs * 10.0 ** (6.0 / 20.0) * np.sqrt(cos_tx * cos_rx)
        coefficients = getattr(raw, order + "_coefficients")
        assert np.all(np.abs(coefficients) > 0.0)
        np.testing.assert_allclose(
            getattr(weighted, order + "_coefficients"),
            coefficients * expected[:, None],
            rtol=1e-6,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            getattr(cached, order + "_coefficients"),
            getattr(weighted, order + "_coefficients"),
            rtol=1e-6,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            getattr(weighted, order + "_delays_s"),
            getattr(raw, order + "_delays_s"),
        )


@pytest.mark.parametrize("num_tx", [1, 2])
@pytest.mark.parametrize(
    "integration_mode",
    [
        "parent_face_quadrature",
        "parent_face_far_field_analytic",
        "parent_face_adaptive_analytic",
    ],
)
def test_coupling_matches_human_po_with_image_endpoints(num_tx, integration_mode):
    human = _front_facing_triangle(2.0, scale=0.03, z_shift=-0.01)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    floor = np.array([[-5.0, -5.0, -1.0], [5.0, -5.0, -1.0], [0.0, 5.0, -1.0]])
    # Static RT diagnostics must not multiply the actual Fresnel reflection.
    reflection_gain = 1e6j
    reflectors = EnvironmentReflectorBank(
        vertices=floor[None],
        centroids=floor.mean(axis=0, keepdims=True),
        normals=np.array([[0.0, 0.0, 1.0]]),
        areas=np.array([50.0]),
        materials=(PhysicalOpticsMaterial(1.0, 0.0, model="pec"),),
        object_names=("floor",),
        face_indices=np.array([0]),
        reflection_gains=np.array([reflection_gain]),
    )
    tx = np.zeros((num_tx, 3))
    if num_tx == 2:
        tx[:, 1] = [-0.1, 0.1]
    rx = np.zeros((1, 3))
    image_tx, image_rx = tx.copy(), rx.copy()
    image_tx[:, 2] = image_rx[:, 2] = -2.0
    events = []
    config = POConfig(
        po_integration_mode=integration_mode,
        po_quadrature_phase_span_scale_rad=np.pi / 8,
        po_quadrature_max_refinement_depth=3,
        po_quadrature_max_subfaces_per_parent=64,
        progress_callback=events.append,
    )
    kwargs = dict(
        vertices=human,
        faces=faces,
        tx_positions=tx,
        rx_positions=rx,
        frequency_hz=60e9,
        human_material=human_skin_material("image-po-human"),
        reflectors=reflectors,
        po_config=config,
        human_visibility_scene=HumanVisibilityScene(faces, human, name="image-po"),
        reflection_scale=0.7,
    )
    result = compute_env_human_coupling_channels(**kwargs)
    cached = compute_env_human_coupling_channels(
        **kwargs,
        human_env_active_sets=result.human_env_active_sets,
        env_human_active_sets=result.env_human_active_sets,
    )
    for order, tx_for_po, rx_for_po in (
        ("human_env", tx, image_rx),
        ("env_human", image_tx, rx),
    ):
        expected = po_channel(
            vertices=human,
            faces=faces,
            tx_positions=tx_for_po,
            rx_positions=rx_for_po,
            frequency_hz=60e9,
            material=kwargs["human_material"],
            visibility="none",
            po_integration_mode=integration_mode,
            po_quadrature_phase_span_scale_rad=np.pi / 8,
            po_quadrature_max_refinement_depth=3,
            po_quadrature_max_subfaces_per_parent=64,
        )
        np.testing.assert_allclose(
            getattr(result, order + "_coefficients"),
            0.7 * expected.coefficients,
            rtol=1e-10,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            getattr(result, order + "_delays_s"),
            expected.delays_s,
        )
        np.testing.assert_allclose(
            getattr(cached, order + "_coefficients"),
            getattr(result, order + "_coefficients"),
            rtol=1e-10,
            atol=1e-15,
        )
        assert getattr(result, order + "_coefficients").shape == (num_tx, 1)

    stage = "quadrature" if integration_mode == "parent_face_quadrature" else "far_field"
    completed = [event for event in events if event["stage"] == f"{stage}_aggregate_done"]
    assert len(completed) == 4  # Both orders, fresh and cached visibility.
    assert all(event["parent_faces_total"] == 1 for event in completed)
    assert all(0 < event["subfaces_total"] <= 64 for event in completed)
    if integration_mode == "parent_face_quadrature":
        assert all(event["subfaces_total"] == 64 for event in completed)


def test_coupling_quadrature_resolves_phase_cancellation_against_dense_surface():
    # An oblique image path spans several wavelengths across this one face.
    # Independently subdivide it uniformly for a dense numerical reference.
    human = _front_facing_triangle(2.0, scale=0.03).astype(np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    centroids, normals, areas = face_geometry(human, faces)
    tx, image_rx = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -2.0])
    triangles = human[None]
    for _ in range(7):
        a, b, d = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        ab, bd, da = (a + b) / 2, (b + d) / 2, (d + a) / 2
        triangles = np.concatenate(
            [
                np.stack([a, ab, da], axis=1),
                np.stack([ab, b, bd], axis=1),
                np.stack([da, bd, d], axis=1),
                np.stack([ab, bd, da], axis=1),
            ]
        )
    dense_vertices = triangles.reshape(-1, 3)
    dense_faces = np.arange(dense_vertices.shape[0]).reshape(-1, 3)
    dc, dn, da = face_geometry(dense_vertices, dense_faces)
    material = PhysicalOpticsMaterial(1.0, 0.0, model="pec")
    reference, _ = vector_facet_face_contributions(
        face_indices=np.arange(len(da)),
        centroids=dc,
        normals=dn,
        areas=da,
        tx_position=tx,
        rx_position=image_rx,
        frequency_hz=60e9,
        material=material,
    )
    kwargs = dict(
        vertices=human,
        faces=faces,
        face_indices=np.array([0]),
        centroids=centroids,
        normals=normals,
        areas=areas,
        tx_position=tx,
        rx_position=image_rx,
        tx_positions=tx[None],
        rx_positions=image_rx[None],
        frequency_hz=60e9,
        material=material,
        backend="numpy",
        precision="float64",
    )
    centroid, _ = env_human_coupling._coupling_face_contributions(
        **kwargs,
        po_config=POConfig(),
    )
    refined, _ = env_human_coupling._coupling_face_contributions(
        **kwargs,
        po_config=POConfig(
            po_integration_mode="parent_face_quadrature",
            po_quadrature_phase_span_scale_rad=np.pi / 8,
            po_quadrature_max_refinement_depth=6,
            po_quadrature_max_subfaces_per_parent=4096,
        ),
    )
    np.testing.assert_allclose(refined[0], reference.sum(), rtol=0.01, atol=1e-15)
    assert abs(refined[0]) < 0.1 * abs(centroid[0])


def test_env_human_coupling_reuses_phase_center_visibility_for_virtual_channels(monkeypatch):
    human_vertices = _front_facing_triangle(1.0)
    human_faces = np.array([[0, 1, 2]], dtype=np.uint32)
    reflector_vertices = _triangle_from_yz(
        -0.5,
        (-3.0, -3.0),
        (3.0, -3.0),
        (0.0, 3.0),
    )
    edge_1 = reflector_vertices[1] - reflector_vertices[0]
    edge_2 = reflector_vertices[2] - reflector_vertices[0]
    normal = np.cross(edge_1, edge_2)
    area = 0.5 * np.linalg.norm(normal)
    normal = normal / np.linalg.norm(normal)
    reflectors = EnvironmentReflectorBank(
        vertices=reflector_vertices[None, :, :],
        centroids=reflector_vertices.mean(axis=0, keepdims=True),
        normals=normal[None, :],
        areas=np.array([area], dtype=float),
        materials=(PhysicalOpticsMaterial(1.0, 0.0, model="pec"),),
        object_names=("reflector",),
        face_indices=np.array([0], dtype=np.int64),
    )
    human_scene = HumanVisibilityScene(
        faces=human_faces,
        vertices=human_vertices,
        name="po-coupling-visibility-cache",
    )
    call_count = 0
    original = env_human_coupling._segments_clear_in_scene

    def counted_segments_clear(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        env_human_coupling,
        "_segments_clear_in_scene",
        counted_segments_clear,
    )

    result = compute_env_human_coupling_channels(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        human_material=human_skin_material("po-coupled-visibility-cache"),
        reflectors=reflectors,
        human_visibility_scene=human_scene,
        reflection_scale=1.0,
    )

    assert result.human_env_coefficients.shape == (2, 1)
    assert result.env_human_coefficients.shape == (2, 1)
    assert result.human_env_delays_s.shape == (2, 1)
    assert result.env_human_delays_s.shape == (2, 1)
    assert np.array_equal(result.human_env_valid, [True])
    assert np.array_equal(result.env_human_valid, [True])

    permuted = compute_env_human_coupling_channels(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        virtual_channel_tx_indices=np.array([1, 0]),
        virtual_channel_rx_indices=np.array([0, 0]),
        frequency_hz=60e9,
        human_material=human_skin_material("po-coupled-visibility-cache-permuted"),
        reflectors=reflectors,
        human_visibility_scene=human_scene,
        reflection_scale=1.0,
        human_env_active_sets=result.human_env_active_sets,
        env_human_active_sets=result.env_human_active_sets,
    )
    assert np.allclose(
        permuted.human_env_coefficients,
        result.human_env_coefficients[::-1],
    )
    assert np.allclose(
        permuted.env_human_coefficients,
        result.env_human_coefficients[::-1],
    )
    assert np.allclose(
        permuted.human_env_delays_s,
        result.human_env_delays_s[::-1],
    )
    assert np.allclose(
        permuted.env_human_delays_s,
        result.env_human_delays_s[::-1],
    )
    # Three geometric legs per order; virtual channels reuse that visibility.
    assert call_count == 6


def test_env_human_coupling_can_reuse_active_sets_without_segment_visibility(monkeypatch):
    human_vertices = _front_facing_triangle(1.0)
    human_faces = np.array([[0, 1, 2]], dtype=np.uint32)
    reflector_vertices = _triangle_from_yz(
        -0.5,
        (-3.0, -3.0),
        (3.0, -3.0),
        (0.0, 3.0),
    )
    edge_1 = reflector_vertices[1] - reflector_vertices[0]
    edge_2 = reflector_vertices[2] - reflector_vertices[0]
    normal = np.cross(edge_1, edge_2)
    area = 0.5 * np.linalg.norm(normal)
    normal = normal / np.linalg.norm(normal)
    reflectors = EnvironmentReflectorBank(
        vertices=reflector_vertices[None, :, :],
        centroids=reflector_vertices.mean(axis=0, keepdims=True),
        normals=normal[None, :],
        areas=np.array([area], dtype=float),
        materials=(PhysicalOpticsMaterial(1.0, 0.0, model="pec"),),
        object_names=("reflector",),
        face_indices=np.array([0], dtype=np.int64),
    )
    human_scene = HumanVisibilityScene(
        faces=human_faces,
        vertices=human_vertices,
        name="po-coupling-active-set-reuse",
    )
    first = compute_env_human_coupling_channels(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=np.array([[0.0, -0.01, 0.0], [0.0, 0.01, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        human_material=human_skin_material("po-coupled-active-set-first"),
        reflectors=reflectors,
        human_visibility_scene=human_scene,
        reflection_scale=1.0,
    )

    def fail_segments_clear(*args, **kwargs):
        raise AssertionError("cached active-set coupling should not retrace segments")

    monkeypatch.setattr(
        env_human_coupling,
        "_segments_clear_in_scene",
        fail_segments_clear,
    )
    human_scene.update_vertices(human_vertices)
    second = compute_env_human_coupling_channels(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=np.array([[0.0, -0.01, 0.0], [0.0, 0.01, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        human_material=human_skin_material("po-coupled-active-set-second"),
        reflectors=reflectors,
        human_visibility_scene=human_scene,
        reflection_scale=1.0,
        human_env_active_sets=first.human_env_active_sets,
        env_human_active_sets=first.env_human_active_sets,
    )

    assert np.array_equal(second.human_env_valid, first.human_env_valid)
    assert np.array_equal(second.env_human_valid, first.env_human_valid)
    assert np.allclose(second.human_env_coefficients, first.human_env_coefficients)
    assert np.allclose(second.env_human_coefficients, first.env_human_coefficients)


def test_env_human_coupling_keeps_clear_path_with_backside_virtual_rx():
    human_vertices = _front_facing_triangle(1.0)
    human_faces = np.array([[0, 1, 2]], dtype=np.uint32)
    human_centroid = human_vertices.mean(axis=0).astype(float)
    human_normal = np.cross(
        human_vertices[1] - human_vertices[0],
        human_vertices[2] - human_vertices[0],
    )
    human_normal = human_normal / np.linalg.norm(human_normal)

    reflection_point = np.array([1.5, 2.0, human_centroid[2]], dtype=float)
    reflected_leg = reflection_point - human_centroid
    virtual_rx = (
        human_centroid
        + (1.0 + np.linalg.norm(reflection_point) / np.linalg.norm(reflected_leg)) * reflected_leg
    )
    assert (
        np.dot(
            human_normal,
            (virtual_rx - human_centroid) / np.linalg.norm(virtual_rx - human_centroid),
        )
        < 0.0
    )

    reflector_normal = virtual_rx / np.linalg.norm(virtual_rx)
    tangent = np.cross(reflector_normal, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(tangent) < 1e-9:
        tangent = np.cross(reflector_normal, np.array([0.0, 1.0, 0.0]))
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(reflector_normal, tangent)
    reflector_vertices = np.array(
        [
            reflection_point + 3.0 * tangent,
            reflection_point + 3.0 * (-0.5 * tangent + 0.8660254038 * bitangent),
            reflection_point + 3.0 * (-0.5 * tangent - 0.8660254038 * bitangent),
        ],
        dtype=np.float32,
    )
    edge_1 = reflector_vertices[1] - reflector_vertices[0]
    edge_2 = reflector_vertices[2] - reflector_vertices[0]
    area = 0.5 * np.linalg.norm(np.cross(edge_1, edge_2))
    reflectors = EnvironmentReflectorBank(
        vertices=reflector_vertices[None, :, :],
        centroids=reflection_point[None, :],
        normals=reflector_normal[None, :],
        areas=np.array([area], dtype=float),
        materials=(PhysicalOpticsMaterial(1.0, 0.0, model="pec"),),
        object_names=("reflector",),
        face_indices=np.array([0], dtype=np.int64),
    )
    human_scene = HumanVisibilityScene(
        faces=human_faces,
        vertices=human_vertices,
        name="po-coupling-virtual-rx-backside",
    )

    result = compute_env_human_coupling_channels(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        human_material=human_skin_material("po-coupled-virtual-rx-backside"),
        reflectors=reflectors,
        human_visibility_scene=human_scene,
        reflection_scale=1.0,
    )

    assert np.array_equal(result.human_env_valid, [True])
    assert np.abs(result.human_env_coefficients[0, 0]) > 0.0


def test_env_human_coupling_rejects_human_blocked_reflector_leg():
    scatterer = _front_facing_triangle(1.0)
    blocker = _triangle_from_yz(
        -0.25,
        (-0.2, -0.1),
        (0.2, -0.1),
        (0.0, 0.2),
    )
    human_vertices = np.vstack([scatterer, blocker])
    human_faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    reflector_vertices = _triangle_from_yz(
        -0.5,
        (-3.0, -3.0),
        (3.0, -3.0),
        (0.0, 3.0),
    )
    edge_1 = reflector_vertices[1] - reflector_vertices[0]
    edge_2 = reflector_vertices[2] - reflector_vertices[0]
    normal = np.cross(edge_1, edge_2)
    area = 0.5 * np.linalg.norm(normal)
    normal = normal / np.linalg.norm(normal)
    reflectors = EnvironmentReflectorBank(
        vertices=reflector_vertices[None, :, :],
        centroids=reflector_vertices.mean(axis=0, keepdims=True),
        normals=normal[None, :],
        areas=np.array([area], dtype=float),
        materials=(PhysicalOpticsMaterial(1.0, 0.0, model="pec"),),
        object_names=("reflector",),
        face_indices=np.array([0], dtype=np.int64),
    )
    human_scene = HumanVisibilityScene(
        faces=human_faces,
        vertices=human_vertices,
        name="po-coupling-blocked-reflector-leg",
    )

    result = compute_env_human_coupling_channels(
        vertices=human_vertices,
        faces=human_faces,
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        human_material=human_skin_material("po-coupled-blocked-human"),
        reflectors=reflectors,
        human_visibility_scene=human_scene,
        reflection_scale=1.0,
    )

    assert result.human_env_valid.shape == (2,)
    assert result.env_human_valid.shape == (2,)
    assert not bool(result.human_env_valid[0])
    assert not bool(result.env_human_valid[0])


def test_po_visibility_returns_expected_fraction_for_deterministic_samples():
    blocker_centroid = _triangle_from_yz(
        1.0,
        (-0.08, 0.08),
        (0.08, 0.08),
        (0.0, 0.22),
    )
    blocker_left = _triangle_from_yz(
        1.0,
        (-0.2, 0.02),
        (0.02, 0.02),
        (-0.02, 0.22),
    )
    back_face = _front_facing_triangle(2.0)
    vertices = np.vstack([blocker_centroid, blocker_left, back_face])
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint32)

    result = po_channel(
        vertices=vertices,
        faces=faces,
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        material=human_skin_material("po-partial"),
        po_adaptive_visibility_sampling=False,
    )

    assert result.visible_sample_counts[2] == 2
    assert np.isclose(result.raw_fractional_visibility[2], 0.5)


def test_po_channel_visibility_defaults_match_public_config():
    import inspect

    defaults = inspect.signature(po_channel).parameters
    config = POConfig()
    assert (
        defaults["po_adaptive_visibility_sampling"].default
        is config.po_adaptive_visibility_sampling
    )
    assert (
        defaults["po_adaptive_visibility_edge_margin"].default
        == config.po_adaptive_visibility_edge_margin
    )


def test_po_visibility_fade_ramps_out_hidden_face():
    vertices_clear = np.vstack(
        [
            _front_facing_triangle(1.0, y_shift=5.0),
            _front_facing_triangle(2.0),
        ]
    )
    vertices_blocked = np.vstack(
        [
            _front_facing_triangle(1.0, scale=2.0),
            _front_facing_triangle(2.0),
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)

    first = po_channel(
        vertices=vertices_clear,
        faces=faces,
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        material=human_skin_material("po-fade"),
        visibility_fade_steps=4,
    )
    second = po_channel(
        vertices=vertices_blocked,
        faces=faces,
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, 0.0]]),
        frequency_hz=60e9,
        material=human_skin_material("po-fade"),
        visibility_state=first.visibility_state,
        visibility_fade_steps=4,
    )

    assert np.isclose(first.effective_weights[1], 1.0)
    assert np.isclose(second.raw_fractional_visibility[1], 0.0)
    assert np.isclose(second.temporal_fade_weights[1], 0.75)
    assert np.isclose(second.effective_weights[1], 0.75)


def test_target_interaction_counts_ignore_inactive_target_slots():
    class DummySceneObject:
        object_id = np.array([7], dtype=np.int32)

    class DummyTarget:
        def ensure_scene_object(self):
            return DummySceneObject()

    class DummyPaths:
        tau = np.zeros(3, dtype=float)
        objects = np.array(
            [
                [[7, 7, 8]],
                [[7, 8, 7]],
            ],
            dtype=np.int32,
        )
        interactions = np.array(
            [
                [
                    [
                        InteractionType.SPECULAR,
                        InteractionType.NONE,
                        InteractionType.SPECULAR,
                    ]
                ],
                [
                    [
                        InteractionType.NONE,
                        InteractionType.SPECULAR,
                        InteractionType.SPECULAR,
                    ]
                ],
            ],
            dtype=np.int32,
        )

    target_counts, total_counts = MmWaveRadarSimulator.target_interaction_counts(
        DummyPaths(),
        [DummyTarget()],
    )

    assert np.array_equal(target_counts, [1, 0, 1])
    assert np.array_equal(total_counts, [1, 1, 2])
    assert np.array_equal(
        MmWaveRadarSimulator.one_target_bounce_mask(
            DummyPaths(),
            [DummyTarget()],
        ),
        [True, True, True],
    )
    assert np.array_equal(
        MmWaveRadarSimulator.single_target_only_bounce_mask(
            DummyPaths(),
            [DummyTarget()],
        ),
        [True, False, False],
    )
