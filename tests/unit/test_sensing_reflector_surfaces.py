# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""RT faces seed image planes; their coplanar surfaces support reflected paths."""

from types import SimpleNamespace
from scipy.spatial.transform import Rotation

from sensing_test_helpers import *  # noqa: F401,F403
import mmWaveRadar.simulation.env_human_coupling as coupling


def _scene_and_static_bank(vertices, faces, selected_faces):
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.uint32)
    mesh = mi.Mesh(
        "coplanar-floor",
        len(vertices),
        len(faces),
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(vertices.astype(np.float32).reshape(-1))
    params["faces"] = mi.UInt32(faces.reshape(-1))
    params.update()
    scene = SimpleNamespace(
        objects={
            "floor": SimpleNamespace(
                object_id=7,
                mi_mesh=mesh,
                radio_material=PhysicalOpticsMaterial(1.0, 0.0, model="pec"),
            )
        }
    )
    selected = np.asarray(selected_faces, dtype=np.int64)
    count = len(selected)
    static = StaticPathBank(
        coefficients=np.ones((1, count), dtype=np.complex128),
        delays_s=np.full(count, 2.0 / c),
        valid=np.ones(count, dtype=bool),
        segment_starts=np.zeros((0, 3)),
        segment_ends=np.zeros((0, 3)),
        segment_path_indices=np.zeros(0, dtype=np.int64),
        path_vertices=vertices[faces[selected]].mean(axis=1)[None],
        path_interactions=np.full((1, count), InteractionType.SPECULAR, dtype=np.uint32),
        path_objects=np.full((1, count), 7, dtype=np.uint32),
        path_primitives=selected[None].astype(np.uint32),
    )
    return scene, static


def _square_floor():
    return (
        np.array([[-2.0, -2.0, -1.0], [2.0, -2.0, -1.0], [2.0, 2.0, -1.0], [-2.0, 2.0, -1.0]]),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
    )


@pytest.mark.parametrize("num_elements", [1, 2])
@pytest.mark.parametrize("human_y", [0.0, 2.0])
def test_image_path_can_cross_other_coplanar_faces_in_fresh_cached_and_array_solves(
    num_elements,
    human_y,
):
    floor, floor_faces = _square_floor()
    # Seed triangle 1 is above y=x. At human_y=0, every reflection goes
    # through triangle 0. At human_y=2 the array rays straddle the diagonal.
    scene, static = _scene_and_static_bank(floor, floor_faces, [1])
    bank = extract_rt_reflector_bank(scene, static, max_reflectors=0)
    assert bank.vertices.shape == (1, 3, 3)
    assert len(bank.support_triangles[0]) == 2
    human = _front_facing_triangle(2.0, scale=0.03, z_shift=-0.01, y_shift=human_y)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    elements = np.zeros((num_elements, 3))
    if num_elements == 2:
        elements[:, 1] = [-0.2, 0.2]
    radar = replace(
        _phase_one_radar(num_chirps=1),
        hardware=RadarHardware.from_positions(elements, elements),
        fmcw=replace(_phase_one_radar(num_chirps=1).fmcw, num_tx=num_elements),
    )
    kwargs = dict(
        vertices=human,
        faces=faces,
        tx_positions=radar.world_tx_positions(),
        rx_positions=radar.world_rx_positions(),
        frequency_hz=60e9,
        human_material=PhysicalOpticsMaterial(1.0, 0.0, model="pec"),
        reflectors=bank,
        human_visibility_scene=HumanVisibilityScene(faces, human, name="surface-path"),
    )
    raw = compute_env_human_coupling_channels(**kwargs)
    weighted = compute_env_human_coupling_channels(**kwargs, radar=radar)
    cached = compute_env_human_coupling_channels(
        **kwargs,
        radar=radar,
        human_env_active_sets=weighted.human_env_active_sets,
        env_human_active_sets=weighted.env_human_active_sets,
    )
    for order in ("human_env", "env_human"):
        coefficients = getattr(raw, order + "_coefficients")
        assert coefficients.shape == (num_elements**2, 1)
        assert np.all(np.abs(coefficients) > 0.0)
        np.testing.assert_allclose(getattr(weighted, order + "_coefficients"), coefficients)
        np.testing.assert_allclose(getattr(cached, order + "_coefficients"), coefficients)
    if human_y == 0.0:
        seed_only = compute_env_human_coupling_channels(
            **{**kwargs, "reflectors": replace(bank, support_triangles=None)},
        )
        assert not np.any(seed_only.human_env_valid)
        assert not np.any(seed_only.env_human_valid)


@pytest.mark.parametrize("rotated", [False, True])
def test_coplanar_rt_hits_produce_one_image_and_minimum_area_uses_whole_surface(rotated):
    floor, faces = _square_floor()
    # Reverse the second triangle's winding: this is still the same plane.
    faces[1] = faces[1, ::-1]
    if rotated:
        rotation = Rotation.from_euler("ZYX", [0.7, -0.4, 0.2]).as_matrix()
        floor = floor @ rotation.T + np.array([3.0, -2.0, 0.5])
    scene, static = _scene_and_static_bank(floor, faces, [1, 0, 1])
    bank = extract_rt_reflector_bank(scene, static, max_reflectors=0, min_area_m2=12.0)
    assert len(bank.vertices) == 1
    np.testing.assert_allclose(bank.areas, [16.0], rtol=1e-6)
    assert bank.support_triangles[0].shape == (2, 3, 3)
    fallback = coupling.extract_environment_reflector_bank(scene, min_area_m2=12.0)
    assert len(fallback.vertices) == 1
    np.testing.assert_allclose(fallback.areas, [16.0], rtol=1e-6)


def test_finite_surface_union_preserves_holes_and_rejects_other_planes():
    # Four strips form a square floor with a central hole.
    outer = np.array([[-2.0, -2.0, -1.0], [2.0, -2.0, -1.0], [2.0, 2.0, -1.0], [-2.0, 2.0, -1.0]])
    inner = outer.copy()
    inner[:, :2] *= 0.25
    vertices = np.vstack([outer, inner, outer + [0.0, 0.0, -0.1]])
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces.extend([[i, j, j + 4], [i, j + 4, i + 4]])
    # A parallel surface at z=-1.1 must not fill the hole at z=-1.
    faces.extend([[8, 9, 10], [8, 10, 11]])
    scene, static = _scene_and_static_bank(vertices, faces, [0])
    bank = extract_rt_reflector_bank(scene, static)
    support = bank.support_triangles[0]
    assert len(support) == 8
    points, valid = coupling._reflection_points_on_surface(
        np.array([0.0, 0.0, -2.0]),
        np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        bank.centroids[0],
        bank.normals[0],
        support,
    )
    np.testing.assert_array_equal(valid, [True, False, False])
    np.testing.assert_allclose(points[0], [1.0, 0.0, -1.0])


def test_densely_triangulated_surface_lookup_preserves_a_hole():
    coordinates = np.linspace(-2.0, 2.0, 9)
    triangles = []
    for i in range(8):
        for j in range(8):
            if i in (3, 4) and j in (3, 4):
                continue  # Leave the central [-.5, .5]^2 hole unfilled.
            x0, x1 = coordinates[i : i + 2]
            y0, y1 = coordinates[j : j + 2]
            a, b = [x0, y0, -1.0], [x1, y0, -1.0]
            d, e = [x1, y1, -1.0], [x0, y1, -1.0]
            triangles.extend([[a, b, d], [a, d, e]])
    xy = np.random.default_rng(7).uniform(-3.0, 3.0, size=(512, 2))
    ends = np.column_stack([2.0 * xy, np.zeros(len(xy))])
    points, valid = coupling._reflection_points_on_surface(
        np.array([0.0, 0.0, -2.0]),
        ends,
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.asarray(triangles),
    )
    expected = np.all(np.abs(xy) <= 2.0, axis=1) & ~np.all(np.abs(xy) < 0.5, axis=1)
    np.testing.assert_array_equal(valid, expected)
    np.testing.assert_allclose(points[:, :2], xy)


def test_quadrature_reflection_is_masked_per_sample_at_surface_boundaries():
    floor, faces = _square_floor()
    scene, static = _scene_and_static_bank(floor, faces, [1])
    bank = extract_rt_reflector_bank(scene, static)
    callback = coupling._image_polarization_callback(
        tx_position=np.array([0.0, 0.0, -2.0]),
        rx_position=np.array([-1.0, 0.0, 0.0]),
        reflector_point=bank.centroids[0],
        reflector_normal=bank.normals[0],
        support_triangles=bank.support_triangles[0],
        material=bank.materials[0],
        frequency_hz=60e9,
        order="env_human",
        scale=1.0,
        polarization_axis=np.array([0.0, 0.0, 1.0]),
    )
    # Different integration points on a parent face may hit different
    # triangles or miss the finite floor. Keep precisely the covered samples.
    incident, _ = callback(np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [6.0, 0.0, 0.0]]))
    np.testing.assert_allclose(np.linalg.norm(incident, axis=1), [1.0, 1.0, 0.0])


def test_cached_visibility_allows_motion_across_coplanar_triangle_boundary():
    floor, floor_faces = _square_floor()
    scene, static = _scene_and_static_bank(floor, floor_faces, [1])
    bank = extract_rt_reflector_bank(scene, static)
    results = []
    for y_shift in (0.0, 3.0):
        human = _front_facing_triangle(2.0, scale=0.03, y_shift=y_shift, z_shift=-0.01)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        kwargs = dict(
            vertices=human,
            faces=faces,
            tx_positions=np.zeros((1, 3)),
            rx_positions=np.zeros((1, 3)),
            frequency_hz=60e9,
            human_material=PhysicalOpticsMaterial(1.0, 0.0, model="pec"),
            reflectors=bank,
            human_visibility_scene=HumanVisibilityScene(faces, human, name="surface-motion"),
        )
        fresh = compute_env_human_coupling_channels(**kwargs)
        if results:
            cached = compute_env_human_coupling_channels(
                **kwargs,
                human_env_active_sets=results[0].human_env_active_sets,
                env_human_active_sets=results[0].env_human_active_sets,
            )
            for order in ("human_env", "env_human"):
                assert np.all(getattr(fresh, order + "_valid"))
                np.testing.assert_allclose(
                    getattr(cached, order + "_coefficients"),
                    getattr(fresh, order + "_coefficients"),
                )
        results.append(fresh)
