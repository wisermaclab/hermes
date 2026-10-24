# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Angle-dependent environment reflection and complex polarization transport."""

from scipy.constants import epsilon_0
from scipy.spatial.transform import Rotation

from sensing_test_helpers import *  # noqa: F401,F403
import mmWaveRadar.simulation.env_human_coupling as coupling
import mmWaveRadar.simulation.physical_optics as po


def _interface_fresnel(cosine, material, frequency=60e9):
    epsilon = material.relative_permittivity - 1j * material.conductivity_s_per_m / (
        2.0 * np.pi * frequency * epsilon_0
    )
    root = np.sqrt(epsilon - (1.0 - cosine**2) + 0j)
    return (
        (cosine - root) / (cosine + root),
        (epsilon * cosine - root) / (epsilon * cosine + root),
    )


@pytest.mark.parametrize("order", ["human_env", "env_human"])
@pytest.mark.parametrize("normal_sign", [-1.0, 1.0])
@pytest.mark.parametrize("angle_deg", [30.0, 60.0, 75.0])
@pytest.mark.parametrize("polarization", ["TE", "TM", "mixed"])
def test_image_field_matches_complex_fresnel_for_actual_incidence(
    order,
    normal_sign,
    angle_deg,
    polarization,
):
    angle = np.deg2rad(angle_deg)
    point = np.array([[2.0 * np.tan(angle), 0.0, 1.0]])
    image, direct = np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 1.0])
    axis = np.array(
        {"TE": [0.0, 1.0, 0.0], "TM": [0.0, 0.0, 1.0], "mixed": [0.0, 1.0, 1.0]}[polarization]
    )
    material = PhysicalOpticsMaterial(4.0, 0.2, model="interface")
    callback = coupling._image_polarization_callback(
        tx_position=direct if order == "human_env" else image,
        rx_position=image if order == "human_env" else direct,
        reflector_normal=np.array([0.0, 0.0, normal_sign]),
        material=material,
        frequency_hz=60e9,
        order=order,
        scale=0.7,
        polarization_axis=axis,
    )
    tx_pol, rx_pol = callback(point)
    transported = rx_pol.conjugate() if order == "human_env" else tx_pol
    r_te, r_tm = _interface_fresnel(np.cos(angle), material)
    expected = (
        0.7 * r_te * axis
        if polarization == "TE"
        else 0.7 * r_tm * np.array([-np.cos(angle), 0.0, np.sin(angle)])
    )
    if polarization == "mixed":
        norm = np.sqrt(1.0 + np.sin(angle) ** 2)
        expected = (
            0.7
            / norm
            * (
                r_te * np.array([0.0, 1.0, 0.0])
                + r_tm * np.sin(angle) * np.array([-np.cos(angle), 0.0, np.sin(angle)])
            )
        )
    np.testing.assert_allclose(transported[0], expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("axis", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
def test_normal_incidence_is_basis_independent_and_finite(axis):
    material = PhysicalOpticsMaterial(4.0, 0.2, model="interface")
    callback = coupling._image_polarization_callback(
        tx_position=np.array([0.0, 0.0, -1.0]),
        rx_position=np.array([-1.0, 0.0, 1.0]),
        reflector_normal=np.array([0.0, 0.0, 1.0]),
        material=material,
        frequency_hz=60e9,
        order="env_human",
        scale=1.0,
        polarization_axis=np.array(axis),
    )
    tx_pol, _ = callback(np.array([[0.0, 0.0, 1.0]]))
    r_te, _ = _interface_fresnel(1.0, material)
    np.testing.assert_allclose(tx_pol[0], r_te * np.array(axis), atol=1e-15)


def test_tm_reflection_vanishes_at_brewster_angle_without_te_power_averaging():
    material = PhysicalOpticsMaterial(4.0, 0.0, model="interface")
    kwargs = dict(
        tx_position=np.array([0.0, 0.0, -1.0]),
        rx_position=np.array([-1.0, 0.0, 1.0]),
        reflector_normal=np.array([0.0, 0.0, 1.0]),
        material=material,
        frequency_hz=60e9,
        order="env_human",
        scale=1.0,
    )
    # tan(theta_B) = sqrt(epsilon_r) = 2; point and real source lie at z=1.
    point = np.array([[4.0, 0.0, 1.0]])
    tm, _ = coupling._image_polarization_callback(
        **kwargs,
        polarization_axis=np.array([0.0, 0.0, 1.0]),
    )(point)
    te, _ = coupling._image_polarization_callback(
        **kwargs,
        polarization_axis=np.array([0.0, 1.0, 0.0]),
    )(point)
    assert np.linalg.norm(tm) < 1e-14
    assert np.linalg.norm(te) == pytest.approx(0.6)


def _floor_setup():
    human = _front_facing_triangle(2.0, scale=0.06, z_shift=-0.02).astype(np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    floor = np.array([[-5.0, -5.0, -1.0], [5.0, -5.0, -1.0], [0.0, 5.0, -1.0]])
    material = PhysicalOpticsMaterial(4.0, 0.2, model="interface")
    reflectors = EnvironmentReflectorBank(
        vertices=floor[None],
        centroids=floor.mean(axis=0, keepdims=True),
        normals=np.array([[0.0, 0.0, 1.0]]),
        areas=np.array([50.0]),
        materials=(material,),
        object_names=("floor",),
        face_indices=np.array([0]),
        reflection_gains=np.array([1e8 + 2e8j]),  # Diagnostic must be ignored.
    )
    return human, faces, reflectors


@pytest.mark.parametrize(
    "mode",
    [
        "face_centroid",
        "parent_face_quadrature",
        "parent_face_far_field_analytic",
    ],
)
@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("specular_scale", [1.0, 0.8])
def test_coupling_applies_complex_fresnel_before_coherent_subface_sum(
    mode,
    backend,
    specular_scale,
    monkeypatch,
):
    if backend == "torch":
        pytest.importorskip("torch")
        monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")
    human, faces, reflectors = _floor_setup()
    reflectors = replace(reflectors, specular_amplitude_scales=np.array([specular_scale]))
    config = POConfig(
        po_integration_mode=mode,
        po_quadrature_phase_span_scale_rad=np.pi / 8,
        po_quadrature_max_refinement_depth=3,
        po_quadrature_max_subfaces_per_parent=64,
    )
    human_material = PhysicalOpticsMaterial(1.0, 0.0, model="pec")
    kwargs = dict(
        vertices=human,
        faces=faces,
        tx_positions=np.zeros((1, 3)),
        rx_positions=np.zeros((1, 3)),
        frequency_hz=60e9,
        human_material=human_material,
        reflectors=reflectors,
        po_config=config,
        human_visibility_scene=HumanVisibilityScene(faces, human, name="fresnel-integration"),
        reflection_scale=0.65,
        backend=backend,
        precision="float64",
    )
    result = compute_env_human_coupling_channels(**kwargs)
    cached = compute_env_human_coupling_channels(
        **kwargs,
        human_env_active_sets=result.human_env_active_sets,
        env_human_active_sets=result.env_human_active_sets,
    )
    for order in ("human_env", "env_human"):
        real, image = np.zeros(3), np.array([0.0, 0.0, -2.0])
        tx, rx = (real, image) if order == "human_env" else (image, real)
        geometry = dict(
            vertices=human,
            faces=faces,
            selected_parent_indices=np.array([0]),
            tx_positions=tx[None],
            rx_positions=rx[None],
            frequency_hz=60e9,
            max_refinement_depth=3,
            max_subfaces_per_parent=64,
        )
        phase_integral = 1.0
        if mode == "face_centroid":
            points, normals, areas = po.face_geometry(human, faces)
        elif mode == "parent_face_quadrature":
            points, normals, areas, _ = po.parent_face_quadrature_geometry(
                **geometry,
                phase_span_scale_rad=np.pi / 8,
            )
        else:
            points, normals, areas, _, triangles, _, _ = (
                po.parent_face_far_field_analytic_geometry(
                    **geometry,
                )
            )
            tx_hat = (tx - points) / np.linalg.norm(tx - points, axis=1)[:, None]
            rx_hat = (rx - points) / np.linalg.norm(rx - points, axis=1)[:, None]
            q = -2.0 * np.pi * 60e9 / c * (tx_hat + rx_hat)[None]
            phase_integral = (
                po._triangle_linear_phase_integrals(triangles, points, q).ravel() / areas
            )
        # With vertical polarization and a horizontal floor, the environment
        # bounce is pure TM. This independent scalar reference is exact here.
        base, _ = po.vector_facet_face_contributions(
            face_indices=np.arange(len(areas)),
            centroids=points,
            normals=normals,
            areas=areas,
            tx_position=tx,
            rx_position=rx,
            frequency_hz=60e9,
            material=human_material,
            backend="numpy",
        )
        cosine = (points[:, 2] + 2.0) / np.linalg.norm(points - image, axis=1)
        _, r_tm = _interface_fresnel(cosine, reflectors.materials[0])
        expected = np.sum(0.65 * specular_scale * r_tm * base * phase_integral)
        np.testing.assert_allclose(
            getattr(result, order + "_coefficients"), expected, rtol=1e-9, atol=1e-15
        )
        np.testing.assert_allclose(
            getattr(cached, order + "_coefficients"), expected, rtol=1e-9, atol=1e-15
        )


@pytest.mark.parametrize("mode", ["face_centroid", "parent_face_quadrature"])
def test_reflected_fields_are_invariant_when_radar_and_scene_rotate_together(mode):
    human, faces, reflectors = _floor_setup()
    orientation = (0.7, -0.4, 0.2)
    rotation = Rotation.from_euler("ZYX", orientation).as_matrix()
    position = np.array([3.0, -2.0, 0.5])
    config = POConfig(
        po_integration_mode=mode,
        po_quadrature_phase_span_scale_rad=np.pi / 8,
        po_quadrature_max_refinement_depth=3,
        po_quadrature_max_subfaces_per_parent=64,
    )
    values = []
    for rotated in (False, True):
        radar = _phase_one_radar(num_chirps=1)
        vertices, bank = human, reflectors
        if rotated:
            radar = replace(radar, position=tuple(position), orientation=orientation)
            vertices = human @ rotation.T + position
            bank = replace(
                reflectors,
                vertices=reflectors.vertices @ rotation.T + position,
                centroids=reflectors.centroids @ rotation.T + position,
                normals=reflectors.normals @ rotation.T,
            )
        values.append(
            compute_env_human_coupling_channels(
                vertices=vertices,
                faces=faces,
                radar=radar,
                tx_positions=radar.world_tx_positions(),
                rx_positions=radar.world_rx_positions(),
                frequency_hz=60e9,
                human_material=PhysicalOpticsMaterial(1.0, 0.0, model="pec"),
                reflectors=bank,
                po_config=config,
                human_visibility_scene=HumanVisibilityScene(
                    faces, vertices, name="rotated-reflection"
                ),
            )
        )
    for order in ("human_env", "env_human"):
        np.testing.assert_allclose(
            getattr(values[1], order + "_coefficients"),
            getattr(values[0], order + "_coefficients"),
            rtol=1e-8,
            atol=1e-15,
        )
