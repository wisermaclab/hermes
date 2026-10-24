# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Independent physical invariants and delay-resolved hybrid regressions."""

from types import SimpleNamespace
from scipy.special import roots_legendre

from sensing_test_helpers import *  # noqa: F401,F403
import mmWaveRadar.simulation.physical_optics as po
import mmWaveRadar.simulation.env_human_coupling as ec
from mmWaveRadar.simulation.channel_gain import apply_path_hardware_gain
from mmWaveRadar.simulation.po_wideband import PODelayResponse
from mmWaveRadar.radar.ti import SimpleCosinePattern, TIDigitizedPattern


def _setup():
    human = _front_facing_triangle(2.0, scale=0.12, z_shift=-0.04).astype(float)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    floor = np.array([[-5.0, -5.0, -1.0], [5.0, -5.0, -1.0], [0.0, 5.0, -1.0]])
    pec = PhysicalOpticsMaterial(1.0, 0.0, model="pec")
    bank = EnvironmentReflectorBank(
        vertices=floor[None],
        centroids=floor.mean(0, keepdims=True),
        normals=np.array([[0.0, 0.0, 1.0]]),
        areas=np.array([50.0]),
        materials=(pec,),
        object_names=("floor",),
        face_indices=np.array([0]),
    )
    return human, faces, pec, bank


def _config(mode="parent_face_quadrature"):
    return POConfig(
        po_integration_mode=mode,
        po_quadrature_phase_span_scale_rad=np.pi / 8,
        po_quadrature_max_refinement_depth=4,
        po_quadrature_max_subfaces_per_parent=256,
    )


def _radar():
    radar = _phase_one_radar(num_chirps=2)
    return replace(
        radar, fmcw=replace(radar.fmcw, slope=68e12, num_adc_samples=225, sampling_frequency=4.5e6)
    )


@pytest.mark.parametrize(
    "mode",
    [
        "face_centroid",
        "parent_face_quadrature",
        "parent_face_adaptive_analytic",
        "parent_face_far_field_analytic",
    ],
)
@pytest.mark.parametrize("axis,reflection_sign", [([0.0, 1.0, 0.0], -1.0), ([0.0, 0.0, 1.0], 1.0)])
@pytest.mark.parametrize("skin", [False, True])
def test_coupling_absolute_field_matches_pec_image_source(mode, axis, reflection_sign, skin):
    # A PEC floor has an exact image source: a tangential polarization axis
    # changes sign and a normal axis retains sign. The reference uses ordinary
    # human PO at the mirrored endpoint, without the coupling polarization
    # callback or its reflection/spreading/gain assembly.
    human, faces, pec, bank = _setup()
    material = human_skin_material("absolute-image-skin") if skin else pec
    config = _config(mode)
    real, image = np.zeros((1, 3)), np.array([[0.0, 0.0, -2.0]])
    coherent_scale = 0.8
    bank = replace(bank, specular_amplitude_scales=np.array([coherent_scale]))
    coupled = compute_env_human_coupling_channels(
        human,
        faces,
        real,
        real,
        60e9,
        material,
        bank,
        human_visibility_scene=HumanVisibilityScene(faces, human, name="absolute-image-human"),
        po_config=config,
        polarization_axis=np.asarray(axis),
    )
    for order, tx, rx in (("human_env", real, image), ("env_human", image, real)):
        reference = po_channel(
            human,
            faces,
            tx,
            rx,
            60e9,
            material,
            visibility="none",
            po_integration_mode=mode,
            po_quadrature_phase_span_scale_rad=config.effective_po_phase_span_scale_rad,
            po_curvature_error_rad=config.po_curvature_error_rad,
            po_quadrature_max_refinement_depth=config.po_quadrature_max_refinement_depth,
            po_quadrature_max_subfaces_per_parent=config.po_quadrature_max_subfaces_per_parent,
            polarization_axis=np.asarray(axis),
        )
        assert np.max(abs(reference.coefficients)) > 1e-12
        np.testing.assert_allclose(
            getattr(coupled, order + "_coefficients"),
            reflection_sign * coherent_scale * reference.coefficients,
            rtol=2e-12,
            atol=1e-20,
        )
        np.testing.assert_allclose(
            getattr(coupled, order + "_delays_s"),
            reference.delays_s,
            rtol=1e-14,
        )


@pytest.mark.parametrize("order", ["human_env", "env_human"])
@pytest.mark.parametrize("channel_order", ["tx_major", "rx_major", "permuted"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("support", ["seed", "union"])
def test_active_set_projection_checks_each_physical_endpoint_once(
    monkeypatch,
    order,
    channel_order,
    precision,
    support,
):
    tx = np.array([[-0.03, 0.0, 0.0], [0.01, 0.002, 0.0], [0.02, -0.002, 0.0]])
    rx = np.array([[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.002, 0.0], [0.02, -0.002, 0.0]])
    hardware = RadarHardware.from_positions(
        tx, rx, virtual_channel_order="rx_major" if channel_order == "rx_major" else "tx_major"
    )
    vc_tx, vc_rx = hardware.virtual_tx_indices(), hardware.virtual_rx_indices()
    if channel_order == "permuted":
        vc_tx, vc_rx = np.roll(vc_tx, 5), np.roll(vc_rx, 5)
    square = np.array([[0.5, -0.5, -1.0], [1.0, -0.5, -1.0], [1.0, 0.5, -1.0], [0.5, 0.5, -1.0]])
    floor = _setup()[3].vertices[0] - [0.0, 0.0, 1.0]
    triangles = square[np.array([[0, 1, 2], [0, 2, 3]])]
    pec = PhysicalOpticsMaterial(1.0, 0.0, model="pec")
    bank = EnvironmentReflectorBank(
        vertices=np.stack([triangles[0], floor]),
        centroids=np.stack([triangles[0].mean(0), floor.mean(0)]),
        normals=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        areas=np.ones(2),
        materials=(pec, pec),
        object_names=("finite", "floor"),
        face_indices=np.zeros(2, int),
        support_triangles=(triangles, floor[None]) if support == "union" else None,
    )
    centers = np.array([[2.0, 0.0, 0.0], [2.0, 0.4, 0.1], [8.0, 0.0, 0.0], [1.4, 0.8, 0.0]])
    active_sets = (
        ec._CouplingActiveSet(-1, np.array([0])),
        ec._CouplingActiveSet(0, np.array([2, 0, 1, 3])),
        ec._CouplingActiveSet(1, np.array([], int)),
        ec._CouplingActiveSet(1, np.array([3, 0])),
        ec._CouplingActiveSet(2, np.array([0])),
    )
    dtype = np.complex64 if precision == "float32" else np.complex128
    center_coefficients = (np.arange(1, 9) + 0.3j * np.arange(8, 0, -1)).astype(dtype)
    expected = np.zeros((12, 8), dtype=dtype)
    expected_delays = np.zeros((12, 8))
    # Independent per-VC reference using full-array phase centers. Endpoint
    # positions straddle the finite reflector edge, so masks differ by antenna.
    for active_set in active_sets:
        reflector, active = active_set.reflector_index, active_set.active_indices
        if reflector not in (0, 1) or not active.size:
            continue
        points = centers[active]
        point, normal = bank.centroids[reflector], ec._normalize(bank.normals[reflector])
        surface = ec._reflector_support(bank, reflector)
        center_tx, center_rx = tx.mean(0), rx.mean(0)
        if order == "human_env":
            center_rx = ec._mirror_point(center_rx, point, normal)
        else:
            center_tx = ec._mirror_point(center_tx, point, normal)
        center_tx_dist, center_rx_dist = np.linalg.norm(
            points - center_tx, axis=1
        ), np.linalg.norm(points - center_rx, axis=1)
        center_lengths = center_tx_dist + center_rx_dist
        center_spreads = np.maximum(center_tx_dist * center_rx_dist, 1e-18)
        out = reflector * 4 + active
        for channel, (t, r) in enumerate(zip(vc_tx, vc_rx)):
            source, receiver = tx[t], rx[r]
            if order == "human_env":
                receiver = ec._mirror_point(receiver, point, normal)
                starts, ends = points, np.broadcast_to(receiver, points.shape)
            else:
                source = ec._mirror_point(source, point, normal)
                starts, ends = np.broadcast_to(source, points.shape), points
            _, finite = ec._reflection_points_on_surface(starts, ends, point, normal, surface)
            tx_dist, rx_dist = np.linalg.norm(points - source, axis=1), np.linalg.norm(
                points - receiver, axis=1
            )
            lengths = tx_dist + rx_dist
            expected[channel, out] = np.where(
                finite,
                center_coefficients[out]
                * np.exp(-2j * np.pi * 60e9 / c * (lengths - center_lengths))
                * (center_spreads / np.maximum(tx_dist * rx_dist, 1e-18)),
                0.0,
            ).astype(dtype)
            expected_delays[channel, out] = lengths / c
    work = []
    original = ec._reflection_points_on_surface

    def record(starts, ends, *args):
        work.append(len(starts))
        return original(starts, ends, *args)

    monkeypatch.setattr(ec, "_reflection_points_on_surface", record)
    actual, delays = ec._coupling_for_pairs_active_sets(
        centers,
        np.tile([-1.0, 0.0, 0.0], (4, 1)),
        np.ones(4),
        np.arange(4),
        tx,
        rx,
        60e9,
        pec,
        bank,
        order=order,
        reflection_scale=1.0,
        active_sets=active_sets,
        center_coefficients=center_coefficients,
        virtual_channel_tx_indices=vc_tx,
        virtual_channel_rx_indices=vc_rx,
        precision=precision,
    )
    np.testing.assert_allclose(
        actual, expected, rtol=2e-7 if precision == "float32" else 2e-12, atol=1e-14
    )
    np.testing.assert_array_equal(delays, expected_delays)
    assert np.any(actual[:, 0] == 0.0) and np.any(actual[:, 0] != 0.0)
    endpoints = 4 if order == "human_env" else 3
    assert work == [4 * endpoints, 2 * endpoints]


@pytest.mark.parametrize("order", ["human_env", "env_human"])
@pytest.mark.parametrize("empty", [False, True])
def test_active_set_projection_keeps_empty_and_fully_rejected_paths_zero(order, empty):
    _, _, material, bank = _setup()
    centers = np.array([[100.0, 100.0, 0.0], [-100.0, -100.0, 0.0]])
    actual, delays = ec._coupling_for_pairs_active_sets(
        centers,
        np.tile([-1.0, 0.0, 0.0], (2, 1)),
        np.ones(2),
        np.arange(2),
        np.zeros((3, 3)),
        np.zeros((4, 3)),
        60e9,
        material,
        bank,
        order=order,
        reflection_scale=1.0,
        active_sets=() if empty else (ec._CouplingActiveSet(0, np.arange(2)),),
        center_coefficients=np.ones(2, complex),
        virtual_channel_tx_indices=np.repeat(np.arange(3), 4),
        virtual_channel_rx_indices=np.tile(np.arange(4), 3),
        precision="float64",
    )
    np.testing.assert_array_equal(actual, np.zeros((12, 2), complex))
    np.testing.assert_array_equal(delays, np.zeros((12, 2)))


@pytest.mark.parametrize("order", ["human_env", "env_human"])
@pytest.mark.parametrize("channel_order", ["tx_major", "rx_major", "permuted"])
@pytest.mark.parametrize("pattern_kind", ["none", "cosine", "digitized"])
def test_coupling_hardware_gain_reuses_physical_antennas_without_changing_channels(
    monkeypatch,
    order,
    channel_order,
    pattern_kind,
):
    _, _, _, bank = _setup()
    hardware = RadarHardware.from_positions(
        [[0.0, 0.0, 0.0], [0.0, 0.002, 0.001], [0.0, -0.001, 0.003]],
        [[0.0, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.002], [0.0, 0.002, 0.001]],
        virtual_channel_order="rx_major" if channel_order == "rx_major" else "tx_major",
    )
    if channel_order == "permuted":
        hardware = replace(
            hardware,
            virtual_channel_tx_indices=np.roll(hardware.virtual_tx_indices(), 5),
            virtual_channel_rx_indices=np.roll(hardware.virtual_rx_indices(), 5),
        )
    pattern = None
    if pattern_kind == "cosine":
        pattern = SimpleCosinePattern(12)
    elif pattern_kind == "digitized":
        # Distinct curves must retain virtual-channel identity after mapping.
        curves = -np.arange(1, 13)[:, None] * np.array([[2.0, 0.0, 1.0]])
        pattern = TIDigitizedPattern(
            "synthetic",
            tuple(map(str, range(12))),
            np.array([60e9]),
            np.array([-180.0, 0.0, 180.0]),
            np.array([-90.0, 0.0, 90.0]),
            curves[None],
            (0.7 * curves)[None],
            {},
        )
    hardware = replace(
        hardware,
        antenna_pattern=pattern,
        scalar_antenna_gain_dbi=6.0,
        virtual_channel_phase_signs=np.where(np.arange(12) % 2, -1.0, 1.0),
    )
    radar = replace(
        _radar(),
        hardware=hardware,
        fmcw=replace(_radar().fmcw, num_tx=3),
        position=(0.1, -0.2, 0.3),
        orientation=(0.1, -0.15, 0.2),
    )
    centers = np.array([[2.0, 0.0, 0.0], [3.0, 1.0, 0.2], [100.0, 100.0, 0.3]])
    tx, rx = radar.world_tx_positions(), radar.world_rx_positions()
    vc_tx, vc_rx = hardware.virtual_tx_indices(), hardware.virtual_rx_indices()
    phase = hardware.channel_gain(include_pattern=False, include_antenna_gain=False)
    expected = np.ones((12, 3), complex) * (1.0 + 0.3j)
    # Independent channel-by-channel reference, including per-channel curves.
    for channel, (t, r) in enumerate(zip(vc_tx, vc_rx)):
        endpoint = rx[r] if order == "human_env" else tx[t]
        image = ec._mirror_points(endpoint[None], bank.centroids[0], bank.normals[0])
        starts, ends = (
            (centers, np.broadcast_to(image, centers.shape))
            if order == "human_env"
            else (np.broadcast_to(image, centers.shape), centers)
        )
        points, finite = ec._reflection_points_on_surface(
            starts, ends, bank.centroids[0], bank.normals[0], bank.vertices
        )
        points = np.where(finite[:, None], points, centers)
        gains = []
        for endpoint_points, local_element in (
            (centers if order == "human_env" else points, hardware.tx_positions[t]),
            (points if order == "human_env" else centers, hardware.rx_positions[r]),
        ):
            local = radar.world_to_local_points(endpoint_points) - local_element
            gains.append(
                hardware.channel_gain(
                    azimuth_deg=np.degrees(np.arctan2(local[:, 1], local[:, 0])),
                    elevation_deg=np.degrees(
                        np.arctan2(local[:, 2], np.hypot(local[:, 0], local[:, 1]))
                    ),
                    frequency_hz=radar.fmcw.carrier_frequency,
                    include_phase=False,
                )[channel]
            )
        expected[channel] *= phase[channel] * np.sqrt(gains[0] * gains[1]) * finite
    work = []
    original = ec._reflection_points_on_surface

    def record(starts, ends, *args):
        work.append(len(starts))
        return original(starts, ends, *args)

    monkeypatch.setattr(ec, "_reflection_points_on_surface", record)
    pattern_work = []
    if pattern_kind == "cosine":
        original_pattern = SimpleCosinePattern.loss_db

        def record_pattern(self, **kwargs):
            pattern_work.append(np.asarray(kwargs["azimuth_deg"]).shape)
            return original_pattern(self, **kwargs)

        monkeypatch.setattr(SimpleCosinePattern, "loss_db", record_pattern)
    actual = np.full((12, 3), 1.0 + 0.3j)
    ec._apply_coupling_hardware_gain(
        actual,
        radar=radar,
        centroids=centers,
        reflectors=bank,
        active_sets=(ec._CouplingActiveSet(0, np.arange(3)),),
        order=order,
        tx_positions=tx,
        rx_positions=rx,
        vc_tx=vc_tx,
        vc_rx=vc_rx,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-15)
    assert np.any(abs(expected) > 0.0) and np.any(expected == 0.0)
    assert work == [3 * (4 if order == "human_env" else 3)]
    if pattern_kind == "cosine":
        assert pattern_work == [(3, 3), (4, 3)]


def test_large_po_plane_has_correct_absolute_specular_phase():
    # Gaussian apodization suppresses panel edges; the stationary-phase limit
    # is a specular PEC channel with coefficient -lambda/(4*pi*2R)*exp(-jk*2R).
    nodes, weights = roots_legendre(4000)
    rho, dr = (nodes + 1.0) * 1.8, weights * 1.8
    angles = np.linspace(0.0, 2 * np.pi, 32, endpoint=False)
    points = np.stack(
        [
            np.full((len(rho), len(angles)), 2.0),
            rho[:, None] * np.cos(angles),
            rho[:, None] * np.sin(angles),
        ],
        -1,
    ).reshape(-1, 3)
    areas = np.broadcast_to(
        (dr * rho * np.exp(-(rho**2) / (2 * 0.6**2)) * 2 * np.pi / 32)[:, None], (len(rho), 32)
    ).ravel()
    a, _ = po.vector_facet_face_contributions(
        np.arange(len(points)),
        points,
        np.tile([-1.0, 0.0, 0.0], (len(points), 1)),
        areas,
        np.zeros(3),
        np.zeros(3),
        60e9,
        PhysicalOpticsMaterial(1.0, 0.0, model="pec"),
    )
    expected = -c / 60e9 / (4 * np.pi * 4.0) * np.exp(-2j * np.pi * 60e9 / c * 4.0)
    assert abs(abs(a.sum() / expected) - 1.0) < 1e-4
    assert abs(np.angle(a.sum() / expected)) < 0.004


@pytest.mark.parametrize("mode", ["face_centroid", "parent_face_quadrature"])
@pytest.mark.parametrize("blocker_center", [[1.5, 0.0, -0.5], [0.5, 0.0, -0.5]])
def test_static_obstruction_blocks_both_reflected_legs_in_fresh_and_cached_solves(
    blocker_center, mode
):
    human, faces, pec, bank = _setup()
    kwargs = dict(
        vertices=human,
        faces=faces,
        tx_positions=np.zeros((1, 3)),
        rx_positions=np.zeros((1, 3)),
        frequency_hz=60e9,
        human_material=pec,
        reflectors=bank,
        radar=_radar(),
        po_config=_config(mode),
        human_visibility_scene=HumanVisibilityScene(faces, human, name="block-human"),
    )
    clear = compute_env_human_coupling_channels(**kwargs)
    assert np.all(clear.human_env_valid) and np.all(clear.env_human_valid)
    blocker = mi.load_dict(
        {
            "type": "scene",
            "blocker": {
                "type": "cube",
                "to_world": mi.ScalarTransform4f()
                .translate(blocker_center)
                .scale([0.1, 0.3, 0.1]),
            },
        }
    )
    for cached in (False, True):
        r = compute_env_human_coupling_channels(
            **kwargs,
            static_visibility_scene=blocker,
            **(
                {}
                if not cached
                else dict(
                    human_env_active_sets=clear.human_env_active_sets,
                    env_human_active_sets=clear.env_human_active_sets,
                )
            ),
        )
        assert not np.any(r.human_env_valid)
        assert not np.any(r.env_human_valid)
        assert not np.any(r.human_env_coefficients)
        assert not np.any(r.env_human_coefficients)
        if r.human_env_delay_response is not None:
            assert not np.any(r.human_env_delay_response.values)
            assert not np.any(r.env_human_delay_response.values)


def test_rt_antenna_gain_uses_arrival_and_applies_scalar_and_phase_once():
    hardware = replace(
        RadarHardware.from_positions([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
        antenna_pattern=SimpleCosinePattern(1, half_power_angle_deg=30.0),
        scalar_antenna_gain_dbi=6.0,
        virtual_channel_phase_signs=np.array([-1.0]),
    )
    values = []
    for departure, arrival in ((0.0, 60.0), (60.0, 0.0)):
        paths = SimpleNamespace(
            theta_t=np.array([np.pi / 2]),
            phi_t=np.deg2rad([departure]),
            theta_r=np.array([np.pi / 2]),
            phi_r=np.deg2rad([arrival]),
        )
        values.append(apply_path_hardware_gain(hardware, np.ones((1, 1), complex), paths=paths))
    expected = -np.sqrt(
        hardware.channel_gain(azimuth_deg=0.0, elevation_deg=0.0, include_phase=False)
        * hardware.channel_gain(azimuth_deg=60.0, elevation_deg=0.0, include_phase=False)
    )
    np.testing.assert_allclose(values[0], expected.reshape(1, 1))
    np.testing.assert_allclose(values[1], expected.reshape(1, 1))


@pytest.mark.parametrize("order", ["human_env", "env_human"])
def test_static_occlusion_is_applied_to_individual_integration_points(order):
    _, _, pec, bank = _setup()
    blocker = mi.load_dict(
        {
            "type": "scene",
            "blocker": {
                "type": "cube",
                "to_world": mi.ScalarTransform4f()
                .translate([1.5, 0.0, -0.5])
                .scale([0.1, 0.2, 0.1]),
            },
        }
    )
    real, image = np.zeros(3), np.array([0.0, 0.0, -2.0])
    callback = ec._image_polarization_callback(
        tx_position=real if order == "human_env" else image,
        rx_position=image if order == "human_env" else real,
        reflector_point=bank.centroids[0],
        reflector_normal=bank.normals[0],
        support_triangles=bank.vertices,
        material=pec,
        frequency_hz=60e9,
        order=order,
        scale=1.0,
        polarization_axis=np.array([0.0, 0.0, 1.0]),
        static_visibility_scene=blocker,
    )
    tx, rx = callback(np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]))
    field = rx if order == "human_env" else tx
    np.testing.assert_allclose(np.linalg.norm(field, axis=1), [0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize(
    "mode", ["face_centroid", "parent_face_quadrature", "parent_face_far_field_analytic"]
)
@pytest.mark.parametrize("axis", [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_skin_coupling_is_complex_reciprocal_for_both_interaction_orders(
    mode, axis, backend, monkeypatch
):
    if backend == "torch":
        pytest.importorskip("torch")
        monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")
    human, faces, _, bank = _setup()
    bank = replace(bank, materials=(PhysicalOpticsMaterial(4.0, 0.2, model="interface"),))
    kwargs = dict(
        vertices=human,
        faces=faces,
        frequency_hz=60e9,
        human_material=human_skin_material("reciprocal-skin"),
        reflectors=bank,
        human_visibility_scene=HumanVisibilityScene(faces, human, name="reciprocal-human"),
        po_config=_config(mode),
        polarization_axis=np.asarray(axis) / np.linalg.norm(axis),
        backend=backend,
        precision="float64",
    )
    tx, rx = np.array([[0.1, -0.2, 0.2]]), np.array([[-0.1, 0.3, -0.1]])
    forward = compute_env_human_coupling_channels(**kwargs, tx_positions=tx, rx_positions=rx)
    reverse = compute_env_human_coupling_channels(**kwargs, tx_positions=rx, rx_positions=tx)
    assert np.any(abs(forward.human_env_coefficients) > 0.0)
    np.testing.assert_allclose(
        forward.human_env_coefficients, reverse.env_human_coefficients, rtol=1e-10, atol=1e-20
    )


def test_quadrature_wideband_adc_matches_individual_subface_delays():
    vertices, faces, pec, _ = _setup()
    radar = _radar()
    cfg = _config()
    bandwidth = radar.fmcw.slope * (radar.fmcw.num_adc_samples - 1) / radar.fmcw.sampling_frequency
    result = po_channel(
        vertices=vertices,
        faces=faces,
        tx_positions=np.zeros((1, 3)),
        rx_positions=np.zeros((1, 3)),
        frequency_hz=60e9,
        material=pec,
        visibility="none",
        po_integration_mode=cfg.po_integration_mode,
        po_quadrature_phase_span_scale_rad=np.pi / 8,
        po_quadrature_max_refinement_depth=4,
        po_quadrature_max_subfaces_per_parent=256,
        max_frequency_offset_hz=bandwidth,
    )
    points, normals, areas, _ = po.parent_face_quadrature_geometry(
        vertices=vertices,
        faces=faces,
        selected_parent_indices=np.array([0]),
        tx_positions=np.zeros((1, 3)),
        rx_positions=np.zeros((1, 3)),
        frequency_hz=60e9,
        phase_span_scale_rad=np.pi / 8,
        max_refinement_depth=4,
        max_subfaces_per_parent=256,
    )
    coeff, tau = po.vector_facet_face_contributions(
        np.arange(len(points)), points, normals, areas, np.zeros(3), np.zeros(3), 60e9, pec
    )
    sim = MmWaveRadarSimulator(compute_backend="numpy", compute_precision="float64")
    expected = sim.synthesize_adc(coeff[None], tau, np.ones(len(points), bool), radar.fmcw)
    actual = sim.synthesize_adc(
        result.coefficients,
        result.delays_s,
        np.array([True]),
        radar.fmcw,
        delay_response=result.delay_response,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-18)
    compressed = sim.synthesize_adc(
        result.coefficients, result.delays_s, np.array([True]), radar.fmcw
    )
    assert np.linalg.norm(compressed - expected) > 0.01 * np.linalg.norm(expected)


@pytest.mark.parametrize("mode", ["parent_face_quadrature", "parent_face_far_field_analytic"])
@pytest.mark.parametrize("displacement,threshold", [(0.0, 5.0), (0.01, 0.0)])
def test_wideband_response_survives_incremental_motion_and_hardware(mode, displacement, threshold):
    vertices, faces, pec, _ = _setup()
    radar = _radar()
    sequence = MeshSequence(
        vertices=np.stack([vertices, vertices + [displacement, 0.0, 0.0]]),
        faces=faces,
        times=np.array([0.0, 1.0]),
    )
    target = MeshTarget("wideband-human", sequence, material=human_skin_material("wideband-skin"))
    cubes = []
    for incremental in (False, True):
        cfg = replace(
            _config(mode),
            po_refresh_frames=int(incremental),
            po_refresh_on_phase_error_rad=threshold,
            po_visibility_refresh_frames=1,
            precomputed_raw_fractional_visibility=np.ones(1),
        )
        sim = MmWaveRadarSimulator(
            solver="po",
            po_config=cfg,
            compute_backend="numpy",
            compute_precision="float64",
            adc_compute_precision="float64",
        )
        cubes.append(sim.run(load_scene(), radar, [target], num_frames=1))
    np.testing.assert_allclose(cubes[0].adc, cubes[1].adc, rtol=1e-10, atol=1e-18)


def test_delay_response_handles_carrier_cancellation_and_multiple_channels():
    vertices, faces, _, _ = _setup()
    radar = _radar()
    bandwidth = radar.fmcw.slope * (radar.fmcw.num_adc_samples - 1) / radar.fmcw.sampling_frequency
    response = PODelayResponse.for_mesh(vertices, faces, 1, bandwidth)
    response.reference_delays_s[:] = 15e-9
    sample_tau = np.array([14.95e-9, 15.05e-9])
    sample_a = np.array([1.0 + 2j, -1.0 - 2j])
    response.accumulate(sample_a, sample_tau, np.zeros(2, int))
    gains = np.array([[1.0], [-2j]])
    response = response.with_channel_gains(gains)
    sim = MmWaveRadarSimulator(compute_backend="numpy", adc_compute_precision="float64")
    expected = sim.synthesize_adc(gains * sample_a, sample_tau, np.ones(2, bool), radar.fmcw)
    actual = response.synthesize_adc(np.array([15e-9]), np.array([True]), radar.fmcw)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    assert response.valid_paths[0]


@pytest.mark.parametrize("mode", ["parent_face_quadrature", "parent_face_far_field_analytic"])
@pytest.mark.parametrize("order", ["human_env", "env_human"])
def test_image_wideband_adc_matches_explicit_frequency_integration(mode, order):
    vertices, faces, pec, bank = _setup()
    radar = _radar()
    radar = replace(
        radar,
        hardware=replace(
            radar.hardware,
            scalar_antenna_gain_dbi=6.0,
            virtual_channel_phase_signs=np.array([-1.0]),
        ),
    )
    result = compute_env_human_coupling_channels(
        vertices,
        faces,
        radar.world_tx_positions(),
        radar.world_rx_positions(),
        60e9,
        pec,
        bank,
        radar=radar,
        po_config=_config(mode),
        human_visibility_scene=HumanVisibilityScene(faces, vertices, name="image-wideband"),
    )
    image, real = np.array([0.0, 0.0, -2.0]), np.zeros(3)
    tx, rx = (real, image) if order == "human_env" else (image, real)
    geometry = dict(
        vertices=vertices,
        faces=faces,
        selected_parent_indices=np.array([0]),
        tx_positions=tx[None],
        rx_positions=rx[None],
        frequency_hz=60e9,
        max_refinement_depth=4,
        max_subfaces_per_parent=256,
    )
    if mode == "parent_face_quadrature":
        points, normals, areas, _ = po.parent_face_quadrature_geometry(
            **geometry, phase_span_scale_rad=np.pi / 8
        )
    else:
        points, normals, areas, _, triangles, _, _ = po.parent_face_far_field_analytic_geometry(
            **geometry
        )
    callback = ec._image_polarization_callback(
        tx_position=tx,
        rx_position=rx,
        reflector_normal=bank.normals[0],
        material=pec,
        frequency_hz=60e9,
        order=order,
        scale=1.0,
        polarization_axis=np.array([0.0, 0.0, 1.0]),
    )
    coeff, tau = po.vector_facet_face_contributions(
        np.arange(len(points)),
        points,
        normals,
        areas,
        tx,
        rx,
        60e9,
        pec,
        polarization_callback=callback,
    )
    offsets = (
        radar.fmcw.slope * np.arange(radar.fmcw.num_adc_samples) / radar.fmcw.sampling_frequency
    )
    expected = np.zeros(len(offsets), complex)
    for i, frequency in enumerate(offsets):
        integral = 1.0
        if mode == "parent_face_far_field_analytic":
            tx_hat = (tx - points) / np.linalg.norm(tx - points, axis=1)[:, None]
            rx_hat = (rx - points) / np.linalg.norm(rx - points, axis=1)[:, None]
            q = -2 * np.pi * (60e9 + frequency) / c * (tx_hat + rx_hat)[None]
            integral = po._triangle_linear_phase_integrals(triangles, points, q).ravel() / areas
        expected[i] = np.sum((coeff * integral).conjugate() * np.exp(2j * np.pi * frequency * tau))
    expected *= -(10 ** (6.0 / 20.0))
    sim = MmWaveRadarSimulator(compute_backend="numpy", adc_compute_precision="float64")
    actual = sim.synthesize_adc(
        getattr(result, order + "_coefficients"),
        getattr(result, order + "_delays_s"),
        getattr(result, order + "_valid"),
        radar.fmcw,
        delay_response=getattr(result, order + "_delay_response"),
    )
    np.testing.assert_allclose(actual[:, 0], expected, rtol=1e-10, atol=1e-18)
    cached = compute_env_human_coupling_channels(
        vertices,
        faces,
        radar.world_tx_positions(),
        radar.world_rx_positions(),
        60e9,
        pec,
        bank,
        radar=radar,
        po_config=_config(mode),
        human_env_active_sets=result.human_env_active_sets,
        env_human_active_sets=result.env_human_active_sets,
        human_visibility_scene=HumanVisibilityScene(faces, vertices, name="image-wideband-cached"),
    )
    np.testing.assert_allclose(
        getattr(cached, order + "_delay_response").values,
        getattr(result, order + "_delay_response").values,
        rtol=1e-12,
        atol=1e-20,
    )


def test_hybrid_rejects_calibrating_direct_po_against_coupled_rt_reference():
    from mmWaveRadar.simulation.hybrid_pipeline import run_calibrated_hybrid_po

    target = MeshTarget(
        "reference-human",
        _single_face_mesh_sequence(x0=2.0),
        material=human_skin_material("reference-skin"),
    )
    with pytest.raises(ValueError, match="single_human_only"):
        run_calibrated_hybrid_po(
            scene=load_scene(),
            radar=_radar(),
            target=target,
            num_frames=1,
            hybrid_coupling=False,
            progress=False,
            hybrid_calibration=POCalibrationConfig(
                mode="rt_sequence", reference_component="human_touch"
            ),
        )


def test_wideband_image_projection_preserves_per_channel_phase_delay_and_gain():
    vertices, faces, pec, bank = _setup()
    radar = _radar()
    hardware = replace(
        RadarHardware.from_positions(
            [[0.0, -0.001, 0.0], [0.0, 0.001, 0.0]], [[0.0, -0.002, 0.0], [0.0, 0.002, 0.0]]
        ),
        scalar_antenna_gain_dbi=6.0,
        virtual_channel_phase_signs=np.array([1.0, -1.0, -1.0, 1.0]),
    )
    array_radar = replace(radar, hardware=hardware, fmcw=replace(radar.fmcw, num_tx=2))
    common = dict(
        vertices=vertices,
        faces=faces,
        frequency_hz=60e9,
        human_material=pec,
        reflectors=bank,
        po_config=_config(),
        human_visibility_scene=HumanVisibilityScene(faces, vertices, name="wideband-array"),
    )
    center = compute_env_human_coupling_channels(
        **common,
        radar=radar,
        tx_positions=radar.world_tx_positions(),
        rx_positions=radar.world_rx_positions(),
    )
    array = compute_env_human_coupling_channels(
        **common,
        radar=array_radar,
        tx_positions=array_radar.world_tx_positions(),
        rx_positions=array_radar.world_rx_positions(),
    )
    offsets = radar.fmcw.slope * np.arange(225) / radar.fmcw.sampling_frequency
    point = vertices.mean(0)
    center_spread = np.linalg.norm(point) * np.linalg.norm(point - [0.0, 0.0, -2.0])
    center_length = np.linalg.norm(point) + np.linalg.norm(point - [0.0, 0.0, -2.0])
    for order in ("human_env", "env_human"):
        base_adc = getattr(center, order + "_delay_response").synthesize_adc(
            getattr(center, order + "_delays_s"), getattr(center, order + "_valid"), radar.fmcw
        )[:, 0]
        actual = getattr(array, order + "_delay_response").synthesize_adc(
            getattr(array, order + "_delays_s"), getattr(array, order + "_valid"), radar.fmcw
        )
        for ch, (tx_i, rx_i) in enumerate(
            zip(hardware.virtual_tx_indices(), hardware.virtual_rx_indices())
        ):
            tx = hardware.tx_positions[tx_i].copy()
            rx = hardware.rx_positions[rx_i].copy()
            if order == "human_env":
                rx[2] = -2.0
            else:
                tx[2] = -2.0
            rt, rr = np.linalg.norm(point - tx), np.linalg.norm(point - rx)
            delta = rt + rr - center_length
            expected = (
                base_adc
                * center_spread
                / (rt * rr)
                * np.exp(2j * np.pi * (60e9 + offsets) * delta / c)
                * 10 ** (6.0 / 20.0)
                * hardware.virtual_channel_phase_signs[ch]
            )
            np.testing.assert_allclose(actual[:, ch], expected, rtol=1e-9, atol=1e-18)
