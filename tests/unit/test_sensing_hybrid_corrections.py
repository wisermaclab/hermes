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
from mmWaveRadar.radar.ti import SimpleCosinePattern


def _setup():
    human = _front_facing_triangle(2., scale=.12, z_shift=-.04).astype(float)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    floor = np.array([[-5., -5., -1.], [5., -5., -1.], [0., 5., -1.]])
    pec = PhysicalOpticsMaterial(1., 0., model="pec")
    bank = EnvironmentReflectorBank(
        vertices=floor[None], centroids=floor.mean(0, keepdims=True),
        normals=np.array([[0., 0., 1.]]), areas=np.array([50.]),
        materials=(pec,), object_names=("floor",), face_indices=np.array([0]),
    )
    return human, faces, pec, bank


def _config(mode="parent_face_quadrature"):
    return HumanPOMobilityConfig(
        po_integration_mode=mode, visibility_fade_chirps=1,
        po_quadrature_phase_span_scale_rad=np.pi/8,
        po_quadrature_max_refinement_depth=4, po_quadrature_max_subfaces_per_parent=256,
    )


def _radar():
    radar = _phase_one_radar(num_chirps=2)
    return replace(radar, fmcw=replace(radar.fmcw, slope=68e12,
        num_adc_samples=225, sampling_frequency=4.5e6))


def test_large_po_plane_has_correct_absolute_specular_phase():
    # Gaussian apodization suppresses panel edges; the stationary-phase limit
    # is a specular PEC channel with coefficient -lambda/(4*pi*2R)*exp(-jk*2R).
    nodes, weights = roots_legendre(4000)
    rho, dr = (nodes + 1.) * 1.8, weights * 1.8
    angles = np.linspace(0., 2*np.pi, 32, endpoint=False)
    points = np.stack([np.full((len(rho), len(angles)), 2.),
        rho[:, None]*np.cos(angles), rho[:, None]*np.sin(angles)], -1).reshape(-1, 3)
    areas = np.broadcast_to((dr*rho*np.exp(-rho**2/(2*.6**2))*2*np.pi/32)[:, None],
                            (len(rho), 32)).ravel()
    a, _ = po.vector_facet_face_contributions(
        np.arange(len(points)), points, np.tile([-1., 0., 0.], (len(points), 1)), areas,
        np.zeros(3), np.zeros(3), 60e9, PhysicalOpticsMaterial(1., 0., model="pec"),
    )
    expected = -c/60e9/(4*np.pi*4.) * np.exp(-2j*np.pi*60e9/c*4.)
    assert abs(abs(a.sum()/expected) - 1.) < 1e-4
    assert abs(np.angle(a.sum()/expected)) < .004


@pytest.mark.parametrize("mode", ["face_centroid", "parent_face_quadrature"])
@pytest.mark.parametrize("blocker_center", [[1.5, 0., -.5], [.5, 0., -.5]])
def test_static_obstruction_blocks_both_reflected_legs_in_fresh_and_cached_solves(blocker_center, mode):
    human, faces, pec, bank = _setup()
    kwargs = dict(vertices=human, faces=faces, tx_positions=np.zeros((1, 3)),
        rx_positions=np.zeros((1, 3)), frequency_hz=60e9, human_material=pec,
        reflectors=bank, radar=_radar(), po_config=_config(mode),
        human_visibility_scene=HumanVisibilityScene(faces, human, name="block-human"))
    clear = compute_env_human_coupling_channels(**kwargs)
    assert np.all(clear.human_env_valid) and np.all(clear.env_human_valid)
    blocker = mi.load_dict({"type": "scene", "blocker": {"type": "cube",
        "to_world": mi.ScalarTransform4f().translate(blocker_center).scale([.1, .3, .1])}})
    for cached in (False, True):
        r = compute_env_human_coupling_channels(**kwargs, static_visibility_scene=blocker,
            **({} if not cached else dict(human_env_active_sets=clear.human_env_active_sets,
                                         env_human_active_sets=clear.env_human_active_sets)))
        assert not np.any(r.human_env_valid)
        assert not np.any(r.env_human_valid)
        assert not np.any(r.human_env_coefficients)
        assert not np.any(r.env_human_coefficients)
        if r.human_env_delay_response is not None:
            assert not np.any(r.human_env_delay_response.values)
            assert not np.any(r.env_human_delay_response.values)


def test_rt_antenna_gain_uses_arrival_and_applies_scalar_and_phase_once():
    hardware = replace(RadarHardware.from_positions([[0., 0., 0.]], [[0., 0., 0.]]),
        antenna_pattern=SimpleCosinePattern(1, half_power_angle_deg=30.),
        scalar_antenna_gain_dbi=6., virtual_channel_phase_signs=np.array([-1.]))
    values = []
    for departure, arrival in ((0., 60.), (60., 0.)):
        paths = SimpleNamespace(theta_t=np.array([np.pi/2]), phi_t=np.deg2rad([departure]),
            theta_r=np.array([np.pi/2]), phi_r=np.deg2rad([arrival]))
        values.append(apply_path_hardware_gain(hardware, np.ones((1, 1), complex), paths=paths))
    expected = -np.sqrt(hardware.channel_gain(azimuth_deg=0., elevation_deg=0., include_phase=False)
                        * hardware.channel_gain(azimuth_deg=60., elevation_deg=0., include_phase=False))
    np.testing.assert_allclose(values[0], expected.reshape(1, 1))
    np.testing.assert_allclose(values[1], expected.reshape(1, 1))


@pytest.mark.parametrize("order", ["human_env", "env_human"])
def test_static_occlusion_is_applied_to_individual_integration_points(order):
    _, _, pec, bank = _setup()
    blocker = mi.load_dict({"type": "scene", "blocker": {"type": "cube",
        "to_world": mi.ScalarTransform4f().translate([1.5, 0., -.5]).scale([.1, .2, .1])}})
    real, image = np.zeros(3), np.array([0., 0., -2.])
    callback = ec._image_polarization_callback(
        tx_position=real if order == "human_env" else image,
        rx_position=image if order == "human_env" else real,
        reflector_point=bank.centroids[0], reflector_normal=bank.normals[0],
        support_triangles=bank.vertices, material=pec, frequency_hz=60e9,
        order=order, scale=1., polarization_axis=np.array([0., 0., 1.]),
        static_visibility_scene=blocker)
    tx, rx = callback(np.array([[2., 0., 0.], [2., 1., 0.]]))
    field = rx if order == "human_env" else tx
    np.testing.assert_allclose(np.linalg.norm(field, axis=1), [0., 1.], atol=1e-12)


@pytest.mark.parametrize("mode", ["face_centroid", "parent_face_quadrature", "parent_face_far_field_analytic"])
@pytest.mark.parametrize("axis", [[0., 0., 1.], [0., 1., 1.]])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_skin_coupling_is_complex_reciprocal_for_both_interaction_orders(mode, axis, backend, monkeypatch):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(po, "_use_torch_backend", lambda backend, precision: backend == "torch")
        monkeypatch.setattr(po, "_torch_device_and_dtypes", lambda precision: (
            torch.device("cpu"), torch.float64, torch.complex128, np.float64, np.complex128))
    human, faces, _, bank = _setup()
    bank = replace(bank, materials=(PhysicalOpticsMaterial(4., .2, model="interface"),))
    kwargs = dict(vertices=human, faces=faces, frequency_hz=60e9,
        human_material=human_skin_material("reciprocal-skin"), reflectors=bank,
        human_visibility_scene=HumanVisibilityScene(faces, human, name="reciprocal-human"),
        po_config=_config(mode), polarization_axis=np.asarray(axis)/np.linalg.norm(axis),
        backend=backend, precision="float64")
    tx, rx = np.array([[.1, -.2, .2]]), np.array([[-.1, .3, -.1]])
    forward = compute_env_human_coupling_channels(**kwargs, tx_positions=tx, rx_positions=rx)
    reverse = compute_env_human_coupling_channels(**kwargs, tx_positions=rx, rx_positions=tx)
    assert np.any(abs(forward.human_env_coefficients) > 0.)
    np.testing.assert_allclose(forward.human_env_coefficients, reverse.env_human_coefficients,
                               rtol=1e-10, atol=1e-20)


def test_quadrature_wideband_adc_matches_individual_subface_delays():
    vertices, faces, pec, _ = _setup()
    radar = _radar(); cfg = _config()
    bandwidth = radar.fmcw.slope*(radar.fmcw.num_adc_samples-1)/radar.fmcw.sampling_frequency
    result = po_channel(vertices=vertices, faces=faces, tx_positions=np.zeros((1, 3)),
        rx_positions=np.zeros((1, 3)), frequency_hz=60e9, material=pec, visibility="none",
        po_integration_mode=cfg.po_integration_mode, po_quadrature_phase_span_scale_rad=np.pi/8,
        po_quadrature_max_refinement_depth=4, po_quadrature_max_subfaces_per_parent=256,
        max_frequency_offset_hz=bandwidth)
    points, normals, areas, _ = po.parent_face_quadrature_geometry(
        vertices=vertices, faces=faces, selected_parent_indices=np.array([0]),
        tx_positions=np.zeros((1, 3)), rx_positions=np.zeros((1, 3)), frequency_hz=60e9,
        phase_span_scale_rad=np.pi/8, max_refinement_depth=4, max_subfaces_per_parent=256)
    coeff, tau = po.vector_facet_face_contributions(np.arange(len(points)), points, normals, areas,
        np.zeros(3), np.zeros(3), 60e9, pec)
    sim = MmWaveRadarSimulator(compute_backend="numpy", compute_precision="float64")
    expected = sim.synthesize_adc(coeff[None], tau, np.ones(len(points), bool), radar.fmcw)
    actual = sim.synthesize_adc(result.coefficients, result.delays_s, np.array([True]),
        radar.fmcw, delay_response=result.delay_response)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-18)
    compressed = sim.synthesize_adc(result.coefficients, result.delays_s, np.array([True]), radar.fmcw)
    assert np.linalg.norm(compressed-expected) > .01*np.linalg.norm(expected)


@pytest.mark.parametrize("mode", ["parent_face_quadrature", "parent_face_far_field_analytic"])
@pytest.mark.parametrize("displacement,threshold", [(0., 5.), (.01, 0.)])
def test_wideband_response_survives_incremental_motion_and_hardware(mode, displacement, threshold):
    vertices, faces, pec, _ = _setup(); radar = _radar()
    sequence = MeshSequence(vertices=np.stack([vertices, vertices + [displacement, 0., 0.]]),
        faces=faces, times=np.array([0., 1.]))
    target = MeshTarget("wideband-human", sequence, material=human_skin_material("wideband-skin"))
    cubes = []
    for incremental in (False, True):
        cfg = replace(_config(mode), incremental_update=incremental,
            incremental_normal_threshold_deg=threshold, incremental_visibility_refresh_chirps=8,
            precomputed_raw_fractional_visibility=np.ones(1))
        sim = MmWaveRadarSimulator(mobility_mode="human_only_po", human_po_config=cfg,
            compute_backend="numpy", compute_precision="float64", adc_compute_precision="float64")
        cubes.append(sim.run(load_scene(), radar, [target], num_frames=1))
    np.testing.assert_allclose(cubes[0].adc, cubes[1].adc, rtol=1e-10, atol=1e-18)


def test_delay_response_handles_carrier_cancellation_and_multiple_channels():
    vertices, faces, _, _ = _setup(); radar = _radar()
    bandwidth = radar.fmcw.slope*(radar.fmcw.num_adc_samples-1)/radar.fmcw.sampling_frequency
    response = PODelayResponse.for_mesh(vertices, faces, 1, bandwidth)
    response.reference_delays_s[:] = 15e-9
    sample_tau = np.array([14.95e-9, 15.05e-9])
    sample_a = np.array([1.+2j, -1.-2j])
    response.accumulate(sample_a, sample_tau, np.zeros(2, int))
    gains = np.array([[1.], [-2j]])
    response = response.with_channel_gains(gains)
    sim = MmWaveRadarSimulator(compute_backend="numpy", adc_compute_precision="float64")
    expected = sim.synthesize_adc(gains*sample_a, sample_tau, np.ones(2, bool), radar.fmcw)
    actual = response.synthesize_adc(np.array([15e-9]), np.array([True]), radar.fmcw)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    assert response.valid_paths[0]


@pytest.mark.parametrize("mode", ["parent_face_quadrature", "parent_face_far_field_analytic"])
@pytest.mark.parametrize("order", ["human_env", "env_human"])
def test_image_wideband_adc_matches_explicit_frequency_integration(mode, order):
    vertices, faces, pec, bank = _setup()
    radar = _radar()
    radar = replace(radar, hardware=replace(radar.hardware,
        scalar_antenna_gain_dbi=6., virtual_channel_phase_signs=np.array([-1.])))
    result = compute_env_human_coupling_channels(
        vertices, faces, radar.world_tx_positions(), radar.world_rx_positions(), 60e9, pec, bank,
        radar=radar, po_config=_config(mode),
        human_visibility_scene=HumanVisibilityScene(faces, vertices, name="image-wideband"))
    image, real = np.array([0., 0., -2.]), np.zeros(3)
    tx, rx = (real, image) if order == "human_env" else (image, real)
    geometry = dict(vertices=vertices, faces=faces, selected_parent_indices=np.array([0]),
        tx_positions=tx[None], rx_positions=rx[None], frequency_hz=60e9,
        max_refinement_depth=4, max_subfaces_per_parent=256)
    if mode == "parent_face_quadrature":
        points, normals, areas, _ = po.parent_face_quadrature_geometry(
            **geometry, phase_span_scale_rad=np.pi/8)
    else:
        points, normals, areas, _, triangles, _, _ = po.parent_face_far_field_analytic_geometry(**geometry)
    callback = ec._image_polarization_callback(tx_position=tx, rx_position=rx,
        reflector_normal=bank.normals[0], material=pec, frequency_hz=60e9,
        order=order, scale=1., polarization_axis=np.array([0., 0., 1.]))
    coeff, tau = po.vector_facet_face_contributions(
        np.arange(len(points)), points, normals, areas, tx, rx, 60e9, pec,
        polarization_callback=callback)
    offsets = radar.fmcw.slope*np.arange(radar.fmcw.num_adc_samples)/radar.fmcw.sampling_frequency
    expected = np.zeros(len(offsets), complex)
    for i, frequency in enumerate(offsets):
        integral = 1.
        if mode == "parent_face_far_field_analytic":
            tx_hat = (tx-points)/np.linalg.norm(tx-points, axis=1)[:, None]
            rx_hat = (rx-points)/np.linalg.norm(rx-points, axis=1)[:, None]
            q = -2*np.pi*(60e9+frequency)/c * (tx_hat+rx_hat)[None]
            integral = po._triangle_linear_phase_integrals(triangles, points, q).ravel()/areas
        expected[i] = np.sum((coeff*integral).conjugate()*np.exp(2j*np.pi*frequency*tau))
    expected *= -10**(6./20.)
    sim = MmWaveRadarSimulator(compute_backend="numpy", adc_compute_precision="float64")
    actual = sim.synthesize_adc(getattr(result, order+"_coefficients"),
        getattr(result, order+"_delays_s"), getattr(result, order+"_valid"), radar.fmcw,
        delay_response=getattr(result, order+"_delay_response"))
    np.testing.assert_allclose(actual[:, 0], expected, rtol=1e-10, atol=1e-18)
    cached = compute_env_human_coupling_channels(
        vertices, faces, radar.world_tx_positions(), radar.world_rx_positions(), 60e9, pec, bank,
        radar=radar, po_config=_config(mode),
        human_env_active_sets=result.human_env_active_sets, env_human_active_sets=result.env_human_active_sets,
        human_visibility_scene=HumanVisibilityScene(faces, vertices, name="image-wideband-cached"))
    np.testing.assert_allclose(getattr(cached, order+"_delay_response").values,
                               getattr(result, order+"_delay_response").values, rtol=1e-12, atol=1e-20)


def test_hybrid_rejects_calibrating_direct_po_against_coupled_rt_reference():
    from mmWaveRadar.simulation.hybrid_pipeline import run_calibrated_hybrid_po
    target = MeshTarget("reference-human", _single_face_mesh_sequence(x0=2.),
                        material=human_skin_material("reference-skin"))
    with pytest.raises(ValueError, match="single_human_only"):
        run_calibrated_hybrid_po(scene=load_scene(), radar=_radar(), target=target,
            num_frames=1, coupling_enabled=False, progress=False,
            calibration_config=POCalibrationConfig(mode="rt_sequence", reference_component="human_touch"))


def test_wideband_image_projection_preserves_per_channel_phase_delay_and_gain():
    vertices, faces, pec, bank = _setup()
    radar = _radar()
    hardware = replace(RadarHardware.from_positions(
        [[0., -.001, 0.], [0., .001, 0.]], [[0., -.002, 0.], [0., .002, 0.]]),
        scalar_antenna_gain_dbi=6., virtual_channel_phase_signs=np.array([1., -1., -1., 1.]))
    array_radar = replace(radar, hardware=hardware, fmcw=replace(radar.fmcw, num_tx=2))
    common = dict(vertices=vertices, faces=faces, frequency_hz=60e9, human_material=pec,
        reflectors=bank, po_config=_config(),
        human_visibility_scene=HumanVisibilityScene(faces, vertices, name="wideband-array"))
    center = compute_env_human_coupling_channels(**common, radar=radar,
        tx_positions=radar.world_tx_positions(), rx_positions=radar.world_rx_positions())
    array = compute_env_human_coupling_channels(**common, radar=array_radar,
        tx_positions=array_radar.world_tx_positions(), rx_positions=array_radar.world_rx_positions())
    offsets = radar.fmcw.slope*np.arange(225)/radar.fmcw.sampling_frequency
    point = vertices.mean(0)
    center_spread = np.linalg.norm(point)*np.linalg.norm(point-[0., 0., -2.])
    center_length = np.linalg.norm(point)+np.linalg.norm(point-[0., 0., -2.])
    for order in ("human_env", "env_human"):
        base_adc = getattr(center, order+"_delay_response").synthesize_adc(
            getattr(center, order+"_delays_s"), getattr(center, order+"_valid"), radar.fmcw)[:, 0]
        actual = getattr(array, order+"_delay_response").synthesize_adc(
            getattr(array, order+"_delays_s"), getattr(array, order+"_valid"), radar.fmcw)
        for ch, (tx_i, rx_i) in enumerate(zip(hardware.virtual_tx_indices(), hardware.virtual_rx_indices())):
            tx = hardware.tx_positions[tx_i].copy(); rx = hardware.rx_positions[rx_i].copy()
            if order == "human_env":
                rx[2] = -2.
            else:
                tx[2] = -2.
            rt, rr = np.linalg.norm(point-tx), np.linalg.norm(point-rx)
            delta = rt+rr-center_length
            expected = (base_adc * center_spread/(rt*rr)
                * np.exp(2j*np.pi*(60e9+offsets)*delta/c)
                * 10**(6./20.) * hardware.virtual_channel_phase_signs[ch])
            np.testing.assert_allclose(actual[:, ch], expected, rtol=1e-9, atol=1e-18)
