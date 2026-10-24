# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Adaptive analytic PO: independent split metrics and persistent parent paths."""

from sensing_test_helpers import *  # noqa: F401,F403
import mmWaveRadar.simulation.physical_optics as po
from mmWaveRadar.simulation import parent_face_adaptive_analytic_geometry


def _equilateral(x=2.0, radius=0.1):
    angle = np.arange(3) * 2.0 * np.pi / 3.0
    return np.column_stack([np.full(3, x), radius * np.cos(angle), radius * np.sin(angle)])


def _geometry(vertices, *, span=3.0, curvature=0.01, depth=8, cap=0, events=None):
    faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
    return parent_face_adaptive_analytic_geometry(
        vertices,
        faces,
        np.arange(len(faces)),
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        60e9,
        span,
        curvature_error_rad=curvature,
        max_refinement_depth=depth,
        max_subfaces_per_parent=cap,
        progress_callback=None if events is None else events.append,
    )


def test_cached_triangle_phase_geometry_batches_nodes_against_numeric_surface_integral():
    from scipy.special import roots_legendre

    rng = np.random.default_rng(51)
    triangles = rng.normal(scale=0.1, size=(7, 3, 3)) + [2.0, 0.2, -0.1]
    centers = triangles.mean(1)
    q = rng.normal(scale=10.0, size=(2, 7, 3))
    scales = np.array([0.0, 0.99, 1.0, 1.1, 2.0])
    actual = po._triangle_phase_integrals_from_geometry(
        po._triangle_linear_phase_geometry(triangles, centers, q), scales[:, None, None]
    )
    nodes, weights = roots_legendre(24)
    s, t = np.meshgrid((nodes + 1.0) / 2.0, (nodes + 1.0) / 2.0, indexing="ij")
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    points = (
        triangles[:, 0, None]
        + s.ravel()[None, :, None] * a[:, None]
        + ((1.0 - s) * t).ravel()[None, :, None] * b[:, None]
    )
    phase = np.einsum("pfd,fnd->pfn", q, points - centers[:, None])
    quadrature_weights = ((weights[:, None] * weights[None, :]) / 4.0 * (1.0 - s)).ravel()
    expected = (
        np.exp(-1j * scales[:, None, None, None] * phase)
        @ quadrature_weights
        * np.linalg.norm(np.cross(a, b), axis=1)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-14)


def test_wideband_analytic_solve_prepares_geometry_once_per_chunk_and_honors_precision(
    monkeypatch,
):
    calls = []
    original = po._triangle_linear_phase_geometry

    def record(*args):
        calls.append(len(args[0]))
        return original(*args)

    monkeypatch.setattr(po, "_triangle_linear_phase_geometry", record)
    results = [
        po.po_channel(
            vertices=_front_facing_triangle(2.0, scale=0.1),
            faces=np.array([[0, 1, 2]]),
            tx_positions=np.zeros((1, 3)),
            rx_positions=np.zeros((1, 3)),
            frequency_hz=60e9,
            material=PhysicalOpticsMaterial(1.0, 0.0, model="pec"),
            visibility="none",
            po_integration_mode="parent_face_far_field_analytic",
            max_frequency_offset_hz=4e9,
            compute_precision=precision,
        )
        for precision in ("float32", "float64")
    ]
    assert len(calls) == 2  # One geometry preparation per solve, across all nodes.
    assert len(results[0].delay_response.frequencies_hz) < len(
        results[1].delay_response.frequencies_hz
    )


def test_split_metrics_match_brute_force_all_pairs_edges_and_factor_one_third():
    rng = np.random.default_rng(42)
    triangles = rng.normal(scale=0.15, size=(7, 3, 3)) + [2.0, 0.2, -0.1]
    tx, rx = rng.normal(scale=0.3, size=(3, 3)), rng.normal(scale=0.3, size=(4, 3))
    curvature, span = np.zeros(7), np.zeros(7)
    for p in tx:
        for q in rx:
            lengths = np.linalg.norm(triangles - p, axis=-1) + np.linalg.norm(
                triangles - q, axis=-1
            )
            center = triangles.mean(axis=1)
            gradient = (center - p) / np.linalg.norm(center - p, axis=1)[:, None] + (
                center - q
            ) / np.linalg.norm(center - q, axis=1)[:, None]
            phases = np.einsum("nvd,nd->nv", triangles - center[:, None], gradient)
            span = np.maximum(span, np.ptp(phases, axis=1))
            for i, j in ((0, 1), (1, 2), (2, 0)):
                midpoint = 0.5 * (triangles[:, i] + triangles[:, j])
                middle_length = np.linalg.norm(midpoint - p, axis=1) + np.linalg.norm(
                    midpoint - q, axis=1
                )
                curvature = np.maximum(
                    curvature, abs(middle_length - 0.5 * (lengths[:, i] + lengths[:, j]))
                )
    np.testing.assert_allclose(
        po._triangle_phase_curvature_rad(triangles, tx, rx, 60e9),
        2.0 * np.pi * 60e9 / (3.0 * c) * curvature,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        po._triangle_linear_phase_span_rad(triangles, tx, rx, 60e9),
        2.0 * np.pi * 60e9 / c * span,
        rtol=1e-12,
        atol=1e-12,
    )


def test_curvature_splits_a_boresight_face_with_zero_linear_phase_span():
    vertices = _equilateral()
    np.testing.assert_allclose(
        po._triangle_linear_phase_span_rad(
            vertices[None], np.zeros((1, 3)), np.zeros((1, 3)), 60e9
        ),
        0.0,
        atol=1e-12,
    )
    error = po._triangle_phase_curvature_rad(
        vertices[None], np.zeros((1, 3)), np.zeros((1, 3)), 60e9
    )[0]
    assert error > 0.01
    assert len(_geometry(vertices, depth=1)[-1]) == 4
    assert len(_geometry(vertices, curvature=error * 1.01, depth=1)[-1]) == 1


def test_phase_span_alone_can_trigger_the_four_child_split():
    vertices = _front_facing_triangle(2.0, scale=0.1, y_shift=1.0)
    span = po._triangle_linear_phase_span_rad(
        vertices[None], np.zeros((1, 3)), np.zeros((1, 3)), 60e9
    )[0]
    assert span > 3.0
    assert len(_geometry(vertices, span=3.0, curvature=10.0, depth=1)[-1]) == 4
    assert len(_geometry(vertices, span=span * 1.01, curvature=10.0, depth=1)[-1]) == 1


def test_recursive_leaves_meet_both_tests_and_preserve_parent_area_and_mesh():
    vertices = np.vstack([_equilateral(), _equilateral(x=2.2) + [0.0, 0.4, 0.0]])
    before = vertices.copy()
    centers, normals, areas, parents, triangles = _geometry(vertices)
    assert len(triangles) > 2
    assert set(parents) == {0, 1}
    np.testing.assert_allclose(
        po._triangle_linear_phase_span_rad(triangles, np.zeros((1, 3)), np.zeros((1, 3)), 60e9)
        <= 3.0 + 1e-12,
        True,
    )
    assert np.all(
        po._triangle_phase_curvature_rad(triangles, np.zeros((1, 3)), np.zeros((1, 3)), 60e9)
        <= 0.01 + 1e-12
    )
    expected = po.face_geometry(vertices, np.array([[0, 1, 2], [3, 4, 5]]))[2]
    np.testing.assert_allclose(np.bincount(parents, weights=areas), expected, rtol=1e-12)
    np.testing.assert_array_equal(vertices, before)


@pytest.mark.parametrize("depth,cap,count", [(0, 0, 1), (1, 0, 4), (8, 1, 1), (8, 4, 4)])
def test_budget_limits_report_unresolved_tolerance(depth, cap, count):
    events = []
    *_, triangles = _geometry(_equilateral(radius=0.5), depth=depth, cap=cap, events=events)
    assert len(triangles) == count
    done = next(e for e in events if e["stage"] == "adaptive_analytic_geometry_done")
    assert done["unresolved_subfaces"] > 0
    assert done["unresolved_parent_faces"] == 1


def test_leaf_integrals_inherit_visibility_and_sum_coherently_to_original_paths():
    vertices = np.vstack(
        [_equilateral(radius=0.04), _equilateral(x=2.2, radius=0.04) + [0.0, 0.15, 0.0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.uint32)
    # Face geometry alone is used here; both faces are made front-facing.
    faces = faces[:, ::-1]
    raw = np.array([0.25, 0.75])
    radar = _phase_one_radar(num_chirps=1)
    base = dict(
        tx_positions=np.zeros((1, 3)),
        rx_positions=np.zeros((1, 3)),
        frequency_hz=60e9,
        material=PhysicalOpticsMaterial(4.0, 0.1),
        visibility_fade_steps=1,
        compute_backend="numpy",
        max_frequency_offset_hz=2e9,
    )
    parent = po.po_channel(
        vertices,
        faces,
        **base,
        po_integration_mode="parent_face_adaptive_analytic",
        po_quadrature_phase_span_scale_rad=3.0,
        po_curvature_error_rad=0.01,
        precomputed_raw_fractional_visibility=raw,
    )
    _, _, _, parents, triangles = parent_face_adaptive_analytic_geometry(
        vertices,
        faces,
        np.arange(2),
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        60e9,
        3.0,
        curvature_error_rad=0.01,
    )
    leaf_vertices = triangles.reshape(-1, 3)
    leaf_faces = np.arange(len(leaf_vertices), dtype=np.uint32).reshape(-1, 3)
    leaves = po.po_channel(
        leaf_vertices,
        leaf_faces,
        **base,
        po_integration_mode="parent_face_far_field_analytic",
        po_quadrature_max_refinement_depth=0,
        precomputed_raw_fractional_visibility=raw[parents],
    )
    expected = np.zeros(2, complex)
    np.add.at(expected, parents, leaves.phase_center_contributions)
    np.testing.assert_allclose(parent.phase_center_contributions, expected, rtol=1e-12, atol=1e-18)
    np.testing.assert_array_equal(parent.face_indices, [0, 1])
    assert parent.coefficients.shape == (1, 2)
    np.testing.assert_allclose(parent.effective_weights, raw)
    fmcw = replace(radar.fmcw, slope=1e12, sampling_frequency=2e3, num_adc_samples=4)
    parent_adc = parent.delay_response.synthesize_adc(parent.delays_s, np.ones(2, bool), fmcw)
    leaf_adc = leaves.delay_response.synthesize_adc(
        leaves.delays_s, np.ones(len(triangles), bool), fmcw
    )
    np.testing.assert_allclose(parent_adc, leaf_adc, rtol=1e-10, atol=1e-16)


@pytest.mark.parametrize("tolerance", [0.0, -0.01, np.nan, np.inf, None])
def test_curvature_tolerance_must_be_finite_positive(tolerance):
    with pytest.raises(ValueError, match="po_curvature_error_rad"):
        POConfig(po_curvature_error_rad=tolerance)


def test_tighter_curvature_converges_to_independent_spherical_wave_integral():
    from numpy.polynomial.legendre import leggauss

    vertices = _equilateral(radius=0.1)
    faces = np.array([[2, 1, 0]], np.uint32)
    material = PhysicalOpticsMaterial(1.0, 0.0, model="pec")
    nodes, weights = leggauss(32)
    nodes, weights = (nodes + 1.0) / 2.0, weights / 2.0
    u, v = np.meshgrid(nodes, nodes, indexing="ij")
    wu, wv = np.meshgrid(weights, weights, indexing="ij")
    tri = vertices[faces[0]]
    points = (
        tri[0] + u[..., None] * (tri[1] - tri[0]) + ((1.0 - u) * v)[..., None] * (tri[2] - tri[0])
    ).reshape(-1, 3)
    _, normals, areas = po.face_geometry(vertices, faces)
    sample_areas = (2.0 * areas[0] * wu * wv * (1.0 - u)).ravel()
    samples, _ = po.vector_facet_face_contributions(
        np.arange(len(points)),
        points,
        np.repeat(normals, len(points), axis=0),
        sample_areas,
        np.zeros(3),
        np.zeros(3),
        60e9,
        material,
    )
    reference = samples.sum()
    errors = []
    for epsilon in (10.0, 0.001):
        result = po.po_channel(
            vertices,
            faces,
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            60e9,
            material,
            visibility="none",
            visibility_fade_steps=1,
            po_integration_mode="parent_face_adaptive_analytic",
            po_quadrature_phase_span_scale_rad=1000.0,
            po_curvature_error_rad=epsilon,
            po_quadrature_max_refinement_depth=8,
        )
        errors.append(abs(result.phase_center_channel - reference) / abs(reference))
    assert errors[0] > 0.1
    assert errors[1] < 0.01


def test_incremental_recompute_uses_the_same_adaptive_policy_as_full_rebuild():
    radar = _phase_one_radar(num_chirps=3, chirp_repetition_time=0.2)
    target = MeshTarget(
        "human",
        _single_face_mesh_sequence(x0=1.0, x1=1.2),
        material=human_skin_material("adaptive-human"),
    )
    config = POConfig(
        po_integration_mode="parent_face_adaptive_analytic",
        po_quadrature_phase_span_scale_rad=3.0,
        po_curvature_error_rad=0.003,
        po_quadrature_max_refinement_depth=3,
        po_quadrature_max_subfaces_per_parent=64,
        po_visibility_samples_per_face=1,
        po_refresh_frames=1,
        po_refresh_on_phase_error_rad=0.0,
    )

    def run(cfg):
        sim = MmWaveRadarSimulator(
            solver="po", po_config=cfg, compute_backend="numpy", compute_precision="float64"
        )
        return sim.run(load_scene(), radar, [target], num_frames=1)

    incremental = run(config)
    full = run(replace(config, po_refresh_frames=0))
    np.testing.assert_allclose(incremental.adc, full.adc, rtol=1e-9, atol=1e-14)
    np.testing.assert_array_equal(
        incremental.metadata.incremental_full_refresh, [[True, False, False]]
    )
    assert incremental.metadata.events[0]["po_curvature_error_rad"] == 0.003


def test_headless_gui_experiment_persists_unresolved_refinement_summary():
    from mmWaveRadar.experiments.human_room import run_human_room_experiment

    result = run_human_room_experiment(
        human_motion=_single_face_mesh_sequence(x0=2.0),
        human_position_m=(2.4, 0.0, 0.0),
        solver="po",
        num_chirps_per_frame=1,
        num_adc_samples=4,
        po_quadrature_max_refinement_depth=0,
        compute_backend="numpy",
    )
    summary = next(
        e for e in result.metadata.events if e.get("reason") == "adaptive_po_refinement"
    )
    assert summary["integration_passes"] == 1
    assert summary["limited_passes"] == 1
    assert summary["max_unresolved_parent_faces"] == 1
    assert summary["max_curvature_error_rad"] > 0.005
    assert result.manifest.solver.parameters["po_quadrature_phase_span_scale_rad"] == 6.0
    assert result.manifest.solver.parameters["po_curvature_error_rad"] == 0.005


def test_phase_span_default_follows_integration_mode():
    analytic = POConfig(po_integration_mode="parent_face_adaptive_analytic")
    assert analytic.po_quadrature_phase_span_scale_rad is None
    assert analytic.effective_po_phase_span_scale_rad == 6.0
    # The default is resolved per use, so changing the mode cannot carry the
    # analytic span into midpoint quadrature.
    quadrature = replace(analytic, po_integration_mode="parent_face_quadrature")
    assert quadrature.effective_po_phase_span_scale_rad == 1.0
    explicit = replace(quadrature, po_quadrature_phase_span_scale_rad=0.5)
    assert explicit.effective_po_phase_span_scale_rad == 0.5
    with pytest.raises(ValueError, match="phase_span"):
        POConfig(po_quadrature_phase_span_scale_rad=0.0)
