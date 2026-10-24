# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Calibration from coherent signals rather than tessellation-dependent powers."""

from types import SimpleNamespace

from sensing_test_helpers import *  # noqa: F401,F403
from mmWaveRadar.simulation import hybrid_pipeline


def _reference_cube(adc, times):
    return RadarCube(
        adc=adc,
        times=times,
        metadata=SensingMetadata(
            events=[{"frame": 0, "chirp": 0, "time": float(times[0, 0])}],
            path_counts=np.ones(times.shape, dtype=np.int64),
            max_displacements=np.zeros(times.shape),
            virtual_channel_order="tx_major",
            solver="rt",
            human_touch_path_power=np.full(times.shape, 999.0),
        ),
    )


@pytest.mark.parametrize("delay_separation", [0.0, 0.5e-9])
def test_sequence_calibration_uses_interference_and_delays_per_channel(delay_separation):
    radar = _phase_one_radar(num_chirps=1)
    radar = replace(radar, fmcw=replace(radar.fmcw, slope=1e12))
    # Opposite channel signs must not cancel across channels. Within each
    # channel the two rays cancel partly, unless their beat tones are orthogonal.
    a = np.array([[1.0, -0.5], [-1.0, 0.5]], dtype=np.complex128)
    adc = MmWaveRadarSimulator().synthesize_adc(
        a,
        np.array([0.5e-9, 0.5e-9 + delay_separation]),
        np.ones(2, dtype=bool),
        radar.fmcw,
        backend="numpy",
        precision="float64",
    )[None, None]
    times = np.zeros((1, 1))
    reference = 3.0 * adc
    baseline = {
        # Full-scene power is deliberately different from the selected component.
        "cube": _reference_cube(100.0 * adc, times),
        "reference_adc_power": {
            "human_touch": np.mean(np.abs(reference) ** 2, axis=2),
            "single_human_only": np.ones((1, 1, 2)),
        },
    }
    gain, metadata = fit_sequence_adc_calibration(
        rt_baseline=baseline,
        po_times=times,
        po_adc=adc,
        config=POCalibrationConfig(
            mode="rt_sequence",
            reference_component="human_touch",
            sample_mode="all_chirps",
            alignment="same_index",
        ),
    )
    assert gain == pytest.approx(3.0)
    assert metadata["raw_po_reference_rms_power"] == pytest.approx(
        0.25 if delay_separation == 0.0 else 1.25,
    )
    assert metadata["reference_power_kind"] == "adc_rms_power"


@pytest.mark.parametrize("integration_mode", ["face_centroid", "parent_face_quadrature"])
def test_po_sequence_calibration_is_invariant_to_splitting_a_small_face(integration_mode):
    class CalibrationSimulator(MmWaveRadarSimulator):
        def _run_rt_baseline_for_calibration(self, scene, radar, targets, *, num_frames, **kwargs):
            del scene, targets, kwargs
            shape = (
                num_frames,
                radar.fmcw.num_chirps_per_frame,
                radar.fmcw.num_adc_samples,
                radar.hardware.num_virtual_channels,
            )
            adc = np.full(shape, 1e-8 + 0j)
            times = np.array(
                [[radar.fmcw.chirp_time(f, k) for k in range(shape[1])] for f in range(num_frames)]
            )
            return {
                "cube": _reference_cube(adc, times),
                "human_reference_adc": adc[0],
                "human_reference_path_counts": np.ones(shape[1], dtype=np.int64),
                "reference_adc_power": {
                    "single_human_only": np.mean(np.abs(adc) ** 2, axis=2),
                },
            }

    triangle = _front_facing_triangle(1.0, scale=1e-4).astype(np.float64)
    a, b, d = triangle
    ab, bd, da = (a + b) / 2, (b + d) / 2, (d + a) / 2
    split = np.array([[a, ab, da], [ab, b, bd], [da, bd, d], [ab, bd, da]])
    radar = _phase_one_radar(num_chirps=2)
    cubes = []
    for vertices in (triangle, split.reshape(-1, 3)):
        faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
        target = MeshTarget(
            "human",
            MeshSequence(
                vertices=np.stack([vertices, vertices]),
                faces=faces,
                times=np.array([0.0, 2.0]),
            ),
            material=human_skin_material("mesh-calibration-human"),
        )
        sim = CalibrationSimulator(
            solver="po",
            compute_backend="numpy",
            compute_precision="float64",
            po_config=POConfig(
                po_visibility_samples_per_face=1,
                po_integration_mode=integration_mode,
                po_calibration=POCalibrationConfig(mode="rt_sequence"),
            ),
        )
        cubes.append(sim.run(load_scene(), radar, [target], num_frames=1))

    gains = [cube.metadata.po_calibration["human_po_amplitude_gain"] for cube in cubes]
    np.testing.assert_allclose(gains[1], gains[0], rtol=1e-6)
    for cube in cubes:
        assert np.mean(np.abs(cube.adc) ** 2) == pytest.approx(1e-16, rel=1e-6)
    # The old diagnostic still shows the expected factor-four power decrease;
    # retaining that diagnostic must not influence the coherent calibration.
    np.testing.assert_allclose(
        cubes[1].metadata.total_path_power, cubes[0].metadata.total_path_power / 4.0, rtol=1e-6
    )


def test_multi_touch_is_not_a_calibration_reference():
    with pytest.raises(ValueError, match="reference_component must be one of"):
        POCalibrationConfig(reference_component="human_multi_touch")


@pytest.mark.parametrize(
    "component,indices",
    [
        ("single_human_only", [0, 1]),
        ("one_human_touch", [0, 1, 2]),
        ("human_touch", [0, 1, 2, 3]),
        ("human_env_coupled", [2]),
    ],
)
def test_rt_reference_power_is_coherent_and_selects_the_requested_paths(component, indices):
    a = np.array([[1.0, -0.5, 0.2j, 0.3, 10.0]], dtype=np.complex128)
    tau = np.array([2e-9, 2e-9, 3e-9, 4e-9, 5e-9])

    class TestSimulator(MmWaveRadarSimulator):
        def trace_scene(self, scene):
            return object()

        def target_interaction_counts(self, paths, targets):
            return np.array([1, 1, 1, 2, 0]), np.array([1, 1, 2, 2, 1])

        def path_arrays_from_paths(self, paths, targets, **kwargs):
            return a, tau, np.ones(5, dtype=bool)

    radar = _phase_one_radar(num_chirps=2)
    simulator = TestSimulator(solver="rt", rt_refresh_frames=0, compute_backend="numpy")
    result = hybrid_pipeline.run_rt_baseline(
        scene=load_scene(),
        radar=radar,
        targets=[
            MeshTarget(
                "human",
                _single_face_mesh_sequence(x0=1.0),
                material=human_skin_material("rt-power-human"),
            )
        ],
        simulator=simulator,
        num_frames=2,
        calibration_reference_component=component,
    )
    mask = np.zeros(5, dtype=bool)
    mask[indices] = True
    expected_adc = simulator.synthesize_adc(a, tau, mask, radar.fmcw, backend="numpy")
    np.testing.assert_allclose(
        result.reference_adc_power[component],
        np.full((2, 2, 1), np.mean(np.abs(expected_adc) ** 2)),
    )


def test_rt_reference_adc_power_packs_each_tdm_channels_own_slot():
    radar = _phase_one_radar(num_chirps=2)
    radar = replace(
        radar,
        hardware=RadarHardware.from_positions(
            [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]], [[0.0, 0.0, 0.0]]
        ),
        fmcw=replace(radar.fmcw, num_tx=2, tx_schedule="tdm"),
    )
    power = np.array([[[1.0, 100.0], [200.0, 4.0], [9.0, 300.0], [400.0, 16.0]]])
    adc = np.ones((1, 4, 4, 2), dtype=np.complex128)
    expanded = SimpleNamespace(
        cube=_reference_cube(adc, np.array([[0.0, 0.5, 1.0, 1.5]])),
        human_reference_adc=adc[0],
        human_reference_path_counts=np.ones(4),
        reference_adc_power={"human_touch": power},
    )
    # Unrelated optional diagnostics are absent from this minimal result.
    from dataclasses import fields

    values = {
        field.name: getattr(expanded, field.name, None)
        for field in fields(hybrid_pipeline.RTBaselineResult)
    }
    result = hybrid_pipeline._collapse_tdm_rt_baseline_result(
        hybrid_pipeline.RTBaselineResult(**values),
        radar,
    )
    np.testing.assert_array_equal(
        result.reference_adc_power["human_touch"], [[[1.0, 4.0], [9.0, 16.0]]]
    )


def test_adc_calibration_rejects_missing_coherent_reference_instead_of_using_path_power():
    adc = np.ones((1, 1, 4, 1), dtype=np.complex128)
    times = np.zeros((1, 1))
    with pytest.raises(ValueError, match="missing coherent RT ADC power"):
        fit_sequence_adc_calibration(
            rt_baseline={"cube": _reference_cube(adc, times)},
            po_times=times,
            po_adc=adc,
            config=POCalibrationConfig(mode="rt_sequence", reference_component="human_touch"),
        )


@pytest.mark.parametrize("bounces,expected_count", [("single", 2), ("any", 3)])
def test_rt_reference_and_solver_share_bounce_masks_at_every_retrace_cadence(
    monkeypatch,
    bounces,
    expected_count,
):
    import mmWaveRadar.simulation.simulator as simulator_module

    radar = _phase_one_radar(num_chirps=3)
    target = MeshTarget(
        "bounce-reference",
        _single_face_mesh_sequence(x0=1.0),
        material=human_skin_material("bounce-reference-skin"),
    )

    class TestSimulator(MmWaveRadarSimulator):
        def trace_scene(self, scene):
            object_id = int(np.asarray(target.ensure_scene_object().object_id).ravel()[0])
            triangle = np.asarray(target.mesh_sequence.vertices_at(0.0), dtype=np.float64)
            centre = triangle.mean(0)
            points = np.zeros((2, 3, 3))
            points[0] = [centre, 0.8 * centre + 0.2 * triangle[0], [2.0, 0.0, 0.0]]
            points[1, 1] = 0.8 * centre + 0.2 * triangle[1]
            interaction = np.full((2, 3), InteractionType.NONE, np.uint32)
            interaction[0] = InteractionType.SPECULAR
            interaction[1, 1] = InteractionType.SPECULAR
            objects = np.full((2, 3), INVALID_SHAPE, np.uint32)
            objects[0] = [object_id, object_id, 99]
            objects[1, 1] = object_id
            primitives = np.full((2, 3), INVALID_PRIMITIVE, np.uint32)
            primitives[0] = 0
            primitives[1, 1] = 0
            lengths = [
                2.0 * np.linalg.norm(centre),
                np.linalg.norm(points[0, 1])
                + np.linalg.norm(points[1, 1] - points[0, 1])
                + np.linalg.norm(points[1, 1]),
                4.0,
            ]
            tau = (np.array(lengths) / c).reshape(1, 1, 3)
            coefficients = np.array([1.0, 3.0, 0.5], complex).reshape(1, 1, 1, 1, 3, 1)
            return SimpleNamespace(
                cir=lambda **kw: (coefficients, tau),
                tau=tau,
                vertices=points.reshape(2, 1, 1, 3, 3),
                interactions=interaction.reshape(2, 1, 1, 3),
                objects=objects.reshape(2, 1, 1, 3),
                primitives=primitives.reshape(2, 1, 1, 3),
            )

    updates = []
    original_update = simulator_module.update_coherent_rt_path_bank

    def update(*args, **kwargs):
        updates.append(True)
        return original_update(*args, **kwargs)

    monkeypatch.setattr(simulator_module, "update_coherent_rt_path_bank", update)
    references = []
    scene = load_scene()
    for cadence in (0, 1):
        settings = dict(
            solver="rt",
            rt_refresh_frames=cadence,
            rt_human_bounces=bounces,
            rt_human_specular_reflection=True,
            rt_max_depth=2,
            compute_backend="numpy",
            compute_precision="float64",
            rt_coherent_transition_config=RTCoherentTransitionConfig(crossfade=True),
        )
        reference = hybrid_pipeline.run_rt_baseline(
            scene=scene,
            radar=radar,
            targets=[target],
            simulator=TestSimulator(**settings),
            num_frames=2,
            calibration_reference_component="human_touch",
        )
        cube = TestSimulator(**settings).run(scene, radar, [target], num_frames=2)
        np.testing.assert_allclose(reference.cube.adc, cube.adc, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(reference.cube.metadata.path_counts, expected_count)
        assert reference.retrace_count == (6 if cadence == 0 else 2)
        np.testing.assert_array_equal(
            reference.human_multi_touch_path_counts, int(bounces == "any")
        )
        references.append(reference)
    np.testing.assert_allclose(
        references[0].cube.adc, references[1].cube.adc, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        references[0].reference_adc_power["human_touch"],
        references[1].reference_adc_power["human_touch"],
        rtol=1e-12,
    )
    # Cross-frame transitions do not exist, so each coherent run updates only
    # its active bank once per chirp (both baseline and solver runs).
    assert len(updates) == 12
