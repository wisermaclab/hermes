# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Calibration from coherent signals rather than tessellation-dependent powers."""

from types import SimpleNamespace

from sensing_test_helpers import *  # noqa: F401,F403
from mmWaveRadar.simulation import hybrid_pipeline


def _reference_cube(adc, times):
    return RadarCube(
        adc=adc, times=times,
        metadata=SensingMetadata(
            events=[{"frame": 0, "chirp": 0, "time": float(times[0, 0])}],
            path_counts=np.ones(times.shape, dtype=np.int64),
            max_displacements=np.zeros(times.shape),
            virtual_channel_order="tx_major", mode="rt_baseline",
            human_touch_path_power=np.full(times.shape, 999.),
        ),
    )


@pytest.mark.parametrize("delay_separation", [0., 0.5e-9])
def test_sequence_calibration_uses_interference_and_delays_per_channel(delay_separation):
    radar = _phase_one_radar(num_chirps=1)
    radar = replace(radar, fmcw=replace(radar.fmcw, slope=1e12))
    # Opposite channel signs must not cancel across channels. Within each
    # channel the two rays cancel partly, unless their beat tones are orthogonal.
    a = np.array([[1., -.5], [-1., .5]], dtype=np.complex128)
    adc = MmWaveRadarSimulator().synthesize_adc(
        a, np.array([.5e-9, .5e-9 + delay_separation]), np.ones(2, dtype=bool),
        radar.fmcw, backend="numpy", precision="float64",
    )[None, None]
    times = np.zeros((1, 1))
    reference = 3. * adc
    baseline = {
        # Full-scene power is deliberately different from the selected component.
        "cube": _reference_cube(100. * adc, times),
        "reference_adc_power": {
            "human_touch": np.mean(np.abs(reference) ** 2, axis=2),
            "single_human_only": np.ones((1, 1, 2)),
        },
    }
    gain, metadata = fit_sequence_adc_calibration(
        rt_baseline=baseline, po_times=times, po_adc=adc,
        config=POCalibrationConfig(
            mode="rt_sequence", reference_component="human_touch",
            sample_mode="all_chirps", alignment="same_index",
        ),
    )
    assert gain == pytest.approx(3.)
    assert metadata["raw_po_reference_rms_power"] == pytest.approx(
        .25 if delay_separation == 0. else 1.25,
    )
    assert metadata["reference_power_kind"] == "adc_rms_power"


@pytest.mark.parametrize("integration_mode", ["face_centroid", "parent_face_quadrature"])
def test_po_sequence_calibration_is_invariant_to_splitting_a_small_face(integration_mode):
    class CalibrationSimulator(MmWaveRadarSimulator):
        def _run_rt_baseline_for_calibration(self, scene, radar, targets, *, num_frames, **kwargs):
            del scene, targets, kwargs
            shape = (num_frames, radar.fmcw.num_chirps_per_frame,
                     radar.fmcw.num_adc_samples, radar.hardware.num_virtual_channels)
            adc = np.full(shape, 1e-8 + 0j)
            times = np.array([[radar.fmcw.chirp_time(f, k) for k in range(shape[1])]
                              for f in range(num_frames)])
            return {
                "cube": _reference_cube(adc, times),
                "human_reference_adc": adc[0],
                "human_reference_path_counts": np.ones(shape[1], dtype=np.int64),
                "reference_adc_power": {
                    "single_human_only": np.mean(np.abs(adc) ** 2, axis=2),
                },
            }

    triangle = _front_facing_triangle(1., scale=1e-4).astype(np.float64)
    a, b, d = triangle
    ab, bd, da = (a + b) / 2, (b + d) / 2, (d + a) / 2
    split = np.array([[a, ab, da], [ab, b, bd], [da, bd, d], [ab, bd, da]])
    radar = _phase_one_radar(num_chirps=2)
    cubes = []
    for vertices in (triangle, split.reshape(-1, 3)):
        faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
        target = MeshTarget(
            "human", MeshSequence(
                vertices=np.stack([vertices, vertices]), faces=faces,
                times=np.array([0., 2.]),
            ), material=human_skin_material("mesh-calibration-human"),
        )
        sim = CalibrationSimulator(
            mobility_mode="human_only_po", compute_backend="numpy",
            compute_precision="float64",
            human_po_config=HumanPOMobilityConfig(
                visibility_samples_per_face=1, visibility_fade_chirps=1,
                po_integration_mode=integration_mode,
                calibration=POCalibrationConfig(mode="rt_sequence"),
            ),
        )
        cubes.append(sim.run(load_scene(), radar, [target], num_frames=1))

    gains = [cube.metadata.po_calibration["human_po_amplitude_gain"] for cube in cubes]
    np.testing.assert_allclose(gains[1], gains[0], rtol=1e-6)
    for cube in cubes:
        assert np.mean(np.abs(cube.adc) ** 2) == pytest.approx(1e-16, rel=1e-6)
    # The old diagnostic still shows the expected factor-four power decrease;
    # retaining that diagnostic must not influence the coherent calibration.
    np.testing.assert_allclose(cubes[1].metadata.total_path_power,
                               cubes[0].metadata.total_path_power / 4., rtol=1e-6)


@pytest.mark.parametrize("component,indices", [
    ("single_human_only", [0, 1]), ("one_human_touch", [0, 1, 2]),
    ("human_touch", [0, 1, 2, 3]), ("human_env_coupled", [2]),
    ("human_multi_touch", [3]),
])
def test_rt_reference_power_is_coherent_and_selects_the_requested_paths(component, indices):
    a = np.array([[1., -.5, .2j, .3, 10.]], dtype=np.complex128)
    tau = np.array([2e-9, 2e-9, 3e-9, 4e-9, 5e-9])

    class TestSimulator(MmWaveRadarSimulator):
        def trace_scene(self, scene):
            return object()

        def target_interaction_counts(self, paths, targets):
            return np.array([1, 1, 1, 2, 0]), np.array([1, 1, 2, 2, 1])

        def path_arrays_from_paths(self, paths, targets, **kwargs):
            return a, tau, np.ones(5, dtype=bool)

    radar = _phase_one_radar(num_chirps=2)
    simulator = TestSimulator(mobility_mode="rt_retrace", compute_backend="numpy")
    result = hybrid_pipeline.run_full_rt_mobility_baseline(
        scene=load_scene(), radar=radar,
        targets=[MeshTarget("human", _single_face_mesh_sequence(x0=1.),
                            material=human_skin_material("rt-power-human"))],
        simulator=simulator, num_frames=2,
        calibration_reference_component=component,
    )
    mask = np.zeros(5, dtype=bool)
    mask[indices] = True
    expected_adc = simulator.synthesize_adc(a, tau, mask, radar.fmcw, backend="numpy")
    np.testing.assert_allclose(result.reference_adc_power[component],
                               np.full((2, 2, 1), np.mean(np.abs(expected_adc) ** 2)))


def test_rt_reference_adc_power_packs_each_tdm_channels_own_slot():
    radar = _phase_one_radar(num_chirps=2)
    radar = replace(
        radar, hardware=RadarHardware.from_positions([[0., 0., 0.], [0., .01, 0.]],
                                                     [[0., 0., 0.]]),
        fmcw=replace(radar.fmcw, num_tx=2, tdm_enabled=True),
    )
    power = np.array([[[1., 100.], [200., 4.], [9., 300.], [400., 16.]]])
    adc = np.ones((1, 4, 4, 2), dtype=np.complex128)
    expanded = SimpleNamespace(
        cube=_reference_cube(adc, np.array([[0., .5, 1., 1.5]])),
        human_reference_adc=adc[0], human_reference_path_counts=np.ones(4),
        reference_adc_power={"human_touch": power},
    )
    # Unrelated optional diagnostics are absent from this minimal result.
    from dataclasses import fields
    values = {field.name: getattr(expanded, field.name, None)
              for field in fields(hybrid_pipeline.FullRTMobilityBaselineResult)}
    result = hybrid_pipeline._collapse_tdm_full_rt_result(
        hybrid_pipeline.FullRTMobilityBaselineResult(**values), radar,
    )
    np.testing.assert_array_equal(result.reference_adc_power["human_touch"],
                                  [[[1., 4.], [9., 16.]]])


def test_adc_calibration_rejects_missing_coherent_reference_instead_of_using_path_power():
    adc = np.ones((1, 1, 4, 1), dtype=np.complex128)
    times = np.zeros((1, 1))
    with pytest.raises(ValueError, match="missing coherent RT ADC power"):
        fit_sequence_adc_calibration(
            rt_baseline={"cube": _reference_cube(adc, times)},
            po_times=times, po_adc=adc,
            config=POCalibrationConfig(mode="rt_sequence", reference_component="human_touch"),
        )
