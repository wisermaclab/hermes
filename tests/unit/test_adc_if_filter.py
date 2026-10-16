# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Receiver passband regressions, including coherent delay-resolved PO."""

from dataclasses import replace
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from mmWaveRadar.radar import FMCWConfig, RadarSensor
from mmWaveRadar.dsp.fft import range_fft
from mmWaveRadar.simulation.adc_synthesis import synthesize_adc_from_paths
from mmWaveRadar.simulation.po_wideband import PODelayResponse


def waveform(**kwargs):
    return replace(FMCWConfig(60e9, 68e12, 58e-6, 65e-6, 4.5e6,
                              128, 1, .1), **kwargs)


@pytest.mark.parametrize("sign", [1., -1.])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_upper_complex_band_survives_but_aliased_far_target_is_removed(sign, backend):
    if backend == "torch":
        pytest.importorskip("torch")
    config = waveform(slope=sign*68e12)
    delays = np.array([.75, 1.125]) * config.sampling_frequency / abs(config.slope)
    counts = {}
    adc = synthesize_adc_from_paths(np.ones((1, 2)), delays, config,
                                   backend=backend, diagnostics=counts)
    spectrum, ranges = range_fft(adc.T, fmcw=config, window=None)
    assert spectrum.shape == (1, 128)
    assert np.argmax(abs(spectrum[0])) == 96
    assert abs(spectrum[0, 16]) < 1e-10
    assert ranges[96] == pytest.approx(299792458.*delays[0]/2.)
    assert counts["tested_path_channels"] == 2
    assert counts["rejected_path_channels"] == 1


@pytest.mark.parametrize("mode,factor,bins", [("complex1x", .9, 128),
                                            ("complex2x", .45, 64), ("real", .45, 64)])
def test_sampling_mode_controls_passband_and_fft(mode, factor, bins):
    config = waveform(adc_mode=mode)
    assert config.effective_if_cutoff_hz == factor*config.sampling_frequency
    adc = synthesize_adc_from_paths(np.array([[1.+1j, 10.]]),
                                   np.array([.25, .95])*config.sampling_frequency/config.slope,
                                   config)
    spectrum, _ = range_fft(adc.T, fmcw=config, window=None)
    assert spectrum.shape[1] == bins
    assert np.argmax(abs(spectrum[0])) == 32
    if mode == "real":
        assert np.all(adc.imag == 0.)


def test_dtype_fallback_and_explicit_complex2x():
    real = np.zeros((2, 32))
    assert range_fft(real, 1e6, 1e12)[0].shape[1] == 16
    assert range_fft(real.astype(complex), 1e6, 1e12)[0].shape[1] == 32
    assert range_fft(real.astype(complex), 1e6, 1e12, adc_mode="complex2x")[0].shape[1] == 16


def test_channel_specific_filter_counts_ignore_invalid_slots():
    config = waveform(if_bandwidth_hz=1e6)
    tau = np.array([[.5, 2., -1.], [2., .5, -1.]])*1e6/config.slope
    counts = {}
    result = synthesize_adc_from_paths(np.array([[1., 10., np.nan], [10., 1., np.nan]]),
                                       tau, config, diagnostics=counts)
    expected = np.exp(2j*np.pi*.5e6*np.arange(128)/config.sampling_frequency)
    np.testing.assert_allclose(result, np.tile(expected[:, None], (1, 2)), atol=1e-12)
    assert counts["tested_path_channels"] == 4
    assert counts["rejected_path_channels"] == 2


def test_board_caps_if_and_array_constructors_preserve_mode():
    config = waveform(sampling_frequency=25e6, adc_mode="complex2x", if_bandwidth_hz=12e6)
    sensor = RadarSensor.from_ti_board("IWR6843ISK", fmcw=config)
    assert sensor.fmcw.adc_mode == "complex2x"
    assert sensor.fmcw.effective_if_cutoff_hz == 10e6
    ura = RadarSensor.from_virtual_ura(rows=2, cols=2, fmcw=config)
    assert ura.fmcw.adc_mode == "complex2x"
    assert ura.fmcw.if_bandwidth_hz == 12e6


@pytest.mark.parametrize("value,mode", [(0, "real"), (1, "complex1x"), (2, "complex2x")])
def test_ti_cli_preserves_adc_mode(tmp_path, value, mode):
    filename = tmp_path / "radar.cfg"
    filename.write_text(f"adcCfg 2 {value}\n"
                        "profileCfg 0 60 7 3 58 0 0 68 1 128 4500 0 0 30\n"
                        "chirpCfg 0 0 0 0 0 0 0 1\n"
                        "frameCfg 0 0 1 0 50 1 0\n")
    sensor = RadarSensor.from_ti_cli_config(str(filename), hardware_model="IWR6843ISK")
    assert sensor.fmcw.adc_mode == mode


@pytest.mark.parametrize("kwargs", [{"adc_mode": "complex"}, {"if_bandwidth_hz": 0.},
                                    {"if_bandwidth_hz": -1.}, {"if_bandwidth_hz": np.inf}])
def test_invalid_receiver_settings(kwargs):
    with pytest.raises(ValueError):
        waveform(**kwargs)


def response_for(config, *, receiver=True):
    response = PODelayResponse.for_mesh(
        np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]]), np.array([[0, 1, 2]]),
        1, config.slope*(config.num_adc_samples-1)/config.sampling_frequency,
        receiver_fmcw=config if receiver else None)
    response.reference_delays_s[:] = 1e-9
    return response


def test_po_filters_subface_delays_before_coherent_parent_sum():
    config = waveform(sampling_frequency=1e6, slope=9e14, num_adc_samples=16)
    response = response_for(config)
    response.accumulate(np.array([1., -1.], complex), np.array([.7e-9, 1.3e-9]),
                        np.array([0, 0]))
    response = response.with_channel_gains(np.ones((2, 1)))
    counts = {}
    actual = synthesize_adc_from_paths(np.zeros((2, 1)), np.array([[1e-9], [1.4e-9]]),
                                       config, delay_response=response, diagnostics=counts)
    expected = np.exp(2j*np.pi*config.slope*.7e-9*np.arange(16)/1e6)
    np.testing.assert_allclose(actual[:, 0], expected, atol=2e-12)
    np.testing.assert_allclose(actual[:, 1], 0., atol=2e-12)
    assert counts["partially_filtered_path_channels"] == 1
    assert counts["rejected_path_channels"] == 1


@pytest.mark.parametrize("sign", [1., -1.])
def test_analytic_triangle_if_clipping_matches_independent_fine_quadrature(sign):
    config = waveform(sampling_frequency=1e6, slope=sign*9e14, num_adc_samples=16)
    response = response_for(config)
    offsets = np.array([[-.3e-9, 0., .3e-9]])
    carrier = 1e9
    coefficient = 1.+.4j
    from mmWaveRadar.simulation.physical_optics import _standard_simplex_exp_integral
    def integral(frequency):
        scale = 2*np.pi*(carrier+frequency)
        return (2.*np.exp(-1j*scale*offsets[:, 0]) * _standard_simplex_exp_integral(
            scale*(offsets[:, 1]-offsets[:, 0]), scale*(offsets[:, 2]-offsets[:, 0])))
    response.accumulate(np.array([coefficient]), np.array([1e-9]), np.array([0]),
                        phase_integral=integral, vertex_delay_offsets_s=offsets,
                        carrier_frequency_hz=carrier)
    taus = np.array([[1e-9], [1.15e-9], [2e-9]])
    response = response.with_channel_gains(np.ones_like(taus, complex))
    counts = {}
    actual = synthesize_adc_from_paths(np.ones_like(taus), taus, config,
                                       delay_response=response, diagnostics=counts)
    nodes, weights = leggauss(40)
    u0, v0 = np.meshgrid((nodes+1)/2., (nodes+1)/2.)
    w = np.outer(weights, weights)/4.
    frequencies = config.slope*np.arange(16)/config.sampling_frequency
    for channel, extent in enumerate([1., .5]):
        u = extent*u0
        v = extent*(1.-u0)*v0/2.
        delta = -.3e-9 + .3e-9*u + .6e-9*v
        jacobian = extent**2*(1.-u0)/2.
        fields = 2.*coefficient*np.sum(
            w[None]*jacobian[None]*np.exp(-2j*np.pi*(carrier+frequencies[:, None, None])*delta),
            axis=(1, 2))
        expected = fields.conjugate()*np.exp(2j*np.pi*frequencies*taus[channel, 0])
        np.testing.assert_allclose(actual[:, channel], expected, atol=5e-12)
    np.testing.assert_array_equal(actual[:, 2], 0.)
    assert counts["partially_filtered_path_channels"] == 2
    assert counts["rejected_path_channels"] == 1


def test_po_keeps_leaf_storage_only_near_receiver_edge():
    config = waveform()
    response = response_for(config)
    # A small facet one metre away is well within the receiver passband.
    response.set_reference(np.array([0]), np.array([[1., 0., 0.]]), np.zeros(3), np.zeros(3))
    response.accumulate(np.ones(1), response.reference_delays_s, np.array([0]))
    assert response.leaf_batches == []


def test_incremental_analytic_po_reclips_current_facets_at_if_edge():
    from sensing_test_helpers import _front_facing_triangle, _phase_one_radar
    from mmWaveRadar.targets import MeshTarget, MeshSequence
    from mmWaveRadar.simulation import MmWaveRadarSimulator, HumanPOMobilityConfig
    from mmWaveRadar.simulation.physical_optics import PhysicalOpticsMaterial

    vertices = _front_facing_triangle(2., scale=.04)
    vertices[:, 0] += np.array([-.01, 0., .01])
    radar = _phase_one_radar(num_chirps=3, chirp_repetition_time=.1)
    center_delay = 2.*np.linalg.norm(vertices.mean(0))/299792458.
    radar = replace(radar, fmcw=replace(radar.fmcw,
                    if_bandwidth_hz=abs(radar.fmcw.slope)*center_delay))
    moved = vertices + [0.0001, 0., 0.]
    target = MeshTarget("if-edge-human", MeshSequence(
        np.stack([vertices, moved]), np.array([[0, 1, 2]], np.uint32), np.array([0., .2])),
        material=PhysicalOpticsMaterial(1., 0., model="pec"))
    config = HumanPOMobilityConfig(
        po_integration_mode="parent_face_adaptive_analytic",
        po_quadrature_max_refinement_depth=2, po_quadrature_max_subfaces_per_parent=16,
        visibility_samples_per_face=1, visibility_fade_chirps=1,
        incremental_full_refresh_chirps=0, incremental_visibility_refresh_chirps=64,
        incremental_centroid_displacement_threshold_m=1.,
    )
    cubes = []
    for incremental in (False, True):
        simulator = MmWaveRadarSimulator(mobility_mode="human_only_po",
            human_po_config=replace(config, incremental_update=incremental),
            compute_backend="numpy", compute_precision="float64")
        cubes.append(simulator._run_human_only_po(radar, [target], num_frames=1))
    np.testing.assert_allclose(cubes[1].adc, cubes[0].adc, rtol=1e-12, atol=1e-20)
    assert np.any(abs(cubes[1].adc) > 0.)
    assert cubes[1].metadata.adc_if_filter["partially_filtered_path_channels"] == 3
