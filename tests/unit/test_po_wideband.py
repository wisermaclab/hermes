# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Wideband ADC backend parity against explicit subface delay synthesis."""

from dataclasses import replace

import numpy as np
import pytest

from mmWaveRadar.radar import FMCWConfig
from mmWaveRadar.simulation import adc_synthesis
from mmWaveRadar.simulation import compute
from mmWaveRadar.simulation.po_wideband import PODelayResponse


@pytest.fixture(params=["cpu", "cuda"])
def torch_device(request, monkeypatch):
    torch = pytest.importorskip("torch")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", request.param)
    return request.param


def response_case(*, sign=1.0, mode="complex1x", num_paths=7):
    config = FMCWConfig(60e9, sign * 68e12, 58e-6, 65e-6, 4.5e6, 225, 1, 0.1, adc_mode=mode)
    response = PODelayResponse.for_mesh(
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
        np.array([[0, 1, 2]]),
        num_paths,
        config.slope * (config.num_adc_samples - 1) / config.sampling_frequency,
    )
    references = np.linspace(2e-9, 25e-9, num_paths)
    response.reference_delays_s[:] = references
    offsets = np.array([-0.02e-9, 0.03e-9])
    rng = np.random.default_rng(14)
    coefficients = rng.normal(size=(num_paths, 2)) + 1j * rng.normal(size=(num_paths, 2))
    response.accumulate(
        coefficients.ravel(),
        (references[:, None] + offsets).ravel(),
        np.repeat(np.arange(num_paths), 2),
    )
    gains = rng.normal(size=(3, num_paths)) + 1j * rng.normal(size=(3, num_paths))
    response = response.with_channel_gains(gains)
    delays = references[None] + np.array([0.0, 0.1e-9, -0.1e-9])[:, None]
    if num_paths:
        delays[1, -1] = -1.0
    frequencies = config.slope * np.arange(config.num_adc_samples) / config.sampling_frequency
    expected = np.zeros((config.num_adc_samples, 3), complex)
    for channel in range(3):
        for parent in range(num_paths):
            if delays[channel, parent] < 0.0:
                continue
            for leaf in range(2):
                expected[:, channel] += (
                    gains[channel, parent] * coefficients[parent, leaf]
                ).conjugate() * np.exp(
                    2j * np.pi * frequencies * (delays[channel, parent] + offsets[leaf])
                )
    if mode == "real":
        expected = expected.real.astype(complex)
    return config, response, delays, expected


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("mode", ["complex1x", "complex2x", "real"])
def test_torch_wideband_adc_matches_explicit_leaf_paths(torch_device, precision, sign, mode):
    config, response, delays, expected = response_case(sign=sign, mode=mode)
    counts = {}
    actual = adc_synthesis.synthesize_adc_from_paths(
        np.zeros_like(delays, complex),
        delays,
        config,
        backend="torch",
        precision=precision,
        delay_response=response,
        diagnostics=counts,
    )
    tolerance = 3e-5 if precision == "float32" else 2e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.dtype == (np.complex64 if precision == "float32" else np.complex128)
    assert counts["tested_path_channels"] == 20
    assert counts["rejected_path_channels"] == 0
    if mode == "real":
        assert np.all(actual.imag == 0.0)


def test_auto_wideband_adc_uses_cuda_when_available(torch_device, monkeypatch):
    if torch_device != "cuda":
        pytest.skip("Requires a CUDA device")
    config, response, delays, expected = response_case()
    calls = []
    original = PODelayResponse._synthesize_adc_torch

    def capture(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PODelayResponse, "_synthesize_adc_torch", capture)
    actual = adc_synthesis.synthesize_adc_from_paths(
        np.zeros_like(delays, complex), delays, config, backend="auto", delay_response=response
    )
    assert calls == [True]
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_auto_wideband_adc_falls_back_without_accelerator(monkeypatch):
    config, response, delays, expected = response_case()
    monkeypatch.delenv("MMWAVE_TORCH_DEVICE", raising=False)
    monkeypatch.setattr(compute, "torch_cuda_is_available", lambda torch: False)
    monkeypatch.setattr(compute, "torch_mps_is_available", lambda torch: False)

    def unexpected_torch(*args, **kwargs):
        pytest.fail("auto must use NumPy without a supported accelerator")

    monkeypatch.setattr(PODelayResponse, "_synthesize_adc_torch", unexpected_torch)
    actual = adc_synthesis.synthesize_adc_from_paths(
        np.zeros_like(delays, complex), delays, config, backend="auto", delay_response=response
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_torch_wideband_adc_chunks_dense_path_banks(torch_device):
    config, response, delays, _ = response_case(num_paths=16000)
    expected = response.synthesize_adc(delays, delays >= 0.0, config)
    actual = response.synthesize_adc(delays, delays >= 0.0, config, backend="torch")
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_torch_wideband_adc_preserves_mixed_if_clipping(torch_device, precision):
    config, response, _, _ = response_case()
    cutoff = config.effective_if_cutoff_hz / abs(config.slope)
    delays = (
        cutoff
        + np.linspace(-0.05e-9, 0.05e-9, 7)[None]
        + np.array([0.0, 0.005e-9, -0.005e-9])[:, None]
    )
    valid = np.ones_like(delays, bool)
    valid[1, 0] = False
    cpu_counts, torch_counts = {}, {}
    expected = response.synthesize_adc(
        delays, valid, config, precision=precision, diagnostics=cpu_counts
    )
    actual = response.synthesize_adc(
        delays, valid, config, backend="torch", precision=precision, diagnostics=torch_counts
    )
    tolerance = 3e-6 if precision == "float32" else 2e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert torch_counts == cpu_counts
    assert 0 < cpu_counts["partially_filtered_path_channels"] < 20
    assert 0 < cpu_counts["rejected_path_channels"] < 20


@pytest.mark.parametrize("num_paths", [0, 7])
def test_torch_wideband_adc_empty_or_invalid_paths(torch_device, num_paths):
    config, response, delays, _ = response_case(num_paths=num_paths)
    actual = response.synthesize_adc(
        delays, np.zeros_like(delays, bool), config, backend="torch", precision="float32"
    )
    np.testing.assert_array_equal(actual, np.zeros((225, 3), np.complex64))


def test_wideband_adc_rejects_uncovered_bandwidth(torch_device):
    config, response, delays, _ = response_case()
    with pytest.raises(ValueError, match="does not cover"):
        response.synthesize_adc(
            delays, delays >= 0.0, replace(config, slope=2 * config.slope), backend="torch"
        )


@pytest.mark.parametrize("bandwidth", [4e9, -4e9])
def test_response_grid_reuses_adequate_order_and_grows_for_larger_meshes(bandwidth):
    vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    faces = np.array([[0, 1, 2]])
    initial = PODelayResponse.for_mesh(vertices, faces, 1, bandwidth)
    # The minimum order steps up, but stays within the reserved headroom.
    fresh = PODelayResponse.for_mesh(vertices * 1.03, faces, 1, bandwidth)
    assert len(fresh.frequencies_hz) > len(initial.frequencies_hz)
    reused = PODelayResponse.for_mesh(vertices * 1.03, faces, 1, bandwidth, previous=initial)
    np.testing.assert_array_equal(reused.frequencies_hz, initial.frequencies_hz)
    shrunk = PODelayResponse.for_mesh(vertices * 0.5, faces, 1, bandwidth, previous=reused)
    np.testing.assert_array_equal(shrunk.frequencies_hz, initial.frequencies_hz)
    grown = PODelayResponse.for_mesh(vertices * 2.0, faces, 1, bandwidth, previous=initial)
    assert len(grown.frequencies_hz) > len(initial.frequencies_hz)
    changed = PODelayResponse.for_mesh(vertices, faces, 1, bandwidth / 2.0, previous=initial)
    assert changed.frequencies_hz[-1] == bandwidth / 2.0
    assert not np.array_equal(changed.frequencies_hz, initial.frequencies_hz)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_sparse_response_preserves_original_ids_for_channel_specific_if_clipping(
    monkeypatch,
    backend,
    sign,
):
    if backend == "torch":
        pytest.importorskip("torch")
        monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")
    config, response, _, _ = response_case(sign=sign)
    cutoff = config.effective_if_cutoff_hz / abs(config.slope)
    tau = (
        cutoff
        + np.linspace(-0.05e-9, 0.05e-9, 7)[None]
        + np.array([0.0, 0.005e-9, -0.005e-9])[:, None]
    )
    valid = np.ones(tau.shape, bool)
    valid[:, 2] = False
    response.channel_gains[:, 4] = 0.0
    reference_counts, actual_counts = {}, {}
    expected = response.synthesize_adc(tau, valid, config, diagnostics=reference_counts)

    ids = np.array([2, 5, 11, 14, 18, 21, 26])
    # Empty, invalid, zero-gain, fully rejected and partial parents interleave.
    padded = replace(
        response,
        values=np.zeros((len(response.frequencies_hz), 29), complex),
        reference_delays_s=np.zeros(29),
        channel_gains=np.zeros((3, 29), complex),
        min_delta_s=np.full(29, np.inf),
        max_delta_s=np.full(29, -np.inf),
        retained_paths=np.ones(29, bool),
        leaf_batches=[
            (ids[parents], coefficients, delta, offsets, carrier)
            for parents, coefficients, delta, offsets, carrier in response.leaf_batches
        ],
    )
    padded.values[:, ids] = response.values
    padded.channel_gains[:, ids] = response.channel_gains
    padded.min_delta_s[ids] = response.min_delta_s
    padded.max_delta_s[ids] = response.max_delta_s
    padded_tau, padded_valid = np.zeros((3, 29)), np.zeros((3, 29), bool)
    padded_tau[:, ids], padded_valid[:, ids] = tau, valid
    actual = padded.synthesize_adc(
        padded_tau, padded_valid, config, backend=backend, diagnostics=actual_counts
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert actual_counts == reference_counts


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_zero_full_response_still_contributes_when_partially_filtered(monkeypatch, backend):
    if backend == "torch":
        pytest.importorskip("torch")
        monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")
    config, response, _, _ = response_case(num_paths=1)
    response.values[:] = 0.0  # A cancelled unfiltered response must not drop IF-edge leaves.
    cutoff = config.effective_if_cutoff_hz / abs(config.slope)
    tau = np.full((3, 1), cutoff)
    frequencies = config.slope * np.arange(config.num_adc_samples) / config.sampling_frequency
    parents, coefficients, offsets, _, _ = response.leaf_batches[0]
    expected = (coefficients[0] * response.channel_gains[:, 0]).conjugate()[None] * np.exp(
        2j * np.pi * frequencies * (cutoff + offsets[0])
    )[:, None]
    actual = response.synthesize_adc(tau, np.ones((3, 1), bool), config, backend=backend)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_precision_specific_grid_reduces_nodes_with_bounded_interpolation_error(sign):
    config = FMCWConfig(60e9, sign * 68e12, 58e-6, 65e-6, 4.5e6, 225, 1, 0.1, if_bandwidth_hz=1e8)
    vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    bandwidth = config.slope * (config.num_adc_samples - 1) / config.sampling_frequency
    grids = [
        PODelayResponse.for_mesh(vertices, [[0, 1, 2]], 1, bandwidth, compute_precision=precision)
        for precision in ("float32", "float64")
    ]
    assert len(grids[0].frequencies_hz) < len(grids[1].frequencies_hz)
    coefficients = np.array([1.0, -0.7j, 0.3 + 0.1j])
    offsets = grids[0].delay_bound_s * np.array([-1.0, 0.3, 1.0])
    tau = 40e-9
    frequencies = config.slope * np.arange(config.num_adc_samples) / config.sampling_frequency
    expected = (
        np.exp(2j * np.pi * frequencies[:, None] * (tau + offsets)) @ coefficients.conjugate()
    )
    for response, tolerance in zip(grids, [1e-8, 1e-13]):
        response.reference_delays_s[:] = tau
        response.accumulate(coefficients, tau + offsets, np.zeros(3, int))
        # Isolate interpolation error from ADC float32 rounding.
        actual = response.synthesize_adc(np.array([tau]), np.array([True]), config)[:, 0]
        assert np.max(abs(actual - expected)) < tolerance * np.sum(abs(coefficients))
    promoted = PODelayResponse.for_mesh(
        vertices, [[0, 1, 2]], 1, bandwidth, previous=grids[0], compute_precision="float64"
    )
    assert len(promoted.frequencies_hz) >= len(grids[1].frequencies_hz)


@pytest.mark.parametrize("sparse", [False, True])
def test_sparse_wideband_synthesis_matches_numpy_on_mps(monkeypatch, sparse):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "mps")
    config, response, tau, _ = response_case(num_paths=53)
    valid = tau >= 0.0
    if sparse:
        valid[:, 1::3] = False
        response.values[:, ::4] = 0.0
        response.channel_gains[:, ::5] = 0.0
    expected = response.synthesize_adc(tau, valid, config, precision="float32")
    actual = response.synthesize_adc(tau, valid, config, backend="torch", precision="float32")
    # Float32 beat-phase error also occurs without compaction. Scale the
    # elementwise bound to the peak so cancelled samples remain meaningful.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-4 * np.max(abs(expected)))
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 3e-5
