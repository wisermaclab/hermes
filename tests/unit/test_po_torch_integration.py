# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Device integration parity, singular limits, and actual PO backend dispatch."""

from copy import deepcopy

import numpy as np
import pytest

from mmWaveRadar.radar import FMCWConfig
from mmWaveRadar.simulation import compute
from mmWaveRadar.simulation import physical_optics as po
from mmWaveRadar.simulation.po_wideband import PODelayResponse


@pytest.fixture(params=["cpu", "cuda", "mps"])
def integration_device(request, monkeypatch):
    torch = pytest.importorskip("torch")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", request.param)
    return request.param


def _runtime(device, precision):
    import torch

    if not compute._supports_compute(torch, device, precision):
        pytest.skip(f"{device} does not support the required {precision} kernels")
    return compute.resolve_torch_runtime("torch", precision)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_complex_cross_matches_numpy_on_selected_device(integration_device, precision):
    import torch

    runtime = _runtime(integration_device, precision)
    rng = np.random.default_rng(58)
    a = rng.normal(size=(4, 8, 3)) + 1j * rng.normal(size=(4, 8, 3))
    b = rng.normal(size=(4, 8, 3)) + 1j * rng.normal(size=(4, 8, 3))
    actual = compute.torch_vector_cross(
        torch.as_tensor(a, dtype=runtime.complex_dtype, device=runtime.device),
        torch.as_tensor(b, dtype=runtime.complex_dtype, device=runtime.device),
    )
    tolerance = 2e-6 if precision == "float32" else 2e-14
    np.testing.assert_allclose(
        actual.cpu().numpy(), np.cross(a, b), rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_simplex_on_device_preserves_singular_and_nearly_equal_limits(
    integration_device, precision
):
    import torch

    runtime = _runtime(integration_device, precision)
    arguments = np.array([-100.0, -3.0, -0.1, -0.001, -1e-7, 0.0, 1e-7, 0.001, 0.1, 3.0, 100.0])
    alpha, beta = np.meshgrid(arguments, arguments)
    expected = po._standard_simplex_exp_integral(alpha, beta)
    actual = po._standard_simplex_exp_integral_torch(
        torch.as_tensor(alpha, dtype=runtime.float_dtype, device=runtime.device),
        torch.as_tensor(beta, dtype=runtime.float_dtype, device=runtime.device),
    )
    assert actual.device.type == runtime.device.type
    if runtime.device.index is not None:
        assert actual.device.index == runtime.device.index
    assert actual.dtype == runtime.complex_dtype
    tolerance = 2e-6 if precision == "float32" else 2e-13
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_device_accumulation_sums_duplicate_unsorted_parents(integration_device, precision):
    runtime = _runtime(integration_device, precision)
    rng = np.random.default_rng(128)
    response = PODelayResponse.for_mesh(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        [[0, 1, 2]],
        47,
        4e9,
        compute_precision=precision,
    )
    response.reference_delays_s[:] = rng.uniform(1e-9, 10e-9, 47)
    parents = rng.integers(0, 47, 1500)
    coefficients = rng.normal(size=1500) + 1j * rng.normal(size=1500)
    delays = response.reference_delays_s[parents] + rng.uniform(-1e-10, 1e-10, 1500)
    expected = deepcopy(response)
    expected.accumulate(coefficients, delays, parents)
    response.accumulate(coefficients, delays, parents, backend="torch", precision=precision)
    magnitudes = np.bincount(parents, weights=abs(coefficients), minlength=47)
    tolerance = 1e-6 if precision == "float32" else 2e-14
    assert np.max(abs(response.values - expected.values) / magnitudes[None]) < tolerance
    np.testing.assert_array_equal(response.min_delta_s, expected.min_delta_s)
    np.testing.assert_array_equal(response.max_delta_s, expected.max_delta_s)
    assert len(response.leaf_batches) == len(expected.leaf_batches)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize(
    "mode",
    ["parent_face_quadrature", "parent_face_adaptive_analytic", "parent_face_far_field_analytic"],
)
def test_full_wideband_po_dispatches_integration_and_matches_numpy(
    integration_device,
    precision,
    mode,
    monkeypatch,
):
    runtime = _runtime(integration_device, precision)
    waveform = FMCWConfig(60e9, 68e12, 58e-6, 65e-6, 4.5e6, 225, 1, 0.1, if_bandwidth_hz=1e8)
    vertices = np.array([[2.0, -0.04, -0.03], [2.0, 0.0, 0.06], [2.0, 0.04, -0.03]])
    arguments = dict(
        vertices=vertices,
        faces=np.array([[0, 1, 2]]),
        tx_positions=np.array([[0.0, -0.001, 0.0], [0.0, 0.001, 0.0]]),
        rx_positions=np.array([[0.0, 0.0, -0.001], [0.0, 0.0, 0.001]]),
        frequency_hz=60e9,
        material=po.PhysicalOpticsMaterial(1.0, 0.0, model="pec"),
        visibility="none",
        po_integration_mode=mode,
        po_quadrature_max_refinement_depth=3,
        po_quadrature_max_subfaces_per_parent=64,
        max_frequency_offset_hz=waveform.slope * 224 / waveform.sampling_frequency,
        receiver_fmcw=waveform,
        compute_precision=precision,
    )
    reference = po.po_channel(**arguments, compute_backend="numpy")
    calls = []
    original = PODelayResponse._accumulate_torch

    def record(self, *args):
        calls.append(args[-1].device)
        return original(self, *args)

    monkeypatch.setattr(PODelayResponse, "_accumulate_torch", record)
    if mode != "parent_face_quadrature":

        def no_numpy_simplex(*args):
            pytest.fail("Torch analytic integration must not use the NumPy simplex kernel")

        monkeypatch.setattr(po, "_standard_simplex_exp_integral", no_numpy_simplex)
    actual = po.po_channel(**arguments, compute_backend="torch")
    assert calls and all(device == runtime.device for device in calls)
    # Isolate integration from ADC backend rounding by mixing both in NumPy.
    expected_adc = reference.delay_response.synthesize_adc(
        reference.delays_s, np.ones(1, bool), waveform
    )
    actual_adc = actual.delay_response.synthesize_adc(actual.delays_s, np.ones(1, bool), waveform)
    tolerance = 2e-3 if precision == "float32" else 2e-9
    assert np.linalg.norm(actual_adc - expected_adc) / np.linalg.norm(expected_adc) < tolerance


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_clipped_analytic_response_stays_on_device(integration_device, precision, monkeypatch):
    import torch

    runtime = _runtime(integration_device, precision)
    response = PODelayResponse.for_mesh(
        [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.03, 0.0]], [[0, 1, 2]], 9, 4e9
    )
    response.reference_delays_s[:] = 1e-9
    offsets = np.array([[-0.3e-9, 0.0, 0.3e-9], [-0.1e-9, 0.2e-9, 0.1e-9]])
    coefficients, parents = np.array([1.0 + 0.4j, -0.3j]), np.array([2, 7])
    carrier = 1e9

    def integral(frequency):
        scale = 2.0 * np.pi * (carrier + frequency)
        return (
            2.0
            * np.exp(-1j * scale * offsets[:, 0])
            * po._standard_simplex_exp_integral(
                scale * (offsets[:, 1] - offsets[:, 0]), scale * (offsets[:, 2] - offsets[:, 0])
            )
        )

    response.accumulate(
        coefficients,
        np.full(2, 1e-9),
        parents,
        phase_integral=integral,
        vertex_delay_offsets_s=offsets,
        carrier_frequency_hz=carrier,
    )
    tau = np.full(9, 1e-9)
    expected = response._filtered_values(parents, tau, 1.05e-9)

    def no_numpy_simplex(*args):
        pytest.fail("Torch IF clipping must not use the NumPy simplex kernel")

    monkeypatch.setattr(po, "_standard_simplex_exp_integral", no_numpy_simplex)
    actual = response._filtered_values(parents, tau, 1.05e-9, runtime=runtime)
    assert isinstance(actual, torch.Tensor) and actual.device.type == runtime.device.type
    if runtime.device.index is not None:
        assert actual.device.index == runtime.device.index
    tolerance = 3e-6 if precision == "float32" else 2e-12
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=tolerance, atol=tolerance)
