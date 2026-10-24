# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Shared compute policy, overrides, capabilities, and caller consistency."""

import builtins

import pytest

from mmWaveRadar.simulation import compute


@pytest.fixture(autouse=True)
def clear_device_override(monkeypatch):
    monkeypatch.delenv("MMWAVE_TORCH_DEVICE", raising=False)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_numpy_does_not_import_torch_or_inspect_devices(monkeypatch, precision):
    original = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            pytest.fail("NumPy backend must not import Torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "unusable-device")
    assert compute.resolve_torch_runtime("numpy", precision) is None


def test_auto_without_optional_torch_uses_numpy(monkeypatch):
    original = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("Torch is not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert compute.resolve_torch_runtime("auto", "float64") is None
    assert compute.select_torch_device() == "cpu"
    with pytest.raises(ImportError, match="not installed"):
        compute.resolve_torch_runtime("torch", "float64")


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_explicit_cpu_override_is_honored_with_cuda_available(monkeypatch, precision):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")
    monkeypatch.setattr(compute, "torch_cuda_is_available", lambda module: True)
    assert compute.resolve_torch_runtime("auto", precision) is None
    runtime = compute.resolve_torch_runtime("torch", precision)
    assert runtime.device == torch.device("cpu")
    assert runtime.phase_dtype == torch.float64
    float_dtype, complex_dtype = compute.numpy_dtypes_for_precision(precision)
    assert runtime.numpy_float_dtype == float_dtype
    assert runtime.numpy_complex_dtype == complex_dtype
    assert compute.select_torch_device() == "cpu"


def test_auto_without_accelerator_and_explicit_torch_cpu_are_distinct(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(compute, "torch_cuda_is_available", lambda module: False)
    monkeypatch.setattr(compute, "torch_mps_is_available", lambda module: False)
    assert compute.resolve_torch_runtime("auto", "float32") is None
    assert compute.resolve_torch_runtime("torch", "float32").device.type == "cpu"


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_cuda_current_device_and_explicit_index(monkeypatch, precision):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(compute, "torch_cuda_is_available", lambda module: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(compute, "_supports_compute", lambda *args: True)
    runtime = compute.resolve_torch_runtime("auto", precision)
    assert str(runtime.device) == "cuda:2"
    assert compute.select_torch_device() == "cuda"
    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cuda:1")
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: None)
    runtime = compute.resolve_torch_runtime("auto", precision)
    assert str(runtime.device) == "cuda:1"
    assert compute.select_torch_device() == "cuda:1"


def test_mps_requires_supported_precision_and_complex_operations(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(compute, "torch_cuda_is_available", lambda module: False)
    monkeypatch.setattr(compute, "torch_mps_is_available", lambda module: True)
    monkeypatch.setattr(
        compute, "_supports_compute", lambda module, device, precision: precision == "float32"
    )
    runtime = compute.resolve_torch_runtime("auto", "float32")
    assert runtime.device.type == "mps"
    assert runtime.phase_dtype == torch.float32
    assert compute.resolve_torch_runtime("auto", "float64") is None
    assert compute.resolve_torch_runtime("torch", "float64").device.type == "cpu"
    monkeypatch.setattr(compute, "_supports_compute", lambda *args: False)
    assert compute.resolve_torch_runtime("auto", "float32") is None


def test_float32_rt_coherence_can_select_mps(monkeypatch):
    torch = pytest.importorskip("torch")
    from mmWaveRadar.simulation import rt_coherence

    monkeypatch.setattr(compute, "torch_cuda_is_available", lambda module: False)
    monkeypatch.setattr(compute, "torch_mps_is_available", lambda module: True)
    monkeypatch.setattr(
        compute, "_supports_compute", lambda module, device, precision: precision == "float32"
    )
    selected = []
    result = object()

    def capture(bank, vertices, *, frequency_hz, runtime):
        selected.append(runtime)
        return result

    monkeypatch.setattr(rt_coherence, "_update_coherent_rt_path_bank_torch", capture)
    assert (
        rt_coherence.update_coherent_rt_path_bank(
            object(), [], backend="auto", precision="float32"
        )
        is result
    )
    assert selected[0].device.type == "mps"
    assert selected[0].float_dtype == torch.float32
    assert selected[0].complex_dtype == torch.complex64


@pytest.mark.parametrize("override", ["bad-device", "cuda:99", "mps"])
def test_unusable_device_override_reports_error(monkeypatch, override):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", override)

    def unavailable(*args, **kwargs):
        raise RuntimeError("device is unavailable")

    monkeypatch.setattr(torch, "empty", unavailable)
    with pytest.raises(ValueError, match="MMWAVE_TORCH_DEVICE"):
        compute.resolve_torch_runtime("auto", "float32")


def test_mps_override_rejects_float64(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "mps")
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="complex float64"):
        compute.resolve_torch_runtime("auto", "float64")


@pytest.mark.parametrize(
    "backend,precision,match",
    [
        ("invalid", "float32", "backend"),
        ("auto", "invalid", "precision"),
        ("numpy", "invalid", "precision"),
    ],
)
def test_invalid_settings_are_rejected_before_device_selection(backend, precision, match):
    with pytest.raises(ValueError, match=match):
        compute.resolve_torch_runtime(backend, precision)


@pytest.mark.parametrize(
    "settings",
    [
        {"compute_backend": "invalid"},
        {"adc_compute_backend": "invalid"},
        {"compute_precision": "invalid"},
        {"adc_compute_precision": "invalid"},
    ],
)
def test_simulator_rejects_invalid_compute_settings_at_construction(settings):
    from mmWaveRadar.simulation import MmWaveRadarSimulator

    with pytest.raises(ValueError, match="backend|precision"):
        MmWaveRadarSimulator(**settings)


def test_runtime_summary_reports_resolved_compute_policy(monkeypatch):
    pytest.importorskip("torch")
    from mmWaveRadar.simulation.runtime import torch_runtime_summary

    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")
    summary = torch_runtime_summary()
    assert summary["torch_device"] == "cpu"
    assert summary["compute_float32"] == {"backend": "numpy", "device": "cpu"}
    assert summary["compute_float64"] == {"backend": "numpy", "device": "cpu"}
