# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Shared NumPy/Torch backend, device, capability, and precision policy."""

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any, Literal
import warnings

import numpy as np

ComputeBackend = Literal["auto", "numpy", "torch"]
ComputePrecision = Literal["float64", "float32"]


def validate_compute_backend(backend, *, name="backend") -> ComputeBackend:
    if backend not in ("auto", "numpy", "torch"):
        raise ValueError(f"{name} must be 'auto', 'numpy', or 'torch'")
    return backend


def numpy_dtypes_for_precision(precision: ComputePrecision) -> tuple[np.dtype, np.dtype]:
    if precision == "float64":
        return np.dtype(np.float64), np.dtype(np.complex128)
    if precision == "float32":
        return np.dtype(np.float32), np.dtype(np.complex64)
    raise ValueError("precision must be 'float64' or 'float32'")


def torch_cuda_is_available(torch) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return bool(torch.cuda.is_available())
        except (AssertionError, RuntimeError):
            return False


def torch_mps_is_available(torch) -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return bool(backend.is_available())
        except (AssertionError, RuntimeError):
            return False


def torch_vector_cross(a, b, *, dim=-1):
    """Cross product with an elementwise complex implementation for MPS."""
    import torch

    if a.device.type == "mps" and (a.is_complex() or b.is_complex()):
        ax, ay, az = a.unbind(dim)
        bx, by, bz = b.unbind(dim)
        return torch.stack((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), dim=dim)
    return torch.linalg.cross(a, b, dim=dim)


@lru_cache(maxsize=32)
def _supports_compute(torch, device: str, precision: ComputePrecision) -> bool:
    """Probe the complex operations shared by ADC, PO, and coherent RT once."""
    kind = torch.device(device).type
    if kind == "cpu":
        return True
    if kind == "mps" and precision == "float64":
        return False
    cdtype = torch.complex64 if precision == "float32" else torch.complex128
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = torch.ones((2, 3), dtype=cdtype, device=device)
            torch.exp(probe)
            probe @ probe.T
            torch_vector_cross(probe, probe, dim=1)
            probe.sum(dim=1)
            real = probe.real.contiguous()
            torch.sinc(real)
            indices = torch.tensor([0, 1, 1], dtype=torch.int64, device=device)
            torch.zeros_like(real).index_add_(1, indices, real)
            mask = real > 0.0
            probe[mask] = probe[mask]
    except (AssertionError, RuntimeError, TypeError, NotImplementedError):
        return False
    return True


def _select_torch_device(torch, *, prefer_gpu=True, prefer_mps=True, precision=None) -> str:
    requested = os.environ.get("MMWAVE_TORCH_DEVICE")
    if requested:
        try:
            device = torch.device(requested)
            if device.type not in ("cpu", "cuda", "mps"):
                raise ValueError("supported devices are cpu, cuda, and mps")
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            torch.empty((1,), device=device)
            if precision is not None and not _supports_compute(torch, str(device), precision):
                raise ValueError(f"device does not support complex {precision} computation")
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"MMWAVE_TORCH_DEVICE={requested!r} is unusable: {exc}") from exc
        return str(device)
    if prefer_gpu and torch_cuda_is_available(torch):
        device = f"cuda:{torch.cuda.current_device()}"
        if precision is None or _supports_compute(torch, device, precision):
            return device
    if prefer_mps and torch_mps_is_available(torch):
        if precision is None or _supports_compute(torch, "mps", precision):
            return "mps"
    return "cpu"


def select_torch_device(*, prefer_gpu: bool = True, prefer_mps: bool = True) -> str:
    """Select a preprocessing device, honoring MMWAVE_TORCH_DEVICE."""
    try:
        import torch
    except ImportError:
        return "cpu"
    device = _select_torch_device(torch, prefer_gpu=prefer_gpu, prefer_mps=prefer_mps)
    # Preserve the public default-device spelling used by preprocessing callers.
    return (
        "cuda"
        if device.startswith("cuda:") and not os.environ.get("MMWAVE_TORCH_DEVICE")
        else device
    )


@dataclass(frozen=True)
class TorchRuntime:
    """One resolved device and its supported dtypes, passed into a kernel."""

    device: Any
    float_dtype: Any
    complex_dtype: Any
    phase_dtype: Any
    numpy_float_dtype: np.dtype
    numpy_complex_dtype: np.dtype


def resolve_torch_runtime(
    backend: ComputeBackend, precision: ComputePrecision
) -> TorchRuntime | None:
    """Return a Torch runtime or None for NumPy, without changing user settings.

    auto selects a supported CUDA/MPS device and otherwise uses NumPy. torch
    requires PyTorch and uses its CPU implementation if no accelerator supports
    the requested precision. numpy never imports Torch or inspects devices.
    An explicit MMWAVE_TORCH_DEVICE applies to every compute kernel.
    """
    validate_compute_backend(backend)
    float_dtype, complex_dtype = numpy_dtypes_for_precision(precision)
    if backend == "numpy":
        return None
    try:
        import torch
    except ImportError:
        if backend == "torch":
            raise
        return None
    device = torch.device(_select_torch_device(torch, precision=precision))
    if backend == "auto" and device.type == "cpu":
        return None
    return TorchRuntime(
        device=device,
        float_dtype=torch.float64 if precision == "float64" else torch.float32,
        complex_dtype=torch.complex128 if precision == "float64" else torch.complex64,
        phase_dtype=torch.float32 if device.type == "mps" else torch.float64,
        numpy_float_dtype=float_dtype,
        numpy_complex_dtype=complex_dtype,
    )
