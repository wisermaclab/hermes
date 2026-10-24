# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Runtime configuration helpers for radar sensing simulations."""

from __future__ import annotations

import os
import platform
import warnings
from typing import Any

import mitsuba as mi

from .compute import (
    resolve_torch_runtime,
    select_torch_device,
    torch_cuda_is_available,
    torch_mps_is_available,
)


def configure_mitsuba_variant(*, prefer_gpu: bool = True) -> str:
    """
    Ensures that a Mitsuba variant is selected.

    The optional ``MMWAVE_MITSUBA_VARIANT`` environment variable can force a
    specific variant. Otherwise, CUDA is preferred when available and LLVM is
    used as the CPU fallback, which works on both x86 and ARM hosts.
    """

    current = mi.variant()
    if current is not None:
        return current

    requested = os.environ.get("MMWAVE_MITSUBA_VARIANT")
    if requested:
        mi.set_variant(requested)
        return mi.variant()

    if prefer_gpu:
        try:
            mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
            return mi.variant()
        except (ImportError, RuntimeError):
            pass

    mi.set_variant("llvm_ad_mono_polarized")
    return mi.variant()


def _torch_cuda_device_count(torch) -> int:
    """Returns the Torch CUDA device count, suppressing backend probe failures."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return int(torch.cuda.device_count())
        except (AssertionError, RuntimeError):
            return 0


def torch_runtime_summary() -> dict[str, Any]:
    """Returns Torch accelerator visibility details for logs and notebooks."""

    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        return {
            "torch_available": False,
            "torch_device": "cpu",
        }

    cuda_available = torch_cuda_is_available(torch)
    cuda_device_count = _torch_cuda_device_count(torch)
    summary: dict[str, Any] = {
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_device": select_torch_device(),
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": cuda_device_count,
        "torch_mps_available": torch_mps_is_available(torch),
    }
    for precision in ("float32", "float64"):
        runtime = resolve_torch_runtime("auto", precision)
        summary[f"compute_{precision}"] = {
            "backend": "numpy" if runtime is None else "torch",
            "device": "cpu" if runtime is None else str(runtime.device),
        }
    requested_device = os.environ.get("MMWAVE_TORCH_DEVICE")
    if requested_device:
        summary["torch_device_requested"] = requested_device
    if summary["torch_cuda_device_count"] > 0:
        try:
            summary["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
        except RuntimeError as exc:
            summary["torch_cuda_device_name"] = f"unavailable: {exc}"
    return summary


def runtime_summary() -> dict[str, Any]:
    """Returns a compact runtime summary for logs and notebooks."""

    variant = configure_mitsuba_variant()
    summary: dict[str, Any] = {
        "mitsuba_variant": variant,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "python": platform.python_version(),
    }
    summary.update(torch_runtime_summary())
    return summary


__all__ = [
    "configure_mitsuba_variant",
    "runtime_summary",
    "select_torch_device",
    "torch_runtime_summary",
]
