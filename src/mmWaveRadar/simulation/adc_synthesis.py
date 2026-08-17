# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""ADC synthesis helpers for FMCW radar path arrays.

The functions in this module turn geometric/path-domain channel coefficients
into one dechirped FMCW fast-time record.  They are intentionally independent
from Sionna path objects so callers can reuse them with RT, PO, hybrid, or
synthetic point-target path banks that expose only coefficient and delay arrays.
"""

from __future__ import annotations

from functools import lru_cache
import warnings

import numpy as np

from .physical_optics import ComputeBackend, ComputePrecision


def synthesize_adc_from_paths(a: np.ndarray, tau: np.ndarray,
                              fmcw, *,
                              backend: ComputeBackend = "numpy",
                              precision: ComputePrecision = "float64") -> np.ndarray:
    r"""
    Synthesizes one chirp of dechirped ADC samples from path arrays.

    The returned samples use the baseband beat model
    ``a * exp(j 2 pi slope tau t_fast)`` and are summed over valid paths for
    each virtual channel. Negative delays are treated as invalid path slots and
    omitted from the sum.

    :param a: Complex path coefficients with shape
        ``[num_virtual_channels, num_paths]``. Each row is one virtual channel
        in the caller's chosen ordering.
    :param tau: Path delays in seconds. A 1-D array with shape ``[num_paths]``
        is shared by every virtual channel; a 2-D array with shape
        ``[num_virtual_channels, num_paths]`` permits channel-specific delays.
    :param fmcw: FMCW configuration object exposing ``num_adc_samples``,
        ``sampling_frequency``, and ``slope`` attributes.
    :param backend: ``"numpy"`` for CPU NumPy, ``"torch"`` for PyTorch, or
        ``"auto"`` to use a supported accelerated torch device when available.
    :param precision: Floating/complex precision used internally and for the
        returned array. ``"float64"`` returns ``complex128`` samples;
        ``"float32"`` returns ``complex64`` samples.
    :returns: Complex ADC matrix with shape
        ``[num_adc_samples, num_virtual_channels]``.
    """
    a = np.asarray(a, dtype=np.complex128)
    tau = np.asarray(tau)
    if np.issubdtype(tau.dtype, np.complexfloating):
        raise ValueError("tau must contain real-valued delays")
    if not np.issubdtype(tau.dtype, np.number):
        try:
            tau = np.asarray(tau, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("tau must contain real-valued delays") from exc
    else:
        tau = np.asarray(tau, dtype=float)
    if a.ndim != 2:
        raise ValueError("a must have shape [num_virtual_channels,"
                         " num_paths]")
    if tau.ndim == 1:
        valid_tau_shape = tau.shape == (a.shape[1],)
    elif tau.ndim == 2:
        valid_tau_shape = tau.shape == a.shape
    else:
        valid_tau_shape = False
    if not valid_tau_shape:
        raise ValueError(
            "tau must have shape [num_paths] or"
            " [num_virtual_channels, num_paths]"
        )
    if not np.all(np.isfinite(tau)):
        raise ValueError("tau must contain only finite delays")
    valid = tau >= 0.0
    coefficient_mask = (
        np.broadcast_to(valid[None, :], a.shape)
        if valid.ndim == 1
        else valid
    )
    finite_coefficients = np.isfinite(a.real) & np.isfinite(a.imag)
    if np.any(coefficient_mask & ~finite_coefficients):
        raise ValueError("a must contain finite coefficients for valid paths")
    return _synthesize_adc(
        a, tau, valid, fmcw, backend=backend, precision=precision)


def _synthesize_adc(a: np.ndarray, tau: np.ndarray, valid: np.ndarray,
                    fmcw, *, backend: ComputeBackend = "numpy",
                    precision: ComputePrecision = "float64") -> np.ndarray:
    """Synthesizes one ADC frame from channel/path coefficients and delays."""

    float_dtype, complex_dtype = (
        _numpy_dtypes_for_precision(precision)
    )
    if backend not in ("auto", "numpy", "torch"):
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    num_adc = fmcw.num_adc_samples
    num_vc = a.shape[0]
    out = np.zeros((num_adc, num_vc), dtype=complex_dtype)
    valid = np.asarray(valid, dtype=bool)
    if tau.size == 0 or not np.any(valid):
        return out

    if _use_torch_backend(backend, precision):
        return _synthesize_adc_torch(
            a, tau, valid, fmcw, precision=precision)

    t_fast = np.arange(num_adc, dtype=float_dtype) / fmcw.sampling_frequency
    if tau.ndim == 1:
        # The common Sionna CIR layout has one delay per path, shared across
        # all virtual channels.  Chunk by path so dense PO path banks do not
        # materialize a giant [path, sample] beat matrix at once.
        valid_indices = np.flatnonzero(valid)
        accum = np.zeros((num_vc, num_adc), dtype=complex_dtype)
        max_elements = 8_000_000
        chunk_size = max(1, max_elements // max(num_adc, 1))
        scale = 2.0 * np.pi * fmcw.slope
        a_all = np.asarray(a, dtype=complex_dtype)
        tau_all = np.asarray(tau, dtype=float_dtype)
        for start in range(0, valid_indices.size, chunk_size):
            stop = min(start + chunk_size, valid_indices.size)
            chunk = valid_indices[start:stop]
            tau_v = tau_all[chunk]
            a_v = a_all[:, chunk]
            beat = np.exp(1j * scale
                          * tau_v[:, None]
                          * t_fast[None, :]).astype(complex_dtype)
            accum += a_v @ beat
        out[:, :] = accum.T.astype(complex_dtype)
        return out

    # Channel-specific delays appear in PO/coupling paths where phase centers
    # can differ per virtual channel.  Mask and sum each channel independently.
    tau_m = np.asarray(tau, dtype=float_dtype)
    a_m = np.asarray(a, dtype=complex_dtype)
    valid_m = valid
    if valid_m.ndim == 1:
        if valid_m.shape != (a.shape[1],):
            raise ValueError("1-D valid mask must have shape [num_paths]")
        valid_m = np.broadcast_to(valid_m[None, :], tau_m.shape)
    elif valid_m.shape != tau_m.shape:
        raise ValueError(
            "valid must have shape [num_paths] or "
            "[num_virtual_channels, num_paths]"
        )
    for channel_index in range(num_vc):
        channel_valid = valid_m[channel_index]
        if not np.any(channel_valid):
            continue
        tau_v = tau_m[channel_index, channel_valid]
        a_v = a_m[channel_index, channel_valid]
        beat = np.exp(1j * 2.0 * np.pi * fmcw.slope
                      * tau_v[:, None]
                      * t_fast[None, :]).astype(complex_dtype)
        out[:, channel_index] = (a_v @ beat).astype(complex_dtype)
    return out


def _numpy_dtypes_for_precision(
    precision: ComputePrecision,
) -> tuple[np.dtype, np.dtype]:
    """Maps compute precision names to NumPy real and complex dtypes."""

    if precision == "float64":
        return np.dtype(np.float64), np.dtype(np.complex128)
    if precision == "float32":
        return np.dtype(np.float32), np.dtype(np.complex64)
    raise ValueError("precision must be 'float64' or 'float32'")


def numpy_dtypes_for_precision(
    precision: ComputePrecision,
) -> tuple[np.dtype, np.dtype]:
    """Returns the NumPy float and complex dtypes for ``precision``.

    This is the public counterpart to the internal dtype helper used by the
    simulator.  It lets callers allocate ADC, coefficient, or profiling arrays
    that match the precision chosen for path and ADC synthesis.
    """

    return _numpy_dtypes_for_precision(precision)


def _torch_cuda_is_available(torch) -> bool:
    """Returns whether CUDA can be queried and used without raising."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return bool(torch.cuda.is_available())
        except (AssertionError, RuntimeError):
            return False


def _use_torch_backend(
    backend: ComputeBackend,
    precision: ComputePrecision = "float64",
) -> bool:
    """Decides whether ADC synthesis should run on a Torch device."""

    if backend == "numpy":
        _numpy_dtypes_for_precision(precision)
        return False
    if backend not in ("auto", "torch"):
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        if backend == "torch":
            raise
        return False
    _numpy_dtypes_for_precision(precision)
    if backend == "torch":
        return True
    if _torch_cuda_is_available(torch):
        return True
    # Apple's MPS backend is only selected after probing complex64 operations;
    # unsupported precision/device combinations fall back to NumPy in auto mode.
    return _torch_mps_available_for_precision(
        torch,
        precision,
    )


def _torch_mps_available_for_precision(
    torch,
    precision: ComputePrecision,
) -> bool:
    """Checks whether Apple MPS supports the complex ops needed here."""

    if precision != "float32":
        return False
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        return False
    try:
        probe = torch.ones((2, 2), dtype=torch.complex64, device="mps")
        torch.exp(probe)
        probe @ probe
    except (RuntimeError, TypeError, NotImplementedError):
        return False
    return True


def _torch_device_and_dtypes(precision: ComputePrecision):
    """Selects a Torch device and matching Torch/NumPy dtypes."""

    import torch  # pylint: disable=import-outside-toplevel

    float_dtype, complex_dtype = (
        _numpy_dtypes_for_precision(precision)
    )
    dtype = torch.float64 if precision == "float64" else torch.float32
    cdtype = torch.complex128 if precision == "float64" else torch.complex64
    if _torch_cuda_is_available(torch):
        device = torch.device("cuda")
    elif _torch_mps_available_for_precision(
        torch,
        precision,
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device, dtype, cdtype, float_dtype, complex_dtype


@lru_cache(maxsize=32)
def _torch_fast_time(
    num_adc_samples: int,
    sampling_frequency: float,
    device,
    dtype,
):
    """Returns the immutable fast-time sample tensor for one FMCW shape."""

    import torch  # pylint: disable=import-outside-toplevel

    return (
        torch.arange(int(num_adc_samples), dtype=dtype, device=device)
        / float(sampling_frequency)
    )


def _synthesize_adc_torch(a: np.ndarray, tau: np.ndarray,
                          valid: np.ndarray, fmcw, *,
                          precision: ComputePrecision = "float64") -> np.ndarray:
    """Torch implementation of ADC synthesis for CUDA/MPS/CPU backends."""

    import torch  # pylint: disable=import-outside-toplevel

    device, dtype, cdtype, float_dtype, complex_dtype = (
        _torch_device_and_dtypes(precision)
    )
    t_fast = _torch_fast_time(
        int(fmcw.num_adc_samples),
        float(fmcw.sampling_frequency),
        device,
        dtype,
    )
    valid = np.asarray(valid, dtype=bool)
    max_elements = 8_000_000
    if tau.ndim == 1:
        valid_indices = np.flatnonzero(valid)
        num_channels = int(a.shape[0])
        num_adc = int(fmcw.num_adc_samples)
        chunk_size = max(1, max_elements // max(num_adc, 1))
        out = torch.zeros((num_channels, num_adc), dtype=cdtype, device=device)
        scale = 2.0 * np.pi * float(fmcw.slope)
        a_np = np.asarray(a, dtype=complex_dtype)
        tau_np = np.asarray(tau, dtype=float_dtype)
        for start in range(0, valid_indices.size, chunk_size):
            stop = min(start + chunk_size, valid_indices.size)
            chunk = valid_indices[start:stop]
            a_chunk = torch.as_tensor(
                a_np[:, chunk],
                dtype=cdtype,
                device=device,
            )
            tau_chunk = torch.as_tensor(
                tau_np[chunk],
                dtype=dtype,
                device=device,
            )
            phase = scale * tau_chunk[:, None] * t_fast[None, :]
            beat = torch.exp(1j * phase).to(cdtype)
            out += a_chunk @ beat
        return out.T.cpu().numpy().astype(complex_dtype)

    a_np = np.asarray(a, dtype=complex_dtype)
    tau_np = np.asarray(tau, dtype=float_dtype)
    if valid.ndim == 1:
        if valid.shape != (a_np.shape[1],):
            raise ValueError("1-D valid mask must have shape [num_paths]")
        valid_np = np.broadcast_to(valid[None, :], tau_np.shape)
    elif valid.shape != tau_np.shape:
        raise ValueError(
            "valid must have shape [num_paths] or "
            "[num_virtual_channels, num_paths]"
        )
    else:
        valid_np = valid

    num_channels, num_paths = a_np.shape
    num_adc = int(fmcw.num_adc_samples)
    # The channel-specific delay path can be enormous for refined PO meshes.
    # Chunk both channels and paths to cap the transient [channel, path, sample]
    # phase/beat tensors and avoid moving the full path bank to the device.
    channel_chunk_size = max(
        1,
        min(num_channels, max_elements // max(num_paths * num_adc, 1)),
    )
    path_chunk_size = max(
        1,
        max_elements // max(channel_chunk_size * num_adc, 1),
    )
    out = torch.zeros((num_adc, num_channels), dtype=cdtype, device=device)
    scale = 2.0 * np.pi * float(fmcw.slope)
    zero = torch.zeros((), dtype=cdtype, device=device)
    for c_start in range(0, num_channels, channel_chunk_size):
        c_stop = min(c_start + channel_chunk_size, num_channels)
        channel_out = torch.zeros(
            (num_adc, c_stop - c_start),
            dtype=cdtype,
            device=device,
        )
        for p_start in range(0, num_paths, path_chunk_size):
            p_stop = min(p_start + path_chunk_size, num_paths)
            a_chunk = torch.as_tensor(
                a_np[c_start:c_stop, p_start:p_stop],
                dtype=cdtype,
                device=device,
            )
            tau_chunk = torch.as_tensor(
                tau_np[c_start:c_stop, p_start:p_stop],
                dtype=dtype,
                device=device,
            )
            valid_chunk = torch.as_tensor(
                np.array(
                    valid_np[c_start:c_stop, p_start:p_stop],
                    dtype=bool,
                    copy=True,
                ),
                dtype=torch.bool,
                device=device,
            )
            phase = scale * tau_chunk[:, :, None] * t_fast[None, None, :]
            beat = torch.exp(1j * phase).to(cdtype)
            weighted = torch.where(
                valid_chunk[:, :, None],
                a_chunk[:, :, None] * beat,
                zero,
            )
            channel_out += torch.sum(weighted, dim=1).T
        out[:, c_start:c_stop] = channel_out
    return out.cpu().numpy().astype(complex_dtype)


__all__ = [
    "numpy_dtypes_for_precision",
    "synthesize_adc_from_paths",
]
