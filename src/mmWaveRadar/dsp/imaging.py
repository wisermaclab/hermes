# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Simple PSF-based radar imaging helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Literal

import numpy as np

from .array_processing import _as_positions_lambda, _direction_axes, steering_matrix


PSFNormalization = Literal["sum", "peak", "none"]
SnapshotMode = Literal["mean_power", "coherent"]


def analytical_array_psf(
    virtual_positions_lambda: np.ndarray,
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    grid_size: int | Iterable[int] = (64, 128),
    channel_weights: np.ndarray | None = None,
    normalization: PSFNormalization = "sum",
) -> Dict[str, Any]:
    """
    Builds an analytical delay-and-sum point-spread function for an array.

    The PSF is evaluated on the direction-cosine lag grid implied by ``u`` and
    optional ``v``. It is centered in the returned array; use ``np.fft.ifftshift``
    before FFT-based convolution/deconvolution.
    """

    positions = _as_positions_lambda(virtual_positions_lambda)
    u_axis, v_axis = _direction_axes(positions, u=u, v=v, grid_size=grid_size)
    u_lag = _centered_lag_axis(u_axis)
    weights = _channel_weights(channel_weights, positions.shape[0])

    if v_axis is None:
        phase = positions[:, 0, None] * u_lag[None, :]
        response = np.sum(weights[:, None] * np.exp(-1j * 2.0 * np.pi * phase), axis=0)
        psf = np.abs(response) ** 2
    else:
        v_lag = _centered_lag_axis(v_axis)
        uu, vv = np.meshgrid(u_lag, v_lag, indexing="xy")
        phase = (
            positions[:, 0, None, None] * uu[None, :, :]
            + positions[:, 1, None, None] * vv[None, :, :]
        )
        response = np.sum(
            weights[:, None, None] * np.exp(-1j * 2.0 * np.pi * phase),
            axis=0,
        )
        psf = np.abs(response) ** 2

    psf = _normalize_psf(psf, normalization)
    result: Dict[str, Any] = {
        "psf": psf,
        "u": np.asarray(u_axis, dtype=float),
        "u_lag": u_lag,
        "normalization": normalization,
    }
    if v_axis is not None:
        result["v"] = np.asarray(v_axis, dtype=float)
        result["v_lag"] = v_lag
        result["valid_mask"] = _valid_direction_mask(u_axis, v_axis)
    return result


def wiener_deconvolution(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    balance: float = 1e-2,
    clip_negative: bool = True,
    eps: float = 1e-18,
) -> Dict[str, Any]:
    """
    Deconvolves ``image`` by a centered PSF using a Wiener filter.

    ``balance`` is the noise-to-signal regularization term in
    ``conj(H) / (|H|^2 + balance)``. Larger values suppress ringing at the cost
    of resolution.
    """

    dirty = np.asarray(image, dtype=float)
    kernel = np.asarray(psf, dtype=float)
    if dirty.shape != kernel.shape:
        raise ValueError("image and psf must have the same shape")
    if dirty.ndim not in (1, 2):
        raise ValueError("image must be 1D or 2D")
    if balance < 0.0:
        raise ValueError("balance must be nonnegative")

    axes = tuple(range(dirty.ndim))
    transfer = np.fft.fftn(np.fft.ifftshift(kernel), s=dirty.shape, axes=axes)
    spectrum = np.fft.fftn(dirty, axes=axes)
    denominator = np.abs(transfer) ** 2 + float(balance) + float(eps)
    estimate = np.fft.ifftn(
        spectrum * np.conj(transfer) / denominator,
        axes=axes,
    )
    restored = np.real(estimate)
    if clip_negative:
        restored = np.maximum(restored, 0.0)

    return {
        "image": restored,
        "dirty_image": dirty,
        "psf": kernel,
        "transfer_function": transfer,
        "balance": float(balance),
    }


def psf_cross_correlation(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    clip_negative: bool = True,
) -> Dict[str, Any]:
    """
    Cross-correlates ``image`` with a centered analytical PSF.

    This is a matched-filter style image: it favors locations whose local shape
    resembles the PSF. Unlike Wiener deconvolution, it does not try to invert the
    PSF and therefore tends to be stable but broader.
    """

    dirty = np.asarray(image, dtype=float)
    kernel = np.asarray(psf, dtype=float)
    if dirty.shape != kernel.shape:
        raise ValueError("image and psf must have the same shape")
    if dirty.ndim not in (1, 2):
        raise ValueError("image must be 1D or 2D")

    axes = tuple(range(dirty.ndim))
    transfer = np.fft.fftn(np.fft.ifftshift(kernel), s=dirty.shape, axes=axes)
    spectrum = np.fft.fftn(dirty, axes=axes)
    estimate = np.fft.ifftn(spectrum * np.conj(transfer), axes=axes)
    correlated = np.real(estimate)
    if clip_negative:
        correlated = np.maximum(correlated, 0.0)

    return {
        "image": correlated,
        "dirty_image": dirty,
        "psf": kernel,
        "transfer_function": transfer,
    }


def backprojection(
    snapshots: np.ndarray,
    virtual_positions_lambda: np.ndarray,
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    grid_size: int | Iterable[int] = (64, 128),
    channel_weights: np.ndarray | None = None,
    snapshot_mode: SnapshotMode = "mean_power",
    mask_invalid_directions: bool = True,
) -> Dict[str, Any]:
    """
    Forms a delay-and-sum backprojection image in direction-cosine space.

    ``snapshots`` is interpreted as ``[..., num_virtual_channels]``. The output
    image is evaluated over direction-cosine axes ``u`` and optional ``v``.
    """

    positions = _as_positions_lambda(virtual_positions_lambda)
    u_axis, v_axis = _direction_axes(positions, u=u, v=v, grid_size=grid_size)
    x = _snapshot_matrix(snapshots, positions.shape[0])
    weights = _channel_weights(channel_weights, positions.shape[0])
    if channel_weights is not None:
        x = x * weights[None, :]

    steering = steering_matrix(positions, u_axis, v_axis, normalize=True)
    steering_flat, out_shape = _flatten_steering(steering)
    response = x @ steering_flat.conj()
    if snapshot_mode == "mean_power":
        image = np.mean(np.abs(response) ** 2, axis=0).reshape(out_shape)
    elif snapshot_mode == "coherent":
        coherent = np.mean(response, axis=0)
        image = (np.abs(coherent) ** 2).reshape(out_shape)
    else:
        raise ValueError("snapshot_mode must be 'mean_power' or 'coherent'")

    valid_mask = None
    if v_axis is not None:
        valid_mask = _valid_direction_mask(u_axis, v_axis)
        if mask_invalid_directions:
            image = np.where(valid_mask, image, 0.0)

    result: Dict[str, Any] = {
        "image": np.asarray(image, dtype=float),
        "u": np.asarray(u_axis, dtype=float),
        "snapshot_mode": snapshot_mode,
    }
    if v_axis is not None:
        result["v"] = np.asarray(v_axis, dtype=float)
        result["valid_mask"] = valid_mask
    peak_index = _peak_index(image, valid_mask if mask_invalid_directions else None)
    result["peak_index"] = peak_index
    if peak_index is not None:
        if v_axis is None:
            result["peak_u"] = float(u_axis[peak_index[0]])
        else:
            result["peak_u"] = float(u_axis[peak_index[1]])
            result["peak_v"] = float(v_axis[peak_index[0]])
    return result


def wiener_psf_image(
    snapshots: np.ndarray,
    virtual_positions_lambda: np.ndarray,
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    grid_size: int | Iterable[int] = (64, 128),
    channel_weights: np.ndarray | None = None,
    snapshot_mode: SnapshotMode = "mean_power",
    psf_normalization: PSFNormalization = "sum",
    wiener_balance: float = 1e-2,
    clip_negative: bool = True,
    mask_invalid_directions: bool = True,
) -> Dict[str, Any]:
    """
    Forms a dirty angle image and Wiener-deconvolves it with an analytical PSF.

    ``snapshots`` is interpreted as ``[..., num_virtual_channels]``. The returned
    image is in direction-cosine coordinates ``u`` and optional ``v``.
    """

    positions = _as_positions_lambda(virtual_positions_lambda)
    u_axis, v_axis = _direction_axes(positions, u=u, v=v, grid_size=grid_size)
    x = _snapshot_matrix(snapshots, positions.shape[0])
    weights = _channel_weights(channel_weights, positions.shape[0])
    if channel_weights is not None:
        x = x * weights[None, :]

    steering = steering_matrix(positions, u_axis, v_axis, normalize=True)
    steering_flat, out_shape = _flatten_steering(steering)
    response = x @ steering_flat.conj()
    if snapshot_mode == "mean_power":
        dirty = np.mean(np.abs(response) ** 2, axis=0).reshape(out_shape)
    elif snapshot_mode == "coherent":
        coherent = np.mean(response, axis=0)
        dirty = (np.abs(coherent) ** 2).reshape(out_shape)
    else:
        raise ValueError("snapshot_mode must be 'mean_power' or 'coherent'")

    psf_result = analytical_array_psf(
        positions,
        u=u_axis,
        v=v_axis,
        channel_weights=weights,
        normalization=psf_normalization,
    )
    psf = np.asarray(psf_result["psf"], dtype=float)
    valid_mask = psf_result.get("valid_mask")
    dirty_for_deconv = np.asarray(dirty, dtype=float)
    if mask_invalid_directions and valid_mask is not None:
        dirty_for_deconv = np.where(valid_mask, dirty_for_deconv, 0.0)

    deconv = wiener_deconvolution(
        dirty_for_deconv,
        psf,
        balance=wiener_balance,
        clip_negative=clip_negative,
    )
    restored = np.asarray(deconv["image"], dtype=float)
    if mask_invalid_directions and valid_mask is not None:
        restored = np.where(valid_mask, restored, 0.0)

    result: Dict[str, Any] = {
        "image": restored,
        "dirty_image": dirty,
        "psf": psf,
        "u": np.asarray(u_axis, dtype=float),
        "psf_result": psf_result,
        "deconvolution": deconv,
        "snapshot_mode": snapshot_mode,
    }
    if v_axis is not None:
        result["v"] = np.asarray(v_axis, dtype=float)
        result["valid_mask"] = valid_mask
    peak_index = _peak_index(restored, valid_mask if mask_invalid_directions else None)
    result["peak_index"] = peak_index
    if peak_index is not None:
        if v_axis is None:
            result["peak_u"] = float(u_axis[peak_index[0]])
        else:
            result["peak_u"] = float(u_axis[peak_index[1]])
            result["peak_v"] = float(v_axis[peak_index[0]])
    return result


def psf_cross_correlation_image(
    snapshots: np.ndarray,
    virtual_positions_lambda: np.ndarray,
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    grid_size: int | Iterable[int] = (64, 128),
    channel_weights: np.ndarray | None = None,
    snapshot_mode: SnapshotMode = "mean_power",
    psf_normalization: PSFNormalization = "sum",
    clip_negative: bool = True,
    mask_invalid_directions: bool = True,
) -> Dict[str, Any]:
    """
    Forms a dirty angle image and cross-correlates it with an analytical PSF.

    This is a simple matched-filter imaging pass in direction-cosine space. It
    is usually best interpreted as a robust localization/sanity-check image,
    not a resolution-enhancing deconvolution.
    """

    positions = _as_positions_lambda(virtual_positions_lambda)
    u_axis, v_axis = _direction_axes(positions, u=u, v=v, grid_size=grid_size)
    x = _snapshot_matrix(snapshots, positions.shape[0])
    weights = _channel_weights(channel_weights, positions.shape[0])
    if channel_weights is not None:
        x = x * weights[None, :]

    steering = steering_matrix(positions, u_axis, v_axis, normalize=True)
    steering_flat, out_shape = _flatten_steering(steering)
    response = x @ steering_flat.conj()
    if snapshot_mode == "mean_power":
        dirty = np.mean(np.abs(response) ** 2, axis=0).reshape(out_shape)
    elif snapshot_mode == "coherent":
        coherent = np.mean(response, axis=0)
        dirty = (np.abs(coherent) ** 2).reshape(out_shape)
    else:
        raise ValueError("snapshot_mode must be 'mean_power' or 'coherent'")

    psf_result = analytical_array_psf(
        positions,
        u=u_axis,
        v=v_axis,
        channel_weights=weights,
        normalization=psf_normalization,
    )
    psf = np.asarray(psf_result["psf"], dtype=float)
    valid_mask = psf_result.get("valid_mask")
    dirty_for_correlation = np.asarray(dirty, dtype=float)
    if mask_invalid_directions and valid_mask is not None:
        dirty_for_correlation = np.where(valid_mask, dirty_for_correlation, 0.0)

    correlation = psf_cross_correlation(
        dirty_for_correlation,
        psf,
        clip_negative=clip_negative,
    )
    matched = np.asarray(correlation["image"], dtype=float)
    if mask_invalid_directions and valid_mask is not None:
        matched = np.where(valid_mask, matched, 0.0)

    result: Dict[str, Any] = {
        "image": matched,
        "dirty_image": dirty,
        "psf": psf,
        "u": np.asarray(u_axis, dtype=float),
        "psf_result": psf_result,
        "correlation": correlation,
        "snapshot_mode": snapshot_mode,
    }
    if v_axis is not None:
        result["v"] = np.asarray(v_axis, dtype=float)
        result["valid_mask"] = valid_mask
    peak_index = _peak_index(matched, valid_mask if mask_invalid_directions else None)
    result["peak_index"] = peak_index
    if peak_index is not None:
        if v_axis is None:
            result["peak_u"] = float(u_axis[peak_index[0]])
        else:
            result["peak_u"] = float(u_axis[peak_index[1]])
            result["peak_v"] = float(v_axis[peak_index[0]])
    return result


def _snapshot_matrix(snapshots: np.ndarray, num_channels: int) -> np.ndarray:
    """Flattens snapshot axes into rows and validates channel count."""

    x = np.asarray(snapshots, dtype=np.complex128)
    if x.ndim == 0:
        raise ValueError("snapshots must include a virtual-channel axis")
    if x.shape[-1] != num_channels:
        raise ValueError("snapshots last axis must match virtual positions")
    return x.reshape((-1, num_channels))


def _flatten_steering(steering: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Flattens steering image axes into columns for matrix products."""

    return steering.reshape((steering.shape[0], -1)), steering.shape[1:]


def _channel_weights(weights: np.ndarray | None, num_channels: int) -> np.ndarray:
    """Returns complex per-channel weights, defaulting to all ones."""

    if weights is None:
        return np.ones(num_channels, dtype=np.complex128)
    out = np.asarray(weights, dtype=np.complex128).reshape(-1)
    if out.shape != (num_channels,):
        raise ValueError("channel_weights must have one value per virtual channel")
    return out


def _centered_lag_axis(axis: np.ndarray) -> np.ndarray:
    """Converts a uniformly spaced direction axis to centered PSF lags."""

    values = np.asarray(axis, dtype=float).reshape(-1)
    if values.size < 1:
        raise ValueError("direction axes must be nonempty")
    if values.size == 1:
        return np.zeros(1, dtype=float)
    diffs = np.diff(values)
    spacing = float(np.mean(diffs))
    if spacing <= 0.0 or not np.allclose(diffs, spacing, rtol=1e-5, atol=1e-8):
        raise ValueError("direction axes must be uniformly spaced for PSF deconvolution")
    center = values.size // 2
    return (np.arange(values.size, dtype=float) - float(center)) * spacing


def _normalize_psf(psf: np.ndarray, normalization: PSFNormalization) -> np.ndarray:
    """Normalizes a PSF by sum, peak, or leaves it unchanged."""

    out = np.asarray(psf, dtype=float)
    if normalization == "sum":
        total = float(np.sum(out))
        return out / total if total > 0.0 else out
    if normalization == "peak":
        peak = float(np.max(out))
        return out / peak if peak > 0.0 else out
    if normalization == "none":
        return out
    raise ValueError("normalization must be 'sum', 'peak', or 'none'")


def _valid_direction_mask(u_axis: np.ndarray, v_axis: np.ndarray) -> np.ndarray:
    """Returns the mask of physically valid 2D direction-cosine samples."""

    u = np.asarray(u_axis, dtype=float)
    v = np.asarray(v_axis, dtype=float)
    return (v[:, None] * v[:, None] + u[None, :] * u[None, :]) <= 1.0 + 1e-12


def _peak_index(image: np.ndarray, valid_mask: np.ndarray | None) -> tuple[int, ...] | None:
    """Returns the strongest finite image index, optionally constrained by a mask."""

    values = np.asarray(image, dtype=float)
    if values.size == 0:
        return None
    if valid_mask is not None:
        masked = np.where(valid_mask, values, -np.inf)
    else:
        masked = values
    if not np.any(np.isfinite(masked)):
        return None
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(masked)), values.shape))


__all__ = [
    "analytical_array_psf",
    "backprojection",
    "psf_cross_correlation",
    "psf_cross_correlation_image",
    "wiener_deconvolution",
    "wiener_psf_image",
]
