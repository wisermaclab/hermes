# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Subspace and adaptive array-processing helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np


def steering_matrix(
    virtual_positions_lambda: np.ndarray,
    u: np.ndarray,
    v: np.ndarray | None = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Builds far-field steering vectors for virtual positions in wavelengths.

    ``virtual_positions_lambda`` may have shape ``[num_channels]``,
    ``[num_channels, 1]``, or ``[num_channels, 2]``. The first coordinate is the
    horizontal aperture coordinate and the optional second coordinate is the
    vertical aperture coordinate. ``u`` and ``v`` are direction cosines.
    """

    positions = _as_positions_lambda(virtual_positions_lambda)
    u = np.asarray(u, dtype=float).reshape(-1)
    if v is None:
        phase = positions[:, 0, None] * u[None, :]
    else:
        v = np.asarray(v, dtype=float).reshape(-1)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        phase = (
            positions[:, 0, None, None] * uu[None, :, :]
            + positions[:, 1, None, None] * vv[None, :, :]
        )
    steering = np.exp(1j * 2.0 * np.pi * phase)
    if normalize:
        steering = steering / np.sqrt(float(positions.shape[0]))
    return steering


def array_covariance(
    snapshots: np.ndarray,
    *,
    diagonal_loading: float = 0.0,
) -> np.ndarray:
    """
    Estimates the spatial covariance matrix from snapshots.

    The last axis is interpreted as the virtual-channel axis; all leading axes
    are flattened into snapshots. ``diagonal_loading`` is relative to the mean
    covariance diagonal power.
    """

    x = _snapshot_matrix(snapshots)
    cov = (x.T @ x.conj()) / float(x.shape[0])
    return _apply_diagonal_loading(cov, diagonal_loading)


def music_pseudospectrum(
    snapshots: np.ndarray,
    virtual_positions_lambda: np.ndarray,
    *,
    num_sources: int = 1,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    grid_size: int | tuple[int, int] = (64, 128),
    diagonal_loading: float = 0.0,
    eps: float = 1e-18,
) -> Dict[str, Any]:
    """
    Computes a MUSIC pseudospectrum over a ULA or URA direction-cosine grid.

    ``snapshots`` must have shape ``[..., num_virtual_channels]``. For a 2D
    virtual array, pass ``virtual_positions_lambda`` as ``[horizontal, vertical]``
    coordinates in wavelengths. If ``v`` is omitted, a 2D grid is used only when
    the position table has nonzero vertical aperture.
    """

    positions = _as_positions_lambda(virtual_positions_lambda)
    u_axis, v_axis = _direction_axes(positions, u=u, v=v, grid_size=grid_size)
    cov = array_covariance(snapshots, diagonal_loading=diagonal_loading)
    _check_covariance_shape(cov, positions.shape[0])

    num_sources = int(num_sources)
    if num_sources < 1 or num_sources >= cov.shape[0]:
        raise ValueError("num_sources must be in [1, num_channels - 1]")

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    noise_subspace = eigenvectors[:, : cov.shape[0] - num_sources]
    steering = steering_matrix(positions, u_axis, v_axis, normalize=True)
    steering_flat, out_shape = _flatten_steering(steering)
    projected = noise_subspace.conj().T @ steering_flat
    denom = np.sum(np.abs(projected) ** 2, axis=0)
    spectrum = (1.0 / np.maximum(denom, float(eps))).reshape(out_shape)

    return _spectrum_result(
        spectrum,
        u_axis=u_axis,
        v_axis=v_axis,
        covariance=cov,
        eigenvalues=eigenvalues,
        steering=steering,
        extra={"noise_subspace": noise_subspace, "num_sources": num_sources},
    )


def mvdr_pseudospectrum(
    snapshots: np.ndarray,
    virtual_positions_lambda: np.ndarray,
    *,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    grid_size: int | tuple[int, int] = (64, 128),
    diagonal_loading: float = 1e-3,
    eps: float = 1e-18,
) -> Dict[str, Any]:
    """
    Computes an MVDR/Capon pseudospectrum over direction-cosine axes.

    ``diagonal_loading`` is relative to the mean covariance diagonal power. A
    small nonzero default is used because range-selected frame snapshots often
    produce rank-deficient covariance estimates.
    """

    positions = _as_positions_lambda(virtual_positions_lambda)
    u_axis, v_axis = _direction_axes(positions, u=u, v=v, grid_size=grid_size)
    x = _snapshot_matrix(snapshots)
    cov_unloaded = (x.T @ x.conj()) / float(x.shape[0])
    loading_value = _diagonal_loading_value(cov_unloaded, diagonal_loading)
    cov = np.asarray(cov_unloaded, dtype=np.complex128).copy()
    if loading_value > 0.0:
        cov += loading_value * np.eye(cov.shape[0], dtype=np.complex128)
    _check_covariance_shape(cov, positions.shape[0])

    steering = steering_matrix(positions, u_axis, v_axis, normalize=True)
    steering_flat, out_shape = _flatten_steering(steering)
    extra: Dict[str, Any] = {"inverse_method": "pinv"}
    if loading_value > 0.0 and x.shape[0] < positions.shape[0]:
        weighted = _loaded_low_rank_inverse_apply(
            x,
            steering_flat,
            loading_value=loading_value,
        )
        eigenvalues = _loaded_low_rank_eigenvalues(
            x,
            num_channels=positions.shape[0],
            loading_value=loading_value,
        )
        extra["inverse_method"] = "woodbury"
    else:
        inv_cov = np.linalg.pinv(cov, hermitian=True)
        weighted = inv_cov @ steering_flat
        eigenvalues = np.linalg.eigvalsh(cov)
        extra["inverse_covariance"] = inv_cov
    denom = np.real(np.sum(steering_flat.conj() * weighted, axis=0))
    spectrum = (1.0 / np.maximum(denom, float(eps))).reshape(out_shape)

    return _spectrum_result(
        spectrum,
        u_axis=u_axis,
        v_axis=v_axis,
        covariance=cov,
        eigenvalues=eigenvalues,
        steering=steering,
        extra=extra,
    )


def _as_positions_lambda(positions: np.ndarray) -> np.ndarray:
    """Normalizes virtual aperture coordinates to ``[channels, 2]`` wavelengths."""

    pos = np.asarray(positions, dtype=float)
    if pos.ndim == 1:
        pos = pos[:, None]
    if pos.ndim != 2 or pos.shape[1] not in (1, 2):
        raise ValueError(
            "virtual_positions_lambda must have shape [num_channels], "
            "[num_channels, 1], or [num_channels, 2]"
        )
    if pos.shape[0] < 2:
        raise ValueError("at least two virtual channels are required")
    if pos.shape[1] == 1:
        pos = np.column_stack([pos[:, 0], np.zeros(pos.shape[0], dtype=float)])
    return pos


def _snapshot_matrix(snapshots: np.ndarray) -> np.ndarray:
    """Flattens leading snapshot axes while preserving the channel axis."""

    x = np.asarray(snapshots, dtype=np.complex128)
    if x.ndim == 0:
        raise ValueError("snapshots must include a virtual-channel axis")
    channels = int(x.shape[-1])
    if channels < 2:
        raise ValueError("at least two virtual channels are required")
    x = x.reshape((-1, channels))
    if x.shape[0] < 1:
        raise ValueError("at least one snapshot is required")
    return x


def _apply_diagonal_loading(cov: np.ndarray, diagonal_loading: float) -> np.ndarray:
    """Applies trace-relative diagonal loading to a covariance matrix."""

    loading = float(diagonal_loading)
    if loading < 0.0:
        raise ValueError("diagonal_loading must be nonnegative")
    if loading == 0.0:
        return np.asarray(cov, dtype=np.complex128)
    cov = np.asarray(cov, dtype=np.complex128).copy()
    diag_power = float(np.real(np.trace(cov)) / max(cov.shape[0], 1))
    if diag_power <= 0.0:
        diag_power = 1.0
    cov += (loading * diag_power) * np.eye(cov.shape[0], dtype=np.complex128)
    return cov


def _diagonal_loading_value(cov: np.ndarray, diagonal_loading: float) -> float:
    """Computes the absolute loading value from a relative loading factor."""

    loading = float(diagonal_loading)
    if loading < 0.0:
        raise ValueError("diagonal_loading must be nonnegative")
    if loading == 0.0:
        return 0.0
    cov = np.asarray(cov, dtype=np.complex128)
    diag_power = float(np.real(np.trace(cov)) / max(cov.shape[0], 1))
    if diag_power <= 0.0:
        diag_power = 1.0
    return loading * diag_power


def _loaded_low_rank_inverse_apply(
    snapshots: np.ndarray,
    rhs: np.ndarray,
    *,
    loading_value: float,
) -> np.ndarray:
    """Applies the inverse of a loaded low-rank covariance via Woodbury."""

    x = np.asarray(snapshots, dtype=np.complex128)
    rhs = np.asarray(rhs, dtype=np.complex128)
    scale = np.sqrt(float(x.shape[0]))
    u_mat = x.T / scale
    gram = u_mat.conj().T @ u_mat
    inner = np.eye(gram.shape[0], dtype=np.complex128) + gram / float(loading_value)
    projected = (u_mat.conj().T @ rhs) / float(loading_value)
    correction = (u_mat / float(loading_value)) @ np.linalg.solve(inner, projected)
    return rhs / float(loading_value) - correction


def _loaded_low_rank_eigenvalues(
    snapshots: np.ndarray,
    *,
    num_channels: int,
    loading_value: float,
) -> np.ndarray:
    """Returns eigenvalues for a diagonally loaded low-rank covariance."""

    x = np.asarray(snapshots, dtype=np.complex128)
    gram = (x.conj() @ x.T) / float(x.shape[0])
    nonzero = np.linalg.eigvalsh(gram)
    eigenvalues = np.full(int(num_channels), float(loading_value), dtype=float)
    count = min(nonzero.size, eigenvalues.size)
    if count:
        eigenvalues[-count:] += np.maximum(nonzero[-count:], 0.0)
    return eigenvalues


def _direction_axes(
    positions: np.ndarray,
    *,
    u: np.ndarray | None,
    v: np.ndarray | None,
    grid_size: int | Iterable[int],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Builds 1D or 2D direction-cosine axes for the aperture geometry."""

    if u is None:
        if np.isscalar(grid_size):
            nu = int(grid_size)
        else:
            seq = tuple(int(x) for x in grid_size)
            nu = seq[-1]
        u_axis = np.linspace(-1.0, 1.0, nu)
    else:
        u_axis = np.asarray(u, dtype=float).reshape(-1)

    if u_axis.size < 1:
        raise ValueError("direction axes must be nonempty")

    if v is None and np.allclose(positions[:, 1], 0.0):
        return u_axis, None

    if v is None:
        if np.isscalar(grid_size):
            nv = int(grid_size)
        else:
            seq = tuple(int(x) for x in grid_size)
            if len(seq) != 2:
                raise ValueError("2D grid_size must be a pair")
            nv = seq[0]
        v_axis = np.linspace(-1.0, 1.0, nv)
    else:
        v_axis = np.asarray(v, dtype=float).reshape(-1)

    if v_axis.size < 1:
        raise ValueError("direction axes must be nonempty")
    return u_axis, v_axis


def _check_covariance_shape(cov: np.ndarray, num_channels: int) -> None:
    """Validates that a covariance matrix matches the virtual-channel count."""

    if cov.shape != (num_channels, num_channels):
        raise ValueError(
            "snapshots last axis must match virtual_positions_lambda channels"
        )


def _flatten_steering(steering: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Flattens steering grid axes into columns and returns the original grid shape."""

    out_shape = steering.shape[1:]
    return steering.reshape((steering.shape[0], -1)), out_shape


def _spectrum_result(
    spectrum: np.ndarray,
    *,
    u_axis: np.ndarray,
    v_axis: np.ndarray | None,
    covariance: np.ndarray,
    eigenvalues: np.ndarray,
    steering: np.ndarray,
    extra: dict[str, Any],
) -> Dict[str, Any]:
    """Packages a pseudospectrum and its diagnostic arrays into one result dict."""

    result: Dict[str, Any] = {
        "map": np.asarray(spectrum, dtype=float),
        "u": np.asarray(u_axis, dtype=float),
        "covariance": covariance,
        "eigenvalues": np.asarray(eigenvalues, dtype=float),
        "steering": steering,
    }
    if v_axis is not None:
        result["v"] = np.asarray(v_axis, dtype=float)
    result.update(extra)
    return result


__all__ = [
    "array_covariance",
    "music_pseudospectrum",
    "mvdr_pseudospectrum",
    "steering_matrix",
]
