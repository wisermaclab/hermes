# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Geometry-first image-source selection by estimated coupling power."""

from dataclasses import replace

import numpy as np

from ..geometry import rotation_matrix
from .env_human_coupling import (
    _empty_reflector_bank,
    _image_polarization_callback,
    _mirror_points,
    _reflection_points_on_surface,
    _reflector_support,
    _segments_clear_in_scene,
    extract_environment_reflector_bank,
)


def motion_volume_bounds(mesh_sequence, time_range_s=None):
    """Bound every key pose and interval endpoint in the selected time window.

    This is an exact bound for linear vertex interpolation and a sampled bound
    for parametric pose interpolation. Stream poses to support lazy providers.
    """
    times = np.asarray(mesh_sequence.times, dtype=float)
    start, stop = (
        (float(times[0]), float(times[-1]))
        if time_range_s is None
        else tuple(map(float, time_range_s))
    )
    if not np.isfinite([start, stop]).all() or stop < start:
        raise ValueError("time_range_s must contain finite increasing endpoints")
    selected = np.unique(np.r_[start, times[(times > start) & (times < stop)], stop])
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for time in selected:
        vertices = np.asarray(mesh_sequence.vertices_at(float(time)), dtype=float)
        lower = np.minimum(lower, vertices.min(axis=0))
        upper = np.maximum(upper, vertices.max(axis=0))
    return lower, upper


def _volume_probes(lower, upper):
    # Extra height levels retain near-floor/leg opportunities for a high radar.
    axes = [np.linspace(lower[i], upper[i], count) for i, count in enumerate((5, 5, 9))]
    return np.unique(np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3), axis=0)


def _endpoint_gain(radar, points, antenna_positions):
    local = radar.world_to_local_points(points.reshape(-1, 3)).reshape(points.shape)
    local -= antenna_positions[:, None, :]
    azimuth = np.degrees(np.arctan2(local[..., 1], local[..., 0]))
    elevation = np.degrees(np.arctan2(local[..., 2], np.hypot(local[..., 0], local[..., 1])))
    return radar.hardware.channel_gain(
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        frequency_hz=radar.fmcw.carrier_frequency,
        include_phase=False,
    )


def _image_power_scores(bank, radar, probes, static_scene, reflection_scale):
    """Peak per-plane power for a unit isotropic scatterer in the human volume.

    Score = |Gamma E|^2 * |G_T G_R| / (R_T^2 R_R^2), in common normalized
    units, using the same combined-pattern split as coupling. It includes
    polarization, coherent material loss, real endpoint beams, finite support,
    static occlusion and both interaction orders. It is a selection estimate,
    not a bound on the coherently integrated human PO/ADC power.
    """
    vc_tx, vc_rx = radar.hardware.virtual_tx_indices(), radar.hardware.virtual_rx_indices()
    tx, rx = radar.world_tx_positions()[vc_tx], radar.world_rx_positions()[vc_rx]
    count, num_probes = len(vc_tx), len(probes)
    points = np.broadcast_to(probes, (count, num_probes, 3))
    flat_points = points.reshape(-1, 3)
    real_tx = np.broadcast_to(tx[:, None, :], points.shape)
    real_rx = np.broadcast_to(rx[:, None, :], points.shape)
    axis = rotation_matrix(radar.orientation)[:, 2]
    scores = np.zeros((len(bank.vertices), 2), dtype=float)
    for index in range(len(bank.vertices)):
        point, normal = bank.centroids[index], bank.normals[index]
        scale = reflection_scale * (
            1.0
            if bank.specular_amplitude_scales is None
            else bank.specular_amplitude_scales[index]
        )
        for order_index, order in enumerate(("human_env", "env_human")):
            tx_po, rx_po = real_tx, real_rx
            if order == "human_env":
                rx_po = np.broadcast_to(
                    _mirror_points(rx, point, normal)[:, None, :], points.shape
                )
                image, direct, endpoint = rx_po, real_tx, real_rx
            else:
                tx_po = np.broadcast_to(
                    _mirror_points(tx, point, normal)[:, None, :], points.shape
                )
                image, direct, endpoint = tx_po, real_rx, real_tx
            hits, finite = _reflection_points_on_surface(
                image.reshape(-1, 3), flat_points, point, normal, _reflector_support(bank, index)
            )
            rt = np.linalg.norm(points - tx_po, axis=-1)
            rr = np.linalg.norm(points - rx_po, axis=-1)
            finite &= ((rt > 1e-6) & (rr > 1e-6)).ravel()
            indices = np.flatnonzero(finite)
            if static_scene is not None and indices.size:
                finite[indices] &= (
                    _segments_clear_in_scene(
                        starts=direct.reshape(-1, 3)[indices],
                        ends=flat_points[indices],
                        scene=static_scene,
                    )
                    & _segments_clear_in_scene(
                        starts=flat_points[indices], ends=hits[indices], scene=static_scene
                    )
                    & _segments_clear_in_scene(
                        starts=endpoint.reshape(-1, 3)[indices],
                        ends=hits[indices],
                        scene=static_scene,
                    )
                )
            if not np.any(finite):
                continue
            indices = np.flatnonzero(finite)
            callback = _image_polarization_callback(
                tx_position=tx_po.reshape(-1, 3)[indices],
                rx_position=rx_po.reshape(-1, 3)[indices],
                reflector_normal=normal,
                material=bank.materials[index],
                frequency_hz=radar.fmcw.carrier_frequency,
                order=order,
                scale=scale,
                polarization_axis=axis,
            )
            tx_pol, rx_pol = callback(flat_points[indices])
            field = rx_pol if order == "human_env" else tx_pol
            reflection_power = np.zeros(len(flat_points), dtype=float)
            reflection_power[indices] = np.sum(abs(field) ** 2, axis=1)
            reflection_power = reflection_power.reshape(count, num_probes)
            hits = hits.reshape(points.shape)
            first, last = (points, hits) if order == "human_env" else (hits, points)
            antenna_power = abs(
                _endpoint_gain(radar, first, radar.hardware.tx_positions[vc_tx])
                * _endpoint_gain(radar, last, radar.hardware.rx_positions[vc_rx])
            )
            valid = finite.reshape(count, num_probes) & (rt > 1e-6) & (rr > 1e-6)
            power = np.divide(
                reflection_power * antenna_power, rt**2 * rr**2, out=np.zeros_like(rt), where=valid
            )
            scores[index, order_index] = float(np.max(power, initial=0.0))
    return scores


def select_image_source_reflectors(
    scene,
    radar,
    target,
    *,
    power_threshold_db=-20.0,
    max_count=4,
    min_area_m2=1e-4,
    reflection_scale=1.0,
    time_range_s=None,
    enabled=True,
):
    """Discover all planes, apply a relative power cutoff and retain the top N.

    Selection is fixed over the run's motion volume, avoiding chirp-to-chirp
    reflector births from changing rankings. The supported maximum is four.
    """
    if isinstance(power_threshold_db, (bool, np.bool_)) or power_threshold_db is None:
        raise ValueError("power_threshold_db must be finite and <= 0")
    try:
        power_threshold_db = float(power_threshold_db)
    except (ValueError, TypeError) as exc:
        raise ValueError("power_threshold_db must be finite and <= 0") from exc
    if not np.isfinite(power_threshold_db) or power_threshold_db > 0:
        raise ValueError("power_threshold_db must be finite and <= 0")
    if (
        isinstance(max_count, (bool, np.bool_))
        or not isinstance(max_count, (int, np.integer))
        or not 1 <= max_count <= 4
    ):
        raise ValueError("max_count must be an integer between 1 and 4")
    bank = (
        extract_environment_reflector_bank(scene, max_reflectors=0, min_area_m2=min_area_m2)
        if enabled
        else _empty_reflector_bank()
    )
    if len(bank.vertices) == 0:
        return replace(
            bank,
            estimated_powers=np.zeros(0),
            relative_power_db=np.zeros(0),
            selection_metadata={
                "method": "image_source_power",
                "power_threshold_db": power_threshold_db,
                "max_count": int(max_count),
                "candidate_count": 0,
                "retained_count": 0,
                "probe_count": 0,
                "candidates": [],
            },
        )
    lower, upper = motion_volume_bounds(target.mesh_sequence, time_range_s)
    probes = _volume_probes(lower, upper)
    order_powers = _image_power_scores(
        bank, radar, probes, getattr(scene, "mi_scene", None), reflection_scale
    )
    powers = np.max(order_powers, axis=1) if len(bank.vertices) else np.zeros(0)
    strongest = float(np.max(powers, initial=0.0))
    relative = np.full(len(powers), -np.inf)
    if strongest > 0.0:
        positive = powers > 0.0
        relative[positive] = 10.0 * np.log10(powers[positive] / strongest)
    above_cutoff = (powers > 0.0) & (relative >= power_threshold_db - 1e-12)
    # Stable tie order follows scene geometry, independently of RT path order.
    ranking = np.argsort(-powers, kind="stable")
    selected = ranking[above_cutoff[ranking]][:max_count]
    keep = np.zeros(len(powers), bool)
    keep[selected] = True
    metadata = {
        "method": "image_source_power",
        "power_threshold_db": power_threshold_db,
        "max_count": int(max_count),
        "candidate_count": len(powers),
        "retained_count": len(selected),
        "probe_count": len(probes),
        "motion_bounds_m": [lower.tolist(), upper.tolist()],
        "score_model": "peak_unit_isotropic_scatterer_power",
        "candidates": [
            {
                "object": bank.object_names[i],
                "face": int(bank.face_indices[i]),
                "estimated_power": float(powers[i]),
                "relative_power_db": float(relative[i]) if powers[i] > 0.0 else None,
                "human_env_power": float(order_powers[i, 0]),
                "env_human_power": float(order_powers[i, 1]),
                "above_power_cutoff": bool(above_cutoff[i]),
                "retained": bool(keep[i]),
            }
            for i in ranking
        ],
    }

    def subset_optional(value):
        return None if value is None else value[selected]

    return replace(
        bank,
        vertices=bank.vertices[selected],
        centroids=bank.centroids[selected],
        normals=bank.normals[selected],
        areas=bank.areas[selected],
        materials=tuple(bank.materials[i] for i in selected),
        object_names=tuple(bank.object_names[i] for i in selected),
        face_indices=bank.face_indices[selected],
        path_indices=subset_optional(bank.path_indices),
        reflection_gains=subset_optional(bank.reflection_gains),
        specular_amplitude_scales=subset_optional(bank.specular_amplitude_scales),
        support_triangles=(
            None
            if bank.support_triangles is None
            else tuple(bank.support_triangles[i] for i in selected)
        ),
        estimated_powers=powers[selected],
        relative_power_db=relative[selected],
        selection_metadata=metadata,
    )
