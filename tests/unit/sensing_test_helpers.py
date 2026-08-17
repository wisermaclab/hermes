# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Shared helpers for sensing unit tests."""

from __future__ import annotations

from dataclasses import replace
from os.path import join
import tempfile

import mitsuba as mi
import numpy as np
import pytest
from scipy.constants import c, pi

from sionna.rt import load_scene
from sionna.rt.constants import InteractionType, INVALID_PRIMITIVE, INVALID_SHAPE
from mmWaveRadar.materials import PhysicalOpticsMaterial, human_skin_material
from mmWaveRadar.radar import FMCWConfig, RadarHardware, RadarSensor
from mmWaveRadar.simulation import (
    EnvironmentReflectorBank,
    HybridStaticEnvPOConfig,
    HumanPOMobilityConfig,
    MmWaveRadarSimulator,
    POCalibrationConfig,
    POIncrementalSweepSetting,
    CoherentRTPathBank,
    CoherentRTPathUpdate,
    PathVisibilityState,
    RadarCube,
    RTCoherentTransitionConfig,
    SENSING_METADATA_SCHEMA_VERSION,
    SensingMetadata,
    StaticPathBank,
    barycentric_coordinates,
    blocked_path_visibility,
    calibration_gain_array,
    compute_env_human_coupling_channels,
    equivalent_bistatic_radial_velocity,
    mesh_sequence_radial_velocity_samples,
    radial_velocity_summary,
    cube_matches_run,
    extract_coherent_rt_path_bank,
    extract_rt_reflector_bank,
    extract_static_path_bank,
    fit_sequence_path_power_calibration,
    load_radar_cube_npz,
    make_incremental_po_config,
    po_channel,
    run_rt_mobility_baseline,
    run_incremental_po_sweep,
    save_radar_cube_npz,
    segment_indices_intersecting_aabb,
    update_coherent_rt_path_bank,
    update_path_visibility_state,
)
import mmWaveRadar.simulation.rt_coherence as rt_coherence
from mmWaveRadar.simulation.physical_optics import HumanVisibilityScene
from mmWaveRadar.targets import (
    MeshSequence,
    MeshTarget,
    interpolate_amass_pose_params,
)
from mmWaveRadar.tutorial_support import load_bedroom_scene


def _mesh_sequence(displacement=0.0):
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    v0 = np.array([
        [1.0, -0.5, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 0.0, 1.0],
    ], dtype=np.float32)
    v1 = v0.copy()
    v1[:, 0] += displacement
    vertices = np.stack([v0, v1], axis=0)
    return MeshSequence(vertices=vertices, faces=faces,
                        times=np.array([0.0, 1.0]))


def _front_facing_triangle(x: float, *, scale: float = 1.0,
                           y_shift: float = 0.0,
                           z_shift: float = 0.0) -> np.ndarray:
    return np.array([
        [x, y_shift - 0.5 * scale, z_shift],
        [x, y_shift, z_shift + scale],
        [x, y_shift + 0.5 * scale, z_shift],
    ], dtype=np.float32)


def _triangle_from_yz(x: float, yz0, yz1, yz2) -> np.ndarray:
    return np.array([
        [x, yz0[0], yz0[1]],
        [x, yz1[0], yz1[1]],
        [x, yz2[0], yz2[1]],
    ], dtype=np.float32)


def _single_face_mesh_sequence(*, x0: float, x1: float | None = None) -> MeshSequence:
    v0 = _front_facing_triangle(x0)
    v1 = _front_facing_triangle(x0 if x1 is None else x1)
    return MeshSequence(
        vertices=np.stack([v0, v1], axis=0),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        times=np.array([0.0, 2.0], dtype=float),
    )


def _phase_one_radar(num_chirps: int = 3, chirp_repetition_time: float = 0.5):
    fmcw = FMCWConfig(carrier_frequency=60e9, slope=1e12,
                      chirp_duration=1e-3,
                      chirp_repetition_time=chirp_repetition_time,
                      sampling_frequency=2e3, num_adc_samples=4,
                      num_chirps_per_frame=num_chirps,
                      frame_period=max(2.0, num_chirps * chirp_repetition_time))
    return RadarSensor(name="radar", position=(0.0, 0.0, 0.0),
                       orientation=(0.0, 0.0, 0.0),
                       hardware=RadarHardware.from_positions(
                           [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
                       fmcw=fmcw)


def _profile_test_cube(
    adc: np.ndarray,
    fmcw: FMCWConfig,
    *,
    wall_scale: float = 1.0,
    recomputed: np.ndarray | None = None,
) -> RadarCube:
    adc = np.asarray(adc, dtype=np.complex128)
    times = np.zeros(adc.shape[:2], dtype=float)
    for frame in range(adc.shape[0]):
        for chirp in range(adc.shape[1]):
            times[frame, chirp] = fmcw.chirp_time(frame, chirp)
    path_counts = np.ones(adc.shape[:2], dtype=np.int64)
    if recomputed is None:
        recomputed = np.zeros(adc.shape[:2], dtype=np.int64)
    metadata = SensingMetadata(
        events=[],
        path_counts=path_counts,
        max_displacements=np.zeros(adc.shape[:2], dtype=float),
        virtual_channel_order="tx_major",
        mode="human_only_po",
        visible_face_counts=path_counts.copy(),
        effective_visible_area_fraction=np.ones(adc.shape[:2], dtype=float),
        runtime_per_chirp_s=np.full(adc.shape[:2], wall_scale, dtype=float),
        runtime_profile_s={"po_incremental_update": float(wall_scale)},
        runtime_profile_counts={"po_incremental_update": 1},
        incremental_recomputed_face_counts=recomputed,
        incremental_phase_updated_face_counts=np.zeros_like(recomputed),
        incremental_full_refresh=np.zeros(adc.shape[:2], dtype=bool),
        incremental_visibility_refresh=np.zeros(adc.shape[:2], dtype=bool),
    )
    return RadarCube(adc=adc, times=times, metadata=metadata)


def _coherent_test_bank(lengths_m, signatures, coefficients=None):
    lengths_m = np.asarray(lengths_m, dtype=np.float64)
    num_paths = int(lengths_m.size)
    if coefficients is None:
        coefficients = np.ones((1, num_paths), dtype=np.complex128)
    path_vertices = np.zeros((1, num_paths, 3), dtype=np.float64)
    path_vertices[0, :, 0] = 0.5 * lengths_m
    interactions = np.full(
        (1, num_paths),
        InteractionType.SPECULAR,
        dtype=np.uint32,
    )
    objects = np.zeros((1, num_paths), dtype=np.uint32)
    primitives = np.arange(num_paths, dtype=np.uint32).reshape(1, num_paths)
    spread_products = np.maximum(0.5 * lengths_m, 1e-18) ** 2
    return CoherentRTPathBank(
        coefficients=np.asarray(coefficients, dtype=np.complex128),
        delays_s=(lengths_m / c).astype(float),
        valid=np.ones(num_paths, dtype=bool),
        anchor_path_lengths_m=lengths_m.reshape(1, num_paths),
        anchor_delay_lengths_m=lengths_m.copy(),
        anchor_spread_products_m=spread_products.reshape(1, num_paths),
        path_vertices=path_vertices,
        path_interactions=interactions,
        path_objects=objects,
        path_primitives=primitives,
        dynamic_target_indices=np.full((1, num_paths), -1, dtype=np.int64),
        dynamic_barycentrics=np.full((1, num_paths, 3), np.nan, dtype=np.float64),
        target_faces=(),
        tx_positions=np.zeros((1, 3), dtype=np.float64),
        rx_positions=np.zeros((1, 3), dtype=np.float64),
        tx_center=np.zeros(3, dtype=np.float64),
        rx_center=np.zeros(3, dtype=np.float64),
        virtual_channel_order="tx_major",
        frequency_hz=60e9,
        path_signatures=tuple(signatures),
    )



__all__ = [name for name in globals() if not name.startswith("__")]
