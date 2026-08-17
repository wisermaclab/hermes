# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Static-target RT/PO experiment service used by the public HERMES GUI."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
import time
from typing import Callable

import numpy as np

from bundle_prepare.safe_npz import (
    DEFAULT_NPZ_LIMITS,
    preflight_npz,
    require_array,
)

from ..dsp import range_fft, range_profile_from_cube
from ..materials import human_skin_material
from ..radar import FMCWConfig, RadarHardware, RadarSensor, get_ti_board_spec
from ..simulation import (
    MmWaveRadarSimulator,
    PhysicalOpticsMaterial,
    po_channel,
    synthesize_adc_from_paths,
)
from ..targets import MeshSequence, MeshTarget, ensure_outward_face_winding
from .manifest import (
    ExperimentManifest,
    RadarExperimentConfig,
    SceneExperimentConfig,
    SolverExperimentConfig,
)
from .solver_diagnostics import (
    SolverDiagnostics,
    summarize_po_surface,
    summarize_rt_paths,
)


TARGET_TYPES = ("plate", "trihedral", "human_mesh")
MATERIAL_PRESETS = ("pec", "aluminum", "concrete")

_MAX_HUMAN_OBJ_BYTES = 64 * 1024**2
_MAX_HUMAN_OBJ_VERTICES = 2_000_000
_MAX_HUMAN_OBJ_TRIANGLES = 4_000_000

_MATERIALS = {
    "pec": {
        "relative_permittivity": 1.0,
        "conductivity_s_per_m": 1.0e7,
        "thickness_m": 0.01,
        "po_model": "pec",
        # Rough PEC proxy: retain an ideal conductor for PO while exposing a
        # modest rough-surface component in Sionna RT.
        "diffuse_coefficient": 0.20,
        "backscattering_lambda": 0.20,
        "color": (0.58, 0.66, 0.76),
    },
    "aluminum": {
        "relative_permittivity": 1.0,
        "conductivity_s_per_m": 3.5e7,
        "thickness_m": 0.003,
        "po_model": "slab",
        "diffuse_coefficient": 0.0,
        "backscattering_lambda": 0.0,
        "color": (0.72, 0.76, 0.80),
    },
    "concrete": {
        "relative_permittivity": 5.31,
        "conductivity_s_per_m": 0.0326,
        "thickness_m": 0.12,
        "po_model": "slab",
        "diffuse_coefficient": 0.35,
        "backscattering_lambda": 0.35,
        "color": (0.62, 0.56, 0.48),
    },
}


def material_preset_parameters(name: str) -> dict[str, object]:
    """Return a copy of one static-target material preset."""

    if name not in _MATERIALS:
        raise ValueError(f"material must be one of {MATERIAL_PRESETS}")
    return dict(_MATERIALS[name])


@dataclass(frozen=True)
class PhysicsMicroscopeResult:
    """Geometry, raw ADC, profiles, and provenance for one RT/PO run."""

    manifest: ExperimentManifest
    target_type: str
    material_name: str
    target_parameters: dict[str, object]
    sensor_parameters: dict[str, object]
    vertices: np.ndarray
    faces: np.ndarray
    face_power: np.ndarray
    face_visibility: np.ndarray
    rt_face_hit_counts: np.ndarray
    po_adc: np.ndarray
    rt_adc: np.ndarray
    adc_times_s: np.ndarray
    ranges_m: np.ndarray
    po_range_profile_power: np.ndarray
    rt_range_profile_power: np.ndarray
    po_runtime_s: float
    rt_runtime_s: float
    visible_face_count: int
    radar_position: np.ndarray
    rt_path_count: int

    @property
    def adc(self) -> np.ndarray:
        """Compatibility alias for the PO cube used by earlier callers."""

        return self.po_adc

    @property
    def range_profile_power(self) -> np.ndarray:
        """Compatibility alias for the PO range profile."""

        return self.po_range_profile_power

    @property
    def runtime_s(self) -> float:
        """Total solver wall time."""

        return float(self.po_runtime_s + self.rt_runtime_s)


def _positive(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _finite(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _rotation_matrix(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    angles = np.asarray((yaw_deg, pitch_deg, roll_deg), dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("orientation angles must be finite")
    yaw, pitch, roll = np.deg2rad(angles)
    rz = np.asarray(
        ((np.cos(yaw), -np.sin(yaw), 0), (np.sin(yaw), np.cos(yaw), 0), (0, 0, 1)),
        dtype=float,
    )
    ry = np.asarray(
        ((np.cos(pitch), 0, np.sin(pitch)), (0, 1, 0), (-np.sin(pitch), 0, np.cos(pitch))),
        dtype=float,
    )
    rx = np.asarray(
        ((1, 0, 0), (0, np.cos(roll), -np.sin(roll)), (0, np.sin(roll), np.cos(roll))),
        dtype=float,
    )
    return rz @ ry @ rx


def _tessellated_quad(
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    *,
    size_u: float,
    size_v: float,
    max_edge_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    nu = max(1, int(np.ceil(size_u / max_edge_m)))
    nv = max(1, int(np.ceil(size_v / max_edge_m)))
    us = np.linspace(0.0, size_u, nu + 1)
    vs = np.linspace(0.0, size_v, nv + 1)
    vertices = np.asarray(
        [origin + u * axis_u + v * axis_v for u in us for v in vs],
        dtype=np.float32,
    )
    faces: list[tuple[int, int, int]] = []
    for iu in range(nu):
        for iv in range(nv):
            v00 = iu * (nv + 1) + iv
            v01 = v00 + 1
            v10 = (iu + 1) * (nv + 1) + iv
            v11 = v10 + 1
            faces.extend(((v00, v01, v11), (v00, v11, v10)))
    return vertices, np.asarray(faces, dtype=np.uint32)


def make_rectangular_plate_mesh(
    *,
    range_m: float,
    lateral_m: float = 0.0,
    vertical_m: float = 0.0,
    width_m: float,
    height_m: float,
    aspect_deg: float = 0.0,
    max_edge_m: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a tessellated plate whose zero-angle normal faces the radar."""

    distance = _positive("range_m", range_m)
    lateral = _finite("lateral_m", lateral_m)
    vertical = _finite("vertical_m", vertical_m)
    width = _positive("width_m", width_m)
    height = _positive("height_m", height_m)
    edge = _positive("max_edge_m", max_edge_m)
    if abs(float(aspect_deg)) >= 90.0:
        raise ValueError("aspect_deg must be finite and strictly between -90 and 90")
    vertices, faces = _tessellated_quad(
        np.asarray((0.0, -0.5 * width, -0.5 * height)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
        size_u=width,
        size_v=height,
        max_edge_m=edge,
    )
    rotation = _rotation_matrix(aspect_deg, pitch_deg, roll_deg)
    vertices = vertices @ rotation.T
    vertices += np.asarray((distance, lateral, vertical))
    return vertices.astype(np.float32), faces


def make_trihedral_corner_mesh(
    *,
    range_m: float,
    lateral_m: float = 0.0,
    vertical_m: float = 0.0,
    edge_m: float,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    max_edge_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create three perpendicular square plates with the opening toward radar."""

    distance = _positive("range_m", range_m)
    lateral = _finite("lateral_m", lateral_m)
    vertical = _finite("vertical_m", vertical_m)
    size = _positive("edge_m", edge_m)
    tessellation = _positive("max_edge_m", max_edge_m)
    panels = (
        ((0, 0, 0), (0, 1, 0), (0, 0, 1), True),
        ((0, 0, 0), (1, 0, 0), (0, 0, 1), False),
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), True),
    )
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    offset = 0
    for origin, axis_u, axis_v, flip in panels:
        vertices, faces = _tessellated_quad(
            np.asarray(origin, dtype=float),
            np.asarray(axis_u, dtype=float),
            np.asarray(axis_v, dtype=float),
            size_u=size,
            size_v=size,
            max_edge_m=tessellation,
        )
        all_vertices.append(vertices)
        all_faces.append((faces[:, [0, 2, 1]] if flip else faces) + offset)
        offset += vertices.shape[0]
    vertices = np.concatenate(all_vertices, axis=0).astype(float)
    faces = np.concatenate(all_faces, axis=0)

    # The panels extend into the positive octant, so their concave opening is
    # +(+x,+y,+z). Align that opening, and its inward-facing normals, with the
    # direction from the reflector to the radar (-x).
    opening = np.ones(3, dtype=float) / np.sqrt(3.0)
    target = np.asarray((-1.0, 0.0, 0.0))
    cross = np.cross(opening, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(opening, target))
    skew = np.asarray(
        ((0, -cross[2], cross[1]), (cross[2], 0, -cross[0]), (-cross[1], cross[0], 0))
    )
    align = np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
    vertices = vertices @ align.T
    vertices = vertices @ _rotation_matrix(yaw_deg, pitch_deg, roll_deg).T
    vertices += np.asarray((distance, lateral, vertical))
    return vertices.astype(np.float32), faces.astype(np.uint32)


def load_human_mesh(
    source: str | Path | bytes,
    *,
    filename: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a static OBJ or pickle-free NPZ triangular mesh."""

    suffix = Path(filename or str(source)).suffix.lower()
    if isinstance(source, bytes):
        if suffix == ".obj" and len(source) > _MAX_HUMAN_OBJ_BYTES:
            raise ValueError("human mesh OBJ exceeds the 64 MiB input limit")
        stream = BytesIO(source)
        if suffix == ".npz":
            arrays = preflight_npz(source, limits=DEFAULT_NPZ_LIMITS)
            require_array(
                arrays,
                "vertices",
                allowed_ndim=(2, 3),
                dtype_kinds=frozenset({"i", "u", "f"}),
                trailing_shape=(3,),
                max_elements=60_000_000,
            )
            require_array(
                arrays,
                "faces",
                allowed_ndim=(2,),
                dtype_kinds=frozenset({"i", "u"}),
                trailing_shape=(3,),
                max_elements=60_000_000,
            )
            archive = np.load(stream, allow_pickle=False)
            with archive:
                vertices = np.array(archive["vertices"], copy=True)
                faces = np.array(archive["faces"], copy=True)
        elif suffix == ".obj":
            text = source.decode("utf-8")
            vertices, faces = _parse_obj(text)
        else:
            raise ValueError("human mesh upload must be an .obj or .npz file")
    else:
        path = Path(source).expanduser()
        suffix = path.suffix.lower()
        if suffix == ".npz":
            arrays = preflight_npz(path, limits=DEFAULT_NPZ_LIMITS)
            require_array(
                arrays,
                "vertices",
                allowed_ndim=(2, 3),
                dtype_kinds=frozenset({"i", "u", "f"}),
                trailing_shape=(3,),
                max_elements=60_000_000,
            )
            require_array(
                arrays,
                "faces",
                allowed_ndim=(2,),
                dtype_kinds=frozenset({"i", "u"}),
                trailing_shape=(3,),
                max_elements=60_000_000,
            )
            with np.load(path, allow_pickle=False) as archive:
                vertices = np.array(archive["vertices"], copy=True)
                faces = np.array(archive["faces"], copy=True)
        elif suffix == ".obj":
            if path.stat().st_size > _MAX_HUMAN_OBJ_BYTES:
                raise ValueError("human mesh OBJ exceeds the 64 MiB input limit")
            vertices, faces = _parse_obj(path.read_text(encoding="utf-8"))
        else:
            raise ValueError("human mesh must be an .obj or .npz file")
    if vertices.ndim == 3:
        if vertices.shape[0] < 1:
            raise ValueError("human mesh sequence is empty")
        vertices = vertices[0]
    vertices = np.asarray(vertices, dtype=np.float32)
    raw_faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise ValueError("human mesh vertices must have shape [V, 3]")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("human mesh vertices must be finite")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
        raise ValueError("human mesh faces must have shape [F, 3]")
    if raw_faces.dtype.kind not in {"i", "u"}:
        raise ValueError("human mesh faces must contain integer indices")
    faces = np.asarray(raw_faces, dtype=np.int64)
    if faces.size == 0 or np.min(faces) < 0 or np.max(faces) >= vertices.shape[0]:
        raise ValueError("human mesh faces contain invalid vertex indices")
    return vertices, ensure_outward_face_winding(vertices, faces)


def _parse_obj(text: str) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line_number, raw in enumerate(StringIO(text), start=1):
        if len(raw) > 4 * 1024**2:
            raise ValueError(
                f"OBJ line {line_number} exceeds the 4 MiB line-size limit"
            )
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            if len(vertices) >= _MAX_HUMAN_OBJ_VERTICES:
                raise ValueError(
                    "human mesh OBJ exceeds the 2,000,000-vertex limit"
                )
            vertices.append(tuple(float(value) for value in parts[1:4]))
        elif parts[0] == "f" and len(parts) >= 4:
            triangle_count = len(parts) - 3
            if len(faces) + triangle_count > _MAX_HUMAN_OBJ_TRIANGLES:
                raise ValueError(
                    "human mesh OBJ exceeds the 4,000,000-triangle limit"
                )
            polygon: list[int] = []
            for token in parts[1:]:
                index = int(token.split("/", maxsplit=1)[0])
                index = len(vertices) + index if index < 0 else index - 1
                polygon.append(index)
            for corner in range(1, len(polygon) - 1):
                faces.append((polygon[0], polygon[corner], polygon[corner + 1]))
        elif parts[0] in {"v", "f"}:
            raise ValueError(f"invalid OBJ geometry at line {line_number}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def _place_loaded_mesh(
    vertices: np.ndarray,
    *,
    range_m: float,
    lateral_m: float,
    vertical_m: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    placed = np.asarray(vertices, dtype=float).copy()
    center = 0.5 * (np.min(placed, axis=0) + np.max(placed, axis=0))
    placed -= center
    placed = placed @ _rotation_matrix(yaw_deg, pitch_deg, roll_deg).T
    position = np.asarray(
        (
            _positive("range_m", range_m),
            _finite("lateral_m", lateral_m),
            _finite("vertical_m", vertical_m),
        )
    )
    placed += position
    return placed.astype(np.float32)


def _radar(
    board_model: str,
    *,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    antenna_pattern_mode: str,
    cosine_3db_beamwidth_deg: float,
    carrier_frequency_hz: float | None,
    slope_hz_per_s: float,
    chirp_duration_s: float,
    chirp_repetition_time_s: float,
    sampling_frequency_hz: float,
    num_adc_samples: int,
    num_chirps_per_frame: int,
    frame_period_s: float,
    tdm_enabled: bool,
    selected_tx_indices: tuple[int, ...] | list[int] | None = None,
    selected_rx_indices: tuple[int, ...] | list[int] | None = None,
) -> RadarSensor:
    board_spec = get_ti_board_spec(board_model)
    orientation_deg = np.asarray((yaw_deg, pitch_deg, roll_deg), dtype=float)
    if not np.all(np.isfinite(orientation_deg)):
        raise ValueError("radar orientation angles must be finite")
    pattern_mode = str(antenna_pattern_mode).casefold().replace("-", "_")
    if pattern_mode not in {"digitized", "none", "cosine"}:
        raise ValueError(
            "antenna_pattern_mode must be 'digitized', 'none', or 'cosine'"
        )
    beamwidth_deg = float(cosine_3db_beamwidth_deg)
    if not np.isfinite(beamwidth_deg) or not 0.0 < beamwidth_deg < 180.0:
        raise ValueError(
            "cosine_3db_beamwidth_deg must be between 0 and 180 degrees"
        )
    hardware = RadarHardware.from_ti_board(
        board_model,
        pattern_mode=pattern_mode,
        cosine_half_power_angle_deg=0.5 * beamwidth_deg,
    )
    if not isinstance(tdm_enabled, (bool, np.bool_)):
        raise ValueError("tdm_enabled must be a boolean")
    tx_indices = (
        tuple(range(hardware.num_tx))
        if selected_tx_indices is None
        else tuple(selected_tx_indices)
    )
    rx_indices = (
        tuple(range(hardware.num_rx))
        if selected_rx_indices is None
        else tuple(selected_rx_indices)
    )
    hardware = hardware.subset_tx(tx_indices).subset_rx(rx_indices)
    fmcw = FMCWConfig(
        carrier_frequency=(
            float(board_spec.design_frequency_hz)
            if carrier_frequency_hz is None
            else carrier_frequency_hz
        ),
        slope=slope_hz_per_s,
        chirp_duration=chirp_duration_s,
        chirp_repetition_time=chirp_repetition_time_s,
        sampling_frequency=sampling_frequency_hz,
        num_adc_samples=num_adc_samples,
        num_chirps_per_frame=num_chirps_per_frame,
        frame_period=frame_period_s,
        num_tx=hardware.num_tx,
        tdm_enabled=bool(tdm_enabled),
    )
    return RadarSensor(
        name="radar",
        hardware=hardware,
        fmcw=fmcw,
        position=(0.0, 0.0, 0.0),
        orientation=tuple(np.deg2rad(orientation_deg)),
        tx_power_dbm=board_spec.tx_power_dbm,
    )


def _materials(target_type: str, material_name: str, diffuse: float):
    if target_type == "human_mesh":
        rt = human_skin_material("gui-human-tissue")
        rt.scattering_coefficient = diffuse
        po = PhysicalOpticsMaterial(
            relative_permittivity=38.0,
            conductivity_s_per_m=1.5,
            thickness_m=0.01,
            model="slab",
        )
        return po, rt
    if material_name not in _MATERIALS:
        raise ValueError(f"material must be one of {MATERIAL_PRESETS}")
    values = _MATERIALS[material_name]
    from sionna.rt import RadioMaterial  # pylint: disable=import-outside-toplevel

    rt = RadioMaterial(
        name=f"gui-{material_name}",
        relative_permittivity=values["relative_permittivity"],
        conductivity=values["conductivity_s_per_m"],
        thickness=values["thickness_m"],
        scattering_coefficient=values["diffuse_coefficient"],
        scattering_pattern="backscattering",
        alpha_r=8,
        alpha_i=20,
        lambda_=values["backscattering_lambda"],
        color=values["color"],
    )
    po = PhysicalOpticsMaterial(
        relative_permittivity=values["relative_permittivity"],
        conductivity_s_per_m=values["conductivity_s_per_m"],
        thickness_m=values["thickness_m"],
        model=values["po_model"],
    )
    return po, rt


def _po_cube(
    vertices: np.ndarray,
    faces: np.ndarray,
    radar: RadarSensor,
    material: PhysicalOpticsMaterial,
    *,
    visibility_samples: int,
    integration_mode: str,
    quadrature_phase_span_scale_rad: float,
    quadrature_max_refinement_depth: int,
    quadrature_max_subfaces_per_parent: int,
    compute_backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = po_channel(
        vertices=vertices,
        faces=faces,
        tx_positions=radar.world_tx_positions(),
        rx_positions=radar.world_rx_positions(),
        frequency_hz=radar.fmcw.carrier_frequency,
        material=material,
        visibility="none",
        visibility_samples_per_face=visibility_samples,
        visibility_fade_chirps=1,
        po_integration_mode=integration_mode,
        po_quadrature_phase_span_scale_rad=quadrature_phase_span_scale_rad,
        po_quadrature_max_refinement_depth=quadrature_max_refinement_depth,
        po_quadrature_max_subfaces_per_parent=(
            quadrature_max_subfaces_per_parent
        ),
        compute_backend=compute_backend,
        compute_precision="float32",
    )
    azimuth_deg, elevation_deg = radar.local_azimuth_elevation(result.face_centroids)
    coefficients = np.asarray(result.coefficients, dtype=np.complex64)
    coefficients *= radar.hardware.channel_gain(
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        frequency_hz=radar.fmcw.carrier_frequency,
    )
    delays = np.asarray(result.delays_s, dtype=float)
    valid = np.any(np.abs(coefficients) > 0.0, axis=0)
    valid &= np.any(delays >= 0.0, axis=0) if delays.ndim == 2 else delays >= 0.0
    chirp_adc = synthesize_adc_from_paths(
        coefficients[:, valid],
        delays[:, valid] if delays.ndim == 2 else delays[valid],
        radar.fmcw,
        backend=compute_backend,
        precision="float32",
    )
    adc = np.repeat(
        chirp_adc[None, None, :, :],
        radar.fmcw.num_chirps_per_frame,
        axis=1,
    )
    face_power = np.zeros(faces.shape[0], dtype=np.float32)
    face_visibility = np.zeros(faces.shape[0], dtype=np.float32)
    indices = np.asarray(result.face_indices, dtype=np.int64)
    contribution_power = np.asarray(
        np.abs(result.phase_center_contributions) ** 2,
        dtype=np.float32,
    )
    face_power[indices] = contribution_power
    contributing = contribution_power > 0.0
    face_visibility[indices[contributing]] = np.asarray(
        result.effective_weights,
        dtype=np.float32,
    )[contributing]
    return adc, face_power, face_visibility


def _profile(adc: np.ndarray, fmcw: FMCWConfig) -> tuple[np.ndarray, np.ndarray]:
    range_cube, ranges_m = range_fft(adc[0], fmcw=fmcw, nfft_mult=4)
    power = range_profile_from_cube(
        range_cube,
        combine_antennas="sum_power",
        combine_chirps="mean",
    )
    return np.asarray(power, dtype=float), np.asarray(ranges_m, dtype=float)


def run_static_target_experiment(
    *,
    target_type: str = "plate",
    range_m: float = 2.0,
    target_y_m: float = 0.0,
    target_z_m: float = 0.0,
    width_m: float = 0.30,
    height_m: float = 0.30,
    corner_edge_m: float = 0.20,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    material_name: str = "pec",
    human_mesh: str | Path | bytes | None = None,
    human_mesh_filename: str | None = None,
    diffuse_reflection_coefficient: float = 0.35,
    fidelity: str = "preview",
    board_model: str = "IWR6843AOPEVM",
    carrier_frequency_hz: float | None = None,
    slope_hz_per_s: float = 68e12,
    chirp_duration_s: float = 58e-6,
    chirp_repetition_time_s: float = 65e-6,
    sampling_frequency_hz: float = 4.5e6,
    num_adc_samples: int = 225,
    num_chirps_per_frame: int | None = None,
    frame_period_s: float = 50e-3,
    tdm_enabled: bool = False,
    selected_tx_indices: tuple[int, ...] | list[int] | None = None,
    selected_rx_indices: tuple[int, ...] | list[int] | None = None,
    radar_yaw_deg: float = 0.0,
    radar_pitch_deg: float = 0.0,
    radar_roll_deg: float = 0.0,
    antenna_pattern_mode: str = "cosine",
    cosine_3db_beamwidth_deg: float = 60.0,
    po_integration_mode: str = "parent_face_quadrature",
    po_quadrature_phase_span_scale_rad: float = 1.0,
    po_quadrature_max_refinement_depth: int = 2,
    po_quadrature_max_subfaces_per_parent: int = 16,
    rt_samples_per_source: int = 100_000,
    rt_max_paths_per_source: int = 5_000,
    rt_max_depth: int = 3,
    compute_backend: str = "numpy",
    cancel_check: Callable[[], bool] | None = None,
) -> PhysicsMicroscopeResult:
    """Simulate one static target with both physical optics and ray tracing."""

    if cancel_check is not None and not callable(cancel_check):
        raise ValueError("cancel_check must be callable or None")

    def check_cancelled() -> None:
        if cancel_check is not None and bool(cancel_check()):
            raise InterruptedError("Simulation cancelled by user")

    check_cancelled()

    if target_type not in TARGET_TYPES:
        raise ValueError(f"target_type must be one of {TARGET_TYPES}")
    if fidelity not in ("preview", "high_fidelity"):
        raise ValueError("fidelity must be 'preview' or 'high_fidelity'")
    if po_integration_mode not in (
        "face_centroid",
        "parent_face_quadrature",
        "parent_face_far_field_analytic",
    ):
        raise ValueError(
            "po_integration_mode must be 'face_centroid', "
            "'parent_face_quadrature', or 'parent_face_far_field_analytic'"
        )
    po_phase_span = float(po_quadrature_phase_span_scale_rad)
    if not np.isfinite(po_phase_span) or po_phase_span <= 0.0:
        raise ValueError(
            "po_quadrature_phase_span_scale_rad must be finite and positive"
        )
    po_refinement_depth = int(po_quadrature_max_refinement_depth)
    if (
        po_refinement_depth != po_quadrature_max_refinement_depth
        or po_refinement_depth < 0
    ):
        raise ValueError(
            "po_quadrature_max_refinement_depth must be a non-negative integer"
        )
    po_subface_cap = int(po_quadrature_max_subfaces_per_parent)
    if (
        po_subface_cap != po_quadrature_max_subfaces_per_parent
        or po_subface_cap < 0
    ):
        raise ValueError(
            "po_quadrature_max_subfaces_per_parent must be a non-negative integer"
        )
    rt_samples = int(rt_samples_per_source)
    if rt_samples != rt_samples_per_source or rt_samples <= 0:
        raise ValueError("rt_samples_per_source must be a positive integer")
    rt_path_cap = int(rt_max_paths_per_source)
    if rt_path_cap != rt_max_paths_per_source or rt_path_cap <= 0:
        raise ValueError("rt_max_paths_per_source must be a positive integer")
    rt_depth = int(rt_max_depth)
    if rt_depth != rt_max_depth or rt_depth <= 0:
        raise ValueError("rt_max_depth must be a positive integer")
    diffuse = float(diffuse_reflection_coefficient)
    if not np.isfinite(diffuse) or diffuse < 0.0 or diffuse > 1.0:
        raise ValueError("diffuse_reflection_coefficient must be in [0, 1]")
    resolved_pattern_mode = (
        str(antenna_pattern_mode).casefold().replace("-", "_")
    )
    resolved_cosine_beamwidth_deg = float(cosine_3db_beamwidth_deg)
    target_position = np.asarray(
        (
            _positive("range_m", range_m),
            _finite("target_y_m", target_y_m),
            _finite("target_z_m", target_z_m),
        ),
        dtype=float,
    )
    mesh_edge_m = 0.02 if fidelity == "preview" else 0.0075
    visibility_samples = 1 if fidelity == "preview" else 4
    num_chirps = (
        16 if fidelity == "preview" else 32
    ) if num_chirps_per_frame is None else num_chirps_per_frame
    if target_type == "plate":
        vertices, faces = make_rectangular_plate_mesh(
            range_m=target_position[0],
            lateral_m=target_position[1],
            vertical_m=target_position[2],
            width_m=width_m,
            height_m=height_m,
            aspect_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            max_edge_m=mesh_edge_m,
        )
        parameters: dict[str, object] = {
            "width_m": float(width_m),
            "height_m": float(height_m),
        }
    elif target_type == "trihedral":
        vertices, faces = make_trihedral_corner_mesh(
            range_m=target_position[0],
            lateral_m=target_position[1],
            vertical_m=target_position[2],
            edge_m=corner_edge_m,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            max_edge_m=mesh_edge_m,
        )
        parameters = {"edge_m": float(corner_edge_m)}
    else:
        if human_mesh is None:
            raise ValueError("human_mesh is required for target_type='human_mesh'")
        vertices, faces = load_human_mesh(
            human_mesh,
            filename=human_mesh_filename,
        )
        vertices = _place_loaded_mesh(
            vertices,
            range_m=target_position[0],
            lateral_m=target_position[1],
            vertical_m=target_position[2],
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )
        parameters = {
            "mesh_filename": Path(human_mesh_filename or str(human_mesh)).name,
            "diffuse_reflection_coefficient": diffuse,
            "face_winding": "normalized_outward_at_load",
        }
        material_name = "human_tissue"
    parameters.update(
        {
            "position_m": [float(value) for value in target_position],
            "range_m": float(target_position[0]),
            "target_y_m": float(target_position[1]),
            "target_z_m": float(target_position[2]),
            "distance_to_radar_m": float(np.linalg.norm(target_position)),
            "yaw_deg": float(yaw_deg),
            "pitch_deg": float(pitch_deg),
            "roll_deg": float(roll_deg),
            "mesh_max_edge_m": float(mesh_edge_m),
            "material": material_name,
        }
    )
    if target_type == "human_mesh":
        parameters.update(
            {
                "relative_permittivity": 38.0,
                "conductivity_s_per_m": 1.5,
                "thickness_m": 0.01,
                "rt_diffuse_reflection_coefficient": diffuse,
                "rt_backscattering_lambda": 0.35,
            }
        )
    else:
        preset = _MATERIALS[material_name]
        parameters.update(
            {
                "relative_permittivity": preset["relative_permittivity"],
                "conductivity_s_per_m": preset["conductivity_s_per_m"],
                "thickness_m": preset["thickness_m"],
                "rt_diffuse_reflection_coefficient": preset[
                    "diffuse_coefficient"
                ],
                "rt_backscattering_lambda": preset[
                    "backscattering_lambda"
                ],
            }
        )
    radar = _radar(
        board_model,
        yaw_deg=radar_yaw_deg,
        pitch_deg=radar_pitch_deg,
        roll_deg=radar_roll_deg,
        antenna_pattern_mode=resolved_pattern_mode,
        cosine_3db_beamwidth_deg=resolved_cosine_beamwidth_deg,
        carrier_frequency_hz=carrier_frequency_hz,
        slope_hz_per_s=slope_hz_per_s,
        chirp_duration_s=chirp_duration_s,
        chirp_repetition_time_s=chirp_repetition_time_s,
        sampling_frequency_hz=sampling_frequency_hz,
        num_adc_samples=num_adc_samples,
        num_chirps_per_frame=num_chirps,
        frame_period_s=frame_period_s,
        tdm_enabled=tdm_enabled,
        selected_tx_indices=selected_tx_indices,
        selected_rx_indices=selected_rx_indices,
    )
    effective_tx_indices = (
        list(range(get_ti_board_spec(board_model).num_tx))
        if selected_tx_indices is None
        else [int(value) for value in selected_tx_indices]
    )
    effective_rx_indices = (
        list(range(get_ti_board_spec(board_model).num_rx))
        if selected_rx_indices is None
        else [int(value) for value in selected_rx_indices]
    )
    po_material, rt_material = _materials(target_type, material_name, diffuse)

    check_cancelled()
    po_started = time.perf_counter()
    po_adc, face_power, face_visibility = _po_cube(
        vertices,
        faces,
        radar,
        po_material,
        visibility_samples=visibility_samples,
        integration_mode=po_integration_mode,
        quadrature_phase_span_scale_rad=po_phase_span,
        quadrature_max_refinement_depth=po_refinement_depth,
        quadrature_max_subfaces_per_parent=po_subface_cap,
        compute_backend=compute_backend,
    )
    po_runtime = time.perf_counter() - po_started
    check_cancelled()

    from sionna.rt import load_scene  # pylint: disable=import-outside-toplevel

    target = MeshTarget(
        name=f"gui_{target_type}",
        mesh_sequence=MeshSequence(
            vertices=vertices[None, :, :],
            faces=faces,
            times=np.asarray([0.0]),
        ),
        material=rt_material,
    )
    rt_simulator = MmWaveRadarSimulator(
        mobility_mode="rt_retrace",
        max_depth=rt_depth,
        samples_per_src=rt_samples,
        max_num_paths_per_src=rt_path_cap,
        coupling_mode="unrestricted",
        diffuse_reflection=True,
        human_specular_reflection=True,
        retrace_once_per_frame=True,
        seed=42,
        compute_backend=compute_backend,
        compute_precision="float32",
        adc_compute_precision="float32",
        cancel_check=cancel_check,
    )
    rt_started = time.perf_counter()
    rt_cube = rt_simulator.run(load_scene(), radar, [target], num_frames=1)
    rt_runtime = time.perf_counter() - rt_started
    check_cancelled()
    rt_adc = np.asarray(rt_cube.adc)
    po_profile, ranges_m = _profile(po_adc, radar.fmcw)
    rt_profile, rt_ranges = _profile(rt_adc, radar.fmcw)
    if not np.allclose(ranges_m, rt_ranges):
        raise RuntimeError("RT and PO range axes do not match")
    manifest = ExperimentManifest(
        name=f"Static target RT/PO: {target_type}",
        scene=SceneExperimentConfig(
            scenario=f"static_{target_type}",
            parameters=parameters,
        ),
        radar=RadarExperimentConfig(
            board_model=board_model,
            profile=f"gui_static_{radar.fmcw.carrier_frequency / 1e9:g}ghz",
            parameters={
                "carrier_frequency_hz": float(radar.fmcw.carrier_frequency),
                "slope_hz_per_s": float(radar.fmcw.slope),
                "chirp_duration_s": float(radar.fmcw.chirp_duration),
                "chirp_repetition_time_s": float(
                    radar.fmcw.chirp_repetition_time
                ),
                "sampling_frequency_hz": float(
                    radar.fmcw.sampling_frequency
                ),
                "num_adc_samples": int(radar.fmcw.num_adc_samples),
                "num_chirps_per_frame": int(radar.fmcw.num_chirps_per_frame),
                "frame_period_s": float(radar.fmcw.frame_period),
                "tdm_enabled": bool(tdm_enabled),
                "tx_indices_zero_based": effective_tx_indices,
                "rx_indices_zero_based": effective_rx_indices,
                "radar_yaw_deg": float(radar_yaw_deg),
                "radar_pitch_deg": float(radar_pitch_deg),
                "radar_roll_deg": float(radar_roll_deg),
                "antenna_pattern_mode": resolved_pattern_mode,
                "cosine_3db_beamwidth_deg": float(
                    resolved_cosine_beamwidth_deg
                ),
            },
        ),
        solver=SolverExperimentConfig(
            mode="hybrid_rt_po",
            fidelity=fidelity,
            parameters={
                "solvers": ["rt", "po"],
                "rt_max_depth": rt_depth,
                "rt_samples_per_source": rt_samples,
                "rt_max_paths_per_source": rt_path_cap,
                "po_visibility_samples_per_face": visibility_samples,
                "po_integration_mode": po_integration_mode,
                "po_quadrature_phase_span_scale_rad": po_phase_span,
                "po_quadrature_max_refinement_depth": po_refinement_depth,
                "po_quadrature_max_subfaces_per_parent": po_subface_cap,
                "compute_backend": compute_backend,
                "amplitude_calibration": "none",
            },
        ),
    )
    sensor_parameters = {
        "board_model": board_model,
        "position": [float(value) for value in radar.position],
        "orientation": [float(value) for value in radar.orientation],
        "orientation_deg": [
            float(radar_yaw_deg),
            float(radar_pitch_deg),
            float(radar_roll_deg),
        ],
        "pattern_mode": resolved_pattern_mode,
        "cosine_3db_beamwidth_deg": resolved_cosine_beamwidth_deg,
        "tdm_enabled": bool(tdm_enabled),
        "tx_indices_zero_based": effective_tx_indices,
        "rx_indices_zero_based": effective_rx_indices,
        "fmcw": {
            "carrier_frequency_hz": float(radar.fmcw.carrier_frequency),
            "slope_hz_per_s": float(radar.fmcw.slope),
            "chirp_duration_s": float(radar.fmcw.chirp_duration),
            "chirp_repetition_time_s": float(
                radar.fmcw.chirp_repetition_time
            ),
            "sampling_frequency_hz": float(radar.fmcw.sampling_frequency),
            "num_adc_samples": int(radar.fmcw.num_adc_samples),
            "num_chirps_per_frame": int(radar.fmcw.num_chirps_per_frame),
            "frame_period_s": float(radar.fmcw.frame_period),
            "num_tx": int(radar.fmcw.num_tx),
            "tdm_enabled": bool(radar.fmcw.tdm_enabled),
        },
    }
    path_counts = np.asarray(rt_cube.metadata.path_counts)
    rt_face_hit_counts = np.asarray(
        rt_simulator.last_target_face_hit_counts.get(
            target.name,
            np.zeros(faces.shape[0], dtype=np.int64),
        ),
        dtype=np.int64,
    )
    return PhysicsMicroscopeResult(
        manifest=manifest,
        target_type=target_type,
        material_name=material_name,
        target_parameters=parameters,
        sensor_parameters=sensor_parameters,
        vertices=vertices,
        faces=faces,
        face_power=face_power,
        face_visibility=face_visibility,
        rt_face_hit_counts=rt_face_hit_counts,
        po_adc=po_adc,
        rt_adc=rt_adc,
        adc_times_s=np.asarray(rt_cube.times),
        ranges_m=ranges_m,
        po_range_profile_power=po_profile,
        rt_range_profile_power=rt_profile,
        po_runtime_s=float(po_runtime),
        rt_runtime_s=float(rt_runtime),
        visible_face_count=int(np.count_nonzero(face_visibility > 0.0)),
        radar_position=np.asarray(radar.position, dtype=float),
        rt_path_count=int(np.max(path_counts)) if path_counts.size else 0,
    )


def run_static_target_diagnostics(
    result: PhysicsMicroscopeResult,
    *,
    top_k: int = 12,
    rt_samples_per_source: int | None = None,
    rt_max_paths_per_source: int | None = None,
    rt_max_depth: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> SolverDiagnostics:
    """Trace one static state and return reusable RT-path/PO-facet diagnostics.

    The main GUI simulation intentionally does not retain Sionna's heavyweight
    ``Paths`` object. This explicit diagnostic action retraces the current
    state only when requested, then converts it to compact NumPy arrays.
    """

    if cancel_check is not None and not callable(cancel_check):
        raise ValueError("cancel_check must be callable or None")
    if cancel_check is not None and bool(cancel_check()):
        raise InterruptedError("Simulation cancelled by user")

    sensor = dict(result.sensor_parameters)
    fmcw = dict(sensor["fmcw"])
    orientation_deg = np.asarray(
        sensor.get("orientation_deg", (0.0, 0.0, 0.0)),
        dtype=float,
    )
    if orientation_deg.shape != (3,):
        raise ValueError("sensor orientation_deg must have shape [3]")
    solver_parameters = dict(result.manifest.solver.parameters)
    samples = int(
        solver_parameters.get("rt_samples_per_source", 100_000)
        if rt_samples_per_source is None
        else rt_samples_per_source
    )
    path_cap = int(
        solver_parameters.get("rt_max_paths_per_source", 5_000)
        if rt_max_paths_per_source is None
        else rt_max_paths_per_source
    )
    depth = int(
        solver_parameters.get("rt_max_depth", 3)
        if rt_max_depth is None
        else rt_max_depth
    )
    radar = _radar(
        str(sensor["board_model"]),
        yaw_deg=float(orientation_deg[0]),
        pitch_deg=float(orientation_deg[1]),
        roll_deg=float(orientation_deg[2]),
        antenna_pattern_mode=str(sensor.get("pattern_mode", "cosine")),
        cosine_3db_beamwidth_deg=float(
            sensor.get("cosine_3db_beamwidth_deg", 60.0)
        ),
        carrier_frequency_hz=float(fmcw["carrier_frequency_hz"]),
        slope_hz_per_s=float(fmcw["slope_hz_per_s"]),
        chirp_duration_s=float(fmcw["chirp_duration_s"]),
        chirp_repetition_time_s=float(fmcw["chirp_repetition_time_s"]),
        sampling_frequency_hz=float(fmcw["sampling_frequency_hz"]),
        num_adc_samples=int(fmcw["num_adc_samples"]),
        num_chirps_per_frame=int(fmcw["num_chirps_per_frame"]),
        frame_period_s=float(fmcw["frame_period_s"]),
        tdm_enabled=bool(sensor.get("tdm_enabled", False)),
        selected_tx_indices=tuple(
            sensor.get("tx_indices_zero_based", ())
        )
        or None,
        selected_rx_indices=tuple(
            sensor.get("rx_indices_zero_based", ())
        )
        or None,
    )
    diffuse = float(
        result.target_parameters.get(
            "rt_diffuse_reflection_coefficient",
            0.0,
        )
    )
    _, rt_material = _materials(
        result.target_type,
        result.material_name,
        diffuse,
    )
    target = MeshTarget(
        name=f"gui_diagnostic_{result.target_type}",
        mesh_sequence=MeshSequence(
            vertices=np.asarray(result.vertices, dtype=np.float32)[None, :, :],
            faces=np.asarray(result.faces),
            times=np.asarray([0.0]),
        ),
        material=rt_material,
    )
    simulator = MmWaveRadarSimulator(
        mobility_mode="rt_retrace",
        max_depth=depth,
        samples_per_src=samples,
        max_num_paths_per_src=path_cap,
        coupling_mode="unrestricted",
        diffuse_reflection=True,
        human_specular_reflection=True,
        seed=42,
        compute_backend="numpy",
        compute_precision="float32",
        adc_compute_precision="float32",
        cancel_check=cancel_check,
    )
    from sionna.rt import load_scene  # pylint: disable=import-outside-toplevel

    paths = simulator.trace_paths(load_scene(), radar, [target], time=0.0)
    if cancel_check is not None and bool(cancel_check()):
        raise InterruptedError("Simulation cancelled by user")
    object_id = int(
        np.asarray(target.ensure_scene_object().object_id).reshape(-1)[0]
    )
    return SolverDiagnostics(
        rt=summarize_rt_paths(
            paths,
            top_k=top_k,
            required_object_ids=(object_id,),
        ),
        po=summarize_po_surface(
            result.vertices,
            result.faces,
            result.face_power,
            result.face_visibility,
            top_k=top_k,
        ),
    )


def run_physics_microscope(
    *,
    aspect_deg: float = 20.0,
    range_m: float = 2.0,
    plate_width_m: float = 0.30,
    plate_height_m: float = 0.30,
    fidelity: str = "preview",
    board_model: str = "IWR6843AOPEVM",
    compute_backend: str = "numpy",
) -> PhysicsMicroscopeResult:
    """Compatibility wrapper for the original plate experiment API."""

    return run_static_target_experiment(
        target_type="plate",
        range_m=range_m,
        width_m=plate_width_m,
        height_m=plate_height_m,
        yaw_deg=aspect_deg,
        fidelity=fidelity,
        board_model=board_model,
        compute_backend=compute_backend,
    )


__all__ = [
    "MATERIAL_PRESETS",
    "material_preset_parameters",
    "PhysicsMicroscopeResult",
    "TARGET_TYPES",
    "load_human_mesh",
    "make_rectangular_plate_mesh",
    "make_trihedral_corner_mesh",
    "run_physics_microscope",
    "run_static_target_experiment",
]
