# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Prepared-room human-motion workflow used by the public GUI."""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
import importlib
import operator
import os
from pathlib import Path
import time
from typing import Callable, Mapping
import xml.etree.ElementTree as ET

import numpy as np

from bundle_prepare.safe_npz import (
    DEFAULT_NPZ_LIMITS,
    preflight_npz,
    require_array,
)

from ..dsp import (
    framewise_range_doppler_map,
    range_fft,
    range_profile_from_cube,
    range_time_map,
)
from ..materials import human_skin_material
from ..radar import FMCWConfig, RadarHardware, RadarSensor, get_ti_board_spec
from ..simulation import (
    HumanPOMobilityConfig,
    MmWaveRadarSimulator,
    POCalibrationConfig,
    PhysicalOpticsMaterial,
    po_channel,
    run_calibrated_hybrid_po,
)
from ..targets import (
    AMASSSMPLMotionSequence,
    MeshSequence,
    MeshTarget,
    ensure_outward_face_winding,
)
from ..tutorial_support import (
    BEDROOM_SCENE_BOXES,
    bedroom_scene_xml_for_boxes,
)
from .manifest import (
    ExperimentManifest,
    RadarExperimentConfig,
    SceneExperimentConfig,
    SolverExperimentConfig,
)
from .solver_diagnostics import (
    POSurfaceDiagnostics,
    RTPathDiagnostics,
    summarize_po_surface,
    summarize_rt_paths,
)


PREPARED_ROOM_PRESETS = ("tutorial_bedroom",)
HUMAN_ROOM_SIMULATION_MODES = (
    "full_rt",
    "coherent_rt",
    "human_only_po",
    "hybrid_po",
)
_ROOM_RADAR_HEIGHT_M = 1.0
_MAX_SCENE_XML_BYTES = 8 * 1024**2
_SUPPORTED_SMPL_MODEL_TYPES = frozenset({"smpl", "smplh", "smplx"})
_SUPPORTED_SMPL_GENDERS = frozenset({"female", "male", "neutral"})
@dataclass(frozen=True)
class HumanRoomPreview:
    """Transformed motion sequence plus lightweight prepared-room geometry."""

    room_preset: str
    room_boxes: tuple[dict[str, object], ...]
    mesh_sequence: object
    human_position_m: np.ndarray
    human_yaw_deg: float
    source_name: str
    motion_payload: Mapping[str, np.ndarray] | None = None
    scene_xml: bytes | None = None
    scene_source_name: str = "tutorial_bedroom"
    motion_coordinate_settings: Mapping[str, object] | None = None


@dataclass(frozen=True)
class HumanRoomResult:
    """One prepared-room RT, PO, or calibrated Hybrid PO result."""

    manifest: ExperimentManifest
    simulation_mode: str
    preview: HumanRoomPreview
    radar_position_m: np.ndarray
    radar_orientation_rad: np.ndarray
    adc: np.ndarray
    adc_times_s: np.ndarray
    ranges_m: np.ndarray
    range_profiles: dict[str, np.ndarray]
    range_time_power: np.ndarray
    range_time_ranges_m: np.ndarray
    range_doppler_power: np.ndarray
    range_doppler_ranges_m: np.ndarray
    velocities_mps: np.ndarray
    components: dict[str, np.ndarray]
    metadata: object
    runtime_s: float
    scene_xml: bytes | None = None
    scene_source_name: str = "tutorial_bedroom"


@dataclass(frozen=True)
class HumanRoomDiagnostics:
    """Compact one-frame RT/PO overlays for a completed dynamic run."""

    frame_index: int
    source_frame_index: int
    rt: RTPathDiagnostics | None
    po: POSurfaceDiagnostics | None
    po_face_power: np.ndarray
    po_face_visibility: np.ndarray


@dataclass(frozen=True)
class LoadedAMASSMotion:
    """Validated human motion from pose parameters or evaluated mesh frames."""

    sequence: object
    payload: Mapping[str, np.ndarray] | None
    source_name: str


class _TransformedMotionSequence:
    """Apply archive axis metadata without eagerly evaluating every frame."""

    def __init__(self, base, *, axes=None, signs=None):
        self._base = base
        self._axes = None if axes is None else np.asarray(axes, dtype=np.int64)
        self._signs = (
            None if signs is None else np.asarray(signs, dtype=np.float32)
        )
        if self._axes is not None and (
            self._axes.shape != (3,)
            or sorted(self._axes.tolist()) != [0, 1, 2]
        ):
            raise ValueError("output_axes must be a permutation of [0, 1, 2]")
        if self._signs is not None and (
            self._signs.shape != (3,)
            or not np.all(np.isfinite(self._signs))
            or np.any(self._signs == 0.0)
        ):
            raise ValueError("output_signs must contain three finite nonzero values")
        self.times = np.asarray(base.times, dtype=float)
        faces = np.asarray(base.faces)
        transform = np.eye(3)[
            self._axes if self._axes is not None else np.arange(3)
        ]
        if self._signs is not None:
            transform = transform * self._signs[:, None]
        if np.linalg.det(transform) < 0.0:
            faces = faces[:, [0, 2, 1]]
        self.faces = ensure_outward_face_winding(
            self.vertices_at(float(self.times[0])),
            faces,
        )

    @property
    def vertices(self) -> np.ndarray:
        return np.stack(
            [self.vertices_at(float(value)) for value in self.times],
            axis=0,
        )

    @property
    def vertex_count(self) -> int:
        return int(self._base.vertex_count)

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def vertices_at(self, time_s: float) -> np.ndarray:
        vertices = np.asarray(self._base.vertices_at(float(time_s)))
        if self._axes is not None:
            vertices = vertices[..., self._axes]
        if self._signs is not None:
            vertices = vertices * self._signs
        return np.asarray(vertices, dtype=np.float32)

    def max_vertex_displacement(
        self,
        time_s: float,
        reference_vertices: np.ndarray,
    ) -> float:
        vertices = self.vertices_at(time_s)
        reference = np.asarray(reference_vertices, dtype=np.float32)
        if reference.shape != vertices.shape:
            raise ValueError("reference_vertices shape mismatch")
        return float(np.max(np.linalg.norm(vertices - reference, axis=1)))


class _PlacedMotionSequence:
    """Place a motion lazily in the radar-relative GUI coordinate frame."""

    def __init__(self, base, *, position, yaw_deg: float):
        self._base = base
        self._time_offset = float(np.asarray(base.times, dtype=float)[0])
        self.times = np.asarray(base.times, dtype=float) - self._time_offset
        initial = np.asarray(base.vertices_at(self._time_offset), dtype=float)
        self._center = 0.5 * (
            np.min(initial, axis=0) + np.max(initial, axis=0)
        )
        self._rotation = _rotation_z(yaw_deg)
        self._position = _vector3("human_position_m", position)
        self.faces = ensure_outward_face_winding(
            self.vertices_at(0.0),
            np.asarray(base.faces),
        )

    @property
    def vertices(self) -> np.ndarray:
        return np.stack(
            [self.vertices_at(float(value)) for value in self.times],
            axis=0,
        )

    @property
    def vertex_count(self) -> int:
        return int(self._base.vertex_count)

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def vertices_at(self, time_s: float) -> np.ndarray:
        vertices = np.asarray(
            self._base.vertices_at(float(time_s) + self._time_offset),
            dtype=float,
        )
        vertices = (vertices - self._center) @ self._rotation.T
        vertices += self._position
        return vertices.astype(np.float32)

    def max_vertex_displacement(
        self,
        time_s: float,
        reference_vertices: np.ndarray,
    ) -> float:
        vertices = self.vertices_at(time_s)
        reference = np.asarray(reference_vertices, dtype=np.float32)
        if reference.shape != vertices.shape:
            raise ValueError("reference_vertices shape mismatch")
        return float(np.max(np.linalg.norm(vertices - reference, axis=1)))


class _FrameWindowMotionSequence:
    """Expose an inclusive source-frame window on radar frame timing."""

    def __init__(
        self,
        base,
        *,
        begin_frame_index: int,
        end_frame_index: int,
        frame_period_s: float,
    ):
        self._base = base
        self.begin_frame_index = int(begin_frame_index)
        self.end_frame_index = int(end_frame_index)
        self.source_frame_indices = np.arange(
            self.begin_frame_index,
            self.end_frame_index + 1,
            dtype=np.int64,
        )
        base_times = np.asarray(base.times, dtype=float)
        frame_period_s = float(frame_period_s)
        if not np.isfinite(frame_period_s) or frame_period_s <= 0.0:
            raise ValueError("frame_period_s must be a finite positive value")
        self._source_times = base_times[self.source_frame_indices]
        self.times = (
            np.arange(self.source_frame_indices.size, dtype=float)
            * frame_period_s
        )
        self._mapping_times = self.times
        self._mapping_source_times = self._source_times
        next_source_index = self.end_frame_index + 1
        if next_source_index < base_times.size:
            self._mapping_times = np.append(
                self._mapping_times,
                self.times[-1] + frame_period_s,
            )
            self._mapping_source_times = np.append(
                self._mapping_source_times,
                base_times[next_source_index],
            )
        self.faces = np.asarray(base.faces)

    @property
    def vertices(self) -> np.ndarray:
        return np.stack(
            [self.vertices_at(float(value)) for value in self.times],
            axis=0,
        )

    @property
    def vertex_count(self) -> int:
        return int(self._base.vertex_count)

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def vertices_at(self, time_s: float) -> np.ndarray:
        time_s = float(time_s)
        source_time_s = float(
            np.interp(
                time_s,
                self._mapping_times,
                self._mapping_source_times,
            )
        )
        return np.asarray(
            self._base.vertices_at(source_time_s),
            dtype=np.float32,
        )

    def max_vertex_displacement(
        self,
        time_s: float,
        reference_vertices: np.ndarray,
    ) -> float:
        vertices = self.vertices_at(time_s)
        reference = np.asarray(reference_vertices, dtype=np.float32)
        if reference.shape != vertices.shape:
            raise ValueError("reference_vertices shape mismatch")
        return float(np.max(np.linalg.norm(vertices - reference, axis=1)))


def _resolve_motion_frame_window(
    *,
    total_frames: int,
    begin_frame_index: int = 0,
    end_frame_index: int | None = None,
    num_frames: int | None = None,
) -> tuple[int, int]:
    """Validate and resolve an inclusive motion-frame window."""

    try:
        total = operator.index(total_frames)
        begin = operator.index(begin_frame_index)
    except TypeError as exc:
        raise ValueError("motion frame indices must be integers") from exc
    if total <= 0:
        raise ValueError("human motion must contain at least one frame")
    if begin < 0 or begin >= total:
        raise ValueError(
            f"begin_frame_index must be in [0, {total - 1}]"
        )
    if end_frame_index is not None and num_frames is not None:
        raise ValueError(
            "specify end_frame_index or num_frames, not both"
        )
    if end_frame_index is not None:
        try:
            end = operator.index(end_frame_index)
        except TypeError as exc:
            raise ValueError("motion frame indices must be integers") from exc
    elif num_frames is not None:
        try:
            count = operator.index(num_frames)
        except TypeError as exc:
            raise ValueError("num_frames must be a positive integer") from exc
        if count <= 0:
            raise ValueError("num_frames must be a positive integer")
        end = begin + count - 1
    else:
        end = begin
    if end < begin or end >= total:
        raise ValueError(
            f"end_frame_index must be in [{begin}, {total - 1}]"
        )
    return int(begin), int(end)


def _rotation_z(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    if not np.isfinite(yaw):
        raise ValueError("human_yaw_deg must be finite")
    return np.asarray(
        (
            (np.cos(yaw), -np.sin(yaw), 0.0),
            (np.sin(yaw), np.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )


def _vector3(name: str, value) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def motion_source_to_radar_transform(
    up_axis: str,
    forward_axis: str,
    *,
    handedness: str = "right",
    length_unit: str = "m",
) -> np.ndarray:
    """Map a declared source frame to radar +x forward, +y left, +z up.

    The returned matrix acts on column vectors and includes the conversion
    to metres. Source axes describe a fixed reference frame, not the person's
    changing heading. Apply this to evaluated meshes, not local joint angles.
    """

    directions = {
        f"{sign}{axis}": multiplier * np.eye(3)[index]
        for index, axis in enumerate("xyz")
        for sign, multiplier in (("+", 1.0), ("-", -1.0))
    }
    if up_axis not in directions or forward_axis not in directions:
        raise ValueError("motion axes must be one of +x, -x, +y, -y, +z, -z")
    up = directions[up_axis]
    forward = directions[forward_axis]
    if np.dot(up, forward) != 0.0:
        raise ValueError("motion up and forward axes must be perpendicular")
    if handedness not in {"right", "left"}:
        raise ValueError("motion handedness must be 'right' or 'left'")
    scales = {"m": 1.0, "cm": 0.01, "mm": 0.001}
    if length_unit not in scales:
        raise ValueError("motion length_unit must be 'm', 'cm', or 'mm'")
    left = np.cross(up, forward)
    if handedness == "left":
        left = -left
    return scales[length_unit] * np.stack((forward, left, up))


def _npz_scalar_string(data, key: str, default: str) -> str:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"AMASS-like {key} must be a scalar string")
    scalar = value.item()
    if isinstance(scalar, (bytes, np.bytes_)):
        scalar = bytes(scalar).decode("utf-8")
    return str(scalar)


def _amass_times(data, frame_count: int) -> np.ndarray:
    if "bundle_times" in data.files:
        times = np.asarray(data["bundle_times"], dtype=float)
    elif "times" in data.files:
        times = np.asarray(data["times"], dtype=float)
    elif "mocap_framerate" in data.files:
        framerate = float(np.asarray(data["mocap_framerate"]).reshape(()))
        if not np.isfinite(framerate) or framerate <= 0.0:
            raise ValueError("AMASS-like mocap_framerate must be positive")
        times = np.arange(frame_count, dtype=float) / framerate
    else:
        raise ValueError(
            "AMASS-like motion requires times, bundle_times, or mocap_framerate"
        )
    if times.shape != (frame_count,):
        raise ValueError("AMASS-like motion times must have shape [T]")
    return times


def _template_faces(
    *,
    smpl_model_dir: str,
    model_type: str,
    gender: str,
) -> np.ndarray:
    try:
        smplx = importlib.import_module("smplx")
    except ImportError as exc:
        raise ImportError(
            "AMASS motion requires optional dependency 'smplx'"
        ) from exc
    model = smplx.create(
        model_path=smpl_model_dir,
        model_type=model_type,
        gender=gender,
        batch_size=1,
        use_pca=False,
    )
    return np.asarray(model.faces, dtype=np.uint32)


def _load_evaluated_mesh_motion(source, arrays, *, source_name: str):
    """Load supplied mesh frames without pairing their faces with SMPL."""

    vertices = require_array(
        arrays,
        "vertices",
        allowed_ndim=(3,),
        dtype_kinds=frozenset({"i", "u", "f"}),
        trailing_shape=(3,),
        max_elements=120_000_000,
    )
    require_array(
        arrays,
        "faces",
        allowed_ndim=(2,),
        dtype_kinds=frozenset({"i", "u"}),
        trailing_shape=(3,),
        max_elements=6_000_000,
        max_bytes=32 * 1024**2,
    )
    for name in ("times", "bundle_times"):
        if name in arrays:
            timing = require_array(
                arrays,
                name,
                allowed_ndim=(1,),
                dtype_kinds=frozenset({"i", "u", "f"}),
                max_elements=10_000_000,
            )
            if timing.shape != (vertices.shape[0],):
                raise ValueError(f"mesh motion {name} must match its frame count")
    if "mocap_framerate" in arrays:
        require_array(
            arrays,
            "mocap_framerate",
            allowed_ndim=(0, 1),
            dtype_kinds=frozenset({"i", "u", "f"}),
            max_elements=1,
        )
    for name, kinds in (
        ("output_axes", frozenset({"i", "u"})),
        ("output_signs", frozenset({"i", "u", "f"})),
    ):
        if name in arrays:
            require_array(
                arrays,
                name,
                allowed_ndim=(1,),
                dtype_kinds=kinds,
                trailing_shape=(3,),
                max_elements=3,
            )

    raw_source = BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    with np.load(raw_source, allow_pickle=False) as data:
        base = MeshSequence(
            vertices=data["vertices"],
            faces=data["faces"],
            times=_amass_times(data, vertices.shape[0]),
        )
        axes = data["output_axes"] if "output_axes" in data.files else None
        signs = data["output_signs"] if "output_signs" in data.files else None
    return LoadedAMASSMotion(
        sequence=_TransformedMotionSequence(base, axes=axes, signs=signs),
        payload=None,
        source_name=source_name,
    )


def load_amass_motion(
    source: str | Path | bytes | bytearray | BytesIO,
    *,
    filename: str | None = None,
    smpl_model_dir: str | Path | None = None,
) -> LoadedAMASSMotion:
    """Load evaluated mesh frames, or evaluate pose-only archives using SMPL."""

    inferred_name = filename
    if inferred_name is None and isinstance(source, (str, Path)):
        inferred_name = Path(source).name
    suffix = Path(inferred_name or "").suffix.casefold()
    if suffix and suffix != ".npz":
        raise ValueError("human motion must be a .npz archive")

    arrays = preflight_npz(source, limits=DEFAULT_NPZ_LIMITS)
    if "vertices" in arrays:
        return _load_evaluated_mesh_motion(
            source,
            arrays,
            source_name=Path(inferred_name or "human_motion.npz").name,
        )
    poses_info = require_array(
        arrays,
        "poses",
        allowed_ndim=(2,),
        dtype_kinds=frozenset({"i", "u", "f"}),
        max_elements=30_000_000,
    )
    trans_info = require_array(
        arrays,
        "trans",
        allowed_ndim=(2,),
        dtype_kinds=frozenset({"i", "u", "f"}),
        trailing_shape=(3,),
        max_elements=30_000_000,
    )
    require_array(
        arrays,
        "betas",
        allowed_ndim=(1,),
        dtype_kinds=frozenset({"i", "u", "f"}),
        max_elements=10_000,
    )
    if "faces" in arrays:
        require_array(
            arrays,
            "faces",
            allowed_ndim=(2,),
            dtype_kinds=frozenset({"i", "u"}),
            trailing_shape=(3,),
            max_elements=6_000_000,
            max_bytes=32 * 1024**2,
        )
    for name, maximum_bytes in (
        ("model_type", 256),
        ("gender", 256),
        ("smpl_model_dir", 4 * 1024),
    ):
        if name in arrays:
            require_array(
                arrays,
                name,
                allowed_ndim=(0,),
                dtype_kinds=frozenset({"S", "U"}),
                max_elements=1,
                max_bytes=maximum_bytes,
            )
    for name in ("times", "bundle_times"):
        if name in arrays:
            timing = require_array(
                arrays,
                name,
                allowed_ndim=(1,),
                dtype_kinds=frozenset({"i", "u", "f"}),
                max_elements=10_000_000,
            )
            if timing.shape != (poses_info.shape[0],):
                raise ValueError(
                    f"AMASS-like {name} must match the pose frame count"
                )
    for name, kinds in (
        ("output_axes", frozenset({"i", "u"})),
        ("output_signs", frozenset({"i", "u", "f"})),
    ):
        if name in arrays:
            require_array(
                arrays,
                name,
                allowed_ndim=(1,),
                dtype_kinds=kinds,
                trailing_shape=(3,),
                max_elements=3,
            )
    if "mocap_framerate" in arrays:
        require_array(
            arrays,
            "mocap_framerate",
            allowed_ndim=(0, 1),
            dtype_kinds=frozenset({"i", "u", "f"}),
            max_elements=1,
        )
    if poses_info.shape[0] <= 0 or trans_info.shape[0] != poses_info.shape[0]:
        raise ValueError("AMASS-like poses and trans frame counts must match")

    raw_source = source
    if isinstance(source, (bytes, bytearray)):
        raw_source = BytesIO(source)
    with np.load(raw_source, allow_pickle=False) as data:
        required = ("poses", "trans", "betas")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(
                f"AMASS-like motion is missing required arrays: {missing}"
            )
        poses = np.asarray(data["poses"], dtype=np.float32)
        translations = np.asarray(data["trans"], dtype=np.float32)
        betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
        if poses.ndim != 2 or poses.shape[0] == 0:
            raise ValueError("AMASS-like poses must have shape [T, pose_dim]")
        times = _amass_times(data, int(poses.shape[0]))
        model_type = _npz_scalar_string(
            data,
            "model_type",
            "smpl",
        ).strip().lower()
        gender = _npz_scalar_string(
            data,
            "gender",
            "neutral",
        ).strip().lower()
        if model_type not in _SUPPORTED_SMPL_MODEL_TYPES:
            raise ValueError(
                "AMASS-like model_type must be one of: "
                + ", ".join(sorted(_SUPPORTED_SMPL_MODEL_TYPES))
            )
        if gender not in _SUPPORTED_SMPL_GENDERS:
            raise ValueError(
                "AMASS-like gender must be one of: "
                + ", ".join(sorted(_SUPPORTED_SMPL_GENDERS))
            )
        archive_model_dir = _npz_scalar_string(data, "smpl_model_dir", "")
        axes = (
            np.asarray(data["output_axes"], dtype=np.int64)
            if "output_axes" in data.files
            else None
        )
        signs = (
            np.asarray(data["output_signs"], dtype=np.float32)
            if "output_signs" in data.files
            else None
        )
        raw_faces = (
            np.asarray(data["faces"])
            if "faces" in data.files
            else None
        )
        if raw_faces is not None:
            if (
                raw_faces.ndim != 2
                or raw_faces.shape[1] != 3
                or raw_faces.dtype.kind not in {"i", "u"}
                or raw_faces.size == 0
                or np.min(raw_faces) < 0
            ):
                raise ValueError(
                    "AMASS-like faces must have non-negative integer shape [F, 3]"
                )
            faces = np.asarray(raw_faces, dtype=np.uint32)
        else:
            faces = None
        payload = {
            "poses": poses.copy(),
            "trans": translations.copy(),
            "betas": betas.copy(),
            "times": (
                np.asarray(data["times"], dtype=float).copy()
                if "times" in data.files
                else np.asarray(times, dtype=float).copy()
            ),
            "bundle_times": (
                np.asarray(times, dtype=float) - float(times[0])
            ),
            "gender": np.asarray(gender),
            "model_type": np.asarray(model_type),
        }
        if "mocap_framerate" in data.files:
            payload["mocap_framerate"] = np.asarray(
                data["mocap_framerate"]
            ).copy()
        if axes is not None:
            payload["output_axes"] = axes.copy()
        if signs is not None:
            payload["output_signs"] = signs.copy()

    configured_model_dir = (
        str(Path(smpl_model_dir).expanduser())
        if smpl_model_dir
        else archive_model_dir
        or os.environ.get("MMWAVE_SMPL_MODEL_DIR", "")
    )
    if not configured_model_dir:
        raise ValueError(
            "AMASS-like motion requires a licensed SMPL model directory"
        )
    if not Path(configured_model_dir).expanduser().is_dir():
        raise FileNotFoundError(
            f"SMPL model directory was not found: {configured_model_dir}"
        )
    if faces is None:
        faces = _template_faces(
            smpl_model_dir=configured_model_dir,
            model_type=model_type,
            gender=gender,
        )
    payload["faces"] = np.asarray(faces, dtype=np.uint32).copy()
    base = AMASSSMPLMotionSequence(
        poses=poses,
        betas=betas,
        translations=translations,
        faces=faces,
        times=times,
        smpl_model_dir=configured_model_dir,
        model_type=model_type,
        gender=gender,
    )
    sequence = _TransformedMotionSequence(base, axes=axes, signs=signs)
    return LoadedAMASSMotion(
        sequence=sequence,
        payload=payload,
        source_name=Path(inferred_name or "amass_motion.npz").name,
    )


def load_human_motion(
    source: str | Path | bytes | bytearray | BytesIO,
    *,
    filename: str | None = None,
    smpl_model_dir: str | Path | None = None,
):
    """Backward-compatible alias returning the evaluated human motion object."""

    return load_amass_motion(
        source,
        filename=filename,
        smpl_model_dir=smpl_model_dir,
    ).sequence


def load_scene_xml(
    source: str | Path | bytes | bytearray | BytesIO,
    *,
    filename: str | None = None,
) -> tuple[bytes, str]:
    """Load and minimally validate a UTF-8 Sionna/Mitsuba scene document."""

    inferred_name = filename
    if inferred_name is None and isinstance(source, (str, Path)):
        inferred_name = Path(source).name
    if Path(inferred_name or "scene.xml").suffix.casefold() != ".xml":
        raise ValueError("dynamic scene must be an .xml file")
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > _MAX_SCENE_XML_BYTES:
            raise ValueError("scene XML exceeds the 8 MiB size limit")
        payload = path.read_bytes()
    elif isinstance(source, BytesIO):
        payload = source.getvalue()
    else:
        payload = bytes(source)
    if len(payload) > _MAX_SCENE_XML_BYTES:
        raise ValueError("scene XML exceeds the 8 MiB size limit")
    try:
        root = ET.fromstring(payload.decode("utf-8"))
    except (UnicodeDecodeError, ET.ParseError) as exc:
        raise ValueError("scene XML must be well-formed UTF-8") from exc
    if root.tag.rsplit("}", maxsplit=1)[-1] != "scene":
        raise ValueError("scene XML root element must be <scene>")
    return payload, Path(inferred_name or "scene.xml").name


def prepared_room_boxes(
    room_preset: str = "tutorial_bedroom",
) -> tuple[dict[str, object], ...]:
    """Return curated room boxes expressed in the radar-relative frame."""

    if room_preset not in PREPARED_ROOM_PRESETS:
        raise ValueError(
            f"room_preset must be one of {PREPARED_ROOM_PRESETS}"
        )
    room_boxes = []
    for source_box in BEDROOM_SCENE_BOXES:
        box = dict(source_box)
        translation = np.asarray(box["translate"], dtype=float)
        translation[2] -= _ROOM_RADAR_HEIGHT_M
        box["translate"] = tuple(float(value) for value in translation)
        room_boxes.append(box)
    return tuple(room_boxes)


def prepare_human_room_preview(
    human_motion: (
        str
        | Path
        | bytes
        | bytearray
        | BytesIO
        | MeshSequence
        | LoadedAMASSMotion
    ),
    *,
    human_motion_filename: str | None = None,
    smpl_model_dir: str | Path | None = None,
    room_preset: str = "tutorial_bedroom",
    scene_xml: str | Path | bytes | bytearray | BytesIO | None = None,
    scene_xml_filename: str | None = None,
    human_position_m: tuple[float, float, float] = (1.6, 0.0, 0.0),
    human_yaw_deg: float = 0.0,
    motion_source_up_axis: str | None = None,
    motion_source_forward_axis: str | None = None,
    motion_source_handedness: str = "right",
    motion_source_length_unit: str = "m",
) -> HumanRoomPreview:
    """Place pose-driven or evaluated mesh motion in a static scene."""

    if room_preset not in PREPARED_ROOM_PRESETS:
        raise ValueError(
            f"room_preset must be one of {PREPARED_ROOM_PRESETS}"
        )
    if isinstance(human_motion, LoadedAMASSMotion):
        sequence = human_motion.sequence
        motion_payload = human_motion.payload
        source_name = human_motion.source_name
    elif isinstance(human_motion, MeshSequence):
        sequence = human_motion
        motion_payload = None
        source_name = Path(
            human_motion_filename or "human_motion.npz"
        ).name
    else:
        loaded = load_amass_motion(
            human_motion,
            filename=human_motion_filename,
            smpl_model_dir=smpl_model_dir,
        )
        sequence = loaded.sequence
        motion_payload = loaded.payload
        source_name = loaded.source_name
    coordinate_settings = None
    if motion_source_up_axis is not None or motion_source_forward_axis is not None:
        transform = motion_source_to_radar_transform(
            motion_source_up_axis,
            motion_source_forward_axis,
            handedness=motion_source_handedness,
            length_unit=motion_source_length_unit,
        )
        # Manual declarations refer to the original evaluated source mesh,
        # overriding output_axes/output_signs rather than applying them twice.
        base = (
            sequence._base
            if isinstance(sequence, _TransformedMotionSequence)
            else sequence
        )
        axes = np.argmax(np.abs(transform), axis=1)
        sequence = _TransformedMotionSequence(
            base,
            axes=axes,
            signs=transform[np.arange(3), axes],
        )
        coordinate_settings = {
            "up_axis": motion_source_up_axis,
            "forward_axis": motion_source_forward_axis,
            "handedness": motion_source_handedness,
            "length_unit": motion_source_length_unit,
            "source_to_radar_matrix": transform.tolist(),
        }
    position = _vector3("human_position_m", human_position_m)
    placed = _PlacedMotionSequence(
        sequence,
        position=position,
        yaw_deg=human_yaw_deg,
    )
    scene_payload = None
    scene_source_name = room_preset
    room_boxes = prepared_room_boxes(room_preset)
    if scene_xml is not None:
        scene_payload, scene_source_name = load_scene_xml(
            scene_xml,
            filename=scene_xml_filename,
        )
        room_boxes = ()
    return HumanRoomPreview(
        room_preset=room_preset,
        room_boxes=room_boxes,
        mesh_sequence=placed,
        human_position_m=position,
        human_yaw_deg=float(human_yaw_deg),
        source_name=source_name,
        motion_payload=motion_payload,
        scene_xml=scene_payload,
        scene_source_name=scene_source_name,
        motion_coordinate_settings=coordinate_settings,
    )


def _radar(
    board_model: str,
    *,
    orientation_deg,
    antenna_pattern_mode: str,
    cosine_3db_beamwidth_deg: float,
    carrier_frequency_hz: float,
    slope_hz_per_s: float,
    chirp_duration_s: float,
    chirp_repetition_time_s: float,
    sampling_frequency_hz: float,
    adc_mode: str = "complex1x",
    if_bandwidth_hz: float | None = None,
    num_adc_samples: int,
    num_chirps_per_frame: int,
    frame_period_s: float,
    tdm_enabled: bool,
    selected_tx_indices: tuple[int, ...] | list[int] | None = None,
    selected_rx_indices: tuple[int, ...] | list[int] | None = None,
) -> RadarSensor:
    orientation = _vector3("radar_orientation_deg", orientation_deg)
    hardware = RadarHardware.from_ti_board(
        board_model,
        pattern_mode=antenna_pattern_mode,
        cosine_half_power_angle_deg=0.5 * float(cosine_3db_beamwidth_deg),
    )
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
        carrier_frequency=float(carrier_frequency_hz),
        slope=float(slope_hz_per_s),
        chirp_duration=float(chirp_duration_s),
        chirp_repetition_time=float(chirp_repetition_time_s),
        sampling_frequency=float(sampling_frequency_hz),
        adc_mode=adc_mode,
        if_bandwidth_hz=if_bandwidth_hz,
        num_adc_samples=int(num_adc_samples),
        num_chirps_per_frame=int(num_chirps_per_frame),
        frame_period=float(frame_period_s),
        num_tx=hardware.num_tx,
        tdm_enabled=bool(tdm_enabled),
    )
    return RadarSensor(
        name="human-room-radar",
        position=(0.0, 0.0, 0.0),
        orientation=tuple(np.deg2rad(orientation)),
        hardware=hardware,
        fmcw=fmcw,
        tx_power_dbm=get_ti_board_spec(board_model).tx_power_dbm,
    )


def _load_environment_scene(room_boxes, *, scene_xml: bytes | None = None):
    """Load an uploaded XML scene or the tutorial room."""

    from sionna.rt import load_scene_from_string

    if scene_xml is not None:
        return load_scene_from_string(
            scene_xml.decode("utf-8"),
            merge_shapes=False,
        )

    xml = bedroom_scene_xml_for_boxes(room_boxes)
    return load_scene_from_string(xml, merge_shapes=False)


def run_human_room_diagnostics(
    result: HumanRoomResult,
    *,
    frame_index: int,
    top_k: int = 12,
    include_rt: bool = True,
    include_po: bool = True,
    cancel_check: Callable[[], bool] | None = None,
) -> HumanRoomDiagnostics:
    """Retrace one completed dynamic frame into compact scene overlays.

    Normal dynamic runs intentionally discard heavyweight Sionna path objects.
    This explicit action reconstructs the selected radar and human state, then
    retains only the strongest RT line segments and per-face PO power.
    """

    if cancel_check is not None and not callable(cancel_check):
        raise ValueError("cancel_check must be callable or None")

    def check_cancelled() -> None:
        if cancel_check is not None and bool(cancel_check()):
            raise InterruptedError("Simulation cancelled by user")

    check_cancelled()
    frame_index = int(frame_index)
    top_k = int(top_k)
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    sequence = result.preview.mesh_sequence
    frame_count = len(sequence.times)
    if frame_index < 0 or frame_index >= frame_count:
        raise ValueError(
            f"frame_index must be in [0, {max(frame_count - 1, 0)}]"
        )

    radar_parameters = dict(result.manifest.radar.parameters)
    solver_parameters = dict(result.manifest.solver.parameters)
    radar = _radar(
        result.manifest.radar.board_model,
        orientation_deg=radar_parameters.get(
            "orientation_deg", (0.0, 0.0, 0.0)
        ),
        antenna_pattern_mode=str(
            radar_parameters.get("antenna_pattern_mode", "cosine")
        ),
        cosine_3db_beamwidth_deg=float(
            radar_parameters.get("cosine_3db_beamwidth_deg", 60.0)
        ),
        carrier_frequency_hz=float(
            radar_parameters.get("carrier_frequency_hz", 60e9)
        ),
        slope_hz_per_s=float(
            radar_parameters.get("slope_hz_per_s", 68e12)
        ),
        chirp_duration_s=float(
            radar_parameters.get("chirp_duration_s", 58e-6)
        ),
        chirp_repetition_time_s=float(
            radar_parameters.get("chirp_repetition_time_s", 65e-6)
        ),
        adc_mode=radar_parameters.get("adc_mode", "complex1x"),
        if_bandwidth_hz=radar_parameters.get("if_bandwidth_hz"),
        sampling_frequency_hz=float(
            radar_parameters.get("sampling_frequency_hz", 4.5e6)
        ),
        num_adc_samples=int(radar_parameters.get("num_adc_samples", 225)),
        num_chirps_per_frame=int(
            radar_parameters.get("num_chirps_per_frame", 16)
        ),
        frame_period_s=float(
            radar_parameters.get("frame_period_s", 50e-3)
        ),
        tdm_enabled=bool(radar_parameters.get("tdm_enabled", False)),
        selected_tx_indices=tuple(
            radar_parameters.get("tx_indices_zero_based", ())
        )
        or None,
        selected_rx_indices=tuple(
            radar_parameters.get("rx_indices_zero_based", ())
        )
        or None,
    )
    time_s = float(sequence.times[frame_index])
    vertices = np.asarray(sequence.vertices_at(time_s), dtype=np.float32)
    faces = np.asarray(sequence.faces, dtype=np.uint32)
    environment_scene = _load_environment_scene(
        result.preview.room_boxes,
        scene_xml=result.scene_xml,
    )

    po_diagnostics = None
    face_power = np.zeros(faces.shape[0], dtype=float)
    face_visibility = np.zeros(faces.shape[0], dtype=float)
    if include_po:
        check_cancelled()
        po_result = po_channel(
            vertices=vertices,
            faces=faces,
            tx_positions=radar.world_tx_positions(),
            rx_positions=radar.world_rx_positions(),
            frequency_hz=radar.fmcw.carrier_frequency,
            material=PhysicalOpticsMaterial(
                relative_permittivity=38.0,
                conductivity_s_per_m=1.5,
                thickness_m=0.01,
                model="slab",
            ),
            visibility="fractional_shadow_fade",
            static_visibility_scene=environment_scene.mi_scene,
            visibility_samples_per_face=int(
                solver_parameters.get("po_visibility_samples_per_face", 1)
            ),
            visibility_fade_chirps=1,
            adaptive_visibility_sampling=True,
            po_integration_mode=str(
                solver_parameters.get(
                    "po_integration_mode", "parent_face_quadrature"
                )
            ),
            po_quadrature_phase_span_scale_rad=float(
                solver_parameters.get(
                    "po_quadrature_phase_span_scale_rad", 1.0
                )
            ),
            po_curvature_error_rad=float(solver_parameters.get("po_curvature_error_rad", 0.01)),
            po_quadrature_max_refinement_depth=int(
                solver_parameters.get(
                    "po_quadrature_max_refinement_depth", 2
                )
            ),
            po_quadrature_max_subfaces_per_parent=int(
                solver_parameters.get(
                    "po_quadrature_max_subfaces_per_parent", 16
                )
            ),
            compute_backend="numpy",
            compute_precision="float32",
        )
        selected_faces = np.asarray(po_result.face_indices, dtype=np.int64)
        contributions = np.asarray(po_result.phase_center_contributions)
        np.add.at(face_power, selected_faces, np.abs(contributions) ** 2)
        face_visibility = np.asarray(
            po_result.effective_weights,
            dtype=float,
        )
        po_diagnostics = summarize_po_surface(
            vertices,
            faces,
            face_power,
            face_visibility,
            top_k=top_k,
        )
        check_cancelled()

    rt_diagnostics = None
    if include_rt:
        material = human_skin_material("gui-human-room-diagnostic-skin")
        material.scattering_coefficient = float(
            result.manifest.scene.parameters.get(
                "diffuse_reflection_coefficient", 0.35
            )
        )
        target = MeshTarget(
            name="gui_human_room_diagnostic_subject",
            mesh_sequence=sequence,
            material=material,
        )
        simulator = MmWaveRadarSimulator(
            mobility_mode="rt_retrace",
            max_depth=int(solver_parameters.get("rt_max_depth", 3)),
            samples_per_src=int(
                solver_parameters.get("rt_samples_per_source", 100_000)
            ),
            max_num_paths_per_src=int(
                solver_parameters.get("rt_max_paths_per_source", 5_000)
            ),
            coupling_mode="unrestricted",
            diffuse_reflection=True,
            human_specular_reflection=True,
            seed=42,
            compute_backend="numpy",
            compute_precision="float32",
            adc_compute_precision="float32",
            cancel_check=cancel_check,
        )
        paths = simulator.trace_paths(
            environment_scene,
            radar,
            [target],
            time=time_s,
        )
        check_cancelled()
        object_id = int(
            np.asarray(target.ensure_scene_object().object_id).reshape(-1)[0]
        )
        rt_diagnostics = summarize_rt_paths(
            paths,
            top_k=top_k,
            required_object_ids=(object_id,),
        )

    source_begin = int(
        result.manifest.scene.parameters.get("motion_begin_frame_index", 0)
    )
    return HumanRoomDiagnostics(
        frame_index=frame_index,
        source_frame_index=source_begin + frame_index,
        rt=rt_diagnostics,
        po=po_diagnostics,
        po_face_power=face_power,
        po_face_visibility=face_visibility,
    )


def _range_profile(adc: np.ndarray, fmcw: FMCWConfig):
    adc = np.asarray(adc)
    if adc.ndim == 4:
        adc = adc.reshape((-1, adc.shape[-2], adc.shape[-1]))
    if adc.ndim != 3:
        raise ValueError(
            "adc must have shape [frame, chirp, sample, channel] or "
            "[chirp, sample, channel]"
        )
    range_cube, ranges_m = range_fft(
        adc,
        fmcw=fmcw,
        nfft_mult=4,
    )
    profile = range_profile_from_cube(
        range_cube,
        combine_antennas="sum_power",
        combine_chirps="mean",
    )
    return np.asarray(profile, dtype=float), np.asarray(ranges_m, dtype=float)


def run_human_room_experiment(
    *,
    human_motion: (
        str
        | Path
        | bytes
        | bytearray
        | BytesIO
        | MeshSequence
        | LoadedAMASSMotion
    ),
    human_motion_filename: str | None = None,
    smpl_model_dir: str | Path | None = None,
    room_preset: str = "tutorial_bedroom",
    scene_xml: str | Path | bytes | bytearray | BytesIO | None = None,
    scene_xml_filename: str | None = None,
    human_position_m: tuple[float, float, float] = (1.6, 0.0, 0.0),
    human_yaw_deg: float = 0.0,
    motion_source_up_axis: str | None = None,
    motion_source_forward_axis: str | None = None,
    motion_source_handedness: str = "right",
    motion_source_length_unit: str = "m",
    radar_orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    board_model: str = "IWR6843AOPEVM",
    carrier_frequency_hz: float = 60e9,
    slope_hz_per_s: float = 68e12,
    chirp_duration_s: float = 58e-6,
    chirp_repetition_time_s: float = 65e-6,
    sampling_frequency_hz: float = 4.5e6,
    adc_mode: str = "complex1x",
    if_bandwidth_hz: float | None = None,
    num_adc_samples: int = 225,
    num_chirps_per_frame: int = 16,
    frame_period_s: float = 50e-3,
    tdm_enabled: bool = False,
    selected_tx_indices: tuple[int, ...] | list[int] | None = None,
    selected_rx_indices: tuple[int, ...] | list[int] | None = None,
    antenna_pattern_mode: str = "cosine",
    cosine_3db_beamwidth_deg: float = 60.0,
    diffuse_reflection_coefficient: float = 0.35,
    simulation_mode: str = "hybrid_po",
    num_frames: int | None = None,
    begin_frame_index: int = 0,
    end_frame_index: int | None = None,
    coupling_enabled: bool = True,
    coupling_power_threshold_db: float = -20.0,
    coupling_max_reflectors: int = 4,
    rt_samples_per_source: int = 100_000,
    rt_max_paths_per_source: int = 5_000,
    rt_max_depth: int = 3,
    po_visibility_samples_per_face: int = 1,
    po_integration_mode: str = "parent_face_adaptive_analytic",
    po_quadrature_phase_span_scale_rad: float | None = None,
    po_curvature_error_rad: float = 0.005,
    po_quadrature_max_refinement_depth: int = 3,
    po_quadrature_max_subfaces_per_parent: int = 64,
    compute_backend: str = "numpy",
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> HumanRoomResult:
    """Run one prepared-room solver mode over a motion-frame window."""

    if simulation_mode not in HUMAN_ROOM_SIMULATION_MODES:
        raise ValueError(
            "simulation_mode must be one of: "
            + ", ".join(HUMAN_ROOM_SIMULATION_MODES)
        )
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError("cancel_check must be callable or None")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    if cancel_check is not None and bool(cancel_check()):
        raise InterruptedError("Simulation cancelled by user")

    preview = prepare_human_room_preview(
        human_motion,
        human_motion_filename=human_motion_filename,
        smpl_model_dir=smpl_model_dir,
        room_preset=room_preset,
        scene_xml=scene_xml,
        scene_xml_filename=scene_xml_filename,
        human_position_m=human_position_m,
        human_yaw_deg=human_yaw_deg,
        motion_source_up_axis=motion_source_up_axis,
        motion_source_forward_axis=motion_source_forward_axis,
        motion_source_handedness=motion_source_handedness,
        motion_source_length_unit=motion_source_length_unit,
    )
    source_motion_frame_count = len(preview.mesh_sequence.times)
    begin_frame_index, end_frame_index = _resolve_motion_frame_window(
        total_frames=source_motion_frame_count,
        begin_frame_index=begin_frame_index,
        end_frame_index=end_frame_index,
        num_frames=num_frames,
    )
    selected_frame_count = end_frame_index - begin_frame_index + 1
    progress_frame_index = 0
    refinement_summary = {
        "reason": "adaptive_po_refinement", "integration_passes": 0,
        "limited_passes": 0, "max_unresolved_parent_faces": 0,
        "max_phase_span_rad": 0., "max_curvature_error_rad": 0.,
    }

    def report_progress(event: dict) -> None:
        nonlocal progress_frame_index
        if event.get("stage") == "adaptive_analytic_geometry_done":
            refinement_summary["integration_passes"] += 1
            refinement_summary["limited_passes"] += int(event["unresolved_subfaces"] > 0)
            refinement_summary["max_unresolved_parent_faces"] = max(
                refinement_summary["max_unresolved_parent_faces"], event["unresolved_parent_faces"])
            for name in ("max_phase_span_rad", "max_curvature_error_rad"):
                refinement_summary[name] = max(refinement_summary[name], event[name])
        if progress_callback is None:
            return
        payload = dict(event)
        if "frame_index" in payload:
            progress_frame_index = int(payload["frame_index"])
        payload.setdefault("frame_index", progress_frame_index)
        payload["simulation_mode"] = simulation_mode
        payload.setdefault("num_frames", selected_frame_count)
        progress_callback(payload)

    report_progress({"frame_index": 0, "solver_mode": "setup"})
    preview = replace(
        preview,
        mesh_sequence=_FrameWindowMotionSequence(
            preview.mesh_sequence,
            begin_frame_index=begin_frame_index,
            end_frame_index=end_frame_index,
            frame_period_s=frame_period_s,
        ),
    )
    radar = _radar(
        board_model,
        orientation_deg=radar_orientation_deg,
        antenna_pattern_mode=antenna_pattern_mode,
        cosine_3db_beamwidth_deg=cosine_3db_beamwidth_deg,
        carrier_frequency_hz=carrier_frequency_hz,
        slope_hz_per_s=slope_hz_per_s,
        chirp_duration_s=chirp_duration_s,
        chirp_repetition_time_s=chirp_repetition_time_s,
        sampling_frequency_hz=sampling_frequency_hz,
        adc_mode=adc_mode,
        if_bandwidth_hz=if_bandwidth_hz,
        num_adc_samples=num_adc_samples,
        num_chirps_per_frame=num_chirps_per_frame,
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
    diffuse = float(diffuse_reflection_coefficient)
    if not np.isfinite(diffuse) or not 0.0 <= diffuse <= 1.0:
        raise ValueError("diffuse_reflection_coefficient must be in [0, 1]")
    material = human_skin_material("gui-human-room-skin")
    material.scattering_coefficient = diffuse
    target = MeshTarget(
        name="gui_human_room_subject",
        mesh_sequence=preview.mesh_sequence,
        material=material,
    )
    human_po = HumanPOMobilityConfig(
        visibility_samples_per_face=int(po_visibility_samples_per_face),
        visibility_fade_chirps=8,
        adaptive_visibility_sampling=True,
        incremental_update=True,
        incremental_visibility_refresh_chirps=64,
        incremental_full_refresh_chirps=0,
        po_integration_mode=po_integration_mode,
        po_quadrature_phase_span_scale_rad=(
            po_quadrature_phase_span_scale_rad
        ),
        po_curvature_error_rad=po_curvature_error_rad,
        po_quadrature_max_refinement_depth=(
            po_quadrature_max_refinement_depth
        ),
        po_quadrature_max_subfaces_per_parent=(
            po_quadrature_max_subfaces_per_parent
        ),
        compute_backend=compute_backend,
        compute_precision="float32",
        progress_callback=report_progress,
    )
    simulator_kwargs = dict(
        max_depth=int(rt_max_depth),
        samples_per_src=int(rt_samples_per_source),
        max_num_paths_per_src=int(rt_max_paths_per_source),
        coupling_mode="unrestricted",
        diffuse_reflection=True,
        seed=42,
        compute_backend=compute_backend,
        compute_precision="float32",
        adc_compute_precision="float32",
        cancel_check=cancel_check,
        progress_callback=report_progress,
    )
    environment_scene = _load_environment_scene(
        preview.room_boxes,
        scene_xml=preview.scene_xml,
    )
    started = time.perf_counter()
    if simulation_mode == "full_rt":
        simulator = MmWaveRadarSimulator(
            mobility_mode="rt_retrace",
            periodic_retrace=True,
            periodic_retrace_period_chirps=1,
            human_specular_reflection=True,
            **simulator_kwargs,
        )
        cube = simulator.run(
            environment_scene,
            radar,
            [target],
            num_frames=selected_frame_count,
        )
    elif simulation_mode == "coherent_rt":
        simulator = MmWaveRadarSimulator(
            mobility_mode="rt_coherent_bank",
            periodic_retrace=True,
            periodic_retrace_period_chirps=int(
                radar.fmcw.num_chirps_per_frame
            ),
            human_specular_reflection=True,
            **simulator_kwargs,
        )
        cube = simulator.run(
            environment_scene,
            radar,
            [target],
            num_frames=selected_frame_count,
        )
    elif simulation_mode == "human_only_po":
        simulator = MmWaveRadarSimulator(
            mobility_mode="human_only_po",
            human_po_config=human_po,
            human_specular_reflection=False,
            **simulator_kwargs,
        )
        cube = simulator.run(
            environment_scene,
            radar,
            [target],
            num_frames=selected_frame_count,
        )
    else:
        # Absolute PO: no RT amplitude fit, so the human terms do not depend
        # on the ray budget or diffuse-scattering settings.
        calibration = POCalibrationConfig(mode="none")
        calibrated = run_calibrated_hybrid_po(
            scene=environment_scene,
            radar=radar,
            target=target,
            num_frames=selected_frame_count,
            calibration_frame_range=(0, selected_frame_count),
            po_visibility_samples_per_face=int(
                po_visibility_samples_per_face
            ),
            coupling_enabled=bool(coupling_enabled),
            coupling_power_threshold_db=coupling_power_threshold_db,
            coupling_max_reflectors=coupling_max_reflectors,
            rt_samples_per_src=int(rt_samples_per_source),
            rt_max_num_paths_per_src=int(rt_max_paths_per_source),
            max_depth=int(rt_max_depth),
            diffuse_reflection=True,
            seed=42,
            compute_backend=compute_backend,
            compute_precision="float32",
            adc_compute_precision="float32",
            progress=False,
            print_reflector_summary=False,
            calibration_config=calibration,
            human_po_config=human_po,
            cancel_check=cancel_check,
            progress_callback=report_progress,
        )
        cube = calibrated.cube
    if refinement_summary["integration_passes"]:
        cube = replace(cube, metadata=replace(
            cube.metadata, events=[*(cube.metadata.events or []), dict(refinement_summary)]))
    runtime_s = time.perf_counter() - started
    components = {
        name: np.asarray(adc)
        for name, adc in (cube.components or {}).items()
        if np.asarray(adc).shape == np.asarray(cube.adc).shape
    }
    total_profile, ranges_m = _range_profile(cube.adc, radar.fmcw)
    profiles = {"total": total_profile}
    for name, adc in components.items():
        profile, component_ranges = _range_profile(adc, radar.fmcw)
        if not np.allclose(ranges_m, component_ranges):
            raise RuntimeError("hybrid component range axes do not match")
        profiles[name] = profile
    range_time_power, range_time_ranges_m = range_time_map(
        cube.adc,
        fmcw=radar.fmcw,
        nfft_mult=4,
        combine_channels="sum_power",
    )
    range_doppler_cube, range_doppler_ranges_m, velocities_mps = (
        framewise_range_doppler_map(
            cube.adc,
            fmcw=radar.fmcw,
        )
    )
    range_doppler_power = np.sum(
        np.abs(range_doppler_cube) ** 2,
        axis=-1,
    )

    manifest = ExperimentManifest(
        name=f"Prepared-room human {simulation_mode}",
        scene=SceneExperimentConfig(
            scenario=(
                "custom_xml"
                if preview.scene_xml is not None
                else room_preset
            ),
            parameters={
                "scene_editable": False,
                "scene_source": preview.scene_source_name,
                "custom_scene_xml": preview.scene_xml is not None,
                "coordinate_frame": "radar_relative",
                "radar_position_m": [0.0, 0.0, 0.0],
                "human_motion_source": preview.source_name,
                "human_motion_format": (
                    "amass_like"
                    if preview.motion_payload is not None
                    else "mesh_sequence_internal"
                ),
                "source_motion_frame_count": source_motion_frame_count,
                "motion_begin_frame_index": begin_frame_index,
                "motion_end_frame_index": end_frame_index,
                "simulation_mode": simulation_mode,
                "human_position_m": preview.human_position_m.tolist(),
                "human_yaw_deg": preview.human_yaw_deg,
                "motion_coordinate_settings": preview.motion_coordinate_settings,
                "diffuse_reflection_coefficient": diffuse,
            },
        ),
        radar=RadarExperimentConfig(
            board_model=board_model,
            profile="gui_human_room",
            parameters={
                "position_m": [0.0, 0.0, 0.0],
                "orientation_deg": list(map(float, radar_orientation_deg)),
                "carrier_frequency_hz": float(radar.fmcw.carrier_frequency),
                "slope_hz_per_s": float(radar.fmcw.slope),
                "chirp_duration_s": float(radar.fmcw.chirp_duration),
                "chirp_repetition_time_s": float(
                    radar.fmcw.chirp_repetition_time
                ),
                "adc_mode": radar.fmcw.adc_mode,
                "if_bandwidth_hz": radar.fmcw.if_bandwidth_hz,
                "effective_if_cutoff_hz": radar.fmcw.effective_if_cutoff_hz,
                "sampling_frequency_hz": float(
                    radar.fmcw.sampling_frequency
                ),
                "num_adc_samples": int(radar.fmcw.num_adc_samples),
                "num_chirps_per_frame": int(
                    radar.fmcw.num_chirps_per_frame
                ),
                "frame_period_s": float(radar.fmcw.frame_period),
                "tdm_enabled": bool(tdm_enabled),
                "tx_indices_zero_based": effective_tx_indices,
                "rx_indices_zero_based": effective_rx_indices,
                "antenna_pattern_mode": antenna_pattern_mode,
                "cosine_3db_beamwidth_deg": float(
                    cosine_3db_beamwidth_deg
                ),
            },
        ),
        solver=SolverExperimentConfig(
            mode={
                "full_rt": "rt_scattering",
                "coherent_rt": "rt_scattering",
                "human_only_po": "target_po",
                "hybrid_po": "hybrid_rt_po",
            }[simulation_mode],
            fidelity="preview",
            parameters={
                "static_environment_solver": (
                    "none" if simulation_mode == "human_only_po" else "rt"
                ),
                "dynamic_human_solver": (
                    "rt"
                    if simulation_mode in ("full_rt", "coherent_rt")
                    else "po"
                ),
                "blocking_enabled": simulation_mode == "hybrid_po",
                "coupling_enabled": bool(
                    simulation_mode == "hybrid_po" and coupling_enabled
                ),
                "coupling_power_threshold_db": coupling_power_threshold_db,
                "coupling_max_reflectors": int(coupling_max_reflectors),
                "rt_samples_per_source": int(rt_samples_per_source),
                "rt_max_paths_per_source": int(rt_max_paths_per_source),
                "rt_max_depth": int(rt_max_depth),
                "po_visibility_samples_per_face": int(
                    po_visibility_samples_per_face
                ),
                "po_integration_mode": po_integration_mode,
                "po_curvature_error_rad": float(po_curvature_error_rad),
                "po_quadrature_phase_span_scale_rad": (
                    human_po.effective_po_phase_span_scale_rad
                ),
                "po_quadrature_max_refinement_depth": int(
                    po_quadrature_max_refinement_depth
                ),
                "po_quadrature_max_subfaces_per_parent": int(
                    po_quadrature_max_subfaces_per_parent
                ),
                "num_frames": selected_frame_count,
                "simulation_mode": simulation_mode,
                "po_calibration_mode": "none",
            },
        ),
    )
    return HumanRoomResult(
        manifest=manifest,
        simulation_mode=simulation_mode,
        preview=preview,
        radar_position_m=np.asarray(radar.position, dtype=float),
        radar_orientation_rad=np.asarray(radar.orientation, dtype=float),
        adc=np.asarray(cube.adc),
        adc_times_s=np.asarray(cube.times),
        ranges_m=ranges_m,
        range_profiles=profiles,
        range_time_power=np.asarray(range_time_power),
        range_time_ranges_m=np.asarray(range_time_ranges_m),
        range_doppler_power=np.asarray(range_doppler_power),
        range_doppler_ranges_m=np.asarray(range_doppler_ranges_m),
        velocities_mps=np.asarray(velocities_mps),
        components=components,
        metadata=cube.metadata,
        runtime_s=float(runtime_s),
        scene_xml=preview.scene_xml,
        scene_source_name=preview.scene_source_name,
    )


__all__ = [
    "HUMAN_ROOM_SIMULATION_MODES",
    "HumanRoomDiagnostics",
    "HumanRoomPreview",
    "HumanRoomResult",
    "LoadedAMASSMotion",
    "PREPARED_ROOM_PRESETS",
    "load_amass_motion",
    "load_human_motion",
    "motion_source_to_radar_transform",
    "load_scene_xml",
    "prepared_room_boxes",
    "prepare_human_room_preview",
    "run_human_room_diagnostics",
    "run_human_room_experiment",
]
