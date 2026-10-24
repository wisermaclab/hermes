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

from .settings import (
    ArraySettings,
    ExperimentExecutionSettings,
    HumanHybridSettings,
    HumanMotionSettings,
    HumanPOSettings,
    HumanRayTracingSettings,
    HumanRoomSettings,
    HumanRoomSolverSettings,
    RadarSettings,
    RoomSceneSettings,
)
from .radar import RadarSettings, build_radar

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
from ..radar import FMCWConfig, RadarSensor, get_ti_board_spec
from ..simulation.types import DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD
from ..simulation import (
    POConfig,
    MmWaveRadarSimulator,
    PhysicalOpticsMaterial,
    po_channel,
    run_calibrated_hybrid_po,
)
from ..targets.mesh import MeshSequenceView, PlacedMeshSequence, TimeWindowMeshSequence

from ..targets import (
    AMASSSMPLMotionSequence,
    MeshSequence,
    MeshTarget,
    ensure_outward_face_winding,
)
from ..tutorial_support import (
    BEDROOM_SCENE_BOXES,
    prepared_room_scene_xml_for_boxes,
)
from ..simulation.subarray import local_array_groups
from ..naming import (
    SOLVERS,
    reject_removed_arguments,
    validate_array_groups,
    validate_array_model,
    validate_hybrid_calibration,
    validate_solver,
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

PREPARED_ROOM_PRESETS = ("empty_room", "tutorial_bedroom")
_ROOM_RADAR_HEIGHT_M = {"empty_room": 1.035, "tutorial_bedroom": 1.0}
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
    solver: str
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
    channel_range_profiles: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class HumanRoomDiagnostics:
    """Compact one-frame RT/PO overlays for a completed dynamic run."""

    frame_index: int
    motion_time_s: float
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


class _TransformedMotionSequence(MeshSequenceView):
    """Apply archive axis metadata without eagerly evaluating every frame."""

    def __init__(self, base, *, axes=None, signs=None):
        self._base = base
        self._axes = None if axes is None else np.asarray(axes, dtype=np.int64)
        self._signs = None if signs is None else np.asarray(signs, dtype=np.float32)
        if self._axes is not None and (
            self._axes.shape != (3,) or sorted(self._axes.tolist()) != [0, 1, 2]
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
        transform = np.eye(3)[self._axes if self._axes is not None else np.arange(3)]
        if self._signs is not None:
            transform = transform * self._signs[:, None]
        if np.linalg.det(transform) < 0.0:
            faces = faces[:, [0, 2, 1]]
        self.faces = ensure_outward_face_winding(
            self.vertices_at(float(self.times[0])),
            faces,
        )

    def vertices_at(self, time_s: float) -> np.ndarray:
        vertices = np.asarray(self._base.vertices_at(float(time_s)))
        if self._axes is not None:
            vertices = vertices[..., self._axes]
        if self._signs is not None:
            vertices = vertices * self._signs
        return np.asarray(vertices, dtype=np.float32)


class _PlacedMotionSequence(PlacedMeshSequence):
    """GUI placement explicitly rebases the source clock to the first pose."""

    def __init__(self, base, *, position, yaw_deg: float):
        super().__init__(base, position=position, yaw_deg=yaw_deg, rebase_time=True)


class _MotionTimeWindow(TimeWindowMeshSequence):
    """Play motion in real time on the radar clock from ``motion_start_s``.

    Radar time ``t`` maps to motion time ``motion_start_s + t``, so every
    chirp samples the pose at its true instant. ``times`` lists the radar
    frame starts, one entry per simulated frame.
    """

    def __init__(
        self,
        base,
        *,
        motion_start_s: float,
        motion_end_s: float,
        num_frames: int,
        frame_period_s: float,
    ):
        self._base = base
        self.motion_start_s = float(motion_start_s)
        self.motion_end_s = float(motion_end_s)
        self.base_sequence = base
        self.start_time_s = self.motion_start_s
        self.end_time_s = self.motion_end_s
        self.clip_to_window = False
        self.frame_period_s = float(frame_period_s)
        self.time_offset_s = float(getattr(base, "time_offset_s", 0.0))
        self.times = np.arange(int(num_frames), dtype=float) * self.frame_period_s
        self.faces = np.asarray(base.faces)

    def motion_time_s(self, frame_index: int) -> float:
        """Motion time at the start of one radar frame [s]."""

        return self.motion_start_s + int(frame_index) * self.frame_period_s


def radar_frame_count(
    motion_start_s: float,
    motion_end_s: float,
    *,
    frame_period_s: float,
    frame_duration_s: float,
) -> int:
    """Count the radar frames whose chirps fit between start and end.

    Frame ``k`` starts at ``motion_start_s + k * frame_period_s`` and its
    chirps last ``frame_duration_s``. The result is zero when the window is
    shorter than one frame's chirps.
    """

    span = float(motion_end_s) - float(motion_start_s) - float(frame_duration_s)
    if span < -1e-12:
        return 0
    return int(np.floor(span / float(frame_period_s) + 1e-9)) + 1


def _resolve_motion_time_window(
    *,
    motion_duration_s: float,
    motion_start_s: float,
    motion_end_s: float | None,
    num_frames: int | None,
    frame_period_s: float,
    frame_duration_s: float,
) -> tuple[float, float, int]:
    """Validate a motion time window and return start, end, and frame count."""

    try:
        start = float(motion_start_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("motion_start_s must be a number of seconds") from exc
    if isinstance(motion_start_s, bool) or not np.isfinite(start):
        raise ValueError("motion_start_s must be a finite number of seconds")
    duration = float(motion_duration_s)
    if start < 0.0 or start > duration + 1e-9:
        raise ValueError(f"motion_start_s must be in [0, {duration:.4f}] s")
    if motion_end_s is not None and num_frames is not None:
        raise ValueError("specify motion_end_s or num_frames, not both")
    if num_frames is not None:
        try:
            count = operator.index(num_frames)
        except TypeError as exc:
            raise ValueError("num_frames must be a positive integer") from exc
        if count <= 0:
            raise ValueError("num_frames must be a positive integer")
        end = start + (count - 1) * float(frame_period_s) + float(frame_duration_s)
    elif motion_end_s is not None:
        try:
            end = float(motion_end_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("motion_end_s must be a number of seconds") from exc
        if isinstance(motion_end_s, bool) or not np.isfinite(end):
            raise ValueError("motion_end_s must be a finite number of seconds")
        count = radar_frame_count(
            start,
            end,
            frame_period_s=frame_period_s,
            frame_duration_s=frame_duration_s,
        )
        if count <= 0:
            raise ValueError(
                f"the window {start:.4f}–{end:.4f} s is shorter than one "
                f"radar frame's chirps ({1e3 * float(frame_duration_s):.3f} ms)"
            )
    else:
        count = 1
        end = start + float(frame_duration_s)
    # A single-pose motion is static; any window holds that pose.
    if duration > 0.0 and end > duration + 1e-9:
        raise ValueError(
            f"the window ends at {end:.4f} s but the motion lasts "
            f"{duration:.4f} s; choose an earlier start, end, or fewer frames"
        )
    return start, float(end), int(count)


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
        raise ValueError("AMASS-like motion requires times, bundle_times, or mocap_framerate")
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
        raise ImportError("AMASS motion requires optional dependency 'smplx'") from exc
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
                raise ValueError(f"AMASS-like {name} must match the pose frame count")
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
            raise ValueError(f"AMASS-like motion is missing required arrays: {missing}")
        poses = np.asarray(data["poses"], dtype=np.float32)
        translations = np.asarray(data["trans"], dtype=np.float32)
        betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
        if poses.ndim != 2 or poses.shape[0] == 0:
            raise ValueError("AMASS-like poses must have shape [T, pose_dim]")
        times = _amass_times(data, int(poses.shape[0]))
        model_type = (
            _npz_scalar_string(
                data,
                "model_type",
                "smpl",
            )
            .strip()
            .lower()
        )
        gender = (
            _npz_scalar_string(
                data,
                "gender",
                "neutral",
            )
            .strip()
            .lower()
        )
        if model_type not in _SUPPORTED_SMPL_MODEL_TYPES:
            raise ValueError(
                "AMASS-like model_type must be one of: "
                + ", ".join(sorted(_SUPPORTED_SMPL_MODEL_TYPES))
            )
        if gender not in _SUPPORTED_SMPL_GENDERS:
            raise ValueError(
                "AMASS-like gender must be one of: " + ", ".join(sorted(_SUPPORTED_SMPL_GENDERS))
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
        raw_faces = np.asarray(data["faces"]) if "faces" in data.files else None
        if raw_faces is not None:
            if (
                raw_faces.ndim != 2
                or raw_faces.shape[1] != 3
                or raw_faces.dtype.kind not in {"i", "u"}
                or raw_faces.size == 0
                or np.min(raw_faces) < 0
            ):
                raise ValueError("AMASS-like faces must have non-negative integer shape [F, 3]")
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
            "bundle_times": (np.asarray(times, dtype=float) - float(times[0])),
            "gender": np.asarray(gender),
            "model_type": np.asarray(model_type),
        }
        if "mocap_framerate" in data.files:
            payload["mocap_framerate"] = np.asarray(data["mocap_framerate"]).copy()
        if axes is not None:
            payload["output_axes"] = axes.copy()
        if signs is not None:
            payload["output_signs"] = signs.copy()

    configured_model_dir = (
        str(Path(smpl_model_dir).expanduser())
        if smpl_model_dir
        else archive_model_dir or os.environ.get("MMWAVE_SMPL_MODEL_DIR", "")
    )
    if not configured_model_dir:
        raise ValueError("AMASS-like motion requires a licensed SMPL model directory")
    if not Path(configured_model_dir).expanduser().is_dir():
        raise FileNotFoundError(f"SMPL model directory was not found: {configured_model_dir}")
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
        raise ValueError(f"room_preset must be one of {PREPARED_ROOM_PRESETS}")
    room_boxes = []
    for source_box in BEDROOM_SCENE_BOXES:
        if room_preset == "empty_room" and source_box["id"] not in {
            "floor",
            "back_wall",
            "left_wall",
            "right_wall",
        }:
            continue
        box = dict(source_box)
        translation = np.asarray(box["translate"], dtype=float)
        translation[2] -= _ROOM_RADAR_HEIGHT_M[room_preset]
        box["translate"] = tuple(float(value) for value in translation)
        room_boxes.append(box)
    return tuple(room_boxes)


def prepare_human_room_preview(
    human_motion: str | Path | bytes | bytearray | BytesIO | MeshSequence | LoadedAMASSMotion,
    *,
    human_motion_filename: str | None = None,
    smpl_model_dir: str | Path | None = None,
    room_preset: str = "tutorial_bedroom",
    scene_xml: str | Path | bytes | bytearray | BytesIO | None = None,
    scene_xml_filename: str | None = None,
    human_position_m: tuple[float, float, float] | None = (1.6, 0.0, 0.0),
    human_yaw_deg: float = 0.0,
    motion_source_up_axis: str | None = None,
    motion_source_forward_axis: str | None = None,
    motion_source_handedness: str = "right",
    motion_source_length_unit: str = "m",
) -> HumanRoomPreview:
    """Place motion in a scene; ``human_position_m=None`` keeps file positions."""

    if room_preset not in PREPARED_ROOM_PRESETS:
        raise ValueError(f"room_preset must be one of {PREPARED_ROOM_PRESETS}")
    if isinstance(human_motion, LoadedAMASSMotion):
        sequence = human_motion.sequence
        motion_payload = human_motion.payload
        source_name = human_motion.source_name
    elif isinstance(human_motion, MeshSequence):
        sequence = human_motion
        motion_payload = None
        source_name = Path(human_motion_filename or "human_motion.npz").name
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
        base = sequence._base if isinstance(sequence, _TransformedMotionSequence) else sequence
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
    placed = _PlacedMotionSequence(
        sequence,
        position=human_position_m,
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
        human_position_m=placed._position.copy(),
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
    tx_schedule: str,
    selected_tx_indices: tuple[int, ...] | list[int] | None = None,
    selected_rx_indices: tuple[int, ...] | list[int] | None = None,
) -> RadarSensor:
    return build_radar(
        RadarSettings(
            board_model=board_model,
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
            tx_schedule=tx_schedule,
            selected_tx_indices=selected_tx_indices,
            selected_rx_indices=selected_rx_indices,
            orientation_deg=orientation_deg,
        ),
        name="human-room-radar",
    )


def _load_environment_scene(room_boxes, *, scene_xml: bytes | None = None):
    """Load an uploaded XML scene or a prepared room."""

    from sionna.rt import load_scene_from_string

    if scene_xml is not None:
        return load_scene_from_string(
            scene_xml.decode("utf-8"),
            merge_shapes=False,
        )

    xml = prepared_room_scene_xml_for_boxes(room_boxes)
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
        raise ValueError(f"frame_index must be in [0, {max(frame_count - 1, 0)}]")

    radar_parameters = dict(result.manifest.radar.parameters)
    solver_parameters = dict(result.manifest.solver.parameters)
    radar = _radar(
        result.manifest.radar.board_model,
        orientation_deg=radar_parameters.get("orientation_deg", (0.0, 0.0, 0.0)),
        antenna_pattern_mode=str(radar_parameters.get("antenna_pattern_mode", "cosine")),
        cosine_3db_beamwidth_deg=float(radar_parameters.get("cosine_3db_beamwidth_deg", 60.0)),
        carrier_frequency_hz=float(radar_parameters.get("carrier_frequency_hz", 60e9)),
        slope_hz_per_s=float(radar_parameters.get("slope_hz_per_s", 68e12)),
        chirp_duration_s=float(radar_parameters.get("chirp_duration_s", 58e-6)),
        chirp_repetition_time_s=float(radar_parameters.get("chirp_repetition_time_s", 65e-6)),
        adc_mode=radar_parameters.get("adc_mode", "complex1x"),
        if_bandwidth_hz=radar_parameters.get("if_bandwidth_hz"),
        sampling_frequency_hz=float(radar_parameters.get("sampling_frequency_hz", 4.5e6)),
        num_adc_samples=int(radar_parameters.get("num_adc_samples", 225)),
        num_chirps_per_frame=int(radar_parameters.get("num_chirps_per_frame", 16)),
        frame_period_s=float(radar_parameters.get("frame_period_s", 50e-3)),
        tx_schedule=str(radar_parameters.get("tx_schedule", "simultaneous")),
        selected_tx_indices=tuple(radar_parameters.get("tx_indices_zero_based", ())) or None,
        selected_rx_indices=tuple(radar_parameters.get("rx_indices_zero_based", ())) or None,
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
            po_visibility_samples_per_face=int(
                solver_parameters.get("po_visibility_samples_per_face", 1)
            ),
            po_adaptive_visibility_sampling=bool(
                solver_parameters.get(
                    "po_adaptive_visibility_sampling", POConfig().po_adaptive_visibility_sampling
                )
            ),
            po_adaptive_visibility_edge_margin=float(
                solver_parameters.get(
                    "po_adaptive_visibility_edge_margin",
                    POConfig().po_adaptive_visibility_edge_margin,
                )
            ),
            po_integration_mode=str(
                solver_parameters.get("po_integration_mode", "parent_face_quadrature")
            ),
            po_quadrature_phase_span_scale_rad=float(
                solver_parameters.get("po_quadrature_phase_span_scale_rad", 1.0)
            ),
            po_curvature_error_rad=float(solver_parameters.get("po_curvature_error_rad", 0.01)),
            po_quadrature_max_refinement_depth=int(
                solver_parameters.get("po_quadrature_max_refinement_depth", 2)
            ),
            po_quadrature_max_subfaces_per_parent=int(
                solver_parameters.get("po_quadrature_max_subfaces_per_parent", 16)
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
            result.manifest.scene.parameters.get("rt_human_diffuse_coefficient", 0.35)
        )
        target = MeshTarget(
            name="gui_human_room_diagnostic_subject",
            mesh_sequence=sequence,
            material=material,
        )
        simulator = MmWaveRadarSimulator(
            solver="rt",
            rt_max_depth=int(solver_parameters.get("rt_max_depth", 3)),
            rt_samples_per_source=int(solver_parameters.get("rt_samples_per_source", 100_000)),
            rt_max_paths_per_source=int(solver_parameters.get("rt_max_paths_per_source", 5_000)),
            rt_human_bounces=str(solver_parameters.get("rt_human_bounces", "single")),
            rt_diffuse_reflection=True,
            rt_human_specular_reflection=True,
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
        object_id = int(np.asarray(target.ensure_scene_object().object_id).reshape(-1)[0])
        rt_diagnostics = summarize_rt_paths(
            paths,
            top_k=top_k,
            required_object_ids=(object_id,),
        )

    motion_start_s = float(result.manifest.scene.parameters.get("motion_start_s", 0.0))
    return HumanRoomDiagnostics(
        frame_index=frame_index,
        motion_time_s=motion_start_s
        + frame_index * float(radar_parameters.get("frame_period_s", 50e-3)),
        rt=rt_diagnostics,
        po=po_diagnostics,
        po_face_power=face_power,
        po_face_visibility=face_visibility,
    )


def _range_profile(adc: np.ndarray, fmcw: FMCWConfig):
    """Return chirp-averaged power as [frame, range bin, channel]."""

    adc = np.asarray(adc)
    frame_count = adc.shape[0] if adc.ndim == 4 else 1
    if adc.ndim == 4:
        adc = adc.reshape((-1, adc.shape[-2], adc.shape[-1]))
    if adc.ndim != 3:
        raise ValueError(
            "adc must have shape [frame, chirp, sample, channel] or [chirp, sample, channel]"
        )
    range_cube, ranges_m = range_fft(
        adc,
        fmcw=fmcw,
        nfft_mult=4,
    )
    profile = range_profile_from_cube(
        range_cube,
        combine_antennas="none",
    )
    profile = profile.reshape((frame_count, -1, *profile.shape[1:])).mean(axis=1)
    return np.asarray(profile, dtype=float), np.asarray(ranges_m, dtype=float)


def run_human_room_experiment(
    *,
    human_motion: str | Path | bytes | bytearray | BytesIO | MeshSequence | LoadedAMASSMotion,
    human_motion_filename: str | None = None,
    smpl_model_dir: str | Path | None = None,
    room_preset: str = "tutorial_bedroom",
    scene_xml: str | Path | bytes | bytearray | BytesIO | None = None,
    scene_xml_filename: str | None = None,
    human_position_m: tuple[float, float, float] | None = (1.6, 0.0, 0.0),
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
    tx_schedule: str = "simultaneous",
    selected_tx_indices: tuple[int, ...] | list[int] | None = None,
    selected_rx_indices: tuple[int, ...] | list[int] | None = None,
    antenna_pattern_mode: str = "cosine",
    cosine_3db_beamwidth_deg: float = 60.0,
    solver: str = "hybrid_po",
    array_model: str = "phase_center",
    array_tx_groups: list[list[int]] | None = None,
    array_rx_groups: list[list[int]] | None = None,
    motion_start_s: float = 0.0,
    motion_end_s: float | None = None,
    num_frames: int | None = None,
    rt_samples_per_source: int = 100_000,
    rt_max_paths_per_source: int = 5_000,
    rt_max_depth: int = 3,
    rt_human_bounces: str = "single",
    rt_human_diffuse_coefficient: float = 0.35,
    rt_refresh_frames: int = 1,
    rt_visibility_refresh_frames: int = 1,
    rt_visibility_fade_frames: int = 0,
    po_refresh_frames: int = 1,
    po_refresh_on_phase_error_rad: float = DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD,
    po_refresh_on_area_fraction: float = POConfig().po_refresh_on_area_fraction,
    po_refresh_on_normal_deg: float = POConfig().po_refresh_on_normal_deg,
    po_visibility_refresh_frames: int = 1,
    po_visibility_fade_frames: int = 0,
    po_visibility_samples_per_face: int = 1,
    po_adaptive_visibility_sampling: bool = POConfig().po_adaptive_visibility_sampling,
    po_adaptive_visibility_edge_margin: float = POConfig().po_adaptive_visibility_edge_margin,
    po_integration_mode: str = "parent_face_adaptive_analytic",
    po_quadrature_phase_span_scale_rad: float | None = None,
    po_curvature_error_rad: float = 0.005,
    po_quadrature_max_refinement_depth: int = 3,
    po_quadrature_max_subfaces_per_parent: int = 64,
    hybrid_coupling: bool = True,
    hybrid_reflector_cutoff_db: float = -20.0,
    hybrid_max_reflectors: int = 4,
    hybrid_calibration: str = "rt_first_pose",
    compute_backend: str = "auto",
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    **removed,
) -> HumanRoomResult:
    """Run one prepared-room solver mode over a motion time window.

    ``motion_start_s`` is the motion time, in seconds after the first pose,
    that coincides with the start of radar frame 0. The motion then plays in
    real time, so every chirp samples the pose at its own instant. The run
    covers every radar frame whose chirps fit before ``motion_end_s``; pass
    ``num_frames`` instead to request a frame count. With neither, one radar
    frame is simulated.

    With ``array_model="subarray"``, ``array_tx_groups`` and
    ``array_rx_groups`` split the active antennas into groups, using the
    same zero-based board indices as ``selected_tx_indices``; ``None`` is
    one group.
    """
    reject_removed_arguments(removed, where="run_human_room_experiment")
    return simulate_human_room(
        HumanRoomSettings(
            motion=HumanMotionSettings(
                human_motion=human_motion,
                human_motion_filename=human_motion_filename,
                smpl_model_dir=smpl_model_dir,
                human_position_m=human_position_m,
                human_yaw_deg=human_yaw_deg,
                motion_source_up_axis=motion_source_up_axis,
                motion_source_forward_axis=motion_source_forward_axis,
                motion_source_handedness=motion_source_handedness,
                motion_source_length_unit=motion_source_length_unit,
                motion_start_s=motion_start_s,
                motion_end_s=motion_end_s,
                num_frames=num_frames,
            ),
            scene=RoomSceneSettings(
                room_preset=room_preset, scene_xml=scene_xml, scene_xml_filename=scene_xml_filename
            ),
            radar=RadarSettings(
                orientation_deg=radar_orientation_deg,
                board_model=board_model,
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
                tx_schedule=tx_schedule,
                selected_tx_indices=selected_tx_indices,
                selected_rx_indices=selected_rx_indices,
                antenna_pattern_mode=antenna_pattern_mode,
                cosine_3db_beamwidth_deg=cosine_3db_beamwidth_deg,
            ),
            solver=HumanRoomSolverSettings(
                solver=solver,
                array=ArraySettings(
                    array_model=array_model,
                    array_tx_groups=array_tx_groups,
                    array_rx_groups=array_rx_groups,
                ),
                rt=HumanRayTracingSettings(
                    rt_samples_per_source=rt_samples_per_source,
                    rt_max_paths_per_source=rt_max_paths_per_source,
                    rt_max_depth=rt_max_depth,
                    rt_human_bounces=rt_human_bounces,
                    rt_human_diffuse_coefficient=rt_human_diffuse_coefficient,
                    rt_refresh_frames=rt_refresh_frames,
                    rt_visibility_refresh_frames=rt_visibility_refresh_frames,
                    rt_visibility_fade_frames=rt_visibility_fade_frames,
                ),
                po=HumanPOSettings(
                    po_refresh_frames=po_refresh_frames,
                    po_refresh_on_phase_error_rad=po_refresh_on_phase_error_rad,
                    po_refresh_on_area_fraction=po_refresh_on_area_fraction,
                    po_refresh_on_normal_deg=po_refresh_on_normal_deg,
                    po_visibility_refresh_frames=po_visibility_refresh_frames,
                    po_visibility_fade_frames=po_visibility_fade_frames,
                    po_visibility_samples_per_face=po_visibility_samples_per_face,
                    po_adaptive_visibility_sampling=po_adaptive_visibility_sampling,
                    po_adaptive_visibility_edge_margin=po_adaptive_visibility_edge_margin,
                    po_integration_mode=po_integration_mode,
                    po_quadrature_phase_span_scale_rad=po_quadrature_phase_span_scale_rad,
                    po_curvature_error_rad=po_curvature_error_rad,
                    po_quadrature_max_refinement_depth=po_quadrature_max_refinement_depth,
                    po_quadrature_max_subfaces_per_parent=po_quadrature_max_subfaces_per_parent,
                ),
                hybrid=HumanHybridSettings(
                    hybrid_coupling=hybrid_coupling,
                    hybrid_reflector_cutoff_db=hybrid_reflector_cutoff_db,
                    hybrid_max_reflectors=hybrid_max_reflectors,
                    hybrid_calibration=hybrid_calibration,
                ),
                execution=ExperimentExecutionSettings(
                    compute_backend=compute_backend,
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                ),
            ),
        )
    )


def simulate_human_room(settings: HumanRoomSettings) -> HumanRoomResult:
    """Run with grouped settings; the legacy keyword entry point remains supported."""
    if not isinstance(settings, HumanRoomSettings):
        raise TypeError("settings must be a HumanRoomSettings")
    return _HumanRoomRun(settings).run()


class _HumanRoomProgress:
    """Progress and refinement counters without retaining the simulation state."""

    def __init__(self, solver, selected_frame_count, progress_callback, refinement_summary):
        self.solver = solver
        self.selected_frame_count = selected_frame_count
        self.progress_callback = progress_callback
        self.refinement_summary = refinement_summary
        self.progress_frame_index = 0

    def __call__(self, event: dict):
        if event.get("stage") == "adaptive_analytic_geometry_done":
            self.refinement_summary["integration_passes"] += 1
            self.refinement_summary["limited_passes"] += int(event["unresolved_subfaces"] > 0)
            self.refinement_summary["max_unresolved_parent_faces"] = max(
                self.refinement_summary["max_unresolved_parent_faces"],
                event["unresolved_parent_faces"],
            )
            for name in ("max_phase_span_rad", "max_curvature_error_rad"):
                self.refinement_summary[name] = max(self.refinement_summary[name], event[name])
        if self.progress_callback is None:
            return
        payload = dict(event)
        if "frame_index" in payload:
            self.progress_frame_index = int(payload["frame_index"])
        payload.setdefault("frame_index", self.progress_frame_index)
        payload["solver"] = self.solver
        payload.setdefault("num_frames", self.selected_frame_count)
        self.progress_callback(payload)


class _HumanRoomRun:
    """Per-run preparation, solver passes and result construction."""

    def __init__(self, settings: HumanRoomSettings):
        self.settings = settings
        self.array_model = self.settings.solver.array.array_model
        self.array_rx_groups = self.settings.solver.array.array_rx_groups
        self.array_tx_groups = self.settings.solver.array.array_tx_groups
        self.hybrid_calibration = self.settings.solver.hybrid.hybrid_calibration
        self.motion_end_s = self.settings.motion.motion_end_s
        self.motion_start_s = self.settings.motion.motion_start_s
        self.solver = self.settings.solver.solver
        self.room_preset = self.settings.scene.room_preset
        self.radar_orientation_deg = self.settings.radar.orientation_deg
        self.board_model = self.settings.radar.board_model
        self.antenna_pattern_mode = self.settings.radar.antenna_pattern_mode
        self.cosine_3db_beamwidth_deg = self.settings.radar.cosine_3db_beamwidth_deg
        self.rt_samples_per_source = self.settings.solver.rt.rt_samples_per_source
        self.rt_max_paths_per_source = self.settings.solver.rt.rt_max_paths_per_source
        self.rt_max_depth = self.settings.solver.rt.rt_max_depth
        self.rt_human_bounces = self.settings.solver.rt.rt_human_bounces
        self.rt_refresh_frames = self.settings.solver.rt.rt_refresh_frames
        self.rt_visibility_refresh_frames = self.settings.solver.rt.rt_visibility_refresh_frames
        self.rt_visibility_fade_frames = self.settings.solver.rt.rt_visibility_fade_frames
        self.po_refresh_frames = self.settings.solver.po.po_refresh_frames
        self.po_refresh_on_phase_error_rad = self.settings.solver.po.po_refresh_on_phase_error_rad
        self.po_refresh_on_area_fraction = self.settings.solver.po.po_refresh_on_area_fraction
        self.po_refresh_on_normal_deg = self.settings.solver.po.po_refresh_on_normal_deg
        self.po_visibility_refresh_frames = self.settings.solver.po.po_visibility_refresh_frames
        self.po_visibility_fade_frames = self.settings.solver.po.po_visibility_fade_frames
        self.po_visibility_samples_per_face = (
            self.settings.solver.po.po_visibility_samples_per_face
        )
        self.po_integration_mode = self.settings.solver.po.po_integration_mode
        self.po_curvature_error_rad = self.settings.solver.po.po_curvature_error_rad
        self.po_quadrature_max_refinement_depth = (
            self.settings.solver.po.po_quadrature_max_refinement_depth
        )
        self.po_quadrature_max_subfaces_per_parent = (
            self.settings.solver.po.po_quadrature_max_subfaces_per_parent
        )
        self.hybrid_coupling = self.settings.solver.hybrid.hybrid_coupling
        self.hybrid_reflector_cutoff_db = self.settings.solver.hybrid.hybrid_reflector_cutoff_db
        self.hybrid_max_reflectors = self.settings.solver.hybrid.hybrid_max_reflectors
        self.compute_backend = self.settings.solver.execution.compute_backend
        self.cancel_check = self.settings.solver.execution.cancel_check
        self.progress_callback = self.settings.solver.execution.progress_callback

    def _stage_prepare(self):
        """Prepare this run using its settings and prior stage outputs."""
        self.solver = validate_solver(self.solver)
        self.array_model = validate_array_model(self.array_model)
        self.array_tx_groups = validate_array_groups(self.array_tx_groups, name="array_tx_groups")
        self.array_rx_groups = validate_array_groups(self.array_rx_groups, name="array_rx_groups")
        if self.array_model == "phase_center" and (
            self.array_tx_groups is not None or self.array_rx_groups is not None
        ):
            raise ValueError("array_tx_groups and array_rx_groups need array_model='subarray'")
        self.hybrid_calibration = validate_hybrid_calibration(self.hybrid_calibration)
        if self.cancel_check is not None and not callable(self.cancel_check):
            raise ValueError("cancel_check must be callable or None")
        if self.progress_callback is not None and not callable(self.progress_callback):
            raise ValueError("progress_callback must be callable or None")
        if self.cancel_check is not None and bool(self.cancel_check()):
            raise InterruptedError("Simulation cancelled by user")

        self.preview = prepare_human_room_preview(
            self.settings.motion.human_motion,
            human_motion_filename=self.settings.motion.human_motion_filename,
            smpl_model_dir=self.settings.motion.smpl_model_dir,
            room_preset=self.room_preset,
            scene_xml=self.settings.scene.scene_xml,
            scene_xml_filename=self.settings.scene.scene_xml_filename,
            human_position_m=self.settings.motion.human_position_m,
            human_yaw_deg=self.settings.motion.human_yaw_deg,
            motion_source_up_axis=self.settings.motion.motion_source_up_axis,
            motion_source_forward_axis=self.settings.motion.motion_source_forward_axis,
            motion_source_handedness=self.settings.motion.motion_source_handedness,
            motion_source_length_unit=self.settings.motion.motion_source_length_unit,
        )
        self.radar = _radar(
            self.board_model,
            orientation_deg=self.radar_orientation_deg,
            antenna_pattern_mode=self.antenna_pattern_mode,
            cosine_3db_beamwidth_deg=self.cosine_3db_beamwidth_deg,
            carrier_frequency_hz=self.settings.radar.carrier_frequency_hz,
            slope_hz_per_s=self.settings.radar.slope_hz_per_s,
            chirp_duration_s=self.settings.radar.chirp_duration_s,
            chirp_repetition_time_s=self.settings.radar.chirp_repetition_time_s,
            sampling_frequency_hz=self.settings.radar.sampling_frequency_hz,
            adc_mode=self.settings.radar.adc_mode,
            if_bandwidth_hz=self.settings.radar.if_bandwidth_hz,
            num_adc_samples=self.settings.radar.num_adc_samples,
            num_chirps_per_frame=self.settings.radar.num_chirps_per_frame,
            frame_period_s=self.settings.radar.frame_period_s,
            tx_schedule=self.settings.radar.tx_schedule,
            selected_tx_indices=self.settings.radar.selected_tx_indices,
            selected_rx_indices=self.settings.radar.selected_rx_indices,
        )
        self.motion_pose_count = len(self.preview.mesh_sequence.times)
        self.motion_duration_s = float(self.preview.mesh_sequence.times[-1])
        self.motion_start_s, self.motion_end_s, self.selected_frame_count = (
            _resolve_motion_time_window(
                motion_duration_s=self.motion_duration_s,
                motion_start_s=self.motion_start_s,
                motion_end_s=self.motion_end_s,
                num_frames=self.settings.motion.num_frames,
                frame_period_s=self.radar.fmcw.frame_period,
                frame_duration_s=self.radar.fmcw.frame_duration,
            )
        )
        self.refinement_summary = {
            "reason": "adaptive_po_refinement",
            "integration_passes": 0,
            "limited_passes": 0,
            "max_unresolved_parent_faces": 0,
            "max_phase_span_rad": 0.0,
            "max_curvature_error_rad": 0.0,
        }

        self.report_progress = _HumanRoomProgress(
            self.solver,
            self.selected_frame_count,
            self.progress_callback,
            self.refinement_summary,
        )
        self.report_progress({"frame_index": 0, "solver_mode": "setup"})
        self.preview = replace(
            self.preview,
            mesh_sequence=_MotionTimeWindow(
                self.preview.mesh_sequence,
                motion_start_s=self.motion_start_s,
                motion_end_s=self.motion_end_s,
                num_frames=self.selected_frame_count,
                frame_period_s=self.radar.fmcw.frame_period,
            ),
        )
        self.effective_tx_indices = (
            list(range(get_ti_board_spec(self.board_model).num_tx))
            if self.settings.radar.selected_tx_indices is None
            else [int(value) for value in self.settings.radar.selected_tx_indices]
        )
        self.effective_rx_indices = (
            list(range(get_ti_board_spec(self.board_model).num_rx))
            if self.settings.radar.selected_rx_indices is None
            else [int(value) for value in self.settings.radar.selected_rx_indices]
        )
        self.local_tx_groups = local_array_groups(
            self.array_tx_groups,
            self.effective_tx_indices,
            name="array_tx_groups",
        )
        self.local_rx_groups = local_array_groups(
            self.array_rx_groups,
            self.effective_rx_indices,
            name="array_rx_groups",
        )
        self.diffuse = float(self.settings.solver.rt.rt_human_diffuse_coefficient)
        if not np.isfinite(self.diffuse) or not 0.0 <= self.diffuse <= 1.0:
            raise ValueError("rt_human_diffuse_coefficient must be in [0, 1]")
        material = human_skin_material("gui-human-room-skin")
        material.scattering_coefficient = self.diffuse
        self.target = MeshTarget(
            name="gui_human_room_subject",
            mesh_sequence=self.preview.mesh_sequence,
            material=material,
        )
        self.human_po = POConfig(
            po_refresh_frames=self.po_refresh_frames,
            po_refresh_on_phase_error_rad=self.po_refresh_on_phase_error_rad,
            po_refresh_on_area_fraction=self.po_refresh_on_area_fraction,
            po_refresh_on_normal_deg=self.po_refresh_on_normal_deg,
            po_visibility_refresh_frames=self.po_visibility_refresh_frames,
            po_visibility_fade_frames=self.po_visibility_fade_frames,
            po_visibility_samples_per_face=int(self.po_visibility_samples_per_face),
            po_adaptive_visibility_sampling=self.settings.solver.po.po_adaptive_visibility_sampling,
            po_adaptive_visibility_edge_margin=self.settings.solver.po.po_adaptive_visibility_edge_margin,
            po_integration_mode=self.po_integration_mode,
            po_quadrature_phase_span_scale_rad=(
                self.settings.solver.po.po_quadrature_phase_span_scale_rad
            ),
            po_curvature_error_rad=self.po_curvature_error_rad,
            po_quadrature_max_refinement_depth=(self.po_quadrature_max_refinement_depth),
            po_quadrature_max_subfaces_per_parent=(self.po_quadrature_max_subfaces_per_parent),
            compute_backend=self.compute_backend,
            compute_precision="float32",
            progress_callback=self.report_progress,
        )
        self.simulator_kwargs = dict(
            array_model=self.array_model,
            array_tx_groups=self.local_tx_groups,
            array_rx_groups=self.local_rx_groups,
            rt_max_depth=int(self.rt_max_depth),
            rt_samples_per_source=int(self.rt_samples_per_source),
            rt_max_paths_per_source=int(self.rt_max_paths_per_source),
            rt_human_bounces=self.rt_human_bounces,
            rt_refresh_frames=self.rt_refresh_frames,
            rt_visibility_refresh_frames=self.rt_visibility_refresh_frames,
            rt_visibility_fade_frames=self.rt_visibility_fade_frames,
            rt_diffuse_reflection=True,
            compute_backend=self.compute_backend,
            compute_precision="float32",
            adc_compute_precision="float32",
            cancel_check=self.cancel_check,
            progress_callback=self.report_progress,
        )
        self.environment_scene = _load_environment_scene(
            self.preview.room_boxes,
            scene_xml=self.preview.scene_xml,
        )

    def _stage_simulate(self):
        """Simulate this run using its settings and prior stage outputs."""
        started = time.perf_counter()
        if self.solver in ("rt", "po"):
            simulator = MmWaveRadarSimulator(
                solver=self.solver,
                po_config=self.human_po,
                rt_human_specular_reflection=self.solver == "rt",
                **self.simulator_kwargs,
            )
            self.cube = simulator.run(
                self.environment_scene,
                self.radar,
                [self.target],
                num_frames=self.selected_frame_count,
            )
        else:
            # The calibrated flow calls solver passes directly, so check the
            # phase-center approximation here as MmWaveRadarSimulator.run does.
            approximation_event = MmWaveRadarSimulator(
                solver="hybrid_po",
                array_model=self.array_model,
                array_tx_groups=self.local_tx_groups,
                array_rx_groups=self.local_rx_groups,
            )._phase_center_approximation(self.radar, [self.target])
            calibrated = run_calibrated_hybrid_po(
                scene=self.environment_scene,
                radar=self.radar,
                target=self.target,
                num_frames=self.selected_frame_count,
                # rt_first_pose fits against the first frame only.
                calibration_frame_range=(
                    0,
                    self.selected_frame_count if self.hybrid_calibration == "rt_sequence" else 1,
                ),
                po_visibility_samples_per_face=int(self.po_visibility_samples_per_face),
                hybrid_coupling=bool(self.hybrid_coupling),
                hybrid_reflector_cutoff_db=self.hybrid_reflector_cutoff_db,
                hybrid_max_reflectors=self.hybrid_max_reflectors,
                hybrid_calibration=self.hybrid_calibration,
                rt_samples_per_source=int(self.rt_samples_per_source),
                rt_max_paths_per_source=int(self.rt_max_paths_per_source),
                rt_max_depth=int(self.rt_max_depth),
                rt_human_bounces=self.rt_human_bounces,
                rt_visibility_refresh_frames=self.rt_visibility_refresh_frames,
                rt_visibility_fade_frames=self.rt_visibility_fade_frames,
                rt_diffuse_reflection=True,
                array_model=self.array_model,
                array_tx_groups=self.local_tx_groups,
                array_rx_groups=self.local_rx_groups,
                compute_backend=self.compute_backend,
                compute_precision="float32",
                adc_compute_precision="float32",
                progress=False,
                print_reflector_summary=False,
                po_config=self.human_po,
                cancel_check=self.cancel_check,
                progress_callback=self.report_progress,
            )
            self.cube = calibrated.cube
            if approximation_event is not None:
                self.cube = replace(
                    self.cube,
                    metadata=replace(
                        self.cube.metadata,
                        events=[*(self.cube.metadata.events or []), approximation_event],
                    ),
                )
        if self.refinement_summary["integration_passes"]:
            self.cube = replace(
                self.cube,
                metadata=replace(
                    self.cube.metadata,
                    events=[*(self.cube.metadata.events or []), dict(self.refinement_summary)],
                ),
            )
        self.runtime_s = time.perf_counter() - started

    def _stage_products(self):
        """Products this run using its settings and prior stage outputs."""
        self.components = {
            name: np.asarray(adc)
            for name, adc in (self.cube.components or {}).items()
            if np.asarray(adc).shape == np.asarray(self.cube.adc).shape
        }
        total_channels, self.ranges_m = _range_profile(self.cube.adc, self.radar.fmcw)
        self.channel_profiles = {"total": total_channels}
        self.profiles = {"total": np.sum(total_channels.mean(axis=0), axis=-1)}
        for self.name, adc in self.components.items():
            profile, component_ranges = _range_profile(adc, self.radar.fmcw)
            if not np.allclose(self.ranges_m, component_ranges):
                raise RuntimeError("hybrid component range axes do not match")
            self.channel_profiles[self.name] = profile
            self.profiles[self.name] = np.sum(profile.mean(axis=0), axis=-1)
        self.range_time_power, self.range_time_ranges_m = range_time_map(
            self.cube.adc,
            fmcw=self.radar.fmcw,
            nfft_mult=4,
            combine_channels="sum_power",
        )
        range_doppler_cube, self.range_doppler_ranges_m, self.velocities_mps = (
            framewise_range_doppler_map(
                self.cube.adc,
                fmcw=self.radar.fmcw,
            )
        )
        self.range_doppler_power = np.sum(
            np.abs(range_doppler_cube) ** 2,
            axis=-1,
        )

    def _stage_manifest(self):
        """Manifest this run using its settings and prior stage outputs."""
        self.manifest = ExperimentManifest(
            name=f"Prepared-room human {self.solver}",
            scene=SceneExperimentConfig(
                scenario=(
                    "custom_xml" if self.preview.scene_xml is not None else self.room_preset
                ),
                parameters={
                    "scene_editable": False,
                    "scene_source": self.preview.scene_source_name,
                    "custom_scene_xml": self.preview.scene_xml is not None,
                    "coordinate_frame": "radar_relative",
                    "radar_position_m": [0.0, 0.0, 0.0],
                    "human_motion_source": self.preview.source_name,
                    "human_motion_format": (
                        "amass_like"
                        if self.preview.motion_payload is not None
                        else "mesh_sequence_internal"
                    ),
                    "motion_pose_count": self.motion_pose_count,
                    "motion_duration_s": self.motion_duration_s,
                    "motion_start_s": self.motion_start_s,
                    "motion_end_s": self.motion_end_s,
                    "solver": self.solver,
                    "human_position_m": self.preview.human_position_m.tolist(),
                    "human_yaw_deg": self.preview.human_yaw_deg,
                    "motion_coordinate_settings": self.preview.motion_coordinate_settings,
                    "rt_human_diffuse_coefficient": self.diffuse,
                },
            ),
            radar=RadarExperimentConfig(
                board_model=self.board_model,
                profile="gui_human_room",
                parameters={
                    "position_m": [0.0, 0.0, 0.0],
                    "orientation_deg": list(map(float, self.radar_orientation_deg)),
                    "carrier_frequency_hz": float(self.radar.fmcw.carrier_frequency),
                    "slope_hz_per_s": float(self.radar.fmcw.slope),
                    "chirp_duration_s": float(self.radar.fmcw.chirp_duration),
                    "chirp_repetition_time_s": float(self.radar.fmcw.chirp_repetition_time),
                    "adc_mode": self.radar.fmcw.adc_mode,
                    "if_bandwidth_hz": self.radar.fmcw.if_bandwidth_hz,
                    "effective_if_cutoff_hz": self.radar.fmcw.effective_if_cutoff_hz,
                    "sampling_frequency_hz": float(self.radar.fmcw.sampling_frequency),
                    "num_adc_samples": int(self.radar.fmcw.num_adc_samples),
                    "num_chirps_per_frame": int(self.radar.fmcw.num_chirps_per_frame),
                    "frame_period_s": float(self.radar.fmcw.frame_period),
                    "tx_schedule": self.radar.fmcw.tx_schedule,
                    "tx_indices_zero_based": self.effective_tx_indices,
                    "rx_indices_zero_based": self.effective_rx_indices,
                    "antenna_pattern_mode": self.antenna_pattern_mode,
                    "cosine_3db_beamwidth_deg": float(self.cosine_3db_beamwidth_deg),
                },
            ),
            solver=SolverExperimentConfig(
                solvers=(self.solver,),
                fidelity="preview",
                parameters={
                    "solver": self.solver,
                    "array_model": self.array_model,
                    "array_tx_groups": (
                        None
                        if self.array_tx_groups is None
                        else [list(group) for group in self.array_tx_groups]
                    ),
                    "array_rx_groups": (
                        None
                        if self.array_rx_groups is None
                        else [list(group) for group in self.array_rx_groups]
                    ),
                    "num_frames": self.selected_frame_count,
                    "rt_samples_per_source": int(self.rt_samples_per_source),
                    "rt_max_paths_per_source": int(self.rt_max_paths_per_source),
                    "rt_max_depth": int(self.rt_max_depth),
                    "rt_human_bounces": self.rt_human_bounces,
                    "rt_refresh_frames": int(self.rt_refresh_frames),
                    "rt_visibility_refresh_frames": int(self.rt_visibility_refresh_frames),
                    "rt_visibility_fade_frames": int(self.rt_visibility_fade_frames),
                    "po_refresh_frames": int(self.po_refresh_frames),
                    "po_refresh_on_phase_error_rad": float(self.po_refresh_on_phase_error_rad),
                    "po_adaptive_visibility_sampling": self.human_po.po_adaptive_visibility_sampling,
                    "po_adaptive_visibility_edge_margin": self.human_po.po_adaptive_visibility_edge_margin,
                    "po_refresh_on_normal_deg": float(self.po_refresh_on_normal_deg),
                    "po_refresh_on_area_fraction": float(self.po_refresh_on_area_fraction),
                    "po_visibility_refresh_frames": int(self.po_visibility_refresh_frames),
                    "po_visibility_fade_frames": int(self.po_visibility_fade_frames),
                    "po_visibility_samples_per_face": int(self.po_visibility_samples_per_face),
                    "po_integration_mode": self.po_integration_mode,
                    "po_curvature_error_rad": float(self.po_curvature_error_rad),
                    "po_quadrature_phase_span_scale_rad": (
                        self.human_po.effective_po_phase_span_scale_rad
                    ),
                    "po_quadrature_max_refinement_depth": int(
                        self.po_quadrature_max_refinement_depth
                    ),
                    "po_quadrature_max_subfaces_per_parent": int(
                        self.po_quadrature_max_subfaces_per_parent
                    ),
                    **(
                        {
                            "hybrid_coupling": bool(self.hybrid_coupling),
                            "hybrid_reflector_cutoff_db": self.hybrid_reflector_cutoff_db,
                            "hybrid_max_reflectors": int(self.hybrid_max_reflectors),
                            "hybrid_calibration": self.hybrid_calibration,
                        }
                        if self.solver == "hybrid_po"
                        else {}
                    ),
                },
            ),
        )

    def _stage_result(self):
        """Result this run using its settings and prior stage outputs."""
        return HumanRoomResult(
            manifest=self.manifest,
            solver=self.solver,
            preview=self.preview,
            radar_position_m=np.asarray(self.radar.position, dtype=float),
            radar_orientation_rad=np.asarray(self.radar.orientation, dtype=float),
            adc=np.asarray(self.cube.adc),
            adc_times_s=np.asarray(self.cube.times),
            ranges_m=self.ranges_m,
            range_profiles=self.profiles,
            channel_range_profiles=self.channel_profiles,
            range_time_power=np.asarray(self.range_time_power),
            range_time_ranges_m=np.asarray(self.range_time_ranges_m),
            range_doppler_power=np.asarray(self.range_doppler_power),
            range_doppler_ranges_m=np.asarray(self.range_doppler_ranges_m),
            velocities_mps=np.asarray(self.velocities_mps),
            components=self.components,
            metadata=self.cube.metadata,
            runtime_s=float(self.runtime_s),
            scene_xml=self.preview.scene_xml,
            scene_source_name=self.preview.scene_source_name,
        )

    def run(self):
        self._stage_prepare()
        self._stage_simulate()
        self._stage_products()
        self._stage_manifest()
        return self._stage_result()


__all__ = [
    "SOLVERS",
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
    "radar_frame_count",
    "run_human_room_diagnostics",
    "run_human_room_experiment",
]
