# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Optional SMPL/AMASS helpers for mmWave radar tutorials and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import importlib
import os

import numpy as np


def load_amass_npz(npz_path: str) -> Dict[str, Any]:
    """
    Loads an AMASS ``.npz`` file and normalizes common fields.

    If ``marker_labels`` is stored as bytes, it is decoded to UTF-8 strings.
    Pickled object arrays are rejected because AMASS inputs may be untrusted.
    """

    try:
        with np.load(npz_path, allow_pickle=False) as data:
            out: Dict[str, Any] = {
                key: np.array(data[key], copy=True) for key in data.files
            }
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc):
            raise
        raise ValueError(
            f"{npz_path} contains a pickled object array; convert it to "
            "numeric, Unicode, or byte-string arrays before loading"
        ) from exc
    labels = out.get("marker_labels")
    if labels is not None and getattr(labels, "dtype", None) is not None:
        if labels.dtype.kind in ("S", "O"):
            out["marker_labels"] = np.array([
                label.decode("utf-8") if isinstance(label, bytes) else str(label)
                for label in labels
            ])
    return out


def visualize_amass_markers(
    amass: Dict[str, Any],
    *,
    frame_idx: int = 0,
    ax=None,
):
    """Plots AMASS marker positions for one frame."""

    marker_data = np.asarray(amass.get("marker_data"))
    if marker_data.ndim != 3 or marker_data.shape[2] != 3:
        raise ValueError("amass['marker_data'] must have shape (T, M, 3)")

    points = marker_data[int(frame_idx)]

    if ax is None:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"AMASS markers frame {int(frame_idx)}")
    return ax



def _canonical_gender(gender: Any) -> str:
    """Normalizes SMPL gender strings to supported model-folder names."""

    value = gender
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8", errors="replace")
    value = str(value) if value is not None else "neutral"
    value = value.lower()
    if value not in ("male", "female", "neutral"):
        value = "neutral"
    return value


def _canonical_model_type(model_type: object) -> str:
    """Normalizes the SMPL-family model name used by this adapter."""

    value = str(model_type).lower()
    if value not in ("smpl", "smplh", "smplx"):
        raise ValueError("model_type must be 'smpl', 'smplh', or 'smplx'")
    return value


def _pose_parameter_slices(
    model_type: str,
    pose_dim: int,
) -> dict[str, slice | bool]:
    """Returns model argument slices for standard SMPL-family pose layouts.

    AMASS normally stores the 156-value SMPL-H layout: global orientation,
    21 body joints, and 15 joints for each hand. SMPL-X's native full-pose
    layout additionally stores jaw and eye rotations before its hand poses.
    """

    model = _canonical_model_type(model_type)
    dimension = int(pose_dim)
    if model == "smpl":
        if dimension < 72:
            raise ValueError("SMPL poses must contain at least 72 values")
        if dimension >= 156:
            return {
                "global_orient": slice(0, 3),
                "body_pose": slice(3, 66),
                "append_zero_hand_joints": True,
            }
        return {
            "global_orient": slice(0, 3),
            "body_pose": slice(3, 72),
            "append_zero_hand_joints": False,
        }

    if dimension == 66:
        return {
            "global_orient": slice(0, 3),
            "body_pose": slice(3, 66),
            "append_zero_hand_joints": False,
        }
    if model == "smplh" and dimension == 156:
        return {
            "global_orient": slice(0, 3),
            "body_pose": slice(3, 66),
            "left_hand_pose": slice(66, 111),
            "right_hand_pose": slice(111, 156),
            "append_zero_hand_joints": False,
        }
    if model == "smplx" and dimension == 156:
        return {
            "global_orient": slice(0, 3),
            "body_pose": slice(3, 66),
            "left_hand_pose": slice(66, 111),
            "right_hand_pose": slice(111, 156),
            "append_zero_hand_joints": False,
        }
    if model == "smplx" and dimension >= 165:
        return {
            "global_orient": slice(0, 3),
            "body_pose": slice(3, 66),
            "jaw_pose": slice(66, 69),
            "leye_pose": slice(69, 72),
            "reye_pose": slice(72, 75),
            "left_hand_pose": slice(75, 120),
            "right_hand_pose": slice(120, 165),
            "append_zero_hand_joints": False,
        }
    expected = "66 or 156" if model == "smplh" else "66, 156, or at least 165"
    raise ValueError(f"{model.upper()} poses must contain {expected} values")


def _model_pose_kwargs(pose_batch, model_type: str, torch_module) -> dict:
    """Maps a batched pose tensor to keyword arguments for ``smplx`` models."""

    layout = _pose_parameter_slices(model_type, int(pose_batch.shape[1]))
    kwargs = {
        name: pose_batch[:, pose_slice]
        for name, pose_slice in layout.items()
        if isinstance(pose_slice, slice)
    }
    if layout["append_zero_hand_joints"]:
        zeros = pose_batch.new_zeros((pose_batch.shape[0], 6))
        kwargs["body_pose"] = torch_module.cat(
            [kwargs["body_pose"], zeros],
            dim=1,
        )
    return kwargs


def _positive_framerate(value: object) -> float:
    """Validates an AMASS sampling rate."""

    try:
        framerate = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mocap_framerate must be finite and positive") from exc
    if not np.isfinite(framerate) or framerate <= 0.0:
        raise ValueError("mocap_framerate must be finite and positive")
    return framerate


def _axis_angle_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    """Converts rotation vectors to unit quaternions in wxyz order."""

    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    scale = np.empty_like(angle)
    small = angle < 1e-12
    scale[~small] = np.sin(half_angle[~small]) / angle[~small]
    scale[small] = 0.5 - angle[small] * angle[small] / 48.0
    quat = np.concatenate([np.cos(half_angle), rotvec * scale], axis=-1)
    return quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-18)


def _quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Converts unit quaternions in wxyz order to rotation vectors."""

    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-18)
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)
    vector = quat[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, quat[..., :1])
    scale = np.empty_like(vector_norm)
    small = vector_norm < 1e-12
    scale[~small] = angle[~small] / vector_norm[~small]
    scale[small] = 2.0
    return vector * scale


def _slerp_quaternion(q0: np.ndarray, q1: np.ndarray, weight: float) -> np.ndarray:
    """Shortest-arc spherical interpolation for quaternions in wxyz order."""

    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    w = float(weight)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)

    linear = dot > 0.9995
    lerp = q0 + w * (q1 - q0)
    lerp = lerp / np.maximum(np.linalg.norm(lerp, axis=-1, keepdims=True), 1e-18)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * w
    sin_theta = np.sin(theta)
    safe_den = np.maximum(sin_theta_0, 1e-18)
    s0 = np.cos(theta) - dot * sin_theta / safe_den
    s1 = sin_theta / safe_den
    slerp = s0 * q0 + s1 * q1
    slerp = slerp / np.maximum(np.linalg.norm(slerp, axis=-1, keepdims=True), 1e-18)
    return np.where(linear, lerp, slerp)


def _slerp_axis_angle(rotvec0: np.ndarray, rotvec1: np.ndarray,
                      weight: float) -> np.ndarray:
    """Spherically interpolates axis-angle rotation vectors."""

    q0 = _axis_angle_to_quaternion(rotvec0)
    q1 = _axis_angle_to_quaternion(rotvec1)
    return _quaternion_to_axis_angle(_slerp_quaternion(q0, q1, weight))


def interpolate_amass_pose_params(
    poses: np.ndarray,
    translations: np.ndarray,
    times: np.ndarray,
    time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolates AMASS pose by joint SLERP and translation by LERP."""

    poses = np.asarray(poses, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    times = np.asarray(times, dtype=float)
    if poses.ndim != 2:
        raise ValueError("poses must have shape [num_times, pose_dim]")
    if translations.shape != (poses.shape[0], 3):
        raise ValueError("translations must have shape [num_times, 3]")
    if times.shape != (poses.shape[0],):
        raise ValueError("times must have shape [num_times]")
    if times.size == 0:
        raise ValueError("times must contain at least one sample")
    if not np.all(np.isfinite(poses)):
        raise ValueError("poses must contain only finite values")
    if not np.all(np.isfinite(translations)):
        raise ValueError("translations must contain only finite values")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain only finite values")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")

    t = float(time)
    if not np.isfinite(t):
        raise ValueError("time must be finite")
    if times.size == 1 or t <= times[0]:
        return poses[0].astype(np.float32), translations[0].astype(np.float32)
    if t >= times[-1]:
        return poses[-1].astype(np.float32), translations[-1].astype(np.float32)

    i1 = int(np.searchsorted(times, t, side="right"))
    i0 = i1 - 1
    weight = float((t - times[i0]) / (times[i1] - times[i0]))
    pose0 = poses[i0]
    pose1 = poses[i1]
    pose_dim = int(poses.shape[1])
    rot_dim = 3 * (pose_dim // 3)
    if rot_dim == 0:
        pose = (1.0 - weight) * pose0 + weight * pose1
    else:
        rotations0 = pose0[:rot_dim].reshape(-1, 3)
        rotations1 = pose1[:rot_dim].reshape(-1, 3)
        rotations = _slerp_axis_angle(rotations0, rotations1, weight).reshape(-1)
        if rot_dim < pose_dim:
            tail = (1.0 - weight) * pose0[rot_dim:] + weight * pose1[rot_dim:]
            pose = np.concatenate([rotations, tail], axis=0)
        else:
            pose = rotations
    translation = (1.0 - weight) * translations[i0] + weight * translations[i1]
    return pose.astype(np.float32), translation.astype(np.float32)


@dataclass
class AMASSSMPLMotionSequence:
    """AMASS motion sequence evaluated through SMPL at arbitrary times.

    Pose interpolation uses per-joint spherical interpolation of AMASS
    axis-angle rotations. Root translation is interpolated linearly. Vertices
    are generated by evaluating the configured SMPL-family model at the
    interpolated pose, which avoids linear interpolation of final vertices.
    """

    poses: np.ndarray
    betas: np.ndarray
    translations: np.ndarray
    faces: np.ndarray
    times: np.ndarray
    smpl_model_dir: str
    model_type: str = "smpl"
    gender: str = "neutral"
    device: str = "cpu"
    _torch: Any = field(init=False, repr=False)
    _model: Any = field(init=False, repr=False)
    _initial_vertices: np.ndarray = field(init=False, repr=False)
    _vertices_cache: np.ndarray | None = field(default=None, init=False,
                                               repr=False)

    def __post_init__(self):
        try:
            torch = importlib.import_module("torch")
            smplx = importlib.import_module("smplx")
        except ImportError as exc:
            raise ImportError(
                "AMASS SMPL motion interpolation requires optional "
                "dependencies 'torch' and 'smplx'. Install them to use "
                "AMASSSMPLMotionSequence."
            ) from exc

        poses = np.asarray(self.poses, dtype=np.float32)
        betas = np.asarray(self.betas, dtype=np.float32)
        translations = np.asarray(self.translations, dtype=np.float32)
        raw_faces = np.asarray(self.faces)
        times = np.asarray(self.times, dtype=float)
        if poses.ndim != 2:
            raise ValueError("poses must have shape [num_times, pose_dim]")
        model_type = _canonical_model_type(self.model_type)
        _pose_parameter_slices(model_type, int(poses.shape[1]))
        if translations.shape != (poses.shape[0], 3):
            raise ValueError("translations must have shape [num_times, 3]")
        if times.shape != (poses.shape[0],):
            raise ValueError("times must have shape [num_times]")
        if times.size == 0:
            raise ValueError("motion sequence must contain at least one sample")
        if not np.all(np.isfinite(poses)):
            raise ValueError("poses must contain only finite values")
        if not np.all(np.isfinite(betas)):
            raise ValueError("betas must contain only finite values")
        if not np.all(np.isfinite(translations)):
            raise ValueError("translations must contain only finite values")
        if not np.all(np.isfinite(times)):
            raise ValueError("times must contain only finite values")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
            raise ValueError("faces must have shape [num_faces, 3]")
        if not np.issubdtype(raw_faces.dtype, np.integer):
            raise ValueError("faces must contain integer vertex indices")
        faces = np.asarray(raw_faces, dtype=np.int64)
        if np.any(faces < 0):
            raise ValueError("faces must contain non-negative vertex indices")
        if betas.ndim != 1:
            betas = betas.reshape(-1)
        if betas.size < 10:
            raise ValueError("betas must contain at least 10 shape coefficients")

        gender = _canonical_gender(self.gender)
        model = smplx.create(
            model_path=self.smpl_model_dir,
            model_type=model_type,
            gender=gender,
            batch_size=1,
            use_pca=False,
        ).to(self.device)
        if hasattr(model, "eval"):
            model.eval()

        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "betas", betas[:10])
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "faces", faces.astype(np.uint32, copy=False))
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "model_type", model_type)
        object.__setattr__(self, "gender", gender)
        object.__setattr__(self, "_torch", torch)
        object.__setattr__(self, "_model", model)
        initial_vertices = self._vertices_from_params(poses[0], translations[0])
        if faces.size and np.any(faces >= initial_vertices.shape[0]):
            raise ValueError("faces reference vertex indices outside the model")
        object.__setattr__(self, "_initial_vertices", initial_vertices)

    @property
    def vertex_count(self) -> int:
        """Number of vertices in one evaluated SMPL mesh."""

        return int(self._initial_vertices.shape[0])

    @property
    def face_count(self) -> int:
        """Number of triangular faces."""

        return int(self.faces.shape[0])

    @property
    def vertices(self) -> np.ndarray:
        """Vertices at the original AMASS sample times.

        Accessing this property evaluates SMPL for every AMASS sample and caches
        the result. Prefer :meth:`vertices_at` for chirp-time simulation.
        """

        if self._vertices_cache is None:
            self._vertices_cache = np.stack(
                [self.vertices_at(float(t)) for t in self.times], axis=0)
        return self._vertices_cache

    def vertices_at(self, time: float) -> np.ndarray:
        """Evaluates SMPL vertices at ``time`` using pose SLERP."""

        pose, translation = interpolate_amass_pose_params(
            self.poses,
            self.translations,
            self.times,
            time,
        )
        return self._vertices_from_params(pose, translation)

    def max_vertex_displacement(self, time: float,
                                reference_vertices: np.ndarray) -> float:
        """Maximum vertex displacement relative to ``reference_vertices``."""

        vertices = self.vertices_at(time)
        ref = np.asarray(reference_vertices, dtype=np.float32)
        if ref.shape != vertices.shape:
            raise ValueError("reference_vertices shape mismatch")
        return float(np.max(np.linalg.norm(vertices - ref, axis=1)))

    def _vertices_from_params(self, pose: np.ndarray,
                              translation: np.ndarray) -> np.ndarray:
        """Runs the SMPL-family model for one pose/translation sample."""

        torch = self._torch
        pose_t = torch.from_numpy(np.asarray(pose, dtype=np.float32)[None, :]).to(
            self.device)
        trans_t = torch.from_numpy(
            np.asarray(translation, dtype=np.float32)[None, :]).to(self.device)
        betas_t = torch.from_numpy(self.betas[None, :]).to(self.device)
        kwargs = dict(
            betas=betas_t,
            transl=trans_t,
        )
        kwargs.update(_model_pose_kwargs(pose_t, self.model_type, torch))
        with torch.no_grad():
            output = self._model(**kwargs)
        return output.vertices.detach().cpu().numpy()[0].astype(
            np.float32, copy=False)


def amass_to_mesh_npz(
    amass_npz_path: str,
    smpl_model_dir: str,
    *,
    model_type: str = "smpl",
    device: str = "cpu",
    out_npz_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts an AMASS sequence into a mesh sequence using SMPL/SMPL-H/SMPL-X.

    Returns ``(faces, vertices, times)`` where ``vertices`` has shape
    ``[num_frames, num_vertices, 3]``.
    """

    try:
        torch = importlib.import_module("torch")
        smplx = importlib.import_module("smplx")
    except ImportError as exc:
        raise ImportError(
            "AMASS-to-mesh conversion requires optional dependencies "
            "'torch' and 'smplx'. Install them to use mmWaveRadar.targets.smpl."
        ) from exc

    data = load_amass_npz(amass_npz_path)

    try:
        poses = data["poses"].astype(np.float32)
        betas = data["betas"].astype(np.float32)
        trans = data["trans"].astype(np.float32)
    except KeyError as exc:
        raise ValueError(f"AMASS file missing required field {exc.args[0]!r}") from exc
    if poses.ndim != 2 or poses.shape[0] == 0:
        raise ValueError("poses must have shape [num_times, pose_dim]")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError("trans must have shape [num_times, 3]")
    betas = betas.reshape(-1)
    if betas.size < 10:
        raise ValueError("betas must contain at least 10 shape coefficients")
    if not (
        np.all(np.isfinite(poses))
        and np.all(np.isfinite(betas))
        and np.all(np.isfinite(trans))
    ):
        raise ValueError("poses, betas, and trans must contain only finite values")
    model_type = _canonical_model_type(model_type)
    _pose_parameter_slices(model_type, int(poses.shape[1]))

    gender = _canonical_gender(data["gender"] if "gender" in data else "neutral")

    num_frames = poses.shape[0]
    if "mocap_framerate" not in data:
        raise ValueError("AMASS file missing mocap_framerate")
    framerate = _positive_framerate(data["mocap_framerate"])
    times = np.arange(num_frames, dtype=float) / framerate

    model = smplx.create(
        model_path=smpl_model_dir,
        model_type=model_type,
        gender=gender,
        batch_size=num_frames,
        use_pca=False,
    ).to(device)

    poses_t = torch.from_numpy(poses).to(device)
    betas_t = torch.from_numpy(betas[:10]).to(device).unsqueeze(0).repeat(
        num_frames, 1)
    trans_t = torch.from_numpy(trans).to(device)

    kwargs = dict(
        betas=betas_t,
        transl=trans_t,
    )
    kwargs.update(_model_pose_kwargs(poses_t, model_type, torch))

    with torch.no_grad():
        output = model(**kwargs)
        vertices = output.vertices.detach().cpu().numpy()

    faces = model.faces.astype(np.int32)

    if out_npz_path:
        os.makedirs(os.path.dirname(out_npz_path) or ".", exist_ok=True)
        np.savez_compressed(
            out_npz_path,
            vertices=vertices,
            faces=faces,
            times=times,
            framerate=framerate,
            src_amass=amass_npz_path,
            gender=gender,
        )

    return faces, vertices, times



def amass_to_smpl_motion_sequence(
    amass_npz_path: str,
    smpl_model_dir: str,
    *,
    model_type: str = "smpl",
    device: str = "cpu",
) -> AMASSSMPLMotionSequence:
    """Builds an AMASS/SMPL sequence with chirp-time pose SLERP."""

    data = load_amass_npz(amass_npz_path)
    if "mocap_framerate" not in data:
        raise ValueError("AMASS file missing mocap_framerate")
    try:
        poses = data["poses"].astype(np.float32)
        trans = data["trans"].astype(np.float32)
        betas = data["betas"].astype(np.float32).reshape(-1)
    except KeyError as exc:
        raise ValueError(f"AMASS file missing required field {exc.args[0]!r}") from exc
    if poses.ndim != 2 or poses.shape[0] == 0:
        raise ValueError("poses must have shape [num_times, pose_dim]")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError("trans must have shape [num_times, 3]")
    if betas.size < 10:
        raise ValueError("betas must contain at least 10 shape coefficients")
    if not (
        np.all(np.isfinite(poses))
        and np.all(np.isfinite(trans))
        and np.all(np.isfinite(betas))
    ):
        raise ValueError("poses, betas, and trans must contain only finite values")
    model_type = _canonical_model_type(model_type)
    _pose_parameter_slices(model_type, int(poses.shape[1]))
    framerate = _positive_framerate(data["mocap_framerate"])
    times = np.arange(poses.shape[0], dtype=float) / framerate
    gender = _canonical_gender(data["gender"] if "gender" in data else "neutral")

    try:
        smplx = importlib.import_module("smplx")
    except ImportError as exc:
        raise ImportError(
            "AMASS SMPL motion interpolation requires optional dependency "
            "'smplx'. Install it to use amass_to_smpl_motion_sequence."
        ) from exc
    template_model = smplx.create(
        model_path=smpl_model_dir,
        model_type=model_type,
        gender=gender,
        batch_size=1,
        use_pca=False,
    )
    faces = template_model.faces.astype(np.uint32)

    return AMASSSMPLMotionSequence(
        poses=poses,
        betas=betas,
        translations=trans,
        faces=faces,
        times=times,
        smpl_model_dir=smpl_model_dir,
        model_type=model_type,
        gender=gender,
        device=device,
    )


def amass_to_mesh_sequence(
    amass_npz_path: str,
    smpl_model_dir: str,
    *,
    model_type: str = "smpl",
    device: str = "cpu",
    out_npz_path: Optional[str] = None,
):
    """
    Converts AMASS motion to a ``MeshSequence`` for ``MeshTarget``.
    """

    from .mesh import MeshSequence  # pylint: disable=import-outside-toplevel

    faces, vertices, times = amass_to_mesh_npz(
        amass_npz_path,
        smpl_model_dir,
        model_type=model_type,
        device=device,
        out_npz_path=out_npz_path,
    )
    return MeshSequence(vertices=vertices, faces=faces, times=times)


__all__ = [
    "AMASSSMPLMotionSequence",
    "amass_to_mesh_npz",
    "amass_to_mesh_sequence",
    "amass_to_smpl_motion_sequence",
    "interpolate_amass_pose_params",
    "load_amass_npz",
    "visualize_amass_markers",
]
