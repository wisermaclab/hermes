# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Mesh targets and motion adapters for mmWave radar simulations."""

from .mesh import (
    MeshSequence,
    MeshTarget,
    ensure_outward_face_winding,
    face_winding_signed_volume,
    human_skin_material,
)
from .smpl import (
    AMASSSMPLMotionSequence,
    amass_to_mesh_npz,
    amass_to_mesh_sequence,
    amass_to_smpl_motion_sequence,
    interpolate_amass_pose_params,
    load_amass_npz,
    visualize_amass_markers,
)

__all__ = [
    "AMASSSMPLMotionSequence",
    "MeshSequence",
    "MeshTarget",
    "ensure_outward_face_winding",
    "face_winding_signed_volume",
    "amass_to_mesh_npz",
    "amass_to_mesh_sequence",
    "amass_to_smpl_motion_sequence",
    "human_skin_material",
    "interpolate_amass_pose_params",
    "load_amass_npz",
    "visualize_amass_markers",
]
