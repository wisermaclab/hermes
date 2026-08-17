# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Path filtering helpers for simulator traced paths and coherent banks."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
from sionna.rt.constants import InteractionType

from ..targets import MeshTarget
from .rt_coherence import CoherentRTPathBank


def single_target_only_bounce_mask(
    paths,
    targets: list[MeshTarget],
) -> np.ndarray:
    """Returns paths with exactly one target interaction and no others."""

    return _single_target_only_bounce_mask(paths, targets)


def _single_target_only_bounce_mask(paths, targets: list[MeshTarget]) -> np.ndarray:
    """Returns paths whose only active interaction is one target hit."""

    target_counts, total_counts = _target_interaction_counts(
        paths,
        targets,
    )
    return (target_counts == 1) & (total_counts == 1)


def _target_interaction_counts(paths, targets: list[MeshTarget]) -> tuple[np.ndarray, np.ndarray]:
    """Counts target and total active interactions for each traced path."""

    num_paths = int(np.asarray(paths.tau).shape[-1])
    if not targets:
        return (
            np.zeros(num_paths, dtype=np.int64),
            np.zeros(num_paths, dtype=np.int64),
        )
    target_ids = {
        int(np.asarray(target.ensure_scene_object().object_id).reshape(-1)[0])
        for target in targets
    }
    if not hasattr(paths, "objects") or not hasattr(paths, "interactions"):
        return (
            np.zeros(num_paths, dtype=np.int64),
            np.zeros(num_paths, dtype=np.int64),
        )
    objects = np.asarray(paths.objects)
    interactions = np.asarray(paths.interactions)
    if objects.size == 0 or interactions.size == 0:
        return (
            np.zeros(num_paths, dtype=np.int64),
            np.zeros(num_paths, dtype=np.int64),
        )
    objects = np.reshape(objects, (objects.shape[0], -1, objects.shape[-1]))
    interactions = np.reshape(
        interactions,
        (interactions.shape[0], -1, interactions.shape[-1]),
    )
    active = interactions != InteractionType.NONE
    total_counts = np.sum(active, axis=(0, 1)).astype(np.int64)
    target_counts = np.zeros(objects.shape[-1], dtype=np.int64)
    for target_id in target_ids:
        target_counts += np.sum(
            active & (objects == target_id),
            axis=(0, 1),
        ).astype(np.int64)
    return target_counts, total_counts


def target_object_ids(targets: Iterable[MeshTarget]) -> set[int]:
    """Returns the Sionna object IDs for mesh targets."""

    return _target_object_ids(targets)


def _target_object_ids(targets: Iterable[MeshTarget]) -> set[int]:
    """Collects Sionna scene object IDs for mesh targets."""

    return {
        int(np.asarray(target.ensure_scene_object().object_id).reshape(-1)[0])
        for target in targets
    }


def target_specular_path_mask_from_arrays(
    interactions: np.ndarray,
    objects: np.ndarray,
    target_ids: set[int],
    *,
    num_paths: int | None = None,
) -> np.ndarray:
    """Returns a mask of paths with a target specular interaction."""

    return _target_specular_path_mask_from_arrays(
        interactions,
        objects,
        target_ids,
        num_paths=num_paths,
    )


def _target_specular_path_mask_from_arrays(
    interactions: np.ndarray,
    objects: np.ndarray,
    target_ids: set[int],
    *,
    num_paths: int | None = None,
) -> np.ndarray:
    """Detects paths with at least one specular interaction on a target object."""

    if num_paths is None:
        num_paths = int(np.asarray(interactions).shape[-1])
    num_paths = int(num_paths)
    if num_paths <= 0 or not target_ids:
        return np.zeros(max(num_paths, 0), dtype=bool)
    interactions = np.asarray(interactions)
    objects = np.asarray(objects)
    if interactions.size == 0 or objects.size == 0:
        return np.zeros(num_paths, dtype=bool)
    interactions = np.reshape(interactions, (-1, num_paths))
    objects = np.reshape(objects, (-1, num_paths))
    target_hit = np.zeros((interactions.shape[0], num_paths), dtype=bool)
    for target_id in target_ids:
        target_hit |= objects == target_id
    return np.any(
        (interactions == InteractionType.SPECULAR) & target_hit,
        axis=0,
    )


def target_specular_path_mask(
    paths,
    targets: Iterable[MeshTarget],
    *,
    num_paths: int,
) -> np.ndarray:
    """Returns a mask of traced paths with target specular interactions."""

    return _target_specular_path_mask(
        paths,
        targets,
        num_paths=num_paths,
    )


def _target_specular_path_mask(
    paths,
    targets: Iterable[MeshTarget],
    *,
    num_paths: int,
) -> np.ndarray:
    """Detects target-specular paths from a Sionna ``Paths`` object."""

    num_paths = int(num_paths)
    if num_paths <= 0 or not targets:
        return np.zeros(max(num_paths, 0), dtype=bool)
    if not hasattr(paths, "objects") or not hasattr(paths, "interactions"):
        return np.zeros(num_paths, dtype=bool)
    target_ids = _target_object_ids(targets)
    return _target_specular_path_mask_from_arrays(
        np.asarray(paths.interactions),
        np.asarray(paths.objects),
        target_ids,
        num_paths=num_paths,
    )


def drop_target_specular_bank(
    bank: CoherentRTPathBank,
    targets: Iterable[MeshTarget],
) -> CoherentRTPathBank:
    """Drops coherent-bank paths with target specular interactions."""

    return _drop_target_specular_bank(
        bank,
        target_object_ids(targets),
    )


def _drop_target_specular_bank(
    bank: CoherentRTPathBank,
    target_ids: set[int],
) -> CoherentRTPathBank:
    """Returns a coherent path bank with target-specular paths invalidated."""

    target_specular = _target_specular_path_mask_from_arrays(
        bank.path_interactions,
        bank.path_objects,
        target_ids,
        num_paths=int(bank.valid.size),
    )
    if not np.any(target_specular):
        return bank
    return replace(
        bank,
        valid=np.asarray(bank.valid, dtype=bool) & ~target_specular,
    )


def one_target_bounce_mask_from_bank(
    bank: CoherentRTPathBank,
    targets: Iterable[MeshTarget],
) -> np.ndarray:
    """Returns paths that touch targets at most once in a coherent bank."""

    return _one_target_bounce_mask_from_bank(
        bank,
        target_object_ids(targets),
    )


def _one_target_bounce_mask_from_bank(
    bank: CoherentRTPathBank,
    target_ids: set[int],
) -> np.ndarray:
    """Returns coherent-bank paths with no more than one target interaction."""

    if not target_ids:
        return np.ones(bank.valid.shape, dtype=bool)
    objects = np.asarray(bank.path_objects)
    interactions = np.asarray(bank.path_interactions)
    if objects.size == 0 or interactions.size == 0:
        return np.ones(bank.valid.shape, dtype=bool)
    active = interactions != InteractionType.NONE
    counts = np.zeros(objects.shape[-1], dtype=np.int64)
    for target_id in target_ids:
        counts += np.sum(
            active & (objects == target_id),
            axis=0,
        ).astype(np.int64)
    return counts <= 1


def one_target_bounce_mask(paths, targets: list[MeshTarget]) -> np.ndarray:
    """Returns paths with at most one target interaction."""

    return _one_target_bounce_mask(paths, targets)


def _one_target_bounce_mask(paths, targets: list[MeshTarget]) -> np.ndarray:
    """Returns traced paths with no more than one target-object interaction."""

    if not targets:
        return np.ones(paths.tau.shape[-1], dtype=bool)
    if not hasattr(paths, "interactions"):
        if not hasattr(paths, "objects"):
            return np.ones(paths.tau.shape[-1], dtype=bool)
        target_ids = {
            int(np.asarray(h.ensure_scene_object().object_id).reshape(-1)[0])
            for h in targets
        }
        objects = np.asarray(paths.objects)
        if objects.size == 0:
            return np.ones(paths.tau.shape[-1], dtype=bool)
        objects = np.reshape(
            objects,
            (objects.shape[0], -1, objects.shape[-1]),
        )
        counts = np.zeros(objects.shape[-1], dtype=np.int64)
        for target_id in target_ids:
            counts += np.sum(
                objects == target_id,
                axis=(0, 1),
            ).astype(np.int64)
        return counts <= 1

    target_counts, _ = _target_interaction_counts(
        paths,
        targets,
    )
    return target_counts <= 1


def target_interaction_counts(
    paths,
    targets: list[MeshTarget],
) -> tuple[np.ndarray, np.ndarray]:
    """Counts target and total interactions for each traced path."""

    return _target_interaction_counts(paths, targets)


_one_human_bounce_mask = _one_target_bounce_mask

__all__ = [
    "drop_target_specular_bank",
    "one_target_bounce_mask",
    "one_target_bounce_mask_from_bank",
    "single_target_only_bounce_mask",
    "target_interaction_counts",
    "target_object_ids",
    "target_specular_path_mask",
    "target_specular_path_mask_from_arrays",
]
