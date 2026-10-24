# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Canonical solver and setting names shared by every HERMES entry point.

The GUI, the JSON/CLI runner, the Python API, validation, and saved bundles
all use these names. Setting prefixes say which solver reads a setting:
``rt_*`` wherever ray tracing runs (the RT solver and the Hybrid PO room),
``po_*`` wherever physical optics runs (the PO solver and the Hybrid PO body
and coupling), ``hybrid_*`` only in Hybrid PO, and no prefix for every solver.
Removed names raise an error that names their replacement.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Literal, Mapping

Solver = Literal["po", "rt", "hybrid_po"]
ArrayModel = Literal["phase_center", "subarray"]
TxSchedule = Literal["simultaneous", "tdm"]
HumanBounces = Literal["single", "any"]
HybridCalibration = Literal["none", "rt_first_pose", "rt_sequence"]

SOLVERS: tuple[str, ...] = ("po", "rt", "hybrid_po")
SOLVER_LABELS: Mapping[str, str] = {"po": "PO", "rt": "RT", "hybrid_po": "Hybrid PO"}
ARRAY_MODELS: tuple[str, ...] = ("phase_center", "subarray")
TX_SCHEDULES: tuple[str, ...] = ("simultaneous", "tdm")
RT_HUMAN_BOUNCES: tuple[str, ...] = ("single", "any")
HYBRID_CALIBRATIONS: tuple[str, ...] = ("none", "rt_first_pose", "rt_sequence")

# Removed solver names and what replaces each.
RENAMED_SOLVERS: Mapping[str, str] = {
    "full_rt": "'rt' with rt_refresh_frames=0",
    "rt_retrace": "'rt' with rt_refresh_frames=0",
    "coherent_rt": "'rt' with rt_refresh_frames=1",
    "rt_coherent_bank": "'rt' with rt_refresh_frames=1",
    "rt_scattering": "'rt'",
    "rt_specular": "'rt'",
    "human_only_po": "'po'",
    "target_po": "'po'",
    "hybrid_static_env_po": "'hybrid_po'",
    "hybrid_rt_po": "'hybrid_po'",
}

# Removed setting names (keyword arguments and JSON keys) and their
# replacements. ``None`` marks a setting that no longer exists.
RENAMED_SETTINGS: Mapping[str, str | None] = {
    "mobility_mode": "solver",
    "simulation_mode": "solver",
    "simulation_modes": "solvers",
    "tdm_enabled": "tx_schedule",
    "max_depth": "rt_max_depth",
    "samples_per_src": "rt_samples_per_source",
    "rt_samples_per_src": "rt_samples_per_source",
    "max_num_paths_per_src": "rt_max_paths_per_source",
    "rt_max_num_paths_per_src": "rt_max_paths_per_source",
    "coupling_mode": "rt_human_bounces",
    "diffuse_reflection": "rt_diffuse_reflection",
    "refraction": "rt_refraction",
    "human_specular_reflection": "rt_human_specular_reflection",
    "seed": "rt_seed",
    "diffuse_reflection_coefficient": "rt_human_diffuse_coefficient",
    "periodic_retrace": "rt_refresh_frames",
    "periodic_retrace_period_chirps": "rt_refresh_frames",
    "retrace_once_per_frame": "rt_refresh_frames",
    "retrace_displacement_fraction": "rt_refresh_on_displacement",
    "visibility_samples_per_face": "po_visibility_samples_per_face",
    "incremental_update": "po_refresh_frames",
    "incremental_full_refresh_chirps": "po_refresh_frames",
    "incremental_visibility_refresh_chirps": "po_visibility_refresh_frames",
    "visibility_fade_chirps": "po_visibility_fade_frames",
    "incremental_normal_threshold_deg": "po_refresh_on_normal_deg",
    "incremental_centroid_displacement_threshold_m": "po_refresh_on_phase_error_rad",
    "incremental_area_relative_threshold": "po_refresh_on_area_fraction",
    "po_refresh_on_centroid_m": "po_refresh_on_phase_error_rad",
    "coupling_enabled": "hybrid_coupling",
    "coupling_power_threshold_db": "hybrid_reflector_cutoff_db",
    "coupling_max_reflectors": "hybrid_max_reflectors",
    "coupling_reflection_scale": "hybrid_reflection_scale",
    "coupling_incremental_update": "po_refresh_frames",
    "coupling_visibility_refresh_chirps": "po_visibility_refresh_frames",
    "blocking_enabled": None,
    "blocking_fade_chirps": "rt_visibility_fade_frames",
    "calibration_config": "hybrid_calibration",
    "human_po_config": "po_config",
    "subarray_tx_indices": "selected_tx_indices (zero-based)",
    "subarray_rx_indices": "selected_rx_indices (zero-based)",
    "hybrid_static_env_po_config": "hybrid_config",
    "base_config": None,
}


def _choice(value, choices: Iterable[str], *, name: str) -> str:
    choices = tuple(choices)
    if isinstance(value, str) and value in choices:
        return value
    raise ValueError(f"{name} must be one of: " + ", ".join(choices))


def validate_solver(value, *, name: str = "solver") -> str:
    """Return a supported solver, or explain a removed or unknown name."""

    if isinstance(value, str) and value in SOLVERS:
        return value
    replacement = RENAMED_SOLVERS.get(value) if isinstance(value, str) else None
    if replacement is not None:
        raise ValueError(
            f"{name} {value!r} was replaced by {replacement}; solvers are " + ", ".join(SOLVERS)
        )
    raise ValueError(f"{name} must be one of: " + ", ".join(SOLVERS))


def validate_array_model(value, *, name: str = "array_model") -> str:
    """Return a supported MIMO array model."""

    if value == "element":
        raise ValueError(
            f"{name} 'element' was replaced by 'subarray'; give each element "
            "its own group in array_tx_groups and array_rx_groups"
        )
    return _choice(value, ARRAY_MODELS, name=name)


def parse_index_spec(text: str) -> tuple[int, ...]:
    """Parse zero-based indices such as ``"0-2,9"`` into ``(0, 1, 2, 9)``.

    Ranges are inclusive and may run downwards. Duplicates keep their first
    position.
    """

    tokens = [token for token in str(text).replace(" ", "").split(",") if token]
    if not tokens:
        raise ValueError("index list must not be empty")
    indices: list[int] = []
    for token in tokens:
        parts = token.split("-")
        if len(parts) == 2 and parts[0] and parts[1]:
            start, stop = int(parts[0]), int(parts[1])
            step = 1 if stop >= start else -1
            indices.extend(range(start, stop + step, step))
        elif len(parts) == 1:
            indices.append(int(token))
        else:
            raise ValueError(f"invalid index range {token!r}")
    return tuple(dict.fromkeys(indices))


def validate_array_groups(groups, *, name: str) -> tuple[tuple[int, ...], ...] | None:
    """Return disjoint, non-empty groups of zero-based antenna indices.

    ``None`` means one group holding every active antenna. Whether the groups
    cover the active antennas is checked against the radar with
    :func:`array_group_partition`.
    """

    if groups is None:
        return None
    if isinstance(groups, (str, bytes)):
        raise ValueError(f"{name} must be a list of index lists")
    result = []
    seen: set[int] = set()
    for group in groups:
        if isinstance(group, (str, bytes)) or not hasattr(group, "__iter__"):
            raise ValueError(f"{name} must be a list of index lists")
        members = tuple(group)
        if not members:
            raise ValueError(f"{name} must not contain an empty group")
        for index in members:
            if isinstance(index, bool) or not hasattr(index, "__index__") or int(index) < 0:
                raise ValueError(f"{name} must contain non-negative zero-based indices")
            if int(index) in seen:
                raise ValueError(f"{name} lists index {int(index)} more than once")
            seen.add(int(index))
        result.append(tuple(int(index) for index in members))
    if not result:
        raise ValueError(f"{name} must contain at least one group")
    return tuple(result)


def array_group_partition(
    groups: tuple[tuple[int, ...], ...] | None,
    count: int,
    *,
    name: str,
) -> tuple[tuple[int, ...], ...]:
    """Return groups that split ``range(count)`` exactly, one group if ``None``."""

    if groups is None:
        return (tuple(range(int(count))),)
    listed = {index for group in groups for index in group}
    expected = set(range(int(count)))
    missing = sorted(expected - listed)
    extra = sorted(listed - expected)
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"leave out {missing}")
        if extra:
            problems.append(f"list {extra}, which are not active")
        raise ValueError(
            f"{name} must split the {count} active antennas into groups; they "
            + " and ".join(problems)
        )
    return groups


def validate_tx_schedule(value, *, name: str = "tx_schedule") -> str:
    """Return a supported transmitter schedule."""

    if isinstance(value, bool):
        raise ValueError(f"{name} replaced the tdm_enabled flag; use 'tdm' or 'simultaneous'")
    return _choice(value, TX_SCHEDULES, name=name)


def saved_tx_schedule(record: Mapping[str, object]) -> str:
    """Read a saved schedule, accepting the ``tdm_enabled`` flag of older files."""

    if "tx_schedule" in record:
        return validate_tx_schedule(record["tx_schedule"])
    return "tdm" if bool(record.get("tdm_enabled", False)) else "simultaneous"


def validate_rt_human_bounces(value, *, name: str = "rt_human_bounces") -> str:
    """Return a supported RT human-bounce filter."""

    return _choice(value, RT_HUMAN_BOUNCES, name=name)


def validate_hybrid_calibration(value, *, name: str = "hybrid_calibration") -> str:
    """Return a supported Hybrid PO amplitude calibration."""

    return _choice(value, HYBRID_CALIBRATIONS, name=name)


def validate_frames(value, *, name: str) -> int:
    """Return a non-negative frame count (0 means every chirp)."""

    if isinstance(value, bool) or not isinstance(value, int) and not (hasattr(value, "__index__")):
        raise ValueError(f"{name} must be a non-negative whole number of frames")
    frames = int(value)
    if frames < 0:
        raise ValueError(f"{name} must be a non-negative whole number of frames")
    return frames


def renamed_setting_message(name: str) -> str | None:
    """Describe a removed setting, or return None for an unknown one."""

    if name not in RENAMED_SETTINGS:
        return None
    replacement = RENAMED_SETTINGS[name]
    if replacement is None:
        return f"{name} was removed"
    return f"{name} was renamed to {replacement}"


def reject_removed_arguments(arguments: Mapping[str, object], *, where: str) -> None:
    """Raise for keyword arguments that are not accepted by ``where``."""

    if not arguments:
        return
    messages = []
    for name in sorted(arguments):
        message = renamed_setting_message(name)
        messages.append(message or f"unexpected argument {name!r}")
    raise TypeError(f"{where}: " + "; ".join(messages))


class _RemovedOption(argparse.Action):
    """Fail on a removed command-line option and name what replaces it."""

    def __init__(self, option_strings, dest, *, replacement: str, **kwargs):
        self.replacement = replacement
        super().__init__(
            option_strings,
            dest=argparse.SUPPRESS,
            nargs="*",
            help=argparse.SUPPRESS,
            **kwargs,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(f"{option_string} was removed; use {self.replacement}")


def add_removed_options(
    parser: argparse.ArgumentParser,
    removed: Mapping[str, str],
) -> None:
    """Register hidden options that fail with their replacement's name."""

    for option, replacement in removed.items():
        parser.add_argument(
            option,
            action=_RemovedOption,
            replacement=replacement,
        )


__all__ = [
    "ARRAY_MODELS",
    "ArrayModel",
    "HYBRID_CALIBRATIONS",
    "HumanBounces",
    "HybridCalibration",
    "RENAMED_SETTINGS",
    "RENAMED_SOLVERS",
    "RT_HUMAN_BOUNCES",
    "SOLVERS",
    "SOLVER_LABELS",
    "Solver",
    "TX_SCHEDULES",
    "TxSchedule",
    "add_removed_options",
    "array_group_partition",
    "parse_index_spec",
    "reject_removed_arguments",
    "renamed_setting_message",
    "saved_tx_schedule",
    "validate_array_groups",
    "validate_array_model",
    "validate_frames",
    "validate_hybrid_calibration",
    "validate_rt_human_bounces",
    "validate_solver",
    "validate_tx_schedule",
]
