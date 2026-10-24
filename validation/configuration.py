# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Validation runtime settings and solver policies."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
from typing import Callable, Literal
import numpy as np
from mmWaveRadar.naming import (
    SOLVERS,
    HybridCalibration,
    validate_frames,
    validate_hybrid_calibration,
    validate_solver,
)
from mmWaveRadar.simulation.types import DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD, POConfig
from bundle_prepare.contract import Bundle as BenchmarkClip
from .settings import resolve_settings

SuiteName = Literal["smoke", "range_time", "range_doppler", "pose_range", "dsp_maps"]


ClutterRemovalMode = Literal["none", "mean"]


DEFAULT_VALIDATION_OUTPUT_DIR = Path(tempfile.gettempdir()) / "mmwave_validation"


DEFAULT_ZERO_DOPPLER_WIDTH_BINS = 1


VALIDATION_SOLVERS = SOLVERS


DEFAULT_VALIDATION_SOLVERS = ("po", "rt")


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Runtime controls for a validation benchmark run."""

    suite: SuiteName = "dsp_maps"
    out_dir: Path = DEFAULT_VALIDATION_OUTPUT_DIR
    simulated_adc_path: Path | None = None
    max_frames: int | None = None
    background_subtract: bool = False
    write_plots: bool = True
    simulate: bool = True
    clutter_removal: ClutterRemovalMode = "none"
    scene_path: str | None = None
    solvers: tuple[str, ...] = DEFAULT_VALIDATION_SOLVERS
    rt_refresh_frames: int = 1
    po_refresh_on_phase_error_rad: float = DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD
    rt_diffuse_reflection: bool = True
    rt_refraction: bool = False
    rt_human_specular_reflection: bool = False
    hybrid_calibration: HybridCalibration = "rt_first_pose"
    smpl_model_dir: str | None = None
    # Zero-based Tx/Rx indices, as everywhere in HERMES; None keeps them all.
    selected_tx_indices: tuple[int, ...] | None = None
    selected_rx_indices: tuple[int, ...] | None = None
    progress: bool = False
    ignore_tdm_timing: bool = False
    solver_settings: dict = field(default_factory=dict)
    target_material: dict = field(default_factory=dict)
    frozen_human_po_amplitude_gain: float | None = None
    cube_observer: Callable | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        resolve_settings(dict(self.solver_settings))
        if self.cube_observer is not None and not callable(self.cube_observer):
            raise ValueError("cube_observer must be callable")
        gain = self.frozen_human_po_amplitude_gain
        if gain is not None and (isinstance(gain, bool) or not np.isfinite(gain) or gain < 0):
            raise ValueError("frozen_human_po_amplitude_gain must be finite and non-negative")
        if gain is not None and self.effective_solver_settings["hybrid_calibration"] != "none":
            raise ValueError("A frozen gain requires hybrid_calibration='none'")
        if isinstance(self.solvers, str):
            raise ValueError("solvers must be a sequence of solver names")
        solvers = tuple(validate_solver(solver, name="solvers entry") for solver in self.solvers)
        if not solvers:
            raise ValueError("solvers must select at least one solver")
        if len(set(solvers)) != len(solvers):
            raise ValueError("solvers must not contain duplicates")
        object.__setattr__(self, "solvers", solvers)
        object.__setattr__(
            self,
            "rt_refresh_frames",
            validate_frames(self.rt_refresh_frames, name="rt_refresh_frames"),
        )
        validate_hybrid_calibration(self.hybrid_calibration)
        for name in ("selected_tx_indices", "selected_rx_indices"):
            object.__setattr__(self, name, zero_based_indices(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "po_refresh_on_phase_error_rad",
            POConfig(
                po_refresh_on_phase_error_rad=self.po_refresh_on_phase_error_rad
            ).po_refresh_on_phase_error_rad,
        )
        smpl_model_dir = self.smpl_model_dir
        if smpl_model_dir is None:
            smpl_model_dir = os.environ.get("MMWAVE_SMPL_MODEL_DIR") or None
        if smpl_model_dir is not None:
            object.__setattr__(
                self,
                "smpl_model_dir",
                str(Path(smpl_model_dir).expanduser()),
            )

    @property
    def effective_solver_settings(self):
        return resolve_settings(
            {
                "rt_refresh_frames": self.rt_refresh_frames,
                "rt_diffuse_reflection": self.rt_diffuse_reflection,
                "rt_refraction": self.rt_refraction,
                "rt_human_specular_reflection": self.rt_human_specular_reflection,
                "po_refresh_on_phase_error_rad": self.po_refresh_on_phase_error_rad,
                "hybrid_calibration": self.hybrid_calibration,
                **dict(self.solver_settings),
            }
        )


def zero_based_indices(values, *, name: str) -> tuple[int, ...] | None:
    """Return unique zero-based antenna indices, or None for every antenna."""

    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of zero-based indices")
    indices = tuple(values)
    if not indices or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        for value in indices
    ):
        raise ValueError(f"{name} must contain non-negative zero-based indices")
    indices = tuple(int(value) for value in indices)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicates")
    return indices


def resolve_clutter_removal(
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
) -> ClutterRemovalMode:
    if config.clutter_removal not in ("none", "mean"):
        raise ValueError("clutter_removal must be 'none' or 'mean'")
    return config.clutter_removal
