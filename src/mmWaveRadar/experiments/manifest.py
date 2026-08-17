# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Versioned, deterministic experiment manifests shared by GUI and batch runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EXPERIMENT_MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_SOLVER_MODES = (
    "rt_specular",
    "rt_scattering",
    "target_po",
    "hybrid_rt_po",
)
SUPPORTED_FIDELITY_TIERS = ("preview", "high_fidelity")
DEFAULT_EXPERIMENT_OUTPUTS = (
    "raw_adc",
    "range_profile",
    "range_doppler",
    "diagnostics",
)


def _plain_mapping(name: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class SceneExperimentConfig:
    """Curated scene identifier and scenario-specific, serializable controls."""

    scenario: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.scenario, str) or not self.scenario.strip():
            raise ValueError("scene.scenario must be a non-empty string")
        object.__setattr__(
            self,
            "parameters",
            _plain_mapping("scene.parameters", self.parameters),
        )


@dataclass(frozen=True)
class RadarExperimentConfig:
    """Radar preset and optional waveform overrides."""

    board_model: str
    profile: str = "default"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.board_model, str) or not self.board_model.strip():
            raise ValueError("radar.board_model must be a non-empty string")
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("radar.profile must be a non-empty string")
        object.__setattr__(
            self,
            "parameters",
            _plain_mapping("radar.parameters", self.parameters),
        )


@dataclass(frozen=True)
class SolverExperimentConfig:
    """Solver choice, execution tier, and explicit numerical controls."""

    mode: str
    fidelity: str = "preview"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in SUPPORTED_SOLVER_MODES:
            raise ValueError(
                "solver.mode must be one of: " + ", ".join(SUPPORTED_SOLVER_MODES)
            )
        if self.fidelity not in SUPPORTED_FIDELITY_TIERS:
            raise ValueError(
                "solver.fidelity must be 'preview' or 'high_fidelity'"
            )
        object.__setattr__(
            self,
            "parameters",
            _plain_mapping("solver.parameters", self.parameters),
        )


@dataclass(frozen=True)
class MeasurementReference:
    """Portable identity and processing controls for a measurement bundle."""

    bundle_id: str
    bundle_schema_version: int = 1
    bundle_fingerprint: str | None = None
    frame_indices: tuple[int, ...] = ()
    background_subtraction: bool = False
    clutter_removal: str = "none"

    def __post_init__(self):
        if not isinstance(self.bundle_id, str) or not self.bundle_id.strip():
            raise ValueError("measurement.bundle_id must be a non-empty string")
        if int(self.bundle_schema_version) <= 0:
            raise ValueError("measurement.bundle_schema_version must be positive")
        indices = tuple(int(index) for index in self.frame_indices)
        if any(index < 0 for index in indices):
            raise ValueError("measurement.frame_indices must be non-negative")
        if len(set(indices)) != len(indices):
            raise ValueError("measurement.frame_indices must not contain duplicates")
        if self.clutter_removal not in ("none", "mean"):
            raise ValueError("measurement.clutter_removal must be 'none' or 'mean'")
        object.__setattr__(self, "frame_indices", indices)
        object.__setattr__(
            self,
            "bundle_schema_version",
            int(self.bundle_schema_version),
        )


@dataclass(frozen=True)
class ExperimentManifest:
    """Complete portable input contract for one HERMES experiment."""

    name: str
    scene: SceneExperimentConfig
    radar: RadarExperimentConfig
    solver: SolverExperimentConfig
    measurement: MeasurementReference | None = None
    outputs: tuple[str, ...] = DEFAULT_EXPERIMENT_OUTPUTS
    seed: int = 42
    schema_version: int = EXPERIMENT_MANIFEST_SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("manifest.name must be a non-empty string")
        if int(self.schema_version) != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "unsupported experiment manifest schema version "
                f"{self.schema_version}"
            )
        outputs = tuple(str(output) for output in self.outputs)
        if not outputs or any(not output for output in outputs):
            raise ValueError("manifest.outputs must contain non-empty names")
        if len(set(outputs)) != len(outputs):
            raise ValueError("manifest.outputs must not contain duplicates")
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with stable field names."""

        # Normalize tuples and other JSON-supported containers so callers see
        # exactly the same structure before and after serialization.
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    def canonical_json(self) -> str:
        """Return canonical JSON used for cache and provenance fingerprints."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def fingerprint(self) -> str:
        """SHA-256 fingerprint of the canonical manifest."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def write_json(self, path: str | Path) -> None:
        """Write a readable manifest without changing its canonical content."""

        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentManifest":
        """Validate and load a manifest mapping."""

        if not isinstance(data, Mapping):
            raise ValueError("experiment manifest must be a JSON object")
        scene = data.get("scene")
        radar = data.get("radar")
        solver = data.get("solver")
        if not isinstance(scene, Mapping):
            raise ValueError("manifest.scene must be an object")
        if not isinstance(radar, Mapping):
            raise ValueError("manifest.radar must be an object")
        if not isinstance(solver, Mapping):
            raise ValueError("manifest.solver must be an object")
        measurement_data = data.get("measurement")
        if measurement_data is not None and not isinstance(
            measurement_data, Mapping
        ):
            raise ValueError("manifest.measurement must be an object or null")
        return cls(
            name=str(data.get("name", "")),
            scene=SceneExperimentConfig(**dict(scene)),
            radar=RadarExperimentConfig(**dict(radar)),
            solver=SolverExperimentConfig(**dict(solver)),
            measurement=(
                None
                if measurement_data is None
                else MeasurementReference(**dict(measurement_data))
            ),
            outputs=tuple(data.get("outputs", DEFAULT_EXPERIMENT_OUTPUTS)),
            seed=int(data.get("seed", 42)),
            schema_version=int(
                data.get(
                    "schema_version",
                    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
                )
            ),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "ExperimentManifest":
        """Load a manifest from a UTF-8 JSON file."""

        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
