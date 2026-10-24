# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Portable software and execution provenance, without importing optional backends."""

from dataclasses import asdict, is_dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.name
    if callable(value):
        return {
            "callable": f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
        }
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def content_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=json_value, allow_nan=False).encode()
    ).hexdigest()


def array_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError("Cannot hash object arrays")
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def simulation_input_hashes(scene, radar, targets):
    """Hash evaluated motion knots and native scene parameters without storing meshes."""
    hashes = {
        "waveform": content_hash(radar.fmcw),
        "radar_pose_and_array": content_hash(
            {
                "position": radar.position,
                "orientation": radar.orientation,
                "tx": radar.world_tx_positions(),
                "rx": radar.world_rx_positions(),
                "tx_power_dbm": radar.tx_power_dbm,
                "board": radar.hardware.board_metadata,
                "hardware": radar.hardware,
            }
        ),
    }
    for index, target in enumerate(targets):
        sequence = target.mesh_sequence
        digest = hashlib.sha256()
        digest.update(array_hash(sequence.faces, sequence.times).encode())
        vertices = getattr(sequence, "vertices", None)
        if isinstance(vertices, np.ndarray):
            digest.update(array_hash(vertices).encode())
        else:
            for t in sequence.times:
                digest.update(array_hash(sequence.vertices_at(float(t))).encode())
        hashes[f"target_{index}_motion"] = digest.hexdigest()
        material = target.material
        if material is not None and is_dataclass(material):
            hashes[f"target_{index}_material"] = content_hash(material)
        elif material is not None:
            hashes[f"target_{index}_material"] = content_hash(
                {
                    key: np.asarray(getattr(material, key)).tolist()
                    for key in (
                        "relative_permittivity",
                        "conductivity",
                        "thickness",
                        "scattering_coefficient",
                        "xpd_coefficient",
                    )
                    if hasattr(material, key)
                }
            )
    native = getattr(scene, "mi_scene", None)
    if native is not None:
        import mitsuba as mi

        parameters = mi.traverse(native)
        hashes["scene_parameters"] = content_hash(
            {key: array_hash(np.asarray(value)) for key, value in parameters.items()}
        )
        materials = getattr(scene, "radio_materials", {})
        hashes["scene_radio_materials"] = content_hash(
            {
                name: {
                    key: np.asarray(getattr(material, key)).tolist()
                    for key in (
                        "relative_permittivity",
                        "conductivity",
                        "scattering_coefficient",
                        "xpd_coefficient",
                    )
                    if hasattr(material, key)
                }
                for name, material in materials.items()
            }
        )
    return hashes


def software_provenance():
    """Report unknown fields explicitly; a source checkout takes version precedence."""
    root = Path(__file__).resolve().parents[2]
    versions = {}
    for package in ("hermes-radar-sim", "sionna-rt", "mitsuba", "drjit", "numpy", "torch"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    project = root / "pyproject.toml"
    if project.is_file():
        match = re.search(r'^version\s*=\s*"([^"]+)"', project.read_text(), re.MULTILINE)
        if match:
            versions["hermes-radar-sim"] = match[1]

    def git(*args):
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    # Do not identify an installed wheel with the unrelated repository around it.
    checkout = project.is_file() and (root / "src/mmWaveRadar/provenance.py").is_file()
    sources = (
        {
            p.relative_to(root).as_posix(): file_sha256(p)
            for p in sorted((root / "src").rglob("*.py"))
        }
        if checkout
        else {}
    )
    return {
        "version": versions["hermes-radar-sim"],
        "git_commit": git("rev-parse", "HEAD") if checkout else None,
        "git_dirty": (
            bool(git("status", "--porcelain", "--untracked-files=normal")) if checkout else None
        ),
        "source_sha256": content_hash(sources) if sources else None,
        "dependencies": versions,
        "python": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
    }


def capture_provenance(*, settings=None, seed=None, input_hashes=None, runtime=None):
    """Additive output record; unknown execution fields remain null, never guessed."""
    mi = sys.modules.get("mitsuba")
    return {
        "schema_version": 1,
        "software": software_provenance(),
        "execution": {
            "seed": seed,
            "mitsuba_variant": mi.variant() if mi is not None else None,
            **(runtime or {}),
        },
        "settings": json.loads(json.dumps(settings or {}, default=json_value, allow_nan=False)),
        "input_hashes": input_hashes or {},
    }


def directory_hashes(root, *, exclude=("provenance.json",)):
    """Hash regular bundle files by portable relative name (including scene assets)."""
    root = Path(root)
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_symlink() and p.relative_to(root).as_posix() not in exclude
    }


def write_bundle_provenance(root):
    """Preserve original provenance; record this preparation and exact packaged inputs."""
    root = Path(root)
    path = root / "provenance.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    record = capture_provenance(input_hashes=directory_hashes(root))
    manifests = {
        p.name: json.loads(p.read_text()) for p in sorted(root.glob("experiment_manifest*.json"))
    }
    if manifests:
        record["experiment_manifests"] = manifests
        record["execution"]["seeds_by_manifest"] = {
            name: value.get("seed") for name, value in manifests.items()
        }
    if previous:
        record["upstream"] = previous
    path.write_text(json.dumps(record, indent=2, default=json_value, allow_nan=False) + "\n")
