# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Dataset-independent handoff checks and CLI plumbing for bundle adapters."""

import json
from functools import partial
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


class ArrayArchive(dict[str, np.ndarray]):
    """In-memory, pickle-free replacement for a NumPy NpzFile."""

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(self)


def load_array_archive(path: Path) -> ArrayArchive:
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected a NumPy .npz archive: {path}")
    with archive:
        return ArrayArchive((key, np.array(archive[key], copy=True)) for key in archive.files)


def frame_id(value: str | int) -> str:
    text = str(value).strip()
    if text.lower().startswith("frame"):
        text = text[5:]
    if not text.isdecimal():
        raise ValueError(f"Invalid frame ID {value!r}; expected a non-negative integer")
    return f"{int(text):06d}"


def resolve_user_path(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_output_root(
    name: str, path: Path | None, *, default_root: Path, prefix: str = ""
) -> Path:
    """Resolve an explicit destination or the adapter's named bundle root."""
    return resolve_user_path(path or default_root / f"{prefix}{name}" / "bundles")


def publish_staged_bundle(
    staged: Path,
    bundle_dir: Path,
    *,
    force: bool,
    ownership_check: Callable[..., bool],
    **identity,
) -> None:
    """Publish a checked bundle with the adapter's ownership policy."""
    from .writer import BundleWriter

    BundleWriter.publish_directory(
        staged,
        bundle_dir,
        force=force,
        ownership_check=partial(ownership_check, **identity),
    )


def scalar_archive_text(data: Mapping, name: str) -> str:
    value = np.asarray(data[name])
    if value.ndim != 0 or value.dtype.kind not in {"S", "U"}:
        raise ValueError(f"SMPL {name} must be a scalar string")
    item = value.item()
    if isinstance(item, bytes):
        try:
            return item.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"SMPL {name} must be valid UTF-8") from exc
    return str(item)


def validate_smpl_arrays(data: Mapping, *, profile: str, flatten_betas: bool = False) -> None:
    """Validate shared SMPL structure; adapters retain identity and clock policy."""
    required = ("poses", "trans", "betas", "gender", "world_up_axis", "world_forward_axis")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"SMPL archive is missing required fields: {missing}")
    poses, trans, betas = (np.asarray(data[key]) for key in ("poses", "trans", "betas"))
    if flatten_betas:
        betas = betas.reshape(-1)
    if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] != 72:
        raise ValueError("SMPL poses must have shape (T, 72) with T > 0")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError("SMPL trans must have shape (T, 3) matching poses")
    if betas.shape != (10,):
        raise ValueError("SMPL betas must have shape (10,)")
    for name, values in (("poses", poses), ("trans", trans), ("betas", betas)):
        if values.dtype.kind not in {"i", "u", "f"} or not np.all(np.isfinite(values)):
            raise ValueError(f"SMPL {name} must contain only finite real values")
        with np.errstate(over="ignore", invalid="ignore"):
            narrowed = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(narrowed)):
            raise ValueError(f"SMPL {name} values must be representable as float32")
    if scalar_archive_text(data, "gender").strip().lower() not in {"neutral", "male", "female"}:
        raise ValueError("SMPL gender must be neutral, male, or female")
    axes = tuple(
        scalar_archive_text(data, key).strip().lower()
        for key in ("world_up_axis", "world_forward_axis")
    )
    if axes != ("z", "-y"):
        raise ValueError(
            "SMPL archive must use canonical AMASS world axes: world_up_axis='z' and world_forward_axis='-y'"
        )
    if "profile" in data:
        observed = scalar_archive_text(data, "profile").strip().lower()
        if observed and observed != profile:
            raise ValueError(f"SMPL archive profile must be {profile}, got {observed!r}")
    if "model_type" in data and scalar_archive_text(data, "model_type").strip().lower() != "smpl":
        raise ValueError("SMPL model_type must be 'smpl'")


def check_bundle_output_integrity(bundle_dir: Path) -> dict:
    from .writer import canonicalize_directory
    from .integrity import check_bundle_integrity

    canonicalize_directory(bundle_dir)
    report = check_bundle_integrity(bundle_dir)
    return {
        "valid": bool(report["valid"]),
        "radar_adc_shape": report["primary_adc_artifact"]["shape"],
        "amass_poses_shape": report["amass_sequence"]["poses_shape"],
        "frame_count": report["frames"]["count"],
    }


def adapter_main(parser_factory, prepare, argv=None) -> int:
    parser = parser_factory()
    args = parser.parse_args(argv)
    try:
        result = prepare(args)
    except (OSError, IndexError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
