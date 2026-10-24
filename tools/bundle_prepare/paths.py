# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Portable resource names and containment for untrusted bundle inputs.

Validate the raw name before constructing a Path: pathlib normalizes away empty
and dot components. Archive names, output names and existing resource paths use
the same lexical rules; filesystem containment additionally rejects symlinks.
"""

from pathlib import Path, PurePosixPath

WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def portable_relative_path(
    value: str,
    *,
    label: str = "path",
    directory: bool = False,
    windows_names: bool = False,
) -> PurePosixPath:
    """Validate an unnormalized POSIX name without allocating or opening files.

    ZIP directory entries may have one trailing slash. Windows device-name and
    trailing-dot restrictions are opt-in for scene assets, preserving NPZ keys.
    Parent traversal, dot/empty components, drive/URI syntax, backslashes and NUL
    bytes are rejected in every policy.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    name = value[:-1] if directory and value.endswith("/") else value
    parts = name.split("/")
    if (
        "\\" in name
        or "\0" in name
        or any(part in {"", ".", ".."} or ":" in part for part in parts)
        or (
            windows_names
            and any(
                part.endswith((".", " "))
                or part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES
                for part in parts
            )
        )
    ):
        raise ValueError(f"{label} must be a safe bundle-relative path: {value!r}")
    return PurePosixPath(name)


def safe_bundle_destination(root: str | Path, value: str, *, label: str) -> Path:
    """Return a contained destination, rejecting existing symlink components."""
    relative = portable_relative_path(value, label=label)
    resolved_root = Path(root).resolve(strict=True)
    cursor = resolved_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink inside the bundle")
    return cursor


def safe_bundle_file(
    root: str | Path,
    value: str,
    *,
    label: str,
    missing_kind: str = "bundle file",
) -> Path:
    """Resolve an existing file inside root without following bundle symlinks."""
    cursor = safe_bundle_destination(root, value, label=label)
    if not cursor.is_file():
        raise FileNotFoundError(f"Referenced {missing_kind} was not found: {value}")
    return cursor
