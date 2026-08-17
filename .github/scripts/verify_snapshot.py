#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University
"""Verify that the public tree is an audited, self-contained snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = "PUBLIC_SNAPSHOT.json"
BOOTSTRAP_ALLOWED_FILES = {
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "SECURITY.md",
}
BOOTSTRAP_ALLOWED_PREFIXES = (".github/",)
TARGET_OWNED_FILES = {
    "SECURITY.md",
}
TARGET_OWNED_PREFIXES = (".github/",)
REQUIRED_BOOTSTRAP_FILES = {
    ".github/CODEOWNERS",
    ".github/scripts/audit_distribution.py",
    ".github/scripts/verify_snapshot.py",
    ".github/workflows/open-sync-pr.yml",
    ".github/workflows/public-ci.yml",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
}
REQUIRED_TARGET_OWNED_FILES = {
    ".github/CODEOWNERS",
    ".github/scripts/audit_distribution.py",
    ".github/scripts/verify_snapshot.py",
    ".github/workflows/open-sync-pr.yml",
    ".github/workflows/public-ci.yml",
    "SECURITY.md",
}
REQUIRED_MANAGED_FILES = {
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
}
FORBIDDEN_PREFIXES = (
    "device/",
    "docs/itu_p2040/",
    "models/",
    "outputs/",
)
FORBIDDEN_SUFFIXES = {
    ".amc",
    ".asf",
    ".c3d",
    ".ckpt",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".obj",
    ".pdf",
    ".pickle",
    ".pkl",
    ".ply",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".cfg",
    ".csv",
    ".css",
    ".html",
    ".in",
    ".ini",
    ".ipynb",
    ".js",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".svg",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
MAX_MANAGED_FILE_BYTES = 2 * 1024 * 1024
PRIVATE_MARKERS = (
    "/Users/",
    "/home/",
    "Dropbox",
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
    "BEGIN EC PRIVATE KEY",
)
CONFLICT_PATTERN = re.compile(r"^(?:<{7}|>{7})(?: |$)", re.MULTILINE)
LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


def fail(messages: list[str]) -> None:
    if messages:
        print("Public snapshot verification failed:", file=sys.stderr)
        for message in messages:
            print(f"- {message}", file=sys.stderr)
        raise SystemExit(1)


def tracked_entries() -> dict[str, str]:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    entries: dict[str, str] = {}
    errors: list[str] = []
    for value in result.stdout.split(b"\0"):
        if not value:
            continue
        try:
            metadata, raw_path = value.split(b"\t", 1)
            mode, _object_id, stage = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            errors.append("Git index contains an unparseable or non-UTF-8 path")
            continue
        if stage != "0":
            errors.append(f"{path}: unresolved Git index stage {stage}")
        if path in entries:
            errors.append(f"{path}: duplicate Git index entry")
        entries[path] = mode
    fail(errors)
    return entries


def bootstrap_allowed(path: str) -> bool:
    return path in BOOTSTRAP_ALLOWED_FILES or path.startswith(
        BOOTSTRAP_ALLOWED_PREFIXES
    )


def target_owned(path: str) -> bool:
    return path in TARGET_OWNED_FILES or path.startswith(TARGET_OWNED_PREFIXES)


def snapshot_managed_data(path: str) -> bool:
    """Return whether a snapshot-managed file belongs to public data.

    The authoritative private exporter audits and byte-pins released data.
    This target-side verifier therefore treats ``data/`` entries as binary-
    safe snapshot payloads while still checking their recorded hash and mode.
    """

    return path.startswith("data/")


def validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a non-empty string")
    if value == "." or "\\" in value:
        raise ValueError(f"unsafe path: {value!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"path contains a control character: {value!r}")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"path must use NFC Unicode normalization: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"unsafe path: {value!r}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_regular_files(
    paths: set[str], entries: dict[str, str], errors: list[str]
) -> None:
    for path in sorted(paths):
        file_path = ROOT / path
        mode = entries.get(path)
        if mode not in ("100644", "100755"):
            errors.append(f"{path}: unsupported Git mode {mode!r}")
        if not file_path.is_file() or file_path.is_symlink():
            errors.append(f"{path}: must be a regular non-symlink file")
            continue
        if file_path.stat().st_size > MAX_MANAGED_FILE_BYTES:
            errors.append(f"{path}: exceeds public file-size limit")
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{path}: governance files must be UTF-8 text")
            continue
        if path.startswith(".github/workflows/"):
            if "pull_request_target" in text:
                errors.append(f"{path}: pull_request_target is not allowed")
            for line_number, line in enumerate(text.splitlines(), start=1):
                match = re.match(r"^\s*uses:\s*([^\s#]+)", line)
                if match is None:
                    continue
                action = match.group(1)
                if action.startswith("./"):
                    continue
                revision = action.rpartition("@")[2]
                if not re.fullmatch(r"[0-9a-f]{40}", revision):
                    errors.append(
                        f"{path}:{line_number}: action must use an immutable SHA"
                    )


def validate_tracked_paths(paths: set[str], errors: list[str]) -> None:
    normalized_prefixes: dict[str, str] = {}
    for path in sorted(paths):
        try:
            validate_relative_path(path)
        except ValueError as exc:
            errors.append(f"tracked file has an invalid path: {exc}")
            continue
        prefix_parts: list[str] = []
        for part in PurePosixPath(path).parts:
            prefix_parts.append(part)
            prefix = "/".join(prefix_parts)
            normalized = unicodedata.normalize("NFC", prefix).casefold()
            previous = normalized_prefixes.get(normalized)
            if previous is not None and previous != prefix:
                errors.append(
                    f"{prefix!r} collides with {previous!r} on "
                    "case-insensitive filesystems"
                )
            normalized_prefixes[normalized] = prefix


def validate_notebook(path: Path, errors: list[str]) -> None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{path.relative_to(ROOT)} is not valid notebook JSON: {exc}")
        return
    if not isinstance(notebook, dict):
        errors.append(f"{path.relative_to(ROOT)} notebook root must be an object")
        return
    metadata = notebook.get("metadata", {})
    if not isinstance(metadata, dict):
        errors.append(f"{path.relative_to(ROOT)} notebook metadata must be an object")
        return
    if metadata.get("widgets"):
        errors.append(f"{path.relative_to(ROOT)} contains saved widget state")
    kernelspec = metadata.get("kernelspec", {})
    if not isinstance(kernelspec, dict):
        errors.append(f"{path.relative_to(ROOT)} kernelspec must be an object")
        kernelspec = {}
    kernel = f"{kernelspec.get('display_name', '')} {kernelspec.get('name', '')}"
    if ".venv" in kernel:
        errors.append(f"{path.relative_to(ROOT)} contains a local kernel name")
    cells = notebook.get("cells", [])
    if not isinstance(cells, list):
        errors.append(f"{path.relative_to(ROOT)} notebook cells must be a list")
        return
    for index, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            errors.append(f"{path.relative_to(ROOT)} cell {index} must be an object")
            continue
        if cell.get("outputs"):
            errors.append(f"{path.relative_to(ROOT)} cell {index} has outputs")
        if cell.get("execution_count") is not None:
            errors.append(
                f"{path.relative_to(ROOT)} cell {index} has an execution count"
            )
        if cell.get("attachments"):
            errors.append(f"{path.relative_to(ROOT)} cell {index} has attachments")


def main() -> None:
    entries = tracked_entries()
    tracked = set(entries)
    errors: list[str] = []
    snapshot_file = ROOT / SNAPSHOT_PATH
    validate_tracked_paths(tracked, errors)

    if not snapshot_file.exists():
        unexpected = sorted(path for path in tracked if not bootstrap_allowed(path))
        missing = sorted(REQUIRED_BOOTSTRAP_FILES - tracked)
        bootstrap_errors = [
            "bootstrap tree contains non-governance file "
            f"{path!r} without {SNAPSHOT_PATH}"
            for path in unexpected
        ]
        bootstrap_errors.extend(
            f"bootstrap tree is missing required file: {path}" for path in missing
        )
        bootstrap_errors.extend(errors)
        validate_regular_files(tracked, entries, bootstrap_errors)
        fail(bootstrap_errors)
        print("Governance-only bootstrap tree verified.")
        return

    if snapshot_file.is_symlink() or not snapshot_file.is_file():
        fail([f"{SNAPSHOT_PATH} must be a regular non-symlink file"])
        return
    if SNAPSHOT_PATH not in tracked:
        errors.append(f"{SNAPSHOT_PATH} must be tracked")
    elif entries[SNAPSHOT_PATH] != "100644":
        errors.append(f"{SNAPSHOT_PATH} must have Git mode 100644")

    try:
        snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail([f"{SNAPSHOT_PATH} is invalid: {exc}"])
        return
    if not isinstance(snapshot, dict):
        fail([f"{SNAPSHOT_PATH} must contain a JSON object"])
        return

    if snapshot.get("schema_version") != 1:
        errors.append("snapshot schema_version must be 1")
    expected_snapshot_keys = {
        "generator",
        "managed_files",
        "policy_sha256",
        "schema_version",
        "source_commit",
        "tree_sha256",
    }
    if set(snapshot) != expected_snapshot_keys:
        errors.append(
            "snapshot keys must be exactly: "
            + ", ".join(sorted(expected_snapshot_keys))
        )
    if snapshot.get("generator") != "internal/public_release/export_public_tree.py":
        errors.append("snapshot generator is not the approved exporter")
    policy_hash = snapshot.get("policy_sha256")
    if not isinstance(policy_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", policy_hash
    ):
        errors.append("policy_sha256 must be a lowercase SHA-256 digest")
    source_commit = snapshot.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", source_commit
    ):
        errors.append("source_commit must be a 40-character lowercase Git SHA")

    records = snapshot.get("managed_files")
    if not isinstance(records, list):
        fail(["managed_files must be a list"])
        return

    managed: dict[str, tuple[str, str]] = {}
    record_paths: list[str] = []
    normalized_paths: dict[str, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"managed_files[{index}] must be an object")
            continue
        if set(record) != {"mode", "path", "sha256"}:
            errors.append(
                f"managed_files[{index}] keys must be exactly: mode, path, sha256"
            )
        try:
            path = validate_relative_path(record.get("path"))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        record_paths.append(path)
        normalized = unicodedata.normalize("NFC", path).casefold()
        if normalized in normalized_paths and normalized_paths[normalized] != path:
            errors.append(
                f"{path}: collides with {normalized_paths[normalized]!r} "
                "on case-insensitive filesystems"
            )
        normalized_paths[normalized] = path
        mode = record.get("mode")
        digest = record.get("sha256")
        if mode not in ("100644", "100755"):
            errors.append(f"{path}: unsupported mode {mode!r}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"{path}: invalid sha256")
            continue
        if path in managed:
            errors.append(f"{path}: duplicate managed-file record")
            continue
        if target_owned(path) or path == SNAPSHOT_PATH:
            errors.append(f"{path}: target-owned paths cannot be snapshot-managed")
        managed[path] = (str(mode), digest)

    if record_paths != sorted(record_paths):
        errors.append("managed_files records must be sorted by path")

    for path in sorted(REQUIRED_MANAGED_FILES - set(managed)):
        errors.append(f"required source-managed file is absent: {path}")
    for path in sorted(REQUIRED_TARGET_OWNED_FILES - tracked):
        errors.append(f"required target-owned file is absent: {path}")

    expected_tracked = set(managed) | {SNAPSHOT_PATH}
    unexpected = sorted(
        tracked - expected_tracked - {path for path in tracked if target_owned(path)}
    )
    missing = sorted(set(managed) - tracked)
    errors.extend(f"unclassified tracked file: {path}" for path in unexpected)
    errors.extend(f"missing managed file: {path}" for path in missing)
    validate_regular_files(
        {path for path in tracked if target_owned(path)}, entries, errors
    )

    for path, (mode, expected_hash) in sorted(managed.items()):
        relative = PurePosixPath(path)
        file_path = ROOT / path
        normalized_path = path.casefold()
        managed_data = snapshot_managed_data(path)
        if not managed_data and any(
            normalized_path.startswith(prefix.casefold())
            for prefix in FORBIDDEN_PREFIXES
        ):
            errors.append(f"{path}: forbidden public path")
        if not managed_data and relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"{path}: forbidden public file type")
        if not managed_data and relative.suffix.lower() not in TEXT_SUFFIXES:
            errors.append(f"{path}: file type is not on the public text allowlist")
        if not file_path.is_file() or file_path.is_symlink():
            errors.append(f"{path}: must be a regular non-symlink file")
            continue
        if not managed_data and file_path.stat().st_size > MAX_MANAGED_FILE_BYTES:
            errors.append(f"{path}: exceeds public file-size limit")
        actual_hash = sha256(file_path)
        if actual_hash != expected_hash:
            errors.append(f"{path}: sha256 does not match the snapshot")
        actual_mode = entries.get(path)
        if actual_mode is not None and actual_mode != mode:
            errors.append(
                f"{path}: Git mode {actual_mode} does not match snapshot mode {mode}"
            )
        if relative.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{path}: expected UTF-8 text")
            continue
        if text.startswith(LFS_POINTER_PREFIX):
            errors.append(f"{path}: Git LFS pointer files are not allowed")
        for marker in PRIVATE_MARKERS:
            if marker in text:
                errors.append(f"{path}: contains private marker {marker!r}")
        if CONFLICT_PATTERN.search(text):
            errors.append(f"{path}: contains merge-conflict markers")
        if path == "THIRD_PARTY_NOTICES.md" and "**BLOCKED" in text:
            errors.append(f"{path}: contains unresolved release blockers")
        if relative.suffix.lower() == ".ipynb":
            validate_notebook(file_path, errors)

    canonical_records = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    aggregate = hashlib.sha256(canonical_records).hexdigest()
    if snapshot.get("tree_sha256") != aggregate:
        errors.append("tree_sha256 does not match managed-file records")

    fail(errors)
    print(f"Verified {len(managed)} managed public files.")


if __name__ == "__main__":
    main()
