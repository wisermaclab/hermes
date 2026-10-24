#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University
"""Reject release archives that contain prohibited or missing payloads."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import zipfile

from approved_fixtures import package_fixture

FORBIDDEN_SUFFIXES = {
    ".amc",
    ".asf",
    ".bin",
    ".c3d",
    ".ckpt",
    ".dat",
    ".dll",
    ".dylib",
    ".h5",
    ".hdf5",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".mat",
    ".npy",
    ".npz",
    ".obj",
    ".onnx",
    ".pdf",
    ".pickle",
    ".pkl",
    ".ply",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".so",
}
REQUIRED_BASENAMES = {"LICENSE", "THIRD_PARTY_NOTICES.md"}
MAX_MEMBER_BYTES = 2 * 1024 * 1024


def validate_member_name(name: str) -> PurePosixPath:
    if not name or "\\" in name or any(ord(char) < 32 for char in name):
        raise ValueError(f"unsafe archive member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name.rstrip("/"):
        raise ValueError(f"unsafe archive member name: {name!r}")
    return path


def audit_names(
    archive: Path, members: Iterable[tuple[str, int, bool, str | None]], errors: list[str]
) -> None:
    seen: set[str] = set()
    basenames: set[str] = set()
    for name, size, is_link, digest in members:
        try:
            path = validate_member_name(name)
        except ValueError as exc:
            errors.append(f"{archive.name}: {exc}")
            continue
        if name in seen:
            errors.append(f"{archive.name}: duplicate member {name!r}")
        seen.add(name)
        basenames.add(path.name)
        if is_link:
            errors.append(f"{archive.name}: link member is not allowed: {name}")
        if size > MAX_MEMBER_BYTES:
            errors.append(f"{archive.name}: member exceeds 2 MiB: {name}")
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            approved = package_fixture(name, wheel=archive.suffix == ".whl")
            if approved is None:
                errors.append(f"{archive.name}: forbidden binary, model, or data file: {name}")
            elif (size, digest) != approved:
                errors.append(f"{archive.name}: synthetic fixture size/hash mismatch: {name}")
    for basename in sorted(REQUIRED_BASENAMES - basenames):
        errors.append(f"{archive.name}: required notice is absent: {basename}")


def audit_wheel(path: Path, errors: list[str]) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                file_type = (info.external_attr >> 16) & 0o170000
                is_link = file_type == stat.S_IFLNK
                if info.flag_bits & 0x1:
                    errors.append(f"{path.name}: encrypted member: {info.filename}")
                approved = package_fixture(info.filename, wheel=True)
                digest = None
                if approved is not None and info.file_size == approved[0] and not is_link:
                    with archive.open(info) as stream:
                        digest = hashlib.sha256(stream.read(approved[0] + 1)).hexdigest()
                members.append((info.filename, info.file_size, is_link, digest))
            audit_names(path, members, errors)
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"{path.name}: invalid wheel archive: {exc}")


def audit_sdist(path: Path, errors: list[str]) -> None:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = []
            for member in archive.getmembers():
                if member.isdir():
                    continue
                approved = package_fixture(member.name, wheel=False)
                digest = None
                if approved is not None and member.size == approved[0] and member.isfile():
                    with archive.extractfile(member) as stream:
                        digest = hashlib.sha256(stream.read(approved[0] + 1)).hexdigest()
                members.append(
                    (member.name, member.size, member.issym() or member.islnk(), digest)
                )
            unsupported = [
                member.name
                for member in archive.getmembers()
                if not member.isdir()
                and not member.isfile()
                and not member.issym()
                and not member.islnk()
            ]
            for name in unsupported:
                errors.append(f"{path.name}: unsupported archive member: {name}")
            audit_names(path, members, errors)
    except (OSError, tarfile.TarError) as exc:
        errors.append(f"{path.name}: invalid source archive: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="?", default=Path("dist"))
    args = parser.parse_args()

    directory = args.directory
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    expected = set(wheels) | set(sdists)
    actual = (
        {path for path in directory.iterdir() if path.is_file()} if directory.is_dir() else set()
    )
    errors: list[str] = []
    if len(wheels) != 1:
        errors.append(f"expected exactly one wheel, found {len(wheels)}")
    if len(sdists) != 1:
        errors.append(f"expected exactly one .tar.gz sdist, found {len(sdists)}")
    for path in sorted(actual - expected):
        errors.append(f"unexpected distribution artifact: {path.name}")
    for path in wheels:
        audit_wheel(path, errors)
    for path in sdists:
        audit_sdist(path, errors)

    if errors:
        print("Distribution audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Verified one wheel and one source distribution.")


if __name__ == "__main__":
    main()
