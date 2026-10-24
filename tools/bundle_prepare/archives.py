# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Resource-bounded extraction for portable bundle archives."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import zipfile

from .paths import portable_relative_path, safe_bundle_destination

_MAX_BUNDLE_ZIP_MEMBERS = 4096
_MAX_BUNDLE_ZIP_MEMBER_BYTES = 2 * 1024**3
_MAX_BUNDLE_ZIP_TOTAL_BYTES = 4 * 1024**3
_MAX_BUNDLE_ZIP_COMPRESSION_RATIO = 200.0


class ZipResourceLimits(Protocol):
    max_zip_members: int
    max_zip_member_bytes: int
    max_zip_total_bytes: int
    max_zip_compression_ratio: float


@dataclass(frozen=True)
class ZipExtractionLimits:
    max_zip_members: int = _MAX_BUNDLE_ZIP_MEMBERS
    max_zip_member_bytes: int = _MAX_BUNDLE_ZIP_MEMBER_BYTES
    max_zip_total_bytes: int = _MAX_BUNDLE_ZIP_TOTAL_BYTES
    max_zip_compression_ratio: float = _MAX_BUNDLE_ZIP_COMPRESSION_RATIO


def extract_bundle_zip(
    source: Path,
    destination: Path,
    *,
    limits: ZipResourceLimits | None = None,
) -> None:
    """Extract a flat portable bundle ZIP without path traversal or symlinks."""

    max_members = _MAX_BUNDLE_ZIP_MEMBERS if limits is None else limits.max_zip_members
    max_member_bytes = (
        _MAX_BUNDLE_ZIP_MEMBER_BYTES if limits is None else limits.max_zip_member_bytes
    )
    max_total_bytes = _MAX_BUNDLE_ZIP_TOTAL_BYTES if limits is None else limits.max_zip_total_bytes
    max_compression_ratio = (
        _MAX_BUNDLE_ZIP_COMPRESSION_RATIO if limits is None else limits.max_zip_compression_ratio
    )
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("bundle ZIP destination must not be a symlink")
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if not members:
            raise ValueError("bundle ZIP is empty")
        if len(members) > max_members:
            raise ValueError(
                "bundle ZIP contains too many members: " f"{len(members):,} > {max_members:,}"
            )
        declared_total = sum(int(member.file_size) for member in members if not member.is_dir())
        if declared_total > max_total_bytes:
            raise ValueError(
                "bundle ZIP uncompressed size exceeds the limit: "
                f"{declared_total:,} > {max_total_bytes:,} bytes"
            )
        seen: set[str] = set()
        extracted_total = 0
        for member in members:
            name = member.filename
            relative = portable_relative_path(
                name, label="unsafe bundle ZIP member", directory=member.is_dir()
            )
            portable_name = relative.as_posix()
            # Case-insensitive filesystems (macOS, Windows) would otherwise let
            # a later member silently overwrite an earlier one.
            if portable_name.casefold() in seen:
                raise ValueError(f"duplicate bundle ZIP member: {name!r}")
            seen.add(portable_name.casefold())
            file_type = (member.external_attr >> 16) & 0o170000
            if file_type == 0o120000:
                raise ValueError(f"bundle ZIP members must not be symlinks: {name}")
            if member.flag_bits & 0x1:
                raise ValueError(f"encrypted bundle ZIP members are not supported: {name}")
            if member.file_size > max_member_bytes:
                raise ValueError(f"bundle ZIP member exceeds the size limit: {name!r}")
            if (
                member.file_size > 0
                and member.file_size / max(member.compress_size, 1) > max_compression_ratio
            ):
                raise ValueError(f"bundle ZIP member has an unsafe compression ratio: {name!r}")
            destination.mkdir(parents=True, exist_ok=True)
            target = safe_bundle_destination(destination, portable_name, label="bundle ZIP member")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            member_bytes = 0
            with archive.open(member) as source_handle, target.open("wb") as output:
                while chunk := source_handle.read(1024 * 1024):
                    member_bytes += len(chunk)
                    extracted_total += len(chunk)
                    if member_bytes > max_member_bytes:
                        raise ValueError(
                            "bundle ZIP member exceeded the size limit while "
                            f"extracting: {name!r}"
                        )
                    if extracted_total > max_total_bytes:
                        raise ValueError(
                            "bundle ZIP exceeded the total size limit while extracting"
                        )
                    output.write(chunk)
