# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Allocation-free structural preflight for untrusted NumPy NPZ archives."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path, PurePosixPath
from typing import BinaryIO
import zipfile

import numpy as np
from numpy.lib import format as npformat


@dataclass(frozen=True)
class NpzLimits:
    """Resource ceilings enforced before NumPy allocates array payloads."""

    max_archive_bytes: int
    max_members: int
    max_member_bytes: int
    max_total_uncompressed_bytes: int
    max_compression_ratio: float
    max_header_bytes: int
    max_array_bytes: int
    max_total_array_bytes: int
    max_array_elements: int
    max_ndim: int
    max_dimension: int
    min_compression_ratio_check_bytes: int = 16 * 1024**2


@dataclass(frozen=True)
class NpyArrayInfo:
    """Shape and dtype metadata read from an NPY header without allocation."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool
    elements: int
    nbytes: int


DEFAULT_NPZ_LIMITS = NpzLimits(
    max_archive_bytes=2 * 1024**3,
    max_members=256,
    max_member_bytes=2 * 1024**3,
    max_total_uncompressed_bytes=4 * 1024**3,
    max_compression_ratio=200.0,
    max_header_bytes=64 * 1024,
    max_array_bytes=2 * 1024**3,
    max_total_array_bytes=4 * 1024**3,
    max_array_elements=250_000_000,
    max_ndim=8,
    max_dimension=50_000_000,
)

GUI_NPZ_LIMITS = NpzLimits(
    max_archive_bytes=256 * 1024**2,
    max_members=64,
    max_member_bytes=256 * 1024**2,
    max_total_uncompressed_bytes=512 * 1024**2,
    max_compression_ratio=100.0,
    max_header_bytes=32 * 1024,
    max_array_bytes=256 * 1024**2,
    max_total_array_bytes=512 * 1024**2,
    max_array_elements=64_000_000,
    max_ndim=5,
    max_dimension=10_000_000,
)


_NPY_HEADER_LENGTH_BYTES = {
    (1, 0): 2,
    (2, 0): 4,
    (3, 0): 4,
}


def _archive_source(
    source: str | Path | bytes | bytearray | memoryview | BinaryIO,
    *,
    max_archive_bytes: int,
):
    if isinstance(source, (str, Path)):
        path = Path(source)
        size = path.stat().st_size
        if size > max_archive_bytes:
            raise ValueError(
                f"NPZ archive is {size:,} bytes; the limit is "
                f"{max_archive_bytes:,} bytes"
            )
        return path, None
    if isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
        if len(payload) > max_archive_bytes:
            raise ValueError(
                f"NPZ archive is {len(payload):,} bytes; the limit is "
                f"{max_archive_bytes:,} bytes"
            )
        stream = BytesIO(payload)
        return stream, stream
    if not hasattr(source, "seek") or not hasattr(source, "tell"):
        raise TypeError("NPZ source must be a path, bytes, or seekable binary file")
    original_position = source.tell()
    source.seek(0, 2)
    size = source.tell()
    source.seek(original_position)
    if size > max_archive_bytes:
        raise ValueError(
            f"NPZ archive is {size:,} bytes; the limit is "
            f"{max_archive_bytes:,} bytes"
        )
    return source, original_position


def _portable_member_name(name: str) -> PurePosixPath:
    relative = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or relative.is_absolute()
        or any(
            part in {"", ".", ".."} or ":" in part
            for part in relative.parts
        )
    ):
        raise ValueError(f"unsafe NPZ member name: {name!r}")
    return relative


def _read_array_header_3_0(
    handle: BinaryIO,
    *,
    header_length: int,
) -> tuple[tuple[int, ...], bool, np.dtype]:
    """Read the UTF-8 NPY 3.0 header unsupported by NumPy's public readers."""

    length_prefix = handle.read(_NPY_HEADER_LENGTH_BYTES[(3, 0)])
    if len(length_prefix) != _NPY_HEADER_LENGTH_BYTES[(3, 0)]:
        raise EOFError("truncated NPY 3.0 header length")
    header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise EOFError("truncated NPY 3.0 header")
    header = header_bytes.decode("utf-8")
    try:
        metadata = ast.literal_eval(header)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("cannot parse NPY 3.0 header") from exc
    if not isinstance(metadata, dict):
        raise ValueError("NPY 3.0 header must contain a dictionary")
    if set(metadata) != {"descr", "fortran_order", "shape"}:
        raise ValueError("NPY 3.0 header has unexpected keys")
    shape = metadata["shape"]
    if not isinstance(shape, tuple) or not all(
        isinstance(value, int) for value in shape
    ):
        raise ValueError("NPY 3.0 header has an invalid shape")
    fortran_order = metadata["fortran_order"]
    if not isinstance(fortran_order, bool):
        raise ValueError("NPY 3.0 header has an invalid fortran_order value")
    try:
        dtype = npformat.descr_to_dtype(metadata["descr"])
    except (TypeError, ValueError) as exc:
        raise ValueError("NPY 3.0 header has an invalid dtype descriptor") from exc
    return shape, fortran_order, dtype


def _read_array_header(
    handle: BinaryIO,
    version: tuple[int, int],
    *,
    max_header_bytes: int,
) -> tuple[tuple[int, ...], bool, np.dtype]:
    """Read one bounded NPY header using only NumPy's public interfaces."""

    length_bytes = _NPY_HEADER_LENGTH_BYTES.get(version)
    if length_bytes is None:
        raise ValueError(f"unsupported NPY format version: {version!r}")
    header_start = handle.tell()
    length_prefix = handle.read(length_bytes)
    if len(length_prefix) != length_bytes:
        raise EOFError("truncated NPY header length")
    header_length = int.from_bytes(length_prefix, byteorder="little")
    if header_length > max_header_bytes:
        raise ValueError(
            f"NPY header is too large: {header_length:,} > "
            f"{max_header_bytes:,} bytes"
        )
    handle.seek(header_start)
    if version == (1, 0):
        return npformat.read_array_header_1_0(
            handle,
            max_header_size=max_header_bytes,
        )
    if version == (2, 0):
        return npformat.read_array_header_2_0(
            handle,
            max_header_size=max_header_bytes,
        )
    return _read_array_header_3_0(handle, header_length=header_length)


def _array_header(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    *,
    limits: NpzLimits,
) -> NpyArrayInfo:
    with archive.open(member) as handle:
        try:
            version = npformat.read_magic(handle)
            shape, fortran_order, dtype = _read_array_header(
                handle,
                version,
                max_header_bytes=limits.max_header_bytes,
            )
        except (EOFError, OSError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid NPY header in NPZ member {member.filename!r}"
            ) from exc
        payload_offset = handle.tell()

    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError(
            f"NPZ array {member.filename!r} must not use an object dtype; "
            "object arrays are incompatible with allow_pickle=False"
        )
    dimensions = tuple(int(value) for value in shape)
    if len(dimensions) > limits.max_ndim:
        raise ValueError(
            f"NPZ array {member.filename!r} has too many dimensions: "
            f"{len(dimensions)} > {limits.max_ndim}"
        )
    if any(value < 0 for value in dimensions):
        raise ValueError(
            f"NPZ array {member.filename!r} contains a negative dimension"
        )
    elements = math.prod(dimensions) if dimensions else 1
    if elements > limits.max_array_elements:
        raise ValueError(
            f"NPZ array {member.filename!r} has too many elements: "
            f"{elements:,} > {limits.max_array_elements:,}"
        )
    if any(value > limits.max_dimension for value in dimensions):
        raise ValueError(
            f"NPZ array {member.filename!r} exceeds a dimension limit"
        )
    nbytes = elements * int(dtype.itemsize)
    if nbytes > limits.max_array_bytes:
        raise ValueError(
            f"NPZ array {member.filename!r} exceeds the array byte limit: "
            f"{nbytes:,} > {limits.max_array_bytes:,}"
        )
    declared_payload_bytes = int(member.file_size) - int(payload_offset)
    if declared_payload_bytes != nbytes:
        raise ValueError(
            f"NPZ member {member.filename!r} payload length does not match "
            "its NPY header"
        )
    return NpyArrayInfo(
        name=member.filename[:-4],
        shape=dimensions,
        dtype=dtype,
        fortran_order=bool(fortran_order),
        elements=elements,
        nbytes=nbytes,
    )


def preflight_npz(
    source: str | Path | bytes | bytearray | memoryview | BinaryIO,
    *,
    limits: NpzLimits = DEFAULT_NPZ_LIMITS,
) -> dict[str, NpyArrayInfo]:
    """Inspect an NPZ central directory and all NPY headers before loading."""

    archive_source, restore_position = _archive_source(
        source,
        max_archive_bytes=limits.max_archive_bytes,
    )
    try:
        with zipfile.ZipFile(archive_source) as archive:
            members = archive.infolist()
            if not members:
                raise ValueError("NPZ archive is empty")
            if len(members) > limits.max_members:
                raise ValueError(
                    f"NPZ archive contains too many members: {len(members):,} "
                    f"> {limits.max_members:,}"
                )
            seen_members: set[str] = set()
            declared_total = 0
            compressed_total = 0
            arrays: dict[str, NpyArrayInfo] = {}
            total_array_bytes = 0
            for member in members:
                relative = _portable_member_name(member.filename)
                portable_name = relative.as_posix()
                if portable_name in seen_members:
                    raise ValueError(
                        f"duplicate NPZ member name: {member.filename!r}"
                    )
                seen_members.add(portable_name)
                file_type = (member.external_attr >> 16) & 0o170000
                if file_type == 0o120000:
                    raise ValueError(
                        f"NPZ members must not be symlinks: {member.filename!r}"
                    )
                if member.flag_bits & 0x1:
                    raise ValueError(
                        f"encrypted NPZ members are not supported: "
                        f"{member.filename!r}"
                    )
                if member.is_dir():
                    continue
                if relative.suffix.casefold() != ".npy":
                    raise ValueError(
                        f"NPZ member must be an NPY array: {member.filename!r}"
                    )
                if member.file_size > limits.max_member_bytes:
                    raise ValueError(
                        f"NPZ member exceeds the size limit: "
                        f"{member.filename!r}"
                    )
                ratio = member.file_size / max(member.compress_size, 1)
                # Small constant-valued arrays commonly compress far beyond a
                # ratio of 100. Absolute member and aggregate ceilings remain
                # authoritative, so reserve the ratio heuristic for payloads
                # large enough to make decompression amplification material.
                if (
                    member.file_size
                    >= limits.min_compression_ratio_check_bytes
                    and ratio > limits.max_compression_ratio
                ):
                    raise ValueError(
                        f"NPZ member has an unsafe compression ratio: "
                        f"{member.filename!r}"
                    )
                declared_total += int(member.file_size)
                compressed_total += int(member.compress_size)
                if declared_total > limits.max_total_uncompressed_bytes:
                    raise ValueError(
                        "NPZ archive exceeds the total uncompressed-size limit"
                    )
                if (
                    declared_total
                    >= limits.min_compression_ratio_check_bytes
                    and declared_total / max(compressed_total, 1)
                    > limits.max_compression_ratio
                ):
                    raise ValueError(
                        "NPZ archive has an unsafe aggregate compression ratio"
                    )
                array = _array_header(archive, member, limits=limits)
                if array.name in arrays:
                    raise ValueError(f"duplicate NPZ array key: {array.name!r}")
                arrays[array.name] = array
                total_array_bytes += array.nbytes
                if total_array_bytes > limits.max_total_array_bytes:
                    raise ValueError(
                        "NPZ arrays exceed the aggregate array-byte limit"
                    )
            if not arrays:
                raise ValueError("NPZ archive contains no arrays")
            return arrays
    except zipfile.BadZipFile as exc:
        raise ValueError("Expected a valid NumPy NPZ archive") from exc
    finally:
        if isinstance(restore_position, int):
            archive_source.seek(restore_position)


def require_array(
    arrays: dict[str, NpyArrayInfo],
    name: str,
    *,
    allowed_ndim: tuple[int, ...] | None = None,
    dtype_kinds: frozenset[str] | None = None,
    trailing_shape: tuple[int, ...] | None = None,
    max_elements: int | None = None,
    max_bytes: int | None = None,
) -> NpyArrayInfo:
    """Validate one preflighted array's inexpensive structural properties."""

    if name not in arrays:
        raise ValueError(f"NPZ archive must contain a {name!r} array")
    info = arrays[name]
    if allowed_ndim is not None and len(info.shape) not in allowed_ndim:
        raise ValueError(
            f"NPZ array {name!r} must have ndim in {allowed_ndim}; "
            f"received shape {info.shape}"
        )
    if dtype_kinds is not None and info.dtype.kind not in dtype_kinds:
        raise ValueError(
            f"NPZ array {name!r} has unsupported dtype {info.dtype}"
        )
    if trailing_shape is not None and (
        len(info.shape) < len(trailing_shape)
        or info.shape[-len(trailing_shape) :] != trailing_shape
    ):
        raise ValueError(
            f"NPZ array {name!r} must end in shape {trailing_shape}; "
            f"received {info.shape}"
        )
    if max_elements is not None and info.elements > max_elements:
        raise ValueError(
            f"NPZ array {name!r} exceeds the element limit: "
            f"{info.elements:,} > {max_elements:,}"
        )
    if max_bytes is not None and info.nbytes > max_bytes:
        raise ValueError(
            f"NPZ array {name!r} exceeds the byte limit: "
            f"{info.nbytes:,} > {max_bytes:,}"
        )
    return info


__all__ = [
    "DEFAULT_NPZ_LIMITS",
    "GUI_NPZ_LIMITS",
    "NpyArrayInfo",
    "NpzLimits",
    "preflight_npz",
    "require_array",
]
