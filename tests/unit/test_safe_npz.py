# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Adversarial tests for allocation-free NPZ preflight."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import warnings
import zipfile

import numpy as np
from numpy.lib import format as npformat
import pytest

from bundle_prepare.safe_npz import GUI_NPZ_LIMITS, preflight_npz


def _npz_bytes(**arrays) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _declared_array_npz(*, name: str, shape: tuple[int, ...]) -> bytes:
    member = BytesIO()
    npformat.write_array_header_1_0(
        member,
        {
            "descr": np.dtype("<f8").str,
            "fortran_order": False,
            "shape": shape,
        },
    )
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(f"{name}.npy", member.getvalue())
    return archive.getvalue()


def _array_member_bytes(
    *,
    version: tuple[int, int],
    values: np.ndarray,
) -> bytes:
    values = np.asarray(values)
    header = {
        "descr": npformat.dtype_to_descr(values.dtype),
        "fortran_order": False,
        "shape": values.shape,
    }
    member = BytesIO()
    if version == (1, 0):
        npformat.write_array_header_1_0(member, header)
    elif version == (2, 0):
        npformat.write_array_header_2_0(member, header)
    elif version == (3, 0):
        magic = b"\x93NUMPY\x03\x00"
        header_text = repr(header).encode("utf-8")
        padding = (-(len(magic) + 4 + len(header_text) + 1)) % 64
        header_bytes = header_text + (b" " * padding) + b"\n"
        member.write(magic)
        member.write(len(header_bytes).to_bytes(4, byteorder="little"))
        member.write(header_bytes)
    else:  # pragma: no cover - test helper misuse
        raise ValueError(f"unsupported test NPY version: {version!r}")
    member.write(values.tobytes(order="C"))
    return member.getvalue()


def _member_npz(*, name: str, payload: bytes) -> bytes:
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(f"{name}.npy", payload)
    return archive.getvalue()


@pytest.mark.parametrize("version", ((1, 0), (2, 0)))
def test_preflight_reads_supported_npy_headers_through_public_api(version):
    values = np.asarray([1.0, 2.0], dtype=np.float64)
    payload = _member_npz(
        name="values",
        payload=_array_member_bytes(version=version, values=values),
    )

    arrays = preflight_npz(payload, limits=GUI_NPZ_LIMITS)

    assert arrays["values"].shape == (2,)
    assert arrays["values"].dtype == values.dtype
    assert arrays["values"].nbytes == values.nbytes


def test_preflight_reads_utf8_npy_3_header_without_private_numpy_api():
    values = np.asarray([(1.0,)], dtype=[("distance_μm", "<f8")])
    member = _array_member_bytes(version=(3, 0), values=values)
    payload = _member_npz(name="values", payload=member)

    arrays = preflight_npz(payload, limits=GUI_NPZ_LIMITS)

    assert arrays["values"].shape == (1,)
    assert arrays["values"].dtype == values.dtype
    assert np.load(BytesIO(member), allow_pickle=False).dtype == values.dtype


def test_preflight_rejects_huge_declared_shape_without_loading_payload():
    payload = _declared_array_npz(name="adc", shape=(64_000_001,))

    with pytest.raises(ValueError, match="too many elements"):
        preflight_npz(payload, limits=GUI_NPZ_LIMITS)


def test_preflight_enforces_per_array_and_aggregate_array_bytes():
    payload = _npz_bytes(
        one=np.arange(8, dtype=np.float64),
        two=np.arange(8, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="array byte limit"):
        preflight_npz(
            payload,
            limits=replace(GUI_NPZ_LIMITS, max_array_bytes=32),
        )
    with pytest.raises(ValueError, match="aggregate array-byte limit"):
        preflight_npz(
            payload,
            limits=replace(
                GUI_NPZ_LIMITS,
                max_array_bytes=128,
                max_total_array_bytes=96,
            ),
        )


def test_preflight_enforces_member_count_size_and_compression_ratio():
    payload = _npz_bytes(
        one=np.zeros(100_000, dtype=np.uint8),
        two=np.zeros(1, dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="too many members"):
        preflight_npz(payload, limits=replace(GUI_NPZ_LIMITS, max_members=1))
    with pytest.raises(ValueError, match="member exceeds the size limit"):
        preflight_npz(
            payload,
            limits=replace(GUI_NPZ_LIMITS, max_member_bytes=1),
        )
    with pytest.raises(ValueError, match="unsafe compression ratio"):
        preflight_npz(
            payload,
            limits=replace(
                GUI_NPZ_LIMITS,
                max_compression_ratio=2.0,
                min_compression_ratio_check_bytes=1,
            ),
        )


def test_preflight_allows_small_benign_highly_compressible_motion():
    payload = _npz_bytes(poses=np.zeros((1000, 72), dtype=np.float32))

    arrays = preflight_npz(payload, limits=GUI_NPZ_LIMITS)

    assert arrays["poses"].shape == (1000, 72)


def test_preflight_rejects_aggregate_ratio_across_small_members():
    payload = _npz_bytes(
        **{
            f"part_{index}": np.zeros(400_000, dtype=np.uint8)
            for index in range(50)
        }
    )

    with pytest.raises(ValueError, match="aggregate compression ratio"):
        preflight_npz(payload, limits=GUI_NPZ_LIMITS)


def test_preflight_rejects_duplicate_and_object_arrays():
    member = BytesIO()
    npformat.write_array(member, np.asarray([1], dtype=np.int8))
    duplicate = BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr("same.npy", member.getvalue())
            archive.writestr("same.npy", member.getvalue())
    with pytest.raises(ValueError, match="duplicate NPZ member"):
        preflight_npz(duplicate.getvalue(), limits=GUI_NPZ_LIMITS)

    objects = _npz_bytes(value=np.asarray([object()], dtype=object))
    with pytest.raises(ValueError, match="object dtype"):
        preflight_npz(objects, limits=GUI_NPZ_LIMITS)


def test_preflight_rejects_malformed_header_and_payload_length():
    malformed = BytesIO()
    with zipfile.ZipFile(malformed, "w") as archive:
        archive.writestr("bad.npy", b"not-an-npy-header")
    with pytest.raises(ValueError, match="invalid NPY header"):
        preflight_npz(malformed.getvalue(), limits=GUI_NPZ_LIMITS)

    missing_payload = _declared_array_npz(name="missing", shape=(2,))
    with pytest.raises(ValueError, match="payload length does not match"):
        preflight_npz(missing_payload, limits=GUI_NPZ_LIMITS)
