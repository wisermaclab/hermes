# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Stable public facade for unified HERMES bundles and DSP comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np

from bundle_prepare.contract import Bundle, load_adc, load_bundle_descriptor
from bundle_prepare.integrity import check_bundle_integrity
from bundle_prepare.safe_npz import (
    DEFAULT_NPZ_LIMITS,
    GUI_NPZ_LIMITS,
    NpzLimits,
    preflight_npz,
)
from validation.metrics import background_subtract_adc, normalized_correlation

from .dsp import framewise_range_doppler_map, range_time_map


HERMES_BUNDLE_SCHEMA_VERSION = 1
_MAX_BUNDLE_ZIP_MEMBERS = 4_096
_MAX_BUNDLE_ZIP_MEMBER_BYTES = 2 * 1024**3
_MAX_BUNDLE_ZIP_TOTAL_BYTES = 4 * 1024**3
_MAX_BUNDLE_ZIP_COMPRESSION_RATIO = 200.0


@dataclass(frozen=True)
class BundleResourceLimits:
    """Optional stricter limits for browser-supplied bundle material."""

    max_zip_members: int
    max_zip_member_bytes: int
    max_zip_total_bytes: int
    max_zip_compression_ratio: float
    max_npz_files: int
    max_npz_archive_total_bytes: int
    max_npz_array_total_bytes: int
    npz_limits: NpzLimits
    max_json_file_bytes: int
    max_xml_file_bytes: int
    max_text_total_bytes: int


GUI_BUNDLE_RESOURCE_LIMITS = BundleResourceLimits(
    max_zip_members=1_024,
    max_zip_member_bytes=256 * 1024**2,
    max_zip_total_bytes=512 * 1024**2,
    max_zip_compression_ratio=100.0,
    max_npz_files=64,
    max_npz_archive_total_bytes=512 * 1024**2,
    max_npz_array_total_bytes=128 * 1024**2,
    npz_limits=GUI_NPZ_LIMITS,
    max_json_file_bytes=8 * 1024**2,
    max_xml_file_bytes=2 * 1024**2,
    max_text_total_bytes=32 * 1024**2,
)


@dataclass(frozen=True)
class RadarProducts:
    """Framework-neutral measured or simulated radar products."""

    adc: np.ndarray
    range_profile_power: np.ndarray
    range_time_power: np.ndarray
    range_time_ranges_m: np.ndarray
    range_doppler_power: np.ndarray
    range_doppler_ranges_m: np.ndarray
    velocities_mps: np.ndarray


@dataclass(frozen=True)
class ComparisonMetrics:
    """Descriptive alignment metrics; these do not certify physical fidelity."""

    range_time_correlation: float
    range_doppler_correlation: float
    complex_correlation: float
    scale_adjusted_complex_nmse: float
    peak_range_error_m: float

    def to_dict(self) -> dict[str, float]:
        """Return plain values suitable for JSON and GUI tables."""

        return {
            "range_time_correlation": float(self.range_time_correlation),
            "range_doppler_correlation": float(self.range_doppler_correlation),
            "complex_correlation": float(self.complex_correlation),
            "scale_adjusted_complex_nmse": float(
                self.scale_adjusted_complex_nmse
            ),
            "peak_range_error_m": float(self.peak_range_error_m),
        }


@dataclass(frozen=True)
class ADCComparison:
    """Primary and candidate products with shared-axis comparison metrics."""

    primary: RadarProducts
    candidate: RadarProducts
    metrics: ComparisonMetrics


@dataclass(frozen=True)
class HermesBundle:
    """Validated bundle plus a stable public summary and product API."""

    root: Path
    bundle: Bundle
    integrity_report: dict[str, Any]
    fingerprint: str
    schema_version: int = HERMES_BUNDLE_SCHEMA_VERSION
    source_path: Path | None = None
    _temporary_directory: Any = None

    @property
    def bundle_id(self) -> str:
        """Return a portable identifier derived from the bundle directory."""

        return self.root.name

    @property
    def primary_adc(self) -> np.ndarray:
        """Return the bundle's primary ADC cube without copying it."""

        return self.bundle.real_adc

    @property
    def profile(self) -> str:
        """Return the declared bundle profile."""

        return self.bundle.profile

    @property
    def data_origin(self) -> str:
        """Return whether the primary ADC is measured or simulated."""

        return self.bundle.data_origin

    @property
    def available_simulations(self) -> tuple[str, ...]:
        """Return solver ADC cubes embedded in the bundle."""

        return tuple(sorted((self.bundle.simulated_adc_paths or {}).keys()))

    @property
    def can_resimulate(self) -> bool:
        """Whether the bundle declares inputs for a fresh human simulation."""

        return self.bundle.can_resimulate

    def load_bundled_simulation(self, solver: str) -> np.ndarray:
        """Load one embedded solver ADC cube."""

        return self.bundle.load_bundled_simulation(solver)

    def primary_products(
        self,
        *,
        background_subtraction: bool = False,
        clutter_removal: str = "none",
    ) -> RadarProducts:
        """Build display products directly from the primary ADC."""

        adc = np.asarray(self.bundle.real_adc)
        if background_subtraction:
            if self.bundle.background_adc is None:
                raise ValueError(
                    "background subtraction requires a background ADC artifact"
                )
            adc = background_subtract_adc(adc, self.bundle.background_adc)
        return compute_radar_products(
            adc,
            self.bundle.sensor_config.fmcw,
            clutter_removal=clutter_removal,
        )

    def compare_adc(
        self,
        candidate_adc: np.ndarray,
        *,
        background_subtraction: bool = False,
        clutter_removal: str = "none",
        channel_indices: tuple[int, ...] | None = None,
    ) -> ADCComparison:
        """Compare candidate ADC with the primary ADC on selected channels."""

        primary_adc = np.asarray(self.bundle.real_adc)
        full_primary_shape = primary_adc.shape
        selected_indices = None
        if channel_indices is not None:
            selected_indices = np.asarray(channel_indices, dtype=np.int64)
            if selected_indices.ndim != 1 or selected_indices.size == 0:
                raise ValueError("channel_indices must select at least one channel")
            if np.unique(selected_indices).size != selected_indices.size:
                raise ValueError("channel_indices must not contain duplicates")
            if np.any(selected_indices < 0) or np.any(
                selected_indices >= full_primary_shape[-1]
            ):
                raise ValueError(
                    "channel_indices contains an index outside the primary ADC "
                    f"channel axis [0, {full_primary_shape[-1] - 1}]"
                )
        if background_subtraction:
            if self.bundle.background_adc is None:
                raise ValueError(
                    "background subtraction requires a background ADC artifact"
                )
            background_adc = np.asarray(self.bundle.background_adc)
            if selected_indices is not None:
                background_adc = background_adc[..., selected_indices]
                primary_adc = primary_adc[..., selected_indices]
            primary_adc = background_subtract_adc(
                primary_adc,
                background_adc,
            )
        elif selected_indices is not None:
            primary_adc = primary_adc[..., selected_indices]
        candidate = np.asarray(candidate_adc)
        if selected_indices is not None and candidate.shape == full_primary_shape:
            candidate = candidate[..., selected_indices]
        if candidate.shape != primary_adc.shape:
            raise ValueError(
                "candidate ADC shape must match primary ADC shape "
                f"({candidate.shape} != {primary_adc.shape})"
            )
        fmcw = self.bundle.sensor_config.fmcw
        primary_products = compute_radar_products(
            primary_adc,
            fmcw,
            clutter_removal=clutter_removal,
        )
        candidate_products = compute_radar_products(
            candidate,
            fmcw,
            clutter_removal=clutter_removal,
        )
        return ADCComparison(
            primary=primary_products,
            candidate=candidate_products,
            metrics=_comparison_metrics(
                primary_adc,
                candidate,
                primary_products,
                candidate_products,
            ),
        )


def load_bundle(
    path: str | Path,
    *,
    resource_limits: BundleResourceLimits | None = None,
) -> HermesBundle:
    """Load and fully validate a HERMES bundle directory or exported ZIP."""

    source_path = Path(path).expanduser().resolve()
    temporary_directory = None
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="hermes-bundle-"
        )
        root = Path(temporary_directory.name)
        _extract_bundle_zip(source_path, root, limits=resource_limits)
    else:
        root = source_path
    if resource_limits is not None:
        _preflight_bundle_resources(root, limits=resource_limits)
    report = check_bundle_integrity(root)
    bundle = Bundle.load(root)
    return HermesBundle(
        root=root,
        bundle=bundle,
        integrity_report=report,
        fingerprint=bundle_fingerprint(root),
        source_path=source_path,
        _temporary_directory=temporary_directory,
    )


def _extract_bundle_zip(
    source: Path,
    destination: Path,
    *,
    limits: BundleResourceLimits | None = None,
) -> None:
    """Extract a flat portable bundle ZIP without path traversal or symlinks."""

    max_members = (
        _MAX_BUNDLE_ZIP_MEMBERS
        if limits is None
        else limits.max_zip_members
    )
    max_member_bytes = (
        _MAX_BUNDLE_ZIP_MEMBER_BYTES
        if limits is None
        else limits.max_zip_member_bytes
    )
    max_total_bytes = (
        _MAX_BUNDLE_ZIP_TOTAL_BYTES
        if limits is None
        else limits.max_zip_total_bytes
    )
    max_compression_ratio = (
        _MAX_BUNDLE_ZIP_COMPRESSION_RATIO
        if limits is None
        else limits.max_zip_compression_ratio
    )
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if not members:
            raise ValueError("bundle ZIP is empty")
        if len(members) > max_members:
            raise ValueError(
                "bundle ZIP contains too many members: "
                f"{len(members):,} > {max_members:,}"
            )
        declared_total = sum(
            int(member.file_size)
            for member in members
            if not member.is_dir()
        )
        if declared_total > max_total_bytes:
            raise ValueError(
                "bundle ZIP uncompressed size exceeds the limit: "
                f"{declared_total:,} > {max_total_bytes:,} bytes"
            )
        seen: set[str] = set()
        extracted_total = 0
        for member in members:
            name = member.filename
            relative = Path(name)
            if (
                relative.is_absolute()
                or "\\" in name
                or any(
                    part in {"", ".", ".."} or ":" in part
                    for part in relative.parts
                )
            ):
                raise ValueError(f"unsafe bundle ZIP member: {name!r}")
            portable_name = relative.as_posix()
            if portable_name in seen:
                raise ValueError(f"duplicate bundle ZIP member: {name!r}")
            seen.add(portable_name)
            file_type = (member.external_attr >> 16) & 0o170000
            if file_type == 0o120000:
                raise ValueError(f"bundle ZIP members must not be symlinks: {name}")
            if member.flag_bits & 0x1:
                raise ValueError(
                    f"encrypted bundle ZIP members are not supported: {name}"
                )
            if member.file_size > max_member_bytes:
                raise ValueError(
                    f"bundle ZIP member exceeds the size limit: {name!r}"
                )
            if (
                member.file_size > 0
                and member.file_size / max(member.compress_size, 1)
                > max_compression_ratio
            ):
                raise ValueError(
                    f"bundle ZIP member has an unsafe compression ratio: {name!r}"
                )
            target = destination / relative
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
                            "bundle ZIP exceeded the total size limit while "
                            "extracting"
                        )
                    output.write(chunk)


def _preflight_bundle_resources(
    root: Path,
    *,
    limits: BundleResourceLimits,
) -> None:
    """Preflight every nested NPZ in an untrusted materialized bundle."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("untrusted bundle root must be a regular directory")
    resolved_root = root.resolve(strict=True)
    bundle_entries = sorted(root.rglob("*"))
    for path in bundle_entries:
        if path.is_symlink():
            raise ValueError("untrusted bundles must not contain symlinks")
        try:
            path.resolve(strict=True).relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("bundle path escapes the bundle root") from exc

    npz_files = [
        path
        for path in bundle_entries
        if path.is_file() and path.suffix.casefold() == ".npz"
    ]
    if len(npz_files) > limits.max_npz_files:
        raise ValueError(
            f"bundle contains too many NPZ files: {len(npz_files):,} > "
            f"{limits.max_npz_files:,}"
        )
    archive_total = 0
    array_total = 0
    for path in npz_files:
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("bundle NPZ path escapes the bundle root") from exc
        archive_total += resolved.stat().st_size
        if archive_total > limits.max_npz_archive_total_bytes:
            raise ValueError("bundle NPZ files exceed the aggregate archive limit")
        arrays = preflight_npz(resolved, limits=limits.npz_limits)
        array_total += sum(info.nbytes for info in arrays.values())
        if array_total > limits.max_npz_array_total_bytes:
            raise ValueError("bundle NPZ arrays exceed the aggregate byte limit")
        if (
            array_total
            >= limits.npz_limits.min_compression_ratio_check_bytes
            and array_total / max(archive_total, 1)
            > limits.npz_limits.max_compression_ratio
        ):
            raise ValueError(
                "bundle NPZ files have an unsafe aggregate compression ratio"
            )

    text_total = 0
    for path in bundle_entries:
        suffix = path.suffix.casefold()
        if suffix not in {".json", ".xml"} or not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("bundle JSON/XML path escapes the bundle root") from exc
        size = resolved.stat().st_size
        limit = (
            limits.max_json_file_bytes
            if suffix == ".json"
            else limits.max_xml_file_bytes
        )
        if size > limit:
            raise ValueError(
                f"bundle {suffix[1:].upper()} file exceeds the size limit: "
                f"{resolved.name!r}"
            )
        text_total += size
        if text_total > limits.max_text_total_bytes:
            raise ValueError("bundle JSON/XML files exceed the aggregate size limit")

    # Parse only the bounded descriptor here so artifact-count/path-alias
    # checks run before integrity validation or any ADC is materialized.
    load_bundle_descriptor(root)


def load_simulated_adc(
    path: str | Path,
    *,
    npz_limits: NpzLimits | None = None,
) -> np.ndarray:
    """Load ADC from a validation-compatible NPZ archive."""

    preflight_npz(
        path,
        limits=DEFAULT_NPZ_LIMITS if npz_limits is None else npz_limits,
    )
    adc, _ = load_adc(Path(path), allowed_ndim=(4,))
    return adc


def bundle_fingerprint(path: str | Path) -> str:
    """Hash every regular bundle file by relative path and content."""

    root = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    for item in files:
        if item.is_symlink():
            raise ValueError("measurement bundles must not contain symlinked files")
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def compute_radar_products(
    adc: np.ndarray,
    fmcw,
    *,
    clutter_removal: str = "none",
) -> RadarProducts:
    """Compute the common GUI products from a measured or simulated ADC cube."""

    if clutter_removal not in ("none", "mean"):
        raise ValueError("clutter_removal must be 'none' or 'mean'")
    raw = np.asarray(adc)
    if raw.ndim != 4:
        raise ValueError("adc must have shape [frames, chirps, samples, channels]")
    remove_mean = clutter_removal == "mean"
    range_time_power, rt_ranges = range_time_map(
        raw,
        fmcw=fmcw,
        remove_mean_clutter=remove_mean,
    )
    rd_cube, rd_ranges, velocities = framewise_range_doppler_map(
        raw,
        fmcw=fmcw,
        num_tx=fmcw.num_tx,
        remove_mean_clutter=remove_mean,
    )
    range_doppler_power = np.sum(np.abs(rd_cube) ** 2, axis=-1)
    return RadarProducts(
        adc=raw,
        range_profile_power=np.mean(range_time_power, axis=0),
        range_time_power=range_time_power,
        range_time_ranges_m=rt_ranges,
        range_doppler_power=range_doppler_power,
        range_doppler_ranges_m=rd_ranges,
        velocities_mps=velocities,
    )


def _comparison_metrics(
    measured_adc: np.ndarray,
    simulated_adc: np.ndarray,
    measured: RadarProducts,
    simulated: RadarProducts,
) -> ComparisonMetrics:
    reference = np.asarray(measured_adc, dtype=np.complex128).reshape(-1)
    estimate = np.asarray(simulated_adc, dtype=np.complex128).reshape(-1)
    denom = float(np.linalg.norm(reference) * np.linalg.norm(estimate))
    complex_correlation = (
        0.0 if denom <= 1e-30 else float(abs(np.vdot(reference, estimate)) / denom)
    )
    estimate_energy = float(np.vdot(estimate, estimate).real)
    reference_energy = float(np.vdot(reference, reference).real)
    if estimate_energy <= 1e-30 or reference_energy <= 1e-30:
        complex_nmse = float("inf")
    else:
        scale = np.vdot(estimate, reference) / estimate_energy
        complex_nmse = float(
            np.linalg.norm(reference - scale * estimate) ** 2 / reference_energy
        )
    measured_peak = int(np.argmax(measured.range_profile_power))
    simulated_peak = int(np.argmax(simulated.range_profile_power))
    peak_error = abs(
        float(measured.range_time_ranges_m[measured_peak])
        - float(simulated.range_time_ranges_m[simulated_peak])
    )
    return ComparisonMetrics(
        range_time_correlation=normalized_correlation(
            measured.range_time_power,
            simulated.range_time_power,
        ),
        range_doppler_correlation=normalized_correlation(
            measured.range_doppler_power,
            simulated.range_doppler_power,
        ),
        complex_correlation=complex_correlation,
        scale_adjusted_complex_nmse=complex_nmse,
        peak_range_error_m=peak_error,
    )


__all__ = [
    "ADCComparison",
    "ComparisonMetrics",
    "BundleResourceLimits",
    "GUI_BUNDLE_RESOURCE_LIMITS",
    "HERMES_BUNDLE_SCHEMA_VERSION",
    "HermesBundle",
    "RadarProducts",
    "bundle_fingerprint",
    "compute_radar_products",
    "load_bundle",
    "load_simulated_adc",
]
