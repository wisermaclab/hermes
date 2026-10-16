# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Panel/Plotly application composed over framework-neutral HERMES services."""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
import hashlib
import html
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import queue
import secrets
import threading
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from bundle_prepare.safe_npz import (
    GUI_NPZ_LIMITS,
    preflight_npz,
    require_array,
)

from ..dsp import angle_map_fft, direction_cosine_valid_mask, range_fft
from ..experiments import (
    HUMAN_ROOM_SIMULATION_MODES,
    HumanRoomPreview,
    MATERIAL_PRESETS,
    load_amass_motion,
    load_scene_xml,
    material_preset_parameters,
    motion_source_to_radar_transform,
    prepare_human_room_preview,
    prepared_room_boxes,
    run_human_room_diagnostics,
    run_human_room_experiment,
    run_static_target_diagnostics,
    run_static_target_experiment,
)
from ..measurements import (
    GUI_BUNDLE_RESOURCE_LIMITS,
    load_bundle,
    load_simulated_adc,
)
from ..radar import (
    FMCWConfig,
    RadarHardware,
    available_ti_boards,
    get_ti_board_spec,
    ti_digitized_pattern_asset_available,
)
from ..simulation import (
    azimuth_elevation_grids,
    calibrate_range_cube_channel_phase,
    virtual_snapshot_grid,
)
from ..simulation.physical_optics import default_phase_span_scale_rad
from ..radar.adc import if_cutoff_hz
from ..targets import MeshSequence
from ..tutorial_support import bedroom_scene_xml_for_boxes


_APP_CSS = """
:root {
  --hermes-bg: var(--background-color);
  --hermes-surface: var(--neutral-fill-card-rest);
  --hermes-border: var(--neutral-stroke-rest);
  --hermes-text: var(--neutral-foreground-rest);
  --hermes-accent: var(--accent-fill-rest);
}

html, body {
  background: var(--hermes-bg);
  color: var(--hermes-text);
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  font-size: 17px;
}

p, li, label, .bk-input, .bk-btn {
  font-size: 16px !important;
  line-height: 1.5;
}

h1 { font-size: 32px !important; }
h2 { font-size: 27px !important; margin-bottom: 8px !important; }
h3 { font-size: 20px !important; }

.bk-tab {
  font-size: 16px !important;
  font-weight: 650;
  padding: 12px 18px !important;
}

.hermes-callout {
  background: rgba(56, 189, 248, 0.09);
  border: 1px solid rgba(56, 189, 248, 0.28);
  border-radius: 8px;
  color: var(--hermes-text);
  padding: 12px 14px;
}

.hermes-compact-hint p, :host(.hermes-compact-hint) p {
  font-size: 14px !important;
  line-height: 1.35;
  margin: 0;
}

.hermes-status {
  color: var(--hermes-text);
  min-height: 48px;
}

.hermes-theme-button .bk-btn {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  color: #f8fbff !important;
}

/*
 * FastListTemplate computes its foreground token inside a design-provider.
 * In light sessions that token is not reliably inherited by Bokeh's Markdown
 * shadow roots. Scope an explicit foreground token to both application roots
 * so Markdown panes remain readable without changing the dark theme.
 */
.hermes-theme-light {
  --hermes-panel: #ffffff;
  --hermes-link: #0369a1;
  --hermes-link-tint: rgba(3, 105, 161, 0.12);
  --code-text-color: #075985;
  --code-bg-color: #e8eff5;
  --hermes-text: #172033;
  --neutral-foreground-rest: #172033;
  --design-background-text-color: #172033;
  color: #172033;
}

.hermes-theme-dark {
  --hermes-panel: #1f242b;
  --hermes-link: #38bdf8;
  --hermes-link-tint: rgba(56, 189, 248, 0.14);
  --hermes-text: #e7eef8;
  --neutral-foreground-rest: #e7eef8;
  --design-background-text-color: #e7eef8;
  color: #e7eef8;
}

.hermes-theme-light .bk-btn-light {
  background: #e2e8f0 !important;
  color: #172033 !important;
  border: 1px solid #94a3b8 !important;
}

/*
 * Settings dialog: one fixed size for every tab, so switching tabs never moves
 * the dialog, its tabs, or its Apply/Cancel footer. A taller tab scrolls
 * inside the content box; a shorter one leaves the footer at the bottom.
 */
fast-dialog#pn-Modal {
  --dialog-height: min(980px, calc(100vh - 32px));
  --dialog-width: min(1040px, calc(100vw - 32px));
}

/* The template places the close button for its default 80% width. */
#pn-Modal .pn-modal-close {
  margin-top: 12px;
  right: calc((100% - min(1040px, 100% - 32px)) / 2 + 12px);
}

#pn-Modal .pn-modal-content {
  height: 100%;
  margin: 0 auto;
  max-width: min(980px, calc(100vw - 48px));
  overflow-x: hidden;
  overflow-y: auto;
  overscroll-behavior: contain;
}

.hermes-settings { min-height: min(980px, calc(100vh - 32px)); }

.hermes-settings-title {
  font-size: 24px;
  font-weight: 650;
  line-height: 1.25;
  padding-top: 8px;
}

.hermes-settings-sub {
  font-size: 14px !important;
  margin: 4px 0 6px;
  opacity: 0.78;
}

.hermes-settings-note p, :host(.hermes-settings-note) p {
  margin: 4px 0;
  opacity: 0.8;
}

.hermes-settings-section {
  border-top: 1px solid var(--hermes-border);
  padding: 8px 0 4px;
}

.hermes-section-head {
  align-items: baseline;
  display: flex;
  flex-wrap: wrap;
  gap: 2px 12px;
  justify-content: space-between;
}

.hermes-used-by {
  font-size: 12.5px;
  opacity: 0.65;
}

.hermes-settings-grid {
  align-items: end;
  display: grid;
}

.hermes-settings-cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.hermes-settings-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.hermes-settings-cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }

.hermes-settings-wide { grid-column: span 2; }

.hermes-settings-split {
  align-items: flex-start;
  gap: 16px;
}

.hermes-settings-readout {
  background: var(--hermes-panel);
  border: 1px solid var(--hermes-border);
  border-left: 4px solid var(--hermes-link);
  border-radius: 10px;
  padding: 8px 2px;
  position: sticky;
  top: 12px;
}

.hermes-settings-footer {
  align-items: center;
  background: var(--hermes-panel);
  border-top: 1px solid var(--hermes-border);
  bottom: 0;
  gap: 10px;
  margin-top: auto;
  padding: 10px 8px;
  position: sticky;
  z-index: 5;
}

@media (max-width: 860px) {
  .hermes-settings-split {
    align-items: stretch;
    flex-direction: column;
  }

  .hermes-settings-readout {
    max-width: none !important;
    min-width: 0 !important;
    position: static;
  }
}

@media (max-width: 640px) {
  .hermes-settings-grid { grid-template-columns: minmax(0, 1fr); }
  .hermes-settings-wide { grid-column: auto; }
}

/*
 * Workflow template: a control rail with its own scroll beside a work column.
 * The work column starts with a sticky run bar, so the run state and actions
 * stay visible while the plots scroll. Nothing in the rail collapses.
 */
.hermes-page {
  align-items: flex-start;
  gap: 20px;
}

.hermes-rail {
  background: var(--hermes-panel);
  border: 1px solid var(--hermes-border);
  border-radius: 10px;
  max-height: calc(100vh - 92px);
  overflow-y: auto;
  overscroll-behavior: contain;
  position: sticky;
  scrollbar-gutter: stable;
  top: 0;
}

.hermes-rail-nav {
  background: var(--hermes-panel);
  border-bottom: 1px solid var(--hermes-border);
  position: sticky;
  top: 0;
  z-index: 3;
}

.hermes-rail-nav-links {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
  padding: 8px 10px;
}

.hermes-rail-nav-links button {
  background: none;
  border: 0;
  border-radius: 6px;
  color: inherit;
  cursor: pointer;
  font: inherit;
  font-size: 14px;
  opacity: 0.72;
  padding: 4px 9px;
  white-space: nowrap;
}

.hermes-rail-nav-links button:hover { opacity: 1; }

.hermes-rail-nav-links button[aria-current="true"] {
  background: var(--hermes-link-tint);
  color: var(--hermes-link);
  font-weight: 650;
  opacity: 1;
}

.hermes-rail-nav-links button:focus-visible {
  outline: 2px solid var(--hermes-link);
  outline-offset: 1px;
}

.hermes-group {
  border-bottom: 1px solid var(--hermes-border);
  padding: 8px 0 10px;
  transition: background-color 0.6s ease;
}

.hermes-group:last-child { border-bottom: 0; }

.hermes-group-flash {
  background: var(--hermes-link-tint);
  transition: none;
}

.hermes-group-title, .hermes-card-title {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  font-weight: 650;
  letter-spacing: 0.09em;
  opacity: 0.75;
  padding-top: 2px;
  text-transform: uppercase;
}

.hermes-runbar {
  background: var(--hermes-panel);
  border: 1px solid var(--hermes-border);
  border-left: 4px solid var(--hermes-link);
  border-radius: 10px;
  box-shadow: 0 12px 24px -20px rgba(0, 0, 0, 0.6);
  padding: 4px 6px 4px 8px;
  position: sticky;
  top: 0;
  z-index: 5;
}

.hermes-runbar-stale { border-left-color: #d69e2e; }

.hermes-runbar-banner {
  align-items: center;
  flex-direction: row;
  flex-wrap: wrap;
  gap: 0 12px;
}

.hermes-banner-text {
  flex: 1 1 320px;
  min-width: 0;
  width: auto !important;
}

.hermes-banner-actions {
  flex: 0 0 auto;
  width: auto !important;
}

.hermes-runbar-heading, .hermes-runbar-status {
  align-items: center;
  flex-wrap: wrap;
  gap: 4px 10px;
}

.hermes-runbar-title {
  font-size: 22px;
  font-weight: 650;
  line-height: 1.25;
  white-space: nowrap;
}

.hermes-runbar-sub {
  font-size: 15px !important;
  margin: 0;
  opacity: 0.78;
}

.hermes-work, .hermes-workspace { min-width: 0; }

/* Contain Plotly's high z-index mode bars below the sticky run bar. */
.hermes-workspace { isolation: isolate; }

.hermes-card {
  background: var(--hermes-panel);
  border: 1px solid var(--hermes-border);
  border-radius: 10px;
  min-width: 0;
  padding: 6px 4px;
}

@media (max-width: 1100px) {
  /* Panel propagates the rail and button widths as minimums; drop them. */
  .hermes-page, .hermes-theme-dark, .hermes-theme-light {
    min-width: 0 !important;
  }

  .hermes-page {
    align-items: stretch;
    flex-direction: column;
  }

  /* The run bar leads, pinned on top; the controls and then the plots follow. */
  .hermes-work { display: contents; }

  .hermes-runbar { order: -1; }

  /* Stretch-width children use a zero flex basis; in a column it would
     collapse their height, so the stacked blocks size to their content. */
  .hermes-rail, .hermes-runbar, .hermes-workspace {
    align-self: stretch !important;
    flex: 0 0 auto !important;
    height: auto !important;
  }

  .hermes-rail {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    max-height: none;
    max-width: none !important;
    min-width: 0 !important;
    overflow: visible;
    position: static;
    width: auto !important;
  }

  .hermes-rail-nav { display: none; }
}

@media (max-width: 700px) {
  .hermes-rail { grid-template-columns: minmax(0, 1fr); }
}

"""

# Rail and Settings controls use a smaller type size so groups stay compact.
_COMPACT_CONTROL_CSS = (
    "label, .bk-input, .bk-btn, p, li {font-size:14px !important;} "
    "label {line-height:1.3; white-space:normal !important;}"
)

_HERMES_ICON_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAALQAAAC0CAYAAAA9zQYyAAACkElEQVR42u3cMa6C"
    "YBCFUZZA7xrs2bzrsbOmk1DYGaPCX9yZU3wbuHNiXhTeNF+uT6lKkxEEtAS0BLQE"
    "tICWgJaAloCWgBbQEtAS0BLQEtACWgJaAloCWgJaQEtAS0BLQEtAC2gJaAloCWgJ"
    "aAEtAS0BLQEtJYBebuuwqh6182ZAAw2049gMaMexGdBA2wxooIF2HKCBdhybAe04"
    "NgMaaJsBDTTQjgM00I5jM6Adx2ZAA20zoIEG2nGABtpxbAa049gMaKBtBjTQQDsO"
    "0EA7js2AdhybAQ20zYAGGmjHARpox7EZ0I5jM6CBthnQQAPtOEAD7Tg2A9pxbAY0"
    "0DYDGmigHQdooB3HZkA7js2ABhpoxwEaaMexGdCOYzOggbYZ0AIa6KHdH0ADXQT0"
    "jvkV0ECXwVwBNdCNQb/DnI4a6KagP2FORg10Q9DfYE5FDXQz0L9gTkQNdCPQ/2BO"
    "Qw10E9BHMCehBroB6DMwp6AGujjoMzEnoAbaJ7RPaKD9DQ000L7lABpo30MDDbRf"
    "CoH2LIdnOYD2tB3QQHseGmigvbECNNDeKQTaW99AA+3/ctgMaKCBdhyggXYcmwHt"
    "ODYDGmibAQ000I4DNNCOYzOgHcdmQANtM6CBBtpxgAbacWwGtOPYDGigbQY00EA7"
    "DtBAO47NgHYcmwENtM2ABhpoxwEaaMexGdCOYzOggbYZ0EAD7ThAA+04NgPacWwG"
    "NNBAOw7QQDuOzYB2HJsBDbTNgAYaaMcBGmjHsRnQjmMzoIG2GdBAA+04QAPtODYD"
    "WgJaAlpAS0BLQEtAS0ALaAloCWgJaAloAS0BLQEtAS0BLaAloCWgJaAloAW0BLQE"
    "tAS0gJaAloCWgJaAFtAS0BLQ0qA2GnhShoQ4FswAAAAASUVORK5CYII="
)


_MIB = 1024 * 1024
_MAX_DROPPED_FILES = 1024
_MAX_DROPPED_FILE_BYTES = 256 * _MIB
_MAX_DROPPED_TOTAL_BYTES = 512 * _MIB
_MAX_STATIC_MESH_UPLOAD_BYTES = 64 * _MIB
_MAX_MOTION_UPLOAD_BYTES = 128 * _MIB
_MAX_SCENE_XML_UPLOAD_BYTES = 2 * _MIB
_MAX_GUI_OBJ_LINE_BYTES = 4 * _MIB
_MAX_GUI_OBJ_VERTICES = 2_000_000
_MAX_GUI_OBJ_TRIANGLES = 4_000_000


def _validated_upload_payload(
    payload: bytes | bytearray | memoryview | str,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    """Return a bounded browser-upload payload or raise a clear error."""

    if isinstance(payload, str):
        value = payload.encode("utf-8")
    elif isinstance(payload, (bytes, bytearray, memoryview)):
        value = bytes(payload)
    else:
        raise TypeError(f"{label} upload must contain bytes")
    if len(value) > int(max_bytes):
        raise ValueError(
            f"{label} upload is {len(value):,} bytes; the limit is "
            f"{int(max_bytes):,} bytes"
        )
    return value


def _preflight_gui_mesh_npz(payload: bytes) -> None:
    """Reject oversized or malformed browser mesh arrays before allocation."""

    arrays = preflight_npz(payload, limits=GUI_NPZ_LIMITS)
    require_array(
        arrays,
        "vertices",
        allowed_ndim=(2, 3),
        dtype_kinds=frozenset({"i", "u", "f"}),
        trailing_shape=(3,),
        max_elements=30_000_000,
        max_bytes=128 * _MIB,
    )
    require_array(
        arrays,
        "faces",
        allowed_ndim=(2,),
        dtype_kinds=frozenset({"i", "u"}),
        trailing_shape=(3,),
        max_elements=30_000_000,
        max_bytes=128 * _MIB,
    )


def _preflight_gui_mesh_obj(payload: bytes) -> None:
    """Bound OBJ expansion before a browser-provided mesh is parsed."""

    vertices = 0
    triangles = 0
    for line_number, raw_line in enumerate(BytesIO(payload), start=1):
        if len(raw_line) > _MAX_GUI_OBJ_LINE_BYTES:
            raise ValueError(
                f"OBJ line {line_number} exceeds the 4 MiB line-size limit"
            )
        parts = raw_line.strip().split()
        if not parts or parts[0].startswith(b"#"):
            continue
        if parts[0] == b"v":
            vertices += 1
            if vertices > _MAX_GUI_OBJ_VERTICES:
                raise ValueError(
                    "human mesh OBJ exceeds the 2,000,000-vertex limit"
                )
        elif parts[0] == b"f":
            if len(parts) < 4:
                raise ValueError(
                    f"invalid OBJ geometry at line {line_number}"
                )
            triangles += len(parts) - 3
            if triangles > _MAX_GUI_OBJ_TRIANGLES:
                raise ValueError(
                    "human mesh OBJ exceeds the 4,000,000-triangle limit"
                )


def _preflight_gui_motion_npz(payload: bytes) -> None:
    """Validate browser pose or mesh motion arrays before allocation."""

    arrays = preflight_npz(payload, limits=GUI_NPZ_LIMITS)
    if "vertices" in arrays:
        vertices = require_array(
            arrays,
            "vertices",
            allowed_ndim=(3,),
            dtype_kinds=frozenset({"i", "u", "f"}),
            trailing_shape=(3,),
            max_elements=30_000_000,
            max_bytes=128 * _MIB,
        )
        require_array(
            arrays,
            "faces",
            allowed_ndim=(2,),
            dtype_kinds=frozenset({"i", "u"}),
            trailing_shape=(3,),
            max_elements=6_000_000,
            max_bytes=32 * _MIB,
        )
        timing_key = next(
            (key for key in ("bundle_times", "times") if key in arrays), None
        )
        if timing_key is not None:
            timing = require_array(
                arrays,
                timing_key,
                allowed_ndim=(1,),
                dtype_kinds=frozenset({"i", "u", "f"}),
                max_elements=10_000_000,
            )
            if timing.shape != (vertices.shape[0],):
                raise ValueError("mesh motion times must match its frame count")
        else:
            require_array(
                arrays,
                "mocap_framerate",
                allowed_ndim=(0, 1),
                dtype_kinds=frozenset({"i", "u", "f"}),
                max_elements=1,
            )
        return
    poses = require_array(
        arrays,
        "poses",
        allowed_ndim=(2,),
        dtype_kinds=frozenset({"i", "u", "f"}),
        max_elements=20_000_000,
        max_bytes=128 * _MIB,
    )
    trans = require_array(
        arrays,
        "trans",
        allowed_ndim=(2,),
        dtype_kinds=frozenset({"i", "u", "f"}),
        trailing_shape=(3,),
        max_elements=20_000_000,
    )
    require_array(
        arrays,
        "betas",
        allowed_ndim=(1,),
        dtype_kinds=frozenset({"i", "u", "f"}),
        max_elements=10_000,
    )
    if "faces" in arrays:
        require_array(
            arrays,
            "faces",
            allowed_ndim=(2,),
            dtype_kinds=frozenset({"i", "u"}),
            trailing_shape=(3,),
            max_elements=6_000_000,
            max_bytes=32 * _MIB,
        )
    for name, maximum_bytes in (
        ("model_type", 256),
        ("gender", 256),
        ("smpl_model_dir", 4 * 1024),
    ):
        if name in arrays:
            require_array(
                arrays,
                name,
                allowed_ndim=(0,),
                dtype_kinds=frozenset({"S", "U"}),
                max_elements=1,
                max_bytes=maximum_bytes,
            )
    for name in ("times", "bundle_times"):
        if name in arrays:
            timing = require_array(
                arrays,
                name,
                allowed_ndim=(1,),
                dtype_kinds=frozenset({"i", "u", "f"}),
                max_elements=10_000_000,
            )
            if timing.shape != (poses.shape[0],):
                raise ValueError(
                    f"AMASS-like {name} must match the pose frame count"
                )
    for name, kinds in (
        ("output_axes", frozenset({"i", "u"})),
        ("output_signs", frozenset({"i", "u", "f"})),
    ):
        if name in arrays:
            require_array(
                arrays,
                name,
                allowed_ndim=(1,),
                dtype_kinds=kinds,
                trailing_shape=(3,),
                max_elements=3,
            )
    if "mocap_framerate" in arrays:
        require_array(
            arrays,
            "mocap_framerate",
            allowed_ndim=(0, 1),
            dtype_kinds=frozenset({"i", "u", "f"}),
            max_elements=1,
        )
    if poses.shape[0] <= 0 or trans.shape[0] != poses.shape[0]:
        raise ValueError("AMASS-like poses and trans frame counts must match")


def _server_path_key(path: str | Path) -> str:
    """Canonicalize a non-empty server path for an exact allowlist check."""

    value = str(path).strip()
    if not value:
        raise ValueError("server path is empty")
    return os.path.normcase(str(Path(value).expanduser().resolve(strict=False)))


def _authorized_server_path(
    path: str | Path,
    *,
    remote_access: bool,
    trusted_paths: set[str],
) -> str:
    """Reject browser-selected server paths in restricted remote sessions."""

    value = str(path)
    if remote_access and _server_path_key(value) not in trusted_paths:
        raise PermissionError(
            "remote GUI sessions cannot read arbitrary server paths; "
            "select an included fixture or upload the file through the browser"
        )
    return value


def _effective_smpl_model_dir(
    browser_value: str | Path,
    configured_value: str | Path,
    *,
    remote_access: bool,
) -> str:
    """Ignore browser-selected pickle roots when the GUI is remotely bound."""

    selected = configured_value if remote_access else browser_value
    return str(Path(selected).expanduser()) if selected else ""


def _materialize_dropped_files(
    uploaded: dict[str, bytes | str],
    destination: Path,
    *,
    max_files: int = _MAX_DROPPED_FILES,
    max_file_bytes: int = _MAX_DROPPED_FILE_BYTES,
    max_total_bytes: int = _MAX_DROPPED_TOTAL_BYTES,
) -> Path:
    """Safely reconstruct a browser-dropped file or folder in temporary storage."""

    if not uploaded:
        raise ValueError("no dropped files were received")
    if len(uploaded) > int(max_files):
        raise ValueError(
            f"dropped upload contains {len(uploaded):,} files; the limit is "
            f"{int(max_files):,}"
        )
    prepared: list[tuple[PurePosixPath, bytes]] = []
    total_bytes = 0
    for raw_name, raw_payload in uploaded.items():
        name = str(raw_name)
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
            raise ValueError(f"unsafe dropped-file path: {name!r}")
        if isinstance(raw_payload, str):
            payload = raw_payload.encode("utf-8")
        elif isinstance(raw_payload, (bytes, bytearray, memoryview)):
            payload = bytes(raw_payload)
        else:
            raise TypeError(f"unsupported dropped-file payload for {name!r}")
        if len(payload) > int(max_file_bytes):
            raise ValueError(
                f"dropped file {name!r} is {len(payload):,} bytes; the "
                f"per-file limit is {int(max_file_bytes):,} bytes"
            )
        total_bytes += len(payload)
        if total_bytes > int(max_total_bytes):
            raise ValueError(
                f"dropped upload is larger than the aggregate "
                f"{int(max_total_bytes):,}-byte limit"
            )
        prepared.append((relative, payload))

    destination.mkdir(parents=True, exist_ok=False)
    written: list[tuple[PurePosixPath, Path]] = []
    for relative, payload in prepared:
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        written.append((relative, target))

    if len(written) == 1 and written[0][0].suffix.casefold() in {
        ".npz",
        ".zip",
    }:
        return written[0][1]
    top_level = {relative.parts[0] for relative, _target in written}
    if len(top_level) == 1 and any(
        len(relative.parts) > 1 for relative, _target in written
    ):
        return destination / next(iter(top_level))
    return destination


def _file_input_stylesheet(plot_template: str, *, compact: bool = True) -> str:
    """Style a clipped native file chooser consistently in both themes."""

    if plot_template == "plotly_dark":
        background, hover, foreground, border = (
            "#182536",
            "#24364a",
            "#e7eef8",
            "#52657a",
        )
    else:
        background, hover, foreground, border = (
            "#f8fafc",
            "#e2e8f0",
            "#172033",
            "#94a3b8",
        )
    if not compact:
        return f"""
input[type=file] {{background: {background}; color: {foreground}; max-width:100%;}}
input[type=file]::file-selector-button {{
  background:{background}; color:{foreground}; border:1px solid {border};
  border-radius:6px; padding:7px 12px; margin-right:8px;
}}
input[type=file]::file-selector-button:hover {{background:{hover};}}
"""
    return f"""
:host {{
  overflow: hidden !important;
}}

label {{
  display: none !important;
}}

.bk-input-group {{
  height: 40px !important;
  overflow: hidden !important;
}}

input[type=file].bk-input {{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  color: transparent !important;
  height: 38px !important;
  max-width: 132px !important;
  min-width: 132px !important;
  overflow: hidden !important;
  padding: 0 !important;
  white-space: nowrap !important;
  width: 132px !important;
}}

input[type=file].bk-input:focus {{
  border: 0 !important;
  box-shadow: none !important;
  outline: 0 !important;
}}

input[type=file].bk-input::file-selector-button,
input[type=file].bk-input::-webkit-file-upload-button {{
  background: {background} !important;
  border: 1px solid {border} !important;
  border-radius: 6px !important;
  box-shadow: none !important;
  color: {foreground} !important;
  height: 38px !important;
  margin: 0 !important;
  padding: 7px 12px !important;
  width: 132px !important;
}}

input[type=file].bk-input::file-selector-button:hover,
input[type=file].bk-input::-webkit-file-upload-button:hover {{
  background: {hover} !important;
}}
"""


def _checkbox_stylesheet(plot_template: str) -> str:
    """Give unchecked and checked boxes sufficient contrast in both themes."""

    if plot_template == "plotly_dark":
        background, border, check, focus = (
            "#0d141f",
            "#8290a6",
            "#06141f",
            "rgba(79, 163, 216, 0.34)",
        )
    else:
        background, border, check, focus = (
            "#ffffff",
            "#64748b",
            "#ffffff",
            "rgba(3, 105, 161, 0.24)",
        )
    accent = "#4fa3d8" if plot_template == "plotly_dark" else "#0369a1"
    return f"""
input[type=checkbox] {{
  appearance: none !important;
  -webkit-appearance: none !important;
  background: {background} !important;
  border: 2px solid {border} !important;
  border-radius: 3px !important;
  display: inline-grid !important;
  height: 18px !important;
  margin-right: 8px !important;
  place-content: center !important;
  width: 18px !important;
}}
input[type=checkbox]::before {{
  border-color: {check} !important;
  border-style: solid !important;
  border-width: 0 2px 2px 0 !important;
  content: "" !important;
  height: 8px !important;
  transform: rotate(45deg) scale(0) !important;
  transition: transform 80ms ease-in-out !important;
  width: 4px !important;
}}
input[type=checkbox]:checked {{
  background: {accent} !important;
  border-color: {accent} !important;
}}
input[type=checkbox]:checked::before {{
  transform: rotate(45deg) scale(1) !important;
}}
input[type=checkbox]:focus-visible {{
  box-shadow: 0 0 0 4px {focus} !important;
  outline: 0 !important;
}}
"""

_PLOT_FONT_FAMILY = (
    "Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
)

# The negative-y component makes increasing range (+x) read left-to-right on
# screen, while the negative-x component keeps front-facing plates visible.
_SCENE_CAMERA_EYE = {"x": -1.45, "y": -1.55, "z": 0.80}
_PO_POWER_LIGHTING = {
    "ambient": 1.0,
    "diffuse": 0.0,
    "specular": 0.0,
    "roughness": 1.0,
    "fresnel": 0.0,
}

_HUMAN_ROOM_MODE_LABELS = {
    "full_rt": "Full RT",
    "coherent_rt": "Coherent RT",
    "human_only_po": "Human-only PO",
    "hybrid_po": "Hybrid PO",
}
_HUMAN_ROOM_MODE_HELP = {
    "full_rt": "Full RT retraces every chirp.",
    "coherent_rt": (
        "Coherent RT updates a path bank between frame retraces."
    ),
    "human_only_po": "Human-only PO excludes environment channels.",
    "hybrid_po": (
        "Hybrid PO includes blocking and coupling with absolute, "
        "uncalibrated PO amplitudes."
    ),
}

_SOLVER_SETTINGS_SCHEMA_VERSION = 1
_SOLVER_SETTINGS_STORAGE_KEY = "hermes.solver-settings.v1"
_PO_DEFAULTS_VERSION = 3
_PO_INTEGRATION_MODES = (
    "parent_face_quadrature",
    "parent_face_adaptive_analytic",
    "parent_face_far_field_analytic",
    "face_centroid",
)
_IMAGE_SOURCE_SELECTION_VERSION = 1
_GUI_SESSION_STORAGE_KEY = "hermes.gui-session.v1"
_GUI_SESSION_DB_NAME = "hermes-gui-session-v1"
_GUI_SESSION_DB_STORE = "uploads"
_THEME_COMPARISON_CACHE_MAX_ENTRIES = 4
_THEME_COMPARISON_CACHE: OrderedDict[str, dict[str, object]] = OrderedDict()
_THEME_COMPARISON_CACHE_LOCK = threading.Lock()
_THEME_GUI_RESULT_CACHE_MAX_ENTRIES = 4
_THEME_GUI_RESULT_CACHE: OrderedDict[str, dict[str, object]] = OrderedDict()
_THEME_GUI_RESULT_CACHE_LOCK = threading.Lock()
_SOLVER_SETTINGS_DEFAULTS = {
    "po_integration_mode": "parent_face_adaptive_analytic",
    "po_quadrature_phase_span_scale_rad": 6.0,
    "po_curvature_error_rad": 0.005,
    "po_quadrature_max_refinement_depth": 3,
    "po_quadrature_max_subfaces_per_parent": 64,
    "coupling_power_threshold_db": -20.0,
    "coupling_max_reflectors": 4,
    "rt_samples_per_source": 100_000,
    "rt_max_paths_per_source": 5_000,
    "rt_max_depth": 3,
}
_RADAR_ACTIVE_ANTENNA_DEFAULTS = {
    "MMWCAS_RF_EVM": {
        "tx": (0, 1, 2, 9),
        "rx": (4, 5, 6, 7),
    },
}
_RT_SOLVER_SETTING_HELP = {
    "rt_samples_per_source": (
        "Number of rays launched from each source during the RT path search. "
        "Higher values can discover weaker or less likely paths, but increase "
        "runtime and memory use."
    ),
    "rt_max_paths_per_source": (
        "Maximum number of discovered propagation paths retained for each "
        "source. Raising this cap reduces path truncation in complex scenes, "
        "but increases memory and downstream computation; it does not launch "
        "more rays."
    ),
    "rt_max_depth": (
        "Maximum number of scene interactions allowed along a path. Larger "
        "values permit additional reflection or scattering bounces, but can "
        "substantially increase search cost."
    ),
}
_SOLVER_SETTINGS_NUMERIC_BOUNDS = {
    "po_quadrature_phase_span_scale_rad": (0.05, 20.0, False),
    "po_curvature_error_rad": (0.000001, 10.0, False),
    "po_quadrature_max_refinement_depth": (0, 16, True),
    "po_quadrature_max_subfaces_per_parent": (0, 4096, True),
    "coupling_power_threshold_db": (-120.0, 0.0, False),
    "coupling_max_reflectors": (1, 4, True),
    "rt_samples_per_source": (100, 2_000_000, True),
    "rt_max_paths_per_source": (100, 2_000_000, True),
    "rt_max_depth": (1, 10, True),
}


def _human_room_running_status(progress: dict) -> str:
    """Describe the current solver pass, frame, and chirp."""

    mode = progress.get("simulation_mode")
    mode_label = _HUMAN_ROOM_MODE_LABELS.get(
        mode, str(mode).replace("_", " ").title()
    )
    parts = [f"**RUNNING** · {mode_label}"]
    solver_mode = progress.get("solver_mode")
    if solver_mode == "setup":
        return " · ".join([*parts, "Preparing"])
    if mode == "hybrid_po":
        pass_label = {
            "human_only_po": "Human PO",
            "hybrid_static_environment": "Static environment",
            "hybrid_coupling": "Human/environment coupling",
        }.get(solver_mode)
        if pass_label is not None:
            parts.append(pass_label)
    frame_index = max(int(progress.get("frame_index", 0)), 0)
    num_frames = max(int(progress.get("num_frames", 1)), 1)
    parts.append(f"frame {min(frame_index + 1, num_frames)}/{num_frames}")
    if "chirp_index" in progress and "num_chirps" in progress:
        chirp_index = max(int(progress["chirp_index"]), 0)
        num_chirps = max(int(progress["num_chirps"]), 1)
        parts.append(f"chirp {min(chirp_index + 1, num_chirps)}/{num_chirps}")
    return " · ".join(parts)


def _rounded_gui_float(value: float) -> float:
    """Normalize browser number-input noise without limiting radar precision."""

    return round(float(value), 6)


def _safe_markdown_text(value: object) -> str:
    """Escape untrusted text before interpolating it into a Markdown pane."""

    return html.escape(str(value), quote=True)


def _safe_markdown_code(value: object) -> str:
    """Escape once inside an inline HTML code element."""

    escaped = _safe_markdown_text(value).replace("`", "&#96;")
    return f"<code>{escaped}</code>"


def _fit_active_tab_height(tabs):
    """Keep eager tab models mounted while sizing only the selected pane."""

    def show_active(_event=None):
        for index, child in enumerate(tabs):
            child.visible = index == tabs.active

    tabs.param.watch(show_active, "active")
    show_active()


def _readable_markdown(root, pn, plot_template):
    tokens = _plot_theme_tokens(plot_template)
    css = (f"code {{color:{tokens['accent']} !important; background:transparent !important;}}"
           "h2 {font-size:27px !important;} h3 {font-size:20px !important;}")
    for pane in root.select(pn.pane.Markdown):
        if css not in pane.stylesheets:
            pane.stylesheets = [*pane.stylesheets, css]
    for card in root.select(pn.Card):
        header = card._header  # pylint: disable=protected-access
        if header is not None:
            header.stylesheets = ["h3 {font-size:20px !important; line-height:1.35; margin:4px 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}"]


def _style_secondary_buttons(root, pn, plot_template):
    dark = plot_template == "plotly_dark"
    background, foreground, border, hover = (
        ("#182536", "#e7eef8", "#64748b", "#26384d") if dark else
        ("#e2e8f0", "#172033", "#94a3b8", "#cbd5e1")
    )
    css = f"""
.bk-btn-light {{
  background: {background} !important; color: {foreground} !important;
  border: 1px solid {border} !important;
}}
.bk-btn-light:hover:enabled {{ background: {hover} !important; }}
.bk-btn-light:disabled {{ opacity: 0.55; }}
"""
    for widget in root.select(pn.widgets.Widget):
        if "color" in widget.param and widget.color == "light":
            widget.stylesheets = [*widget.stylesheets, css]


def _rail_nav_component(pn, items):
    """Jump links that scroll the enclosing control rail to one group.

    ``items`` pairs a group anchor (see ``_rail_group``) with its link label.
    The current group is marked while the rail scrolls; nothing collapses.
    """

    import param

    class RailNav(pn.reactive.ReactiveHTML):
        items = param.List(default=[])

        _template = (
            '<nav id="nav" class="hermes-rail-nav-links" '
            'aria-label="Jump to a group of controls"></nav>'
        )
        _scripts = {
            "render": """
const ancestor = (node, name) => {
  while (node != null) {
    if (node.classList?.contains(name)) return node
    node = node instanceof ShadowRoot ? node.host : node.parentNode
  }
  return null
}
const deepFind = (root, name) => {
  for (const element of root?.querySelectorAll?.('*') ?? []) {
    if (element.classList.contains(name)) return element
    if (element.shadowRoot != null) {
      const nested = deepFind(element.shadowRoot, name)
      if (nested != null) return nested
    }
  }
  return null
}
state.rail = () => ancestor(nav, 'hermes-rail')
state.bar = () => ancestor(nav, 'hermes-rail-nav')
state.group = anchor => {
  const rail = state.rail()
  return rail == null ? null : deepFind(rail.shadowRoot ?? rail, 'hermes-group-' + anchor)
}
state.buttons = []
nav.replaceChildren()
for (const [anchor, label] of data.items) {
  const button = document.createElement('button')
  button.type = 'button'
  button.textContent = label
  button.dataset.anchor = anchor
  button.addEventListener('click', () => state.jump(anchor))
  nav.appendChild(button)
  state.buttons.push(button)
}
state.mark = anchor => state.buttons.forEach((button, index) => button.setAttribute(
  'aria-current',
  String(anchor == null ? index === 0 : button.dataset.anchor === anchor),
))
// In narrow windows the rail stops scrolling and the page scrolls instead.
state.scrolls = rail => rail != null
  && rail.scrollHeight > rail.clientHeight + 1
  && getComputedStyle(rail).overflowY !== 'visible'
state.jump = anchor => {
  const rail = state.rail()
  const target = state.group(anchor)
  if (target == null) return
  state.mark(anchor)
  state.hold = Date.now() + 700
  const behavior = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ? 'auto' : 'smooth'
  if (state.scrolls(rail)) {
    const top = rail.scrollTop + target.getBoundingClientRect().top
      - rail.getBoundingClientRect().top - (state.bar()?.offsetHeight ?? 0)
    rail.scrollTo({top: Math.max(0, top), behavior})
  } else {
    target.scrollIntoView({block: 'start', behavior})
  }
  target.classList.add('hermes-group-flash')
  setTimeout(() => target.classList.remove('hermes-group-flash'), 450)
}
state.spy = () => {
  const rail = state.rail()
  if (!state.scrolls(rail) || Date.now() < (state.hold ?? 0)) return
  const line = rail.getBoundingClientRect().top + (state.bar()?.offsetHeight ?? 0) + 16
  let current = null
  for (const [anchor] of data.items) {
    const target = state.group(anchor)
    if (target != null && target.getBoundingClientRect().top <= line) current = anchor
  }
  if (rail.scrollTop + rail.clientHeight >= rail.scrollHeight - 2) {
    current = data.items[data.items.length - 1][0]
  }
  state.mark(current)
}
state.mark(null)
state.attach = () => {
  const rail = state.rail()
  if (rail == null) {
    state.timer = setTimeout(state.attach, 250)
    return
  }
  state.attached = rail
  rail.addEventListener('scroll', state.spy, {passive: true})
}
state.attach()
""",
            "remove": """
clearTimeout(state.timer)
state.attached?.removeEventListener('scroll', state.spy)
""",
        }

    return RailNav(
        items=[list(item) for item in items],
        css_classes=["hermes-rail-nav"],
        sizing_mode="stretch_width",
        margin=0,
    )


def _rail_group(pn, anchor, title, *objects, heading=None):
    """Group rail controls under one small, always-visible heading."""

    if heading is None:
        heading = pn.pane.HTML(
            '<div class="hermes-group-title" role="heading" aria-level="3">'
            f"{html.escape(title)}</div>",
            sizing_mode="stretch_width",
            margin=(0, 10),
        )
    return pn.Column(
        heading,
        *objects,
        name=title,
        sizing_mode="stretch_width",
        margin=0,
        css_classes=["hermes-group", f"hermes-group-{anchor}"],
    )


def _settings_section(pn, title, used_by, *objects):
    """One Settings section: a small heading, the workflows it affects, controls."""

    heading = pn.pane.HTML(
        '<div class="hermes-section-head">'
        '<span class="hermes-group-title" role="heading" aria-level="3">'
        f"{html.escape(title)}</span>"
        f'<span class="hermes-used-by">Used by {html.escape(used_by)}</span>'
        "</div>",
        sizing_mode="stretch_width",
        margin=(0, 10),
    )
    return pn.Column(
        heading,
        *objects,
        name=title,
        sizing_mode="stretch_width",
        margin=0,
        css_classes=["hermes-settings-section"],
    )


def _settings_grid(pn, *widgets, columns=2):
    """Lay settings out in a fixed number of columns so rows stay balanced."""

    return pn.Column(
        *widgets,
        sizing_mode="stretch_width",
        margin=0,
        css_classes=["hermes-settings-grid", f"hermes-settings-cols-{columns}"],
    )


def _settings_note(pn, text):
    """A short explanatory line at the top of a Settings tab."""

    return pn.pane.Markdown(
        text,
        sizing_mode="stretch_width",
        css_classes=["hermes-settings-note"],
    )


def _compact_controls(root, pn):
    """Apply the smaller control type size to every control inside ``root``."""

    for component in root.select(pn.viewable.Viewable):
        if isinstance(component, (pn.widgets.Widget, pn.pane.Markdown)):
            if _COMPACT_CONTROL_CSS not in component.stylesheets:
                component.stylesheets = [*component.stylesheets, _COMPACT_CONTROL_CSS]


def _run_bar(pn, *, title, subtitle, help_text, status, actions=(), banner):
    """Sticky page header: title, help, run status, actions and stale state."""

    heading = pn.Row(
        pn.pane.HTML(
            f'<div class="hermes-runbar-title" role="heading" aria-level="2">'
            f"{html.escape(title)}</div>",
            margin=(0, 4),
            styles={"flex": "0 0 auto", "width": "max-content"},
            stylesheets=[":host {width:max-content !important;}"],
        ),
        pn.widgets.TooltipIcon(
            value=help_text, width=22, height=22, sizing_mode="fixed", margin=0,
        ),
        pn.pane.HTML(
            f'<p class="hermes-runbar-sub">{html.escape(subtitle)}</p>',
            margin=(0, 4),
            sizing_mode="stretch_width",
            min_width=200,
        ),
        sizing_mode="stretch_width",
        margin=0,
        css_classes=["hermes-runbar-heading"],
    )
    status_row = pn.Row(
        status,
        *actions,
        sizing_mode="stretch_width",
        margin=0,
        css_classes=["hermes-runbar-status"],
    )
    bar = pn.Column(
        heading,
        status_row,
        banner,
        sizing_mode="stretch_width",
        margin=0,
        css_classes=["hermes-runbar"],
    )

    def show_stale_edge(event):
        classes = [name for name in bar.css_classes if name != "hermes-runbar-stale"]
        bar.css_classes = [*classes, "hermes-runbar-stale"] if event.new else classes

    banner.param.watch(show_stale_edge, "visible")
    return bar


def _runtime_stage_table(stages):
    """Show the largest measured stages; overlapping stages are not additive."""
    values = []
    for key, value in stages.items():
        seconds = float(value)
        if np.isfinite(seconds) and seconds > 0:
            # rt_* entries are aliases of the legacy full_rt_* stages.
            if key.startswith("rt_") and "full_" + key in stages:
                continue
            values.append((str(key), seconds))
    if not values:
        return ""
    threshold = max(0.001, max(seconds for _, seconds in values) * 0.001)
    largest = sorted((entry for entry in values if entry[1] >= threshold),
                     key=lambda entry: (-entry[1], entry[0]))[:10]
    if not largest:
        return ""
    rows = ["| Stage | Seconds |", "| --- | ---: |"]
    for key, seconds in largest:
        words = [word.upper() if word in {"rt", "po", "adc", "bvh", "fft"} else word
                 for word in key.split("_")]
        label = " ".join(words)
        label = label[0].upper() + label[1:]
        rows.append(f"| {_safe_markdown_text(label)} | {seconds:,.3f} |")
    rows.append("\nLargest measured stages; overlapping timings are not additive. "
                "Entries below 1 ms or 0.1% of the largest stage are omitted. "
                "All stages are available under Advanced.")
    return "\n".join(rows)


def _modal_events_component(pn):
    """Report native modal dismissal and label the icon-only buttons."""

    import param

    class ModalEvents(pn.reactive.ReactiveHTML):
        dismissed = param.Integer(default=0)
        _template = '<span id="events" style="display:none;"></span>'
        _scripts = {
            "render": """
const modal = document.getElementById('pn-Modal')
if (modal != null) {
  let wasOpen = modal.style.display !== 'none' && modal.style.display !== ''
  state.observer = new MutationObserver(() => {
    const open = modal.style.display !== 'none' && modal.style.display !== ''
    if (wasOpen && !open) data.dismissed += 1
    wasOpen = open
  })
  state.observer.observe(modal, {attributes: true, attributeFilter: ['style']})
  state.escape = event => {
    if (event.key === 'Escape' && wasOpen) modal.style.display = 'none'
  }
  document.addEventListener('keydown', state.escape)
}
let attempts = 0
state.buttonObservers = []
const watched = new WeakSet()
const labelButtons = () => {
  const found = new Set()
  const labels = {
    'hermes-icon-settings': 'Radar and solver settings',
    'hermes-icon-theme': 'Switch colour theme',
    'hermes-icon-download': 'Download simulated ADC',
  }
  const findButton = root => {
    const button = root.querySelector('button')
    if (button != null) return button
    for (const element of root.querySelectorAll('*')) {
      if (element.shadowRoot) {
        const nested = findButton(element.shadowRoot)
        if (nested != null) return nested
      }
    }
    return null
  }
  const visit = root => {
    for (const element of root.querySelectorAll('*')) {
      for (const [name, label] of Object.entries(labels)) {
        if (element.classList.contains(name)) {
          const widgetRoot = element.shadowRoot ?? element
          const button = findButton(widgetRoot)
          if (button != null) {
            button.setAttribute('aria-label', label)
            button.setAttribute('title', label)
            found.add(name)
            if (!watched.has(widgetRoot)) {
              watched.add(widgetRoot)
              const observer = new MutationObserver(() => {
                const current = findButton(widgetRoot)
                current?.setAttribute('aria-label', label)
                current?.setAttribute('title', label)
              })
              observer.observe(widgetRoot, {childList: true, subtree: true})
              state.buttonObservers.push(observer)
            }
          }
        }
      }
      if (element.shadowRoot) visit(element.shadowRoot)
    }
  }
  visit(document)
  if (found.size < 3 && ++attempts < 20) state.timer = setTimeout(labelButtons, 250)
}
labelButtons()
""",
            "remove": """
state.observer?.disconnect()
for (const observer of state.buttonObservers ?? []) observer.disconnect()
document.removeEventListener('keydown', state.escape)
clearTimeout(state.timer)
""",
        }

    return ModalEvents()


def _result_settings_fingerprint(values: dict) -> str:
    """Identify solver inputs independently of widget events and theme."""

    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    encoded = json.dumps(canonical(values), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rounded_target_size(value: float) -> float:
    """Normalize target dimensions to their two-decimal UI precision."""

    return round(float(value), 2)


def _validated_solver_settings_payload(payload: object) -> dict[str, object]:
    """Return supported, in-range values from one browser storage payload."""

    if not isinstance(payload, dict):
        return {}
    if payload.get("schema_version") != _SOLVER_SETTINGS_SCHEMA_VERSION:
        return {}
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        return {}

    validated: dict[str, object] = {}
    integration_mode = settings.get("po_integration_mode")
    if integration_mode in _PO_INTEGRATION_MODES:
        validated["po_integration_mode"] = integration_mode

    for name, (lower, upper, integer) in _SOLVER_SETTINGS_NUMERIC_BOUNDS.items():
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not np.isfinite(numeric) or not lower <= numeric <= upper:
            continue
        if integer:
            converted = int(numeric)
            if numeric != converted:
                continue
            validated[name] = converted
        else:
            validated[name] = numeric
    if payload.get("po_defaults_version") != _PO_DEFAULTS_VERSION:
        # Migrate earlier GUI profiles once, then retain subsequent PO edits.
        validated.update({
            name: value
            for name, value in _SOLVER_SETTINGS_DEFAULTS.items()
            if name.startswith("po_")
        })
    return validated


def _validated_po_phase_span_by_mode(payload: object) -> dict[str, float]:
    """Return valid remembered spans from a current solver-settings payload."""

    if not isinstance(payload, dict):
        return {}
    if (payload.get("schema_version") != _SOLVER_SETTINGS_SCHEMA_VERSION
            or payload.get("po_defaults_version") != _PO_DEFAULTS_VERSION):
        return {}
    spans = payload.get("po_phase_span_by_mode")
    if not isinstance(spans, dict):
        return {}
    lower, upper, _ = _SOLVER_SETTINGS_NUMERIC_BOUNDS[
        "po_quadrature_phase_span_scale_rad"
    ]
    validated = {}
    for mode, value in spans.items():
        if (mode not in _PO_INTEGRATION_MODES or isinstance(value, bool)
                or not isinstance(value, (int, float))):
            continue
        numeric = float(value)
        if np.isfinite(numeric) and lower <= numeric <= upper:
            validated[mode] = numeric
    return validated


def _gui_modules():
    try:
        import panel as pn
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError(
            "The HERMES GUI requires optional dependencies. Install them with "
            '`python -m pip install "hermes-radar-sim[gui]"`.'
        ) from exc
    return pn, go, make_subplots


def _solver_settings_storage_component(pn):
    """Bridge versioned solver settings to this browser's local storage."""

    import param

    class SolverSettingsStorage(pn.reactive.ReactiveHTML):
        loaded_settings = param.Dict(default={})
        saved_settings = param.Dict(default={})

        _template = '<span id="storage" style="display:none;"></span>'
        _scripts = {
            "render": f"""
try {{
  const raw = window.localStorage.getItem("{_SOLVER_SETTINGS_STORAGE_KEY}")
  if (raw !== null) {{
    const saved = JSON.parse(raw)
    if (saved === null || typeof saved !== "object" || Array.isArray(saved)) {{
      throw new Error("Stored solver settings are not an object")
    }}
    data.loaded_settings = {{
      ...saved,
      load_sequence: Date.now(),
    }}
  }}
}} catch (error) {{
  console.warn("Ignoring invalid HERMES solver settings", error)
  window.localStorage.removeItem("{_SOLVER_SETTINGS_STORAGE_KEY}")
}}
""",
            "saved_settings": f"""
try {{
  const saved = data.saved_settings
  if (saved != null && Object.keys(saved).length > 0) {{
    window.localStorage.setItem(
      "{_SOLVER_SETTINGS_STORAGE_KEY}",
      JSON.stringify(saved)
    )
  }}
}} catch (error) {{
  console.warn("Unable to save HERMES solver settings", error)
}}
""",
        }

    return SolverSettingsStorage()


def _gui_session_storage_component(pn):
    """Preserve navigation, controls, and uploads across theme reloads."""

    import param

    class GuiSessionStorage(pn.reactive.ReactiveHTML):
        loaded_state = param.Dict(default={})
        loaded_uploads = param.Dict(default={})
        saved_state = param.Dict(default={})
        saved_uploads = param.Dict(default={})

        _template = '<span id="storage" style="display:none;"></span>'
        _scripts = {
            "render": f"""
let restored = null
try {{
  const raw = window.sessionStorage.getItem("{_GUI_SESSION_STORAGE_KEY}")
  if (raw !== null) {{
    restored = JSON.parse(raw)
    if (restored === null || typeof restored !== "object" || Array.isArray(restored)) {{
      throw new Error("Stored GUI session state is not an object")
    }}
    data.loaded_state = {{...restored, load_sequence: Date.now()}}
  }}
}} catch (error) {{
  console.warn("Ignoring invalid HERMES GUI session state", error)
  window.sessionStorage.removeItem("{_GUI_SESSION_STORAGE_KEY}")
}}

if (
  restored != null
  && (
    restored.has_uploads === true
    || restored.has_motion === true
  )
) {{
  const request = window.indexedDB.open("{_GUI_SESSION_DB_NAME}", 1)
  request.onupgradeneeded = event => {{
    const database = event.target.result
    if (!database.objectStoreNames.contains("{_GUI_SESSION_DB_STORE}")) {{
      database.createObjectStore("{_GUI_SESSION_DB_STORE}")
    }}
  }}
  request.onsuccess = event => {{
    const database = event.target.result
    const transaction = database.transaction(
      "{_GUI_SESSION_DB_STORE}", "readonly"
    )
    const store = transaction.objectStore("{_GUI_SESSION_DB_STORE}")
    const uploadsLookup = store.get("uploads")
    uploadsLookup.onsuccess = () => {{
      const saved = uploadsLookup.result
      if (saved != null && typeof saved === "object") {{
        data.loaded_uploads = {{...saved, load_sequence: Date.now()}}
        return
      }}
      // Backward-compatible recovery for sessions created before all three
      // GUI uploads were persisted together.
      const motionLookup = store.get("motion")
      motionLookup.onsuccess = () => {{
        const motion = motionLookup.result
        if (
          motion != null
          && typeof motion.filename === "string"
          && typeof motion.payload === "string"
        ) {{
          data.loaded_uploads = {{
            motion,
            load_sequence: Date.now(),
          }}
        }}
      }}
    }}
    transaction.oncomplete = () => database.close()
  }}
  request.onerror = () => {{
    console.warn("Unable to restore HERMES GUI uploads", request.error)
  }}
}}
""",
            "saved_state": f"""
try {{
  const saved = data.saved_state
  if (saved != null && Object.keys(saved).length > 0) {{
    window.sessionStorage.setItem(
      "{_GUI_SESSION_STORAGE_KEY}",
      JSON.stringify(saved)
    )
  }}
}} catch (error) {{
  console.warn("Unable to save HERMES GUI session state", error)
}}
""",
            "saved_uploads": f"""
const saved = data.saved_uploads
const request = window.indexedDB.open("{_GUI_SESSION_DB_NAME}", 1)
request.onupgradeneeded = event => {{
  const database = event.target.result
  if (!database.objectStoreNames.contains("{_GUI_SESSION_DB_STORE}")) {{
    database.createObjectStore("{_GUI_SESSION_DB_STORE}")
  }}
}}
request.onsuccess = event => {{
  const database = event.target.result
  const transaction = database.transaction(
    "{_GUI_SESSION_DB_STORE}", "readwrite"
  )
  const store = transaction.objectStore("{_GUI_SESSION_DB_STORE}")
  if (saved != null && Object.keys(saved).length > 0) {{
    store.put(saved, "uploads")
  }} else {{
    store.delete("uploads")
  }}
  store.delete("motion")
  transaction.oncomplete = () => database.close()
}}
request.onerror = () => {{
  console.warn("Unable to save HERMES GUI uploads", request.error)
}}
""",
        }

    return GuiSessionStorage()


def _cache_theme_comparison(
    *,
    bundle_fingerprint: str,
    comparison_kind: str,
    candidates: list[tuple[str, str, str, str, np.ndarray]],
    background_subtraction: bool,
    clutter_removal: str,
    channel_indices: tuple[int, ...] | None,
    configuration_signature: tuple[object, ...] | None = None,
    result_settings: dict | None = None,
) -> str:
    """Retain completed candidate ADC briefly across a theme reload."""

    key = secrets.token_urlsafe(24)
    payload: dict[str, object] = {
        "bundle_fingerprint": str(bundle_fingerprint),
        "comparison_kind": str(comparison_kind),
        "candidates": [
            (
                str(candidate_key),
                str(label),
                str(heading),
                str(detail),
                np.asarray(adc),
            )
            for candidate_key, label, heading, detail, adc in candidates
        ],
        "background_subtraction": bool(background_subtraction),
        "clutter_removal": str(clutter_removal),
        "channel_indices": channel_indices,
        "configuration_signature": configuration_signature,
        "result_settings": result_settings,
    }
    with _THEME_COMPARISON_CACHE_LOCK:
        _THEME_COMPARISON_CACHE[key] = payload
        _THEME_COMPARISON_CACHE.move_to_end(key)
        while len(_THEME_COMPARISON_CACHE) > _THEME_COMPARISON_CACHE_MAX_ENTRIES:
            _THEME_COMPARISON_CACHE.popitem(last=False)
    return key


def _get_theme_comparison(
    key: str,
    *,
    bundle_fingerprint: str,
) -> dict[str, object] | None:
    """Return a cached comparison only for the same primary bundle."""

    if not key:
        return None
    with _THEME_COMPARISON_CACHE_LOCK:
        payload = _THEME_COMPARISON_CACHE.get(key)
        if payload is None:
            return None
        if payload.get("bundle_fingerprint") != str(bundle_fingerprint):
            return None
        _THEME_COMPARISON_CACHE.move_to_end(key)
        return payload


def _cache_theme_gui_results(**results: object) -> str:
    """Retain completed static/dynamic results across one theme reload."""

    key = secrets.token_urlsafe(24)
    with _THEME_GUI_RESULT_CACHE_LOCK:
        _THEME_GUI_RESULT_CACHE[key] = dict(results)
        _THEME_GUI_RESULT_CACHE.move_to_end(key)
        while len(_THEME_GUI_RESULT_CACHE) > _THEME_GUI_RESULT_CACHE_MAX_ENTRIES:
            _THEME_GUI_RESULT_CACHE.popitem(last=False)
    return key


def _get_theme_gui_results(key: str) -> dict[str, object] | None:
    """Return cached GUI solver results for an opaque browser-session key."""

    if not key:
        return None
    with _THEME_GUI_RESULT_CACHE_LOCK:
        payload = _THEME_GUI_RESULT_CACHE.get(key)
        if payload is None:
            return None
        _THEME_GUI_RESULT_CACHE.move_to_end(key)
        return payload


def _plot_template(session_args) -> str:
    """Resolve the Plotly theme used by Panel's query-string theme switch."""

    raw = session_args.get("theme") if session_args else None
    value = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return "plotly_white" if str(value).strip("'\"") == "default" else "plotly_dark"


def _plot_theme_tokens(plot_template: str) -> dict[str, str]:
    """Return high-contrast colors shared by every Plotly figure."""

    if plot_template == "plotly_dark":
        return {
            "background": "#101b2d",
            "foreground": "#e7eef8",
            "title": "#f8fbff",
            "accent": "#38bdf8",
            "grid": "rgba(148,163,184,0.18)",
            "axis": "rgba(148,163,184,0.42)",
            "blue": "#59c3ff",
            "orange": "#ffb454",
            "green": "#69db9c",
            "radar_green": "#39d98a",
            "purple": "#d980fa",
            "indigo": "#8c9eff",
            "total": "#f7f9fb",
        }
    return {
        "background": "#ffffff",
        "foreground": "#172033",
        "title": "#101827",
        "accent": "#0369a1",
        "grid": "#b6c2d0",
        "axis": "#64748b",
        "blue": "#0369a1",
        "orange": "#b45309",
        "green": "#047857",
        "radar_green": "#047857",
        "purple": "#7e22ce",
        "indigo": "#4338ca",
        "total": "#0f172a",
    }


def _comparison_json_stylesheet(plot_template: str) -> str:
    """Keep comparison metrics readable inside the JSON pane shadow root."""

    tokens = _plot_theme_tokens(plot_template)
    return f"""
:host, pre, code, span,
.json-formatter-row,
.json-formatter-string,
.json-formatter-number,
.json-formatter-boolean,
.json-formatter-null {{
  color: {tokens["foreground"]} !important;
}}
.json-formatter-key {{
  color: {tokens["accent"]} !important;
}}
"""


def _style_figure(figure, *, plot_template: str):
    """Apply the fixed HERMES visual language to every Plotly figure."""

    tokens = _plot_theme_tokens(plot_template)
    dark = plot_template == "plotly_dark"
    figure.update_layout(
        template=plot_template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=tokens["background"],
        font={
            "family": _PLOT_FONT_FAMILY,
            "size": 15,
            "color": tokens["foreground"],
        },
        title_font={
            "size": 20,
            "color": tokens["title"],
        },
        hoverlabel={"font": {"size": 15}},
        legend={"font": {"color": tokens["foreground"]}},
    )
    axis_style = {
        "title_font": {"size": 16, "color": tokens["foreground"]},
        "tickfont": {"size": 14, "color": tokens["foreground"]},
        "gridcolor": tokens["grid"],
    }
    if not dark:
        axis_style.update(
            linecolor=tokens["axis"],
            tickcolor=tokens["axis"],
            zerolinecolor=tokens["axis"],
            showline=True,
        )
    figure.update_xaxes(**axis_style)
    figure.update_yaxes(**axis_style)
    figure.update_annotations(
        font={
            "family": _PLOT_FONT_FAMILY,
            "size": 15,
            "color": tokens["foreground"],
        }
    )
    if any(
        trace.type in {"mesh3d", "scatter3d"} for trace in figure.data
    ):
        scene_axis_style = {
            "backgroundcolor": tokens["background"],
            "gridcolor": tokens["grid"],
            "linecolor": tokens["axis"],
            "zerolinecolor": tokens["axis"],
            "showbackground": True,
            "title_font": {"size": 13, "color": tokens["foreground"]},
            "tickfont": {"size": 11, "color": tokens["foreground"]},
            "nticks": 4,
            "tickformat": ".2~f",
        }
        figure.update_scenes(
            bgcolor=tokens["background"],
            xaxis=scene_axis_style,
            yaxis=scene_axis_style,
            zaxis=scene_axis_style,
            xaxis_title_text="x [m]",
            yaxis_title_text="y [m]",
            zaxis_title_text="z [m]",
        )
    return figure


def _empty_figure(go, plot_template):
    figure = go.Figure()
    figure.add_annotation(text="Load a bundle to view its products", showarrow=False)
    figure.update_layout(xaxis_visible=False, yaxis_visible=False,
                         margin={"l": 20, "r": 20, "t": 30, "b": 20})
    return _style_figure(figure, plot_template=plot_template)


def _relative_db(values: np.ndarray, *, floor_db: float = -80.0) -> np.ndarray:
    power = np.asarray(values, dtype=float)
    peak = max(float(np.max(power)), 1e-30)
    return np.maximum(10.0 * np.log10(np.maximum(power, 1e-30) / peak), floor_db)


def _absolute_power_db(values: np.ndarray) -> np.ndarray:
    """Convert native simulator power to dB without a reference normalization."""

    power = np.asarray(values, dtype=float)
    return 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))


def _absolute_db_limits(
    series: tuple[np.ndarray, ...],
    *,
    dynamic_range_db: float,
    headroom_db: float,
) -> tuple[float, float]:
    """Choose stable shared limits without letting a weak solver crush the plot."""

    peaks = [float(np.max(_absolute_power_db(values))) for values in series]
    strongest_peak = max(peaks)
    upper = 5.0 * np.ceil((strongest_peak + headroom_db) / 5.0)
    lower = upper - float(dynamic_range_db)
    return float(lower), float(upper)


def _face_colors(face_power: np.ndarray) -> list[str]:
    from matplotlib import colormaps, colors

    power = np.asarray(face_power, dtype=float)
    if not np.any(power > 0.0):
        return ["#000000"] * power.size
    db = _relative_db(power, floor_db=-60.0)
    normalized = np.clip((db + 60.0) / 60.0, 0.0, 1.0)
    cmap = colormaps.get_cmap("inferno")
    return [
        colors.to_hex(cmap(float(level)), keep_alpha=False)
        if value > 0.0 else "#000000"
        for value, level in zip(power, normalized)
    ]


def _target_face_colors(result) -> list[str]:
    """Return relative PO power colors for one static target."""

    return _face_colors(result.face_power)


def _scene_grid_step(span_m: float) -> float:
    """Return a stable, human-readable grid interval for a scene span."""

    raw_step = max(float(span_m) / 6.0, 1e-3)
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1.0:
        multiplier = 1.0
    elif normalized <= 2.0:
        multiplier = 2.0
    elif normalized <= 5.0:
        multiplier = 5.0
    else:
        multiplier = 10.0
    return float(multiplier * magnitude)


def _target_position(result) -> np.ndarray:
    """Return the target's rotation anchor in scene coordinates."""

    parameters = result.target_parameters
    position = parameters.get(
        "position_m",
        (
            parameters["range_m"],
            parameters.get("target_y_m", 0.0),
            parameters.get("target_z_m", 0.0),
        ),
    )
    vector = np.asarray(position, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("target position must contain three finite values")
    return vector


def _target_trace_update(
    result,
    *,
    sequence: int,
    include_geometry: bool = True,
) -> dict[str, object]:
    """Serialize a mesh-only result update that leaves the 3D frame untouched."""

    vertices = np.asarray(result.vertices)
    face_colors = _target_face_colors(result)
    pivot = _target_position(result)
    update = {
        "sequence": int(sequence),
        "facecolor": face_colors,
        "meta": {
            "orientation_pivot": [float(value) for value in pivot],
            "yaw_deg": float(result.target_parameters["yaw_deg"]),
            "pitch_deg": float(result.target_parameters["pitch_deg"]),
            "roll_deg": float(result.target_parameters["roll_deg"]),
        },
    }
    if include_geometry:
        update.update(
            {
                "x": vertices[:, 0].tolist(),
                "y": vertices[:, 1].tolist(),
                "z": vertices[:, 2].tolist(),
            }
        )
    return update


def _scene_interaction_component(pn, plot):
    """Wrap a Plotly scene with a live Shift-drag target-rotation gesture."""

    import param

    class SceneInteraction(pn.reactive.ReactiveHTML):
        children = param.List(default=[])
        target_gesture = param.Dict(default={})
        target_result = param.Dict(default={})

        _template = (
            '<div id="scene" style="height:100%; position:relative; width:100%;">'
            "${children}"
            '<div id="target_overlay" title="Shift-drag to rotate target" '
            'style="background:transparent; inset:0; pointer-events:none; '
            'position:absolute; touch-action:none; z-index:1000;"></div>'
            "</div>"
        )
        _scripts = {
            "render": """
state.findPlot = (root) => {
  if (root == null) return null
  const direct = root.querySelector?.(".js-plotly-plot")
  if (direct != null) return direct
  for (const element of root.querySelectorAll?.("*") ?? []) {
    if (element.shadowRoot != null) {
      const nested = state.findPlot(element.shadowRoot)
      if (nested != null) return nested
    }
  }
  return null
}
const multiply = (left, right) => left.map(
  row => right[0].map(
    (_, column) => row.reduce(
      (sum, value, index) => sum + value * right[index][column], 0
    )
  )
)
const matrixVector = (matrix, vector) => matrix.map(
  row => row.reduce(
    (sum, value, index) => sum + value * vector[index], 0
  )
)
const transpose = matrix => matrix[0].map(
  (_, column) => matrix.map(row => row[column])
)
const rotation = (yawDeg, pitchDeg, rollDeg) => {
  const yaw = yawDeg * Math.PI / 180
  const pitch = pitchDeg * Math.PI / 180
  const roll = rollDeg * Math.PI / 180
  const rz = [
    [Math.cos(yaw), -Math.sin(yaw), 0],
    [Math.sin(yaw), Math.cos(yaw), 0],
    [0, 0, 1],
  ]
  const ry = [
    [Math.cos(pitch), 0, Math.sin(pitch)],
    [0, 1, 0],
    [-Math.sin(pitch), 0, Math.cos(pitch)],
  ]
  const rx = [
    [1, 0, 0],
    [0, Math.cos(roll), -Math.sin(roll)],
    [0, Math.sin(roll), Math.cos(roll)],
  ]
  return multiply(multiply(rz, ry), rx)
}
const clamp = (value, lower, upper) => Math.min(
  upper, Math.max(lower, value)
)
const install = () => {
  if (state.removed) return
  const plot = state.findPlot(scene)
  if (plot == null || typeof plot.on !== "function") {
    state.installTimer = setTimeout(install, 50)
    return
  }
  state.plot = plot
  const emitGesture = () => {
    data.target_gesture = {
      sequence: (data.target_gesture.sequence ?? 0) + 1,
      yaw_deg: state.nextYaw,
      pitch_deg: state.nextPitch,
    }
  }
  const applyPreview = (yawDeg, pitchDeg) => {
    const current = rotation(
      state.startYaw, state.startPitch, state.startRoll
    )
    const next = rotation(yawDeg, pitchDeg, state.startRoll)
    const undoCurrent = transpose(current)
    const xs = []
    const ys = []
    const zs = []
    for (let index = 0; index < state.startX.length; index++) {
      const relative = [
        state.startX[index] - state.pivot[0],
        state.startY[index] - state.pivot[1],
        state.startZ[index] - state.pivot[2],
      ]
      const local = matrixVector(undoCurrent, relative)
      const rotated = matrixVector(next, local)
      xs.push(rotated[0] + state.pivot[0])
      ys.push(rotated[1] + state.pivot[1])
      zs.push(rotated[2] + state.pivot[2])
    }
    return window.Plotly.restyle(
      state.plot, {x: [xs], y: [ys], z: [zs]}, [0]
    )
  }
  const applySettledPreview = preview => {
    state.previewInFlight = true
    Promise.resolve(applyPreview(preview.yaw, preview.pitch))
      .catch(error => console.error("Target preview failed", error))
      .finally(() => {
        state.previewInFlight = false
        if (state.finalPreview != null) {
          const finalPreview = state.finalPreview
          state.finalPreview = null
          applySettledPreview(finalPreview)
          return
        }
        if (state.shiftDragging && state.previewPending) {
          requestPreview()
        }
      })
  }
  const requestPreview = () => {
    state.previewPending = true
    if (state.previewInFlight || state.previewFrame != null) return
    state.previewFrame = requestAnimationFrame(() => {
      state.previewFrame = null
      if (!state.shiftDragging) {
        state.previewPending = false
        return
      }
      const preview = {yaw: state.nextYaw, pitch: state.nextPitch}
      state.previewPending = false
      applySettledPreview(preview)
    })
  }
  const finish = event => {
    if (!state.shiftDragging) return
    event?.preventDefault?.()
    if (event?.type?.startsWith("pointer")) {
      event.stopImmediatePropagation()
    }
    state.shiftDragging = false
    window.removeEventListener("pointermove", state.pointerMove, true)
    window.removeEventListener("pointerup", state.pointerUp, true)
    window.removeEventListener("pointercancel", state.pointerCancel, true)
    if (
      state.pointerId != null
      && target_overlay.hasPointerCapture?.(state.pointerId)
    ) {
      target_overlay.releasePointerCapture(state.pointerId)
    }
    if (state.previewFrame != null) {
      cancelAnimationFrame(state.previewFrame)
      state.previewFrame = null
    }
    state.previewPending = false
    const finalPreview = {yaw: state.nextYaw, pitch: state.nextPitch}
    if (state.previewInFlight) {
      // Replace every intermediate request with exactly one final draw.
      state.finalPreview = finalPreview
    } else {
      applySettledPreview(finalPreview)
    }
    emitGesture()
    target_overlay.style.cursor = "grab"
    target_overlay.style.pointerEvents = state.shiftHeld ? "auto" : "none"
  }
  state.pointerMove = event => {
    if (!state.shiftDragging) return
    if ((event.buttons & 1) === 0) {
      finish(event)
      return
    }
    event.preventDefault()
    event.stopImmediatePropagation()
    const sensitivity = 0.35
    state.nextYaw = Math.round(10 * clamp(
      state.startYaw + sensitivity * (event.clientX - state.startClientX),
      -75, 75
    )) / 10
    state.nextPitch = Math.round(10 * clamp(
      state.startPitch - sensitivity * (event.clientY - state.startClientY),
      -75, 75
    )) / 10
    requestPreview()
  }
  state.pointerUp = event => finish(event)
  state.pointerCancel = event => {
    state.nextYaw = state.startYaw
    state.nextPitch = state.startPitch
    finish(event)
  }
  state.pointerDown = event => {
    if (!state.shiftHeld || event.button !== 0) return
    if (
      state.shiftDragging
      || state.previewInFlight
      || state.finalPreview != null
    ) {
      event.preventDefault()
      event.stopImmediatePropagation()
      return
    }
    const activePlot = state.findPlot(scene)
    if (activePlot == null) return
    state.plot = activePlot
    const trace = state.plot.data?.[0]
    if (trace == null) return
    event.preventDefault()
    event.stopImmediatePropagation()
    state.shiftDragging = true
    state.pointerId = event.pointerId
    target_overlay.setPointerCapture?.(event.pointerId)
    target_overlay.style.cursor = "grabbing"
    state.startClientX = event.clientX
    state.startClientY = event.clientY
    state.startX = Array.from(trace.x)
    state.startY = Array.from(trace.y)
    state.startZ = Array.from(trace.z)
    const meta = trace.meta ?? {}
    state.startYaw = Number(meta.yaw_deg ?? 0)
    state.startPitch = Number(meta.pitch_deg ?? 0)
    state.startRoll = Number(meta.roll_deg ?? 0)
    state.nextYaw = state.startYaw
    state.nextPitch = state.startPitch
    state.previewPending = false
    state.finalPreview = null
    state.pivot = Array.from(
      meta.orientation_pivot ?? [0, 0, 0]
    ).map(Number)
    window.addEventListener("pointermove", state.pointerMove, true)
    window.addEventListener("pointerup", state.pointerUp, true)
    window.addEventListener("pointercancel", state.pointerCancel, true)
  }
  state.keyDown = event => {
    if (event.key !== "Shift") return
    state.shiftHeld = true
    target_overlay.style.cursor = "grab"
    target_overlay.style.pointerEvents = "auto"
  }
  state.keyUp = event => {
    if (event.key !== "Shift") return
    state.shiftHeld = false
    if (!state.shiftDragging) {
      target_overlay.style.pointerEvents = "none"
    }
  }
  state.windowBlur = () => {
    state.shiftHeld = false
    if (state.shiftDragging) {
      finish()
    } else {
      target_overlay.style.pointerEvents = "none"
    }
  }
  state.lostPointerCapture = event => {
    if (state.shiftDragging) finish(event)
  }
  // The overlay, rather than Plotly's drag layer, owns every Shift gesture.
  target_overlay.addEventListener("pointerdown", state.pointerDown, true)
  target_overlay.addEventListener(
    "lostpointercapture", state.lostPointerCapture, true
  )
  window.addEventListener("keydown", state.keyDown, true)
  window.addEventListener("keyup", state.keyUp, true)
  window.addEventListener("blur", state.windowBlur, true)
}
state.removed = false
state.shiftHeld = false
state.previewInFlight = false
state.previewPending = false
state.finalPreview = null
state.installTimer = setTimeout(install, 0)
""",
            "target_result": """
const update = data.target_result
const activePlot = state.findPlot?.(scene)
if (activePlot != null) state.plot = activePlot
if (
  state.plot != null
  && update != null
  && update.sequence !== state.resultSequence
) {
  state.resultSequence = update.sequence
  const traceUpdate = {
    facecolor: [update.facecolor],
    meta: [update.meta],
  }
  if (update.x != null && update.y != null && update.z != null) {
    traceUpdate.x = [update.x]
    traceUpdate.y = [update.y]
    traceUpdate.z = [update.z]
  }
  window.Plotly.restyle(
    state.plot,
    traceUpdate,
    [0],
  )
}
""",
            "remove": """
state.removed = true
clearTimeout(state.installTimer)
if (state.previewFrame != null) cancelAnimationFrame(state.previewFrame)
if (state.pointerDown != null) {
  target_overlay.removeEventListener("pointerdown", state.pointerDown, true)
  target_overlay.removeEventListener(
    "lostpointercapture", state.lostPointerCapture, true
  )
}
if (state.pointerMove != null) {
  window.removeEventListener("pointermove", state.pointerMove, true)
  window.removeEventListener("pointerup", state.pointerUp, true)
  window.removeEventListener("pointercancel", state.pointerCancel, true)
}
if (state.keyDown != null) {
  window.removeEventListener("keydown", state.keyDown, true)
  window.removeEventListener("keyup", state.keyUp, true)
  window.removeEventListener("blur", state.windowBlur, true)
}
""",
        }

    return SceneInteraction(
        children=[plot],
        height=500,
        min_height=500,
        sizing_mode="stretch_width",
        name="3D target interaction",
    )


def _dynamic_scene_interaction_component(pn, plot):
    """Keep dynamic mesh redraws from interrupting Plotly camera gestures."""

    import param

    class DynamicSceneInteraction(pn.reactive.ReactiveHTML):
        children = param.List(default=[])

        _template = (
            '<div id="scene" style="height:100%; position:relative; '
            'width:100%;">${children}</div>'
        )
        _scripts = {
            "render": """
state.findPlot = root => {
  if (root == null) return null
  const direct = root.querySelector?.(".js-plotly-plot")
  if (direct != null) return direct
  for (const element of root.querySelectorAll?.("*") ?? []) {
    if (element.shadowRoot != null) {
      const nested = state.findPlot(element.shadowRoot)
      if (nested != null) return nested
    }
  }
  return null
}
const install = () => {
  if (state.removed) return
  const plot = state.findPlot(scene)
  if (plot == null || window.Plotly == null) {
    state.installTimer = setTimeout(install, 50)
    return
  }

  // Panel applies server-side Figure.plotly_restyle calls through the global
  // Plotly.restyle entry point. Guard only this plot and retain only its most
  // recent mesh update while Plotly owns a camera gesture.
  const registryKey = "__hermesDynamicSceneRestyleGuard"
  let guard = window[registryKey]
  if (guard == null) {
    const originalRestyle = window.Plotly.restyle.bind(window.Plotly)
    const originalUpdate = window.Plotly.update.bind(window.Plotly)
    const records = new WeakMap()
    const cloneCamera = camera => {
      if (camera == null || typeof camera !== "object") return null
      return JSON.parse(JSON.stringify(camera))
    }
    const readCamera = target => {
      const liveCamera = target?._fullLayout?.scene?._scene?.getCamera?.()
      return cloneCamera(
        liveCamera
        ?? target?.layout?.scene?.camera
        ?? target?._fullLayout?.scene?.camera
      )
    }
    const runLatest = (target, record) => {
      if (
        record.interacting
        || record.inFlight
        || record.pending == null
      ) return Promise.resolve(target)
      const args = record.pending
      record.pending = null
      record.inFlight = true
      // Plotly can rebuild a WebGL scene when mesh coordinates change. Send
      // the most recently observed user camera in the same update so no
      // playback frame can restore the figure's initial perspective or zoom.
      const camera = readCamera(target) ?? record.camera
      if (camera != null) record.camera = cloneCamera(camera)
      const update = (
        camera == null
          ? originalRestyle(target, ...args)
          : originalUpdate(
              target,
              args[0],
              {"scene.camera": camera},
              args[1],
            )
      )
      return Promise.resolve(update)
        .catch(error => console.error(
          "Dynamic-scene frame refresh failed", error
        ))
        .finally(() => {
          record.inFlight = false
          if (!record.interacting && record.pending != null) {
            requestAnimationFrame(() => runLatest(target, record))
          }
        })
    }
    const guardedRestyle = (target, ...args) => {
      const record = records.get(target)
      if (record == null) return originalRestyle(target, ...args)
      record.pending = args
      return runLatest(target, record)
    }
    guard = {
      cloneCamera,
      originalRestyle,
      originalUpdate,
      readCamera,
      records,
      guardedRestyle,
      runLatest,
    }
    window[registryKey] = guard
    window.Plotly.restyle = guardedRestyle
  }

  state.plot = plot
  state.guard = guard
  state.record = {
    camera: guard.readCamera(plot),
    interacting: false,
    inFlight: false,
    pending: null,
  }
  guard.records.set(plot, state.record)

  state.captureCamera = eventData => {
    if (state.record == null) return
    const eventCamera = eventData?.["scene.camera"]
    const camera = (
      eventCamera != null
        ? state.guard.cloneCamera(eventCamera)
        : state.guard.readCamera(state.plot)
    )
    if (camera != null) state.record.camera = camera
  }

  state.flush = () => {
    if (state.record == null) return
    state.record.interacting = false
    state.guard.runLatest(state.plot, state.record)
  }
  state.beginInteraction = () => {
    if (state.record != null) state.record.interacting = true
  }
  state.pointerDown = event => {
    if (event.button == null || event.button === 0) {
      clearTimeout(state.wheelTimer)
      state.beginInteraction()
    }
  }
  state.pointerEnd = () => {
    // Let Plotly finish recording the new camera before drawing the newest
    // queued human frame.
    requestAnimationFrame(() => requestAnimationFrame(state.flush))
  }
  state.wheel = () => {
    state.beginInteraction()
    clearTimeout(state.wheelTimer)
    state.wheelTimer = setTimeout(state.flush, 180)
  }
  state.windowBlur = () => state.flush()

  plot.addEventListener("pointerdown", state.pointerDown, true)
  plot.addEventListener("wheel", state.wheel, true)
  plot.on("plotly_relayout", state.captureCamera)
  plot.on("plotly_relayouting", state.captureCamera)
  window.addEventListener("pointerup", state.pointerEnd, true)
  window.addEventListener("pointercancel", state.pointerEnd, true)
  window.addEventListener("blur", state.windowBlur, true)
}
state.removed = false
state.installTimer = setTimeout(install, 0)
""",
            "remove": """
state.removed = true
clearTimeout(state.installTimer)
clearTimeout(state.wheelTimer)
if (state.record != null) {
  state.record.interacting = false
  state.record.pending = null
}
if (state.plot != null) {
  state.guard?.records?.delete(state.plot)
  state.plot.removeEventListener("pointerdown", state.pointerDown, true)
  state.plot.removeEventListener("wheel", state.wheel, true)
  state.plot.removeListener?.("plotly_relayout", state.captureCamera)
  state.plot.removeListener?.("plotly_relayouting", state.captureCamera)
}
if (state.pointerEnd != null) {
  window.removeEventListener("pointerup", state.pointerEnd, true)
  window.removeEventListener("pointercancel", state.pointerEnd, true)
  window.removeEventListener("blur", state.windowBlur, true)
}
""",
        }

    return DynamicSceneInteraction(
        children=[plot],
        height=570,
        min_height=570,
        sizing_mode="stretch_width",
        name="Dynamic scene camera interaction",
    )


def _confirmation_dialog_component(pn):
    """Return a browser modal that emits a sequence when stop is confirmed."""

    import param

    class ConfirmationDialog(pn.reactive.ReactiveHTML):
        confirmation_sequence = param.Integer(default=0)
        open = param.Boolean(default=False)

        _dom_events = {
            "cancel": ["click"],
            "confirm": ["click"],
        }

        _template = """
<div id="backdrop" style="align-items:center; background:rgba(2,6,23,0.72);
  display:none; inset:0; justify-content:center; padding:24px;
  position:fixed; z-index:10000;">
  <div style="background:var(--hermes-surface); border:1px solid
    var(--hermes-border); border-radius:12px; box-shadow:0 24px 70px
    rgba(0,0,0,0.45); color:var(--hermes-text); max-width:480px;
    padding:24px; width:100%;">
    <h3 style="margin:0 0 10px;">Stop the running simulation?</h3>
    <p style="margin:0 0 20px;">The current solver kernel will finish, then
      HERMES will stop at the next safe boundary. No partial ADC cube will be
      exported.</p>
    <div style="display:flex; gap:12px; justify-content:flex-end;">
      <button id="cancel" type="button" class="bk-btn bk-btn-default">
        Keep running
      </button>
      <button id="confirm" type="button" class="bk-btn bk-btn-danger">
        Stop simulation
      </button>
    </div>
  </div>
</div>
"""
        _scripts = {
            "render": "backdrop.style.display = data.open ? 'flex' : 'none'",
            "open": "backdrop.style.display = data.open ? 'flex' : 'none'",
        }

        def _cancel_click(self, _event):
            self.open = False

        def _confirm_click(self, _event):
            self.confirmation_sequence += 1
            self.open = False

    return ConfirmationDialog(
        height=0,
        width=0,
        sizing_mode="fixed",
        name="Stop simulation confirmation",
    )


def _physics_scene_figure(result, go, *, plot_template: str):
    palette = _plot_theme_tokens(plot_template)
    vertices = np.asarray(result.vertices)
    faces = np.asarray(result.faces, dtype=int)
    pivot = _target_position(result)
    sensor_parameters = getattr(result, "sensor_parameters", {})
    orientation = np.asarray(
        sensor_parameters.get("orientation", (0.0, 0.0, 0.0)),
        dtype=float,
    )
    if orientation.shape != (3,) or not np.all(np.isfinite(orientation)):
        orientation = np.zeros(3, dtype=float)
    radar_yaw, radar_pitch, radar_roll = orientation
    boresight_direction = np.asarray(
        (
            np.cos(radar_yaw) * np.cos(radar_pitch),
            np.sin(radar_yaw) * np.cos(radar_pitch),
            -np.sin(radar_pitch),
        ),
        dtype=float,
    )
    boresight_length = float(
        np.clip(0.22 * float(np.linalg.norm(pivot)), 0.18, 0.55)
    )
    boresight_endpoint = boresight_length * boresight_direction
    local_up_direction = np.asarray(
        (
            np.cos(radar_yaw) * np.sin(radar_pitch) * np.cos(radar_roll)
            + np.sin(radar_yaw) * np.sin(radar_roll),
            np.sin(radar_yaw) * np.sin(radar_pitch) * np.cos(radar_roll)
            - np.cos(radar_yaw) * np.sin(radar_roll),
            np.cos(radar_pitch) * np.cos(radar_roll),
        ),
        dtype=float,
    )
    local_up_endpoint = 0.55 * boresight_length * local_up_direction
    # A pivot-centered bounding sphere is invariant under yaw/pitch/roll.
    # Quantizing upward to 0.1 mm prevents float32 rotation noise from changing
    # the scene ranges and visually moving the coordinate frame.
    radius = float(np.max(np.linalg.norm(vertices - pivot, axis=1)))
    radius = max(float(np.ceil(radius * 10_000.0) / 10_000.0), 0.05)
    lower = np.minimum(
        np.minimum(np.minimum(pivot - radius, 0.0), boresight_endpoint),
        local_up_endpoint,
    )
    upper = np.maximum(
        np.maximum(np.maximum(pivot + radius, 0.0), boresight_endpoint),
        local_up_endpoint,
    )
    span = max(float(np.max(upper - lower)), 0.25)
    padding = 0.08 * span
    ranges = {
        "x": [float(lower[0] - padding), float(upper[0] + padding)],
        "y": [float(lower[1] - padding), float(upper[1] + padding)],
        "z": [float(lower[2] - padding), float(upper[2] + padding)],
    }
    range_spans = {
        axis: bounds[1] - bounds[0] for axis, bounds in ranges.items()
    }
    largest_range_span = max(range_spans.values())
    # Plotly's data-aspect normalization preserves scene-box volume rather than
    # forcing its longest side to one. Reproduce that normalization explicitly
    # so the frame fills the viewport while remaining invariant under rotation.
    aspect_normalizer = float(
        np.cbrt(np.prod(np.asarray(list(range_spans.values()), dtype=float)))
    )
    aspect_ratio = {
        axis: axis_span / aspect_normalizer
        for axis, axis_span in range_spans.items()
    }
    grid_step = _scene_grid_step(largest_range_span)
    human_faces = result.target_type == "human_mesh"
    face_colors = _target_face_colors(result)
    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            facecolor=face_colors,
            opacity=1.0,
            flatshading=True,
            name=result.target_type.replace("_", " ").title(),
            showlegend=False,
            meta={
                "orientation_pivot": [float(value) for value in pivot],
                "yaw_deg": float(result.target_parameters["yaw_deg"]),
                "pitch_deg": float(result.target_parameters["pitch_deg"]),
                "roll_deg": float(result.target_parameters["roll_deg"]),
            },
            hovertemplate="Facet %{pointNumber}<extra></extra>",
            lighting=_PO_POWER_LIGHTING,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[None, None],
            y=[None, None],
            z=[None, None],
            mode="markers",
            marker={
                "size": 0.1,
                "color": [-60.0, 0.0],
                "colorscale": "Inferno",
                "cmin": -60.0,
                "cmax": 0.0,
                "showscale": True,
                "colorbar": {
                    "title": {"text": "PO relative<br>power [dB]"},
                    "tickvals": [-60.0, -40.0, -20.0, 0.0],
                    "ticksuffix": " dB",
                    "thickness": 16,
                },
            },
            name="PO relative facet power",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker={
                "size": 7,
                "color": palette["radar_green"],
                "symbol": "diamond",
            },
            text=["Radar (0, 0, 0)"],
            textfont={"color": palette["foreground"]},
            textposition="top center",
            name="Radar",
            showlegend=False,
            hovertemplate="Radar<br>(0, 0, 0) m<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[0.0, float(local_up_endpoint[0])],
            y=[0.0, float(local_up_endpoint[1])],
            z=[0.0, float(local_up_endpoint[2])],
            mode="lines",
            line={"width": 4, "color": palette["blue"]},
            name="Radar local up",
            showlegend=False,
            hovertemplate=(
                "Radar local +z"
                f"<br>roll {np.rad2deg(radar_roll):.1f}°"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[0.0, float(boresight_endpoint[0])],
            y=[0.0, float(boresight_endpoint[1])],
            z=[0.0, float(boresight_endpoint[2])],
            mode="lines",
            line={"width": 6, "color": palette["radar_green"]},
            name="Radar boresight",
            showlegend=False,
            hovertemplate=(
                "Radar boresight"
                f"<br>yaw {np.rad2deg(radar_yaw):.1f}°"
                f"<br>pitch {np.rad2deg(radar_pitch):.1f}°"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title=(
            "PO facet contribution · "
            f"{result.target_type.replace('_', ' ').title()}"
        ),
        scene={
            "xaxis": {
                "title": "x · range forward [m]",
                "range": ranges["x"],
                "tick0": 0.0,
                "dtick": grid_step,
            },
            "yaxis": {
                "title": "y · lateral, radar-left [m]",
                "range": ranges["y"],
                "tick0": 0.0,
                "dtick": grid_step,
            },
            "zaxis": {
                "title": "z · vertical, up [m]",
                "range": ranges["z"],
                "tick0": 0.0,
                "dtick": grid_step,
            },
            "aspectmode": "manual",
            "aspectratio": aspect_ratio,
            "dragmode": "orbit",
            "camera": {
                "eye": dict(_SCENE_CAMERA_EYE),
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
        },
        uirevision="hermes-scene-camera",
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        height=520,
        showlegend=human_faces,
        legend=(
            {"orientation": "h", "y": 0.99, "x": 0.01}
            if human_faces
            else None
        ),
    )
    return _style_figure(figure, plot_template=plot_template)


def _box_mesh(center, half_extent):
    """Return vertices and triangular faces for one axis-aligned room box."""

    center = np.asarray(center, dtype=float)
    half_extent = np.asarray(half_extent, dtype=float)
    signs = np.asarray(
        [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ],
        dtype=float,
    )
    vertices = center[None, :] + signs * half_extent[None, :]
    faces = np.asarray(
        [
            (0, 1, 2),
            (0, 2, 3),
            (4, 6, 5),
            (4, 7, 6),
            (0, 4, 5),
            (0, 5, 1),
            (1, 5, 6),
            (1, 6, 2),
            (2, 6, 7),
            (2, 7, 3),
            (3, 7, 4),
            (3, 4, 0),
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _motion_source_interval_s(times_s: np.ndarray) -> float | None:
    """Return the robust positive interval of a motion timeline."""

    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size < 2:
        return None
    intervals = np.diff(times)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0.0)]
    if intervals.size == 0:
        return None
    return float(np.median(intervals))


def _motion_playback_step(times_s: np.ndarray) -> int:
    """Choose a source-frame stride for an interactive 20--25 Hz preview."""

    times = np.asarray(times_s, dtype=float).reshape(-1)
    source_interval_s = _motion_source_interval_s(times)
    if source_interval_s is None:
        return 1
    source_rate_hz = 1.0 / source_interval_s
    if source_rate_hz <= 25.0 * (1.0 + 1.0e-6):
        return 1

    # Prefer 25 Hz when it divides the source rate, otherwise 20 Hz. This
    # yields 25 Hz for 50/100 Hz motion and 20 Hz for 60/120 Hz motion while
    # retaining the original source-frame indices for display and simulation.
    for target_rate_hz in (25.0, 20.0):
        stride = max(int(np.rint(source_rate_hz / target_rate_hz)), 1)
        display_rate_hz = source_rate_hz / stride
        if np.isclose(
            display_rate_hz,
            target_rate_hz,
            rtol=0.01,
            atol=0.05,
        ):
            return min(stride, max(int(times.size - 1), 1))

    stride = max(int(np.rint(source_rate_hz / 22.5)), 1)
    return min(stride, max(int(times.size - 1), 1))


def _motion_playback_interval_ms(times_s: np.ndarray) -> int:
    """Choose a responsive interval for the decimated motion preview."""

    source_interval_s = _motion_source_interval_s(times_s)
    if source_interval_s is None:
        return 100
    display_interval_s = source_interval_s * _motion_playback_step(times_s)
    return int(np.clip(np.rint(1000.0 * display_interval_s), 40, 1000))


def _motion_fast_playback_step(times_s: np.ndarray) -> int:
    """Choose a frame stride for approximately four-times-speed playback."""

    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size < 2 or _motion_source_interval_s(times) is None:
        return 1
    normal_stride = _motion_playback_step(times)
    stride = max(4 * normal_stride, normal_stride + 1)
    return min(stride, int(times.size - 1))


def _human_room_scene_title(preview, frame_index: int) -> str:
    """Return the dynamic-scene title for a source motion frame."""

    if preview is None:
        return "Dynamic scene · upload human motion to place the subject"
    selected_frame = int(
        np.clip(frame_index, 0, len(preview.mesh_sequence.times) - 1)
    )
    return (
        f"{preview.scene_source_name} · {preview.source_name} · "
        f"frame {selected_frame}"
    )


def _human_floor_overlay_data(vertices, room_boxes) -> dict:
    """Mark human vertices below a floor's top surface within its footprint."""

    vertices = np.asarray(vertices, dtype=float)
    clearance = np.full(len(vertices), np.nan, dtype=float)
    floor_boxes = [
        box for box in room_boxes if "floor" in str(box["id"]).lower()
    ]
    for box in floor_boxes:
        center = np.asarray(box["translate"], dtype=float)
        half_size = np.abs(np.asarray(box["scale"], dtype=float))
        over_floor = np.all(
            np.abs(vertices[:, :2] - center[:2]) <= half_size[:2] + 1e-6,
            axis=1,
        )
        floor_top = center[2] + half_size[2]
        clearance[over_floor] = np.fmin(
            clearance[over_floor], vertices[over_floor, 2] - floor_top
        )

    # A 0.1 mm tolerance avoids flagging float32 roundoff at floor contact.
    penetrating = clearance < -1e-4
    checked = np.isfinite(clearance)
    if not floor_boxes:
        status = "**Floor check unavailable:** no floor geometry in this preview."
    elif not np.any(checked):
        status = "**Outside floor footprint:** human vertices are not over the floor."
    elif np.any(penetrating):
        depth_cm = -100.0 * float(np.min(clearance[penetrating]))
        status = (
            f"**Floor penetration: {depth_cm:.2f} cm** · "
            f"{np.count_nonzero(penetrating):,} below-floor vertices shown in red."
        )
    else:
        gap_cm = max(100.0 * float(np.min(clearance[checked])), 0.0)
        status = f"**No floor penetration** · floor clearance: {gap_cm:.2f} cm."
    return {
        "x": vertices[penetrating, 0],
        "y": vertices[penetrating, 1],
        "z": vertices[penetrating, 2],
        "customdata": -100.0 * clearance[penetrating],
        "meta": {"floor_status": status, "floor_boxes": floor_boxes},
    }


def _update_human_room_scene_frame(
    figure,
    preview,
    *,
    frame_index: int,
    update_title: bool = True,
) -> bool:
    """Restyle the human mesh and floor markers during motion playback.

    Rebuilding the complete Plotly room for every player tick can make browser
    input outrun server rendering.  A Plotly restyle sends only the moving
    vertices and floor markers, preserving the room, radar, and current camera.
    """

    if figure is None or preview is None:
        return False
    human_trace_indices = [
        index
        for index, trace in enumerate(figure.data)
        if trace.name == "Human mesh"
    ]
    if len(human_trace_indices) != 1 or any(
        trace.name == "Top target-touching RT rays" for trace in figure.data
    ):
        return False
    sequence = preview.mesh_sequence
    selected_frame = int(np.clip(frame_index, 0, len(sequence.times) - 1))
    vertices = np.asarray(
        sequence.vertices_at(float(sequence.times[selected_frame])),
        dtype=float,
    )
    trace_index = human_trace_indices[0]
    trace = figure.data[trace_index]
    if len(trace.x) != vertices.shape[0]:
        return False
    # A data-only restyle leaves Plotly's active 3-D camera gesture alone.
    # Sending a layout update on every player tick cancels orbit/zoom drags in
    # the browser, so the title is refreshed only for non-playing changes.
    updates = {
        "x": [vertices[:, 0]],
        "y": [vertices[:, 1]],
        "z": [vertices[:, 2]],
    }
    trace_indices = [trace_index]
    for index, overlay in enumerate(figure.data):
        if overlay.name == "Below-floor vertices":
            overlay_data = _human_floor_overlay_data(
                vertices, overlay.meta["floor_boxes"]
            )
            for key, value in overlay_data.items():
                updates.setdefault(key, [getattr(trace, key)]).append(value)
            trace_indices.append(index)
            break
    # The camera-interaction guard retains only the latest restyle, so the
    # mesh and floor markers must travel together in a single frame update.
    figure.plotly_restyle(updates, trace_indices)
    if update_title:
        figure.plotly_relayout(
            {"title.text": _human_room_scene_title(preview, selected_frame)}
        )
    return True


def _human_room_window_values(
    *,
    last_frame: int,
    previous_begin: int,
    previous_end: int,
    reset_window: bool,
) -> tuple[int, int]:
    """Return a safe inclusive source-frame window for the GUI controls."""

    last_frame = max(int(last_frame), 0)
    if reset_window:
        return 0, 0
    begin_frame = min(max(int(previous_begin), 0), last_frame)
    end_frame = min(max(int(previous_end), begin_frame), last_frame)
    return begin_frame, end_frame


def _lock_widget_disabled_states(widgets) -> tuple[tuple[object, bool], ...]:
    """Disable unique widgets and return the states needed to restore them."""

    states = []
    seen = set()
    for widget in widgets:
        identity = id(widget)
        if identity in seen:
            continue
        seen.add(identity)
        states.append((widget, bool(widget.disabled)))
        widget.disabled = True
    return tuple(states)


def _restore_widget_disabled_states(states) -> None:
    """Restore widget disabled states captured before a long-running action."""

    for widget, disabled in states:
        widget.disabled = bool(disabled)


def _human_room_scene_figure(
    preview,
    go,
    *,
    frame_index: int,
    radar_orientation_deg=(0.0, 0.0, 0.0),
    room_boxes_override=None,
    diagnostics=None,
    title_override: str | None = None,
    plot_template: str,
):
    """Render the prepared room, selected motion frame, and fixed radar."""

    palette = _plot_theme_tokens(plot_template)
    figure = go.Figure()
    if preview is not None:
        room_boxes = preview.room_boxes
    elif room_boxes_override is not None:
        room_boxes = room_boxes_override
    else:
        room_boxes = prepared_room_boxes()
    if preview is not None and getattr(preview, "scene_xml", None) is not None:
        room_boxes = _scene_xml_boxes(preview.scene_xml)
    room_palette = {
        "floor": "#5d7083",
        "back_wall": "#74879a",
        "left_wall": "#74879a",
        "right_wall": "#8699ac",
        "bed": "#496d8c",
        "nightstand": "#8b6b4e",
        "wardrobe": "#795c44",
    }
    for box in room_boxes:
        vertices, faces = _box_mesh(box["translate"], box["scale"])
        object_name = str(box["id"]).replace("_", " ").title()
        material_name = str(
            box.get("material_display", box.get("material", "Unspecified"))
        )
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=room_palette.get(str(box["id"]), "#74879a"),
                opacity=(
                    0.18
                    if "wall" in str(box["id"])
                    or "floor" in str(box["id"]).lower()
                    else 0.45
                ),
                flatshading=True,
                name=object_name,
                showlegend=False,
                hovertemplate=(
                    f"Object: {object_name}<br>Material: {material_name}"
                    "<extra></extra>"
                ),
            )
        )
    human_vertices = None
    if preview is not None:
        sequence = preview.mesh_sequence
        selected_frame = int(np.clip(frame_index, 0, len(sequence.times) - 1))
        vertices = np.asarray(
            sequence.vertices_at(float(sequence.times[selected_frame])),
            dtype=float,
        )
        faces = np.asarray(sequence.faces, dtype=np.int64)
        human_vertices = vertices
        mesh_options = {"color": palette["orange"]}
        has_po_power = False
        if diagnostics is not None and diagnostics.po is not None:
            po_power = np.asarray(diagnostics.po_face_power, dtype=float)
            if po_power.shape == (faces.shape[0],):
                mesh_options = {"facecolor": _face_colors(po_power)}
                has_po_power = True
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                **mesh_options,
                opacity=1.0 if has_po_power else 0.92,
                flatshading=True,
                name="PO face power" if has_po_power else "Human mesh",
                showlegend=has_po_power,
                hovertemplate=(
                    "Object: Human facet %{pointNumber}<br>"
                    "Material: Human skin<extra></extra>"
                ),
                lighting=_PO_POWER_LIGHTING if has_po_power else {
                    "ambient": 0.68,
                    "diffuse": 0.82,
                    "specular": 0.08,
                    "roughness": 0.86,
                },
            )
        )
        figure.add_trace(
            go.Scatter3d(
                **_human_floor_overlay_data(vertices, room_boxes),
                mode="markers",
                marker={"size": 5, "color": "#ef4444"},
                name="Below-floor vertices",
                showlegend=False,
                hovertemplate=(
                    "Floor penetration: %{customdata:.2f} cm<br>"
                    "z: %{z:.3f} m<extra></extra>"
                ),
            )
        )
    if diagnostics is not None and diagnostics.rt is not None:
        starts = np.asarray(diagnostics.rt.segment_starts_m, dtype=float)
        ends = np.asarray(diagnostics.rt.segment_ends_m, dtype=float)
        if starts.size:
            x_values: list[float | None] = []
            y_values: list[float | None] = []
            z_values: list[float | None] = []
            for start, end in zip(starts, ends):
                x_values.extend((float(start[0]), float(end[0]), None))
                y_values.extend((float(start[1]), float(end[1]), None))
                z_values.extend((float(start[2]), float(end[2]), None))
            figure.add_trace(
                go.Scatter3d(
                    x=x_values,
                    y=y_values,
                    z=z_values,
                    mode="lines",
                    line={"width": 5, "color": palette["orange"]},
                    name="Top target-touching RT rays",
                    hovertemplate="RT segment<extra></extra>",
                )
            )
    orientation = np.deg2rad(
        np.asarray(radar_orientation_deg, dtype=float)
    )
    yaw, pitch, _roll = orientation
    direction = np.asarray(
        (
            np.cos(yaw) * np.cos(pitch),
            np.sin(yaw) * np.cos(pitch),
            -np.sin(pitch),
        )
    )
    endpoint = 0.55 * direction
    figure.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker={
                "size": 7,
                "color": palette["radar_green"],
                "symbol": "diamond",
            },
            text=["Radar (0, 0, 0)"],
            textfont={"color": palette["foreground"]},
            textposition="top center",
            name="Radar",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[0.0, endpoint[0]],
            y=[0.0, endpoint[1]],
            z=[0.0, endpoint[2]],
            mode="lines",
            line={"width": 6, "color": palette["radar_green"]},
            name="Radar boresight",
            showlegend=False,
        )
    )
    title = (
        str(title_override)
        if title_override is not None
        else _human_room_scene_title(preview, frame_index)
    )
    bounds = []
    for box in room_boxes:
        translate = np.asarray(box["translate"], dtype=float)
        scale = np.asarray(box["scale"], dtype=float)
        bounds.extend((translate - scale, translate + scale))
    bounds.extend((np.zeros(3, dtype=float), endpoint))
    if human_vertices is not None and human_vertices.size:
        bounds.extend(
            (
                np.min(human_vertices, axis=0),
                np.max(human_vertices, axis=0),
            )
        )
    bounds_array = np.asarray(bounds, dtype=float)
    lower = np.min(bounds_array, axis=0)
    upper = np.max(bounds_array, axis=0)
    span = np.maximum(upper - lower, np.asarray((2.0, 2.0, 2.0)))
    padding = np.maximum(0.04 * span, np.asarray((0.35, 0.35, 0.25)))
    lower -= padding
    upper += padding
    figure.update_layout(
        template="plotly_dark",
        title=title,
        scene={
            "xaxis": {
                "title": "x · range forward [m]",
                "range": [float(lower[0]), float(upper[0])],
            },
            "yaxis": {
                "title": "y · radar-left [m]",
                "range": [float(lower[1]), float(upper[1])],
            },
            "zaxis": {
                "title": "z · relative height [m]",
                "range": [float(lower[2]), float(upper[2])],
            },
            "aspectmode": "data",
            "dragmode": "orbit",
            "camera": {
                "eye": {"x": -1.35, "y": -1.55, "z": 0.9},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
        },
        uirevision="hermes-human-room-camera",
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        height=570,
        showlegend=diagnostics is not None,
    )
    return _style_figure(figure, plot_template=plot_template)


def _solver_diagnostics_scene_figure(
    result,
    diagnostics,
    go,
    *,
    plot_template: str,
):
    """Overlay selected target-touching RT paths on the shared scene."""

    palette = _plot_theme_tokens(plot_template)
    figure = _physics_scene_figure(
        result,
        go,
        plot_template=plot_template,
    )
    starts = np.asarray(diagnostics.rt.segment_starts_m, dtype=float)
    ends = np.asarray(diagnostics.rt.segment_ends_m, dtype=float)
    if starts.size:
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        z_values: list[float | None] = []
        for start, end in zip(starts, ends):
            x_values.extend((float(start[0]), float(end[0]), None))
            y_values.extend((float(start[1]), float(end[1]), None))
            z_values.extend((float(start[2]), float(end[2]), None))
        figure.add_trace(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="lines",
                line={"width": 5, "color": palette["orange"]},
                name="Top target-touching RT paths",
                hovertemplate="RT segment<extra></extra>",
            )
        )
    figure.update_layout(
        title="Selected target-touching RT paths",
        showlegend=bool(starts.size),
        legend={"orientation": "h", "y": 0.99, "x": 0.01},
    )
    return figure


def _solver_diagnostics_summary(diagnostics) -> str:
    """Format compact solver-diagnostic counts without duplicate tables."""

    return (
        f"**RT valid link-paths:** "
        f"`{diagnostics.rt.valid_link_path_count:,}` · "
        f"**PO visible facets:** `{diagnostics.po.visible_face_count:,}`"
    )


def _path_depth_figure(diagnostics, go, *, plot_template: str):
    palette = _plot_theme_tokens(plot_template)
    histogram = np.asarray(diagnostics.rt.depth_histogram, dtype=np.int64)
    figure = go.Figure(
        go.Bar(
            x=np.arange(histogram.size),
            y=histogram,
            marker_color=palette["orange"],
            name="RT link-path count",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title="RT interaction-depth histogram",
        xaxis_title="Number of interactions",
        yaxis_title="Valid link-path count",
        height=300,
        margin={"l": 65, "r": 20, "t": 50, "b": 55},
    )
    return _style_figure(figure, plot_template=plot_template)


def _human_room_profile_figure(result, go, *, plot_template: str):
    """Plot one room-simulation mode on a shared absolute power scale."""

    palette = _plot_theme_tokens(plot_template)
    simulation_mode = getattr(result, "simulation_mode", "hybrid_po")
    mode_label = _HUMAN_ROOM_MODE_LABELS.get(
        simulation_mode,
        simulation_mode.replace("_", " ").title(),
    )
    labels = {
        "total": mode_label,
        "human_po": "Human PO",
        "static_environment_blocked": "Static environment RT",
        "static_environment_unblocked": "Static environment RT (unblocked)",
        "human_env": "Human → environment",
        "env_human": "Environment → human",
    }
    colors = (
        palette["total"],
        palette["orange"],
        palette["blue"],
        palette["indigo"],
        palette["green"],
        palette["purple"],
    )
    series = list(result.range_profiles.items())
    lower_db, upper_db = _absolute_db_limits(
        [np.asarray(values) for _, values in series],
        dynamic_range_db=100.0,
        headroom_db=3.0,
    )
    figure = go.Figure()
    for color, (name, values) in zip(colors, series):
        figure.add_trace(
            go.Scatter(
                x=result.ranges_m,
                y=_absolute_power_db(values),
                mode="lines",
                line={
                    "color": color,
                    "width": 3.0 if name == "total" else 2.0,
                    "dash": "solid",
                },
                name=labels.get(name, name.replace("_", " ").title()),
            )
        )
    figure.update_layout(
        template="plotly_dark",
        title=f"{mode_label} range profile",
        xaxis_title="Range [m]",
        yaxis_title="Power [dB, simulator units]",
        yaxis_range=[lower_db, upper_db],
        height=360,
        margin={"l": 60, "r": 20, "t": 52, "b": 55},
    )
    return _style_figure(figure, plot_template=plot_template)


def _hybrid_po_component_profile_figure(result, go, *, plot_template: str):
    """Compare the human and blocked-environment Hybrid PO components."""

    palette = _plot_theme_tokens(plot_template)
    component_styles = (
        ("human_po", "Human only", palette["orange"]),
        (
            "static_environment_blocked",
            "Environment RT with human blockage",
            palette["blue"],
        ),
        ("human_env", "Human → environment", palette["green"]),
        ("env_human", "Environment → human", palette["purple"]),
    )
    available = [
        (name, label, color, np.asarray(result.range_profiles[name]))
        for name, label, color in component_styles
        if name in result.range_profiles
    ]
    if not available:
        return None
    lower_db, upper_db = _absolute_db_limits(
        tuple(values for _name, _label, _color, values in available),
        dynamic_range_db=100.0,
        headroom_db=3.0,
    )
    figure = go.Figure()
    for _name, label, color, values in available:
        figure.add_trace(
            go.Scatter(
                x=result.ranges_m,
                y=_absolute_power_db(values),
                mode="lines",
                line={"color": color, "width": 2.5, "dash": "solid"},
                name=label,
            )
        )
    figure.update_layout(
        template="plotly_dark",
        title="Hybrid PO component range profiles",
        xaxis_title="Range [m]",
        yaxis_title="Power [dB, simulator units]",
        yaxis_range=[lower_db, upper_db],
        height=380,
        margin={"l": 60, "r": 20, "t": 52, "b": 55},
    )
    return _style_figure(figure, plot_template=plot_template)


def _human_room_rt_segment_histogram(results, go, *, plot_template: str):
    """Aggregate RT path interaction depths as path-segment histograms."""

    palette = _plot_theme_tokens(plot_template)
    colors = (palette["blue"], palette["orange"])
    series = []
    for result in results:
        if result.simulation_mode not in ("full_rt", "coherent_rt"):
            continue
        raw_histogram = getattr(
            result.metadata,
            "path_depth_histogram",
            None,
        )
        if raw_histogram is None:
            continue
        histogram = np.asarray(raw_histogram, dtype=np.int64)
        if histogram.ndim == 0 or histogram.shape[-1] == 0:
            continue
        if histogram.ndim > 1:
            histogram = np.sum(
                histogram,
                axis=tuple(range(histogram.ndim - 1)),
            )
        series.append((result, histogram))
    if not series:
        return None

    figure = go.Figure()
    for color, (result, histogram) in zip(colors, series):
        mode_label = _HUMAN_ROOM_MODE_LABELS.get(
            result.simulation_mode,
            result.simulation_mode.replace("_", " ").title(),
        )
        figure.add_trace(
            go.Bar(
                x=np.arange(histogram.size, dtype=int) + 1,
                y=histogram,
                marker_color=color,
                name=mode_label,
            )
        )
    figure.update_layout(
        template="plotly_dark",
        title="RT path segment-count histogram",
        xaxis_title="Path segments (interactions + 1)",
        yaxis_title="Valid RT path count over simulated frames",
        barmode="group",
        height=340,
        margin={"l": 70, "r": 20, "t": 52, "b": 60},
    )
    return _style_figure(figure, plot_template=plot_template)


def _range_profile_figure(result, go, *, plot_template: str):
    palette = _plot_theme_tokens(plot_template)
    po_db = _absolute_power_db(result.po_range_profile_power)
    rt_db = _absolute_power_db(result.rt_range_profile_power)
    lower_db, upper_db = _absolute_db_limits(
        (
            np.asarray(result.po_range_profile_power),
            np.asarray(result.rt_range_profile_power),
        ),
        dynamic_range_db=100.0,
        headroom_db=3.0,
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.ranges_m,
            y=po_db,
            mode="lines",
            line={"color": palette["blue"], "width": 2.5},
            name="Physical optics (PO)",
            hovertemplate=(
                "Range %{x:.3f} m<br>"
                "Power %{y:.1f} dB (sim.)<extra>PO</extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.ranges_m,
            y=rt_db,
            mode="lines",
            line={
                "color": palette["orange"],
                "width": 2.5,
                "dash": "solid",
            },
            name="Ray tracing (RT)",
            hovertemplate=(
                "Range %{x:.3f} m<br>"
                "Power %{y:.1f} dB (sim.)<extra>RT</extra>"
            ),
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title="RT and PO range profiles · absolute simulated power",
        xaxis_title="Range [m]",
        yaxis_title="Power [dB, simulator units]",
        yaxis_range=[lower_db, upper_db],
        margin={"l": 60, "r": 20, "t": 50, "b": 55},
        height=330,
    )
    return _style_figure(figure, plot_template=plot_template)


def _angle_fft_range_products(result) -> dict[str, object]:
    """Cache range FFT products for interactive RT/PO angle maps."""

    parameters = result.sensor_parameters["fmcw"]
    hardware = RadarHardware.from_ti_board(
        result.sensor_parameters["board_model"],
        pattern_mode="none",
    )
    hardware = hardware.subset_tx(
        result.sensor_parameters.get(
            "tx_indices_zero_based",
            list(range(hardware.num_tx)),
        )
    ).subset_rx(
        result.sensor_parameters.get(
            "rx_indices_zero_based",
            list(range(hardware.num_rx)),
        )
    )
    fmcw = FMCWConfig(
        carrier_frequency=parameters["carrier_frequency_hz"],
        slope=parameters["slope_hz_per_s"],
        chirp_duration=parameters["chirp_duration_s"],
        chirp_repetition_time=parameters["chirp_repetition_time_s"],
        sampling_frequency=parameters["sampling_frequency_hz"],
        adc_mode=parameters.get("adc_mode", "complex1x"),
        if_bandwidth_hz=parameters.get("if_bandwidth_hz"),
        num_adc_samples=parameters["num_adc_samples"],
        num_chirps_per_frame=parameters["num_chirps_per_frame"],
        frame_period=parameters["frame_period_s"],
        num_tx=parameters["num_tx"],
        tdm_enabled=bool(
            result.sensor_parameters.get("tdm_enabled", False)
        ),
    )
    products: dict[str, object] = {
        "hardware": hardware,
        "fmcw": fmcw,
    }
    common_ranges = None
    for solver, adc in (("po", result.po_adc), ("rt", result.rt_adc)):
        range_cube, ranges_m = range_fft(
            np.asarray(adc)[0],
            fmcw=fmcw,
            window="hann",
            nfft_mult=4,
        )
        if range_cube.shape[-1] != hardware.num_virtual_channels:
            raise ValueError(
                f"{solver.upper()} ADC has {range_cube.shape[-1]} channels, "
                f"but {hardware.name} defines {hardware.num_virtual_channels}"
            )
        products[f"{solver}_range_cube"] = (
            calibrate_range_cube_channel_phase(range_cube, hardware)
        )
        if common_ranges is None:
            common_ranges = np.asarray(ranges_m, dtype=float)
        elif not np.allclose(common_ranges, ranges_m):
            raise RuntimeError("RT and PO angle-map range axes do not match")
    products["ranges_m"] = common_ranges
    centroid = np.mean(np.asarray(result.vertices, dtype=float), axis=0)
    centroid_range_m = float(
        np.linalg.norm(centroid - np.asarray(result.radar_position, dtype=float))
    )
    products["centroid_range_m"] = centroid_range_m
    products["centroid_range_bin"] = int(
        np.argmin(np.abs(common_ranges - centroid_range_m))
    )
    return products


def _selected_angle_fft_power(
    products: dict[str, object],
    *,
    solver: str,
    first_bin: int,
    last_bin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate angle-FFT power over an inclusive range-bin interval."""

    ranges_m = np.asarray(products["ranges_m"])
    lower = max(0, min(int(first_bin), ranges_m.size - 1))
    upper = max(lower, min(int(last_bin), ranges_m.size - 1))
    range_cube = np.asarray(products[f"{solver}_range_cube"])
    hardware = products["hardware"]
    fmcw = products["fmcw"]
    accumulated = None
    u_axis = None
    v_axis = None
    snapshot_count = 0
    # Chunk the selected bins to bound memory for a full-range selection.
    for chunk_start in range(lower, upper + 1, 16):
        chunk_stop = min(chunk_start + 16, upper + 1)
        selected = range_cube[:, chunk_start:chunk_stop, :]
        virtual_grid, grid_metadata = virtual_snapshot_grid(
            selected,
            hardware,
            wavelength=fmcw.wavelength,
            align_origin=True,
            return_metadata=True,
        )
        angle = angle_map_fft(
            virtual_grid,
            fc_hz=fmcw.carrier_frequency,
            window=None,
            fft_size=(64, 128),
            power=True,
            spacing_y_lambda=grid_metadata["spacing_y_lambda"],
            spacing_z_lambda=grid_metadata["spacing_z_lambda"],
        )
        power = np.asarray(angle["map"], dtype=float)
        if "u" in angle and "v" in angle:
            spatial_dimensions = 2
            reduced = np.sum(power, axis=tuple(range(power.ndim - 2)))
            chunk_u_axis = np.asarray(angle["u"], dtype=float)
            chunk_v_axis = np.asarray(angle["v"], dtype=float)
        elif "u" in angle:
            spatial_dimensions = 1
            reduced = np.sum(power, axis=tuple(range(power.ndim - 1)))[
                None, :
            ]
            chunk_u_axis = np.asarray(angle["u"], dtype=float)
            chunk_v_axis = np.asarray([0.0])
        elif "v" in angle:
            spatial_dimensions = 1
            reduced = np.sum(power, axis=tuple(range(power.ndim - 1)))[
                :, None
            ]
            chunk_u_axis = np.asarray([0.0])
            chunk_v_axis = np.asarray(angle["v"], dtype=float)
        else:
            raise ValueError(
                f"{hardware.name} does not have an angle-resolving aperture"
            )
        accumulated = reduced if accumulated is None else accumulated + reduced
        snapshot_count += int(
            np.prod(power.shape[: power.ndim - spatial_dimensions])
        )
        u_axis = chunk_u_axis
        v_axis = chunk_v_axis
    if accumulated is None or u_axis is None or v_axis is None:
        raise RuntimeError("angle FFT range-bin selection is empty")
    return accumulated / max(snapshot_count, 1), u_axis, v_axis


def _angle_map_figure(
    products,
    make_subplots,
    go,
    *,
    range_bins: tuple[int, int],
    plot_template: str,
):
    """Build shared-scale RT/PO direction-cosine angle-FFT heatmaps."""

    first_bin, last_bin = sorted(int(value) for value in range_bins)
    po_power, u_axis, v_axis = _selected_angle_fft_power(
        products,
        solver="po",
        first_bin=first_bin,
        last_bin=last_bin,
    )
    rt_power, rt_u_axis, rt_v_axis = _selected_angle_fft_power(
        products,
        solver="rt",
        first_bin=first_bin,
        last_bin=last_bin,
    )
    if not np.allclose(u_axis, rt_u_axis) or not np.allclose(v_axis, rt_v_axis):
        raise RuntimeError("RT and PO angle axes do not match")
    azimuth_deg, elevation_deg = azimuth_elevation_grids(
        {"u": u_axis, "v": v_axis}
    )
    hover_angles = np.stack((azimuth_deg, elevation_deg), axis=-1)
    valid = direction_cosine_valid_mask(u_axis, v_axis)
    po_db = np.where(valid, _absolute_power_db(po_power), np.nan)
    rt_db = np.where(valid, _absolute_power_db(rt_power), np.nan)
    lower_db, upper_db = _absolute_db_limits(
        (po_power[valid], rt_power[valid]),
        dynamic_range_db=60.0,
        headroom_db=2.0,
    )

    ranges_m = np.asarray(products["ranges_m"], dtype=float)
    first_bin = max(0, min(first_bin, ranges_m.size - 1))
    last_bin = max(first_bin, min(last_bin, ranges_m.size - 1))
    title_range = (
        f"bin {first_bin} · {ranges_m[first_bin]:.3f} m"
        if first_bin == last_bin
        else (
            f"bins {first_bin}–{last_bin} · "
            f"{ranges_m[first_bin]:.3f}–{ranges_m[last_bin]:.3f} m"
        )
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Physical optics (PO)", "Ray tracing (RT)"),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    for column, power_db in enumerate((po_db, rt_db), start=1):
        figure.add_trace(
            go.Heatmap(
                z=power_db,
                x=u_axis,
                y=v_axis,
                customdata=hover_angles,
                colorscale="Turbo",
                zmin=lower_db,
                zmax=upper_db,
                showscale=column == 2,
                colorbar={"title": "dB (sim.)"} if column == 2 else None,
                hovertemplate=(
                    "u %{x:.3f}<br>"
                    "v %{y:.3f}<br>"
                    "Azimuth %{customdata[0]:.1f}°<br>"
                    "Elevation %{customdata[1]:.1f}°<br>"
                    "Power %{z:.1f} dB (sim.)<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
    figure.update_xaxes(title_text="Horizontal direction cosine u", row=1, col=1)
    figure.update_xaxes(title_text="Horizontal direction cosine u", row=1, col=2)
    figure.update_yaxes(title_text="Vertical direction cosine v", row=1, col=1)
    map_label = (
        "2D angle FFT"
        if u_axis.size > 1 and v_axis.size > 1
        else "Angle FFT · single-axis aperture"
    )
    figure.update_layout(
        template="plotly_dark",
        title=f"{map_label} · {title_range} · absolute simulated power",
        height=420,
        margin={"l": 65, "r": 35, "t": 75, "b": 55},
    )
    return _style_figure(figure, plot_template=plot_template)


def _adc_trace_figure(products, go, *, title: str, plot_template: str,
                      frame_index: int = 0, chirp_index: int = 0, channel_index: int = 0):
    palette = _plot_theme_tokens(plot_template)
    adc = np.asarray(products.adc)
    figure = go.Figure()
    selection = (int(frame_index), int(chirp_index), int(channel_index))
    if any(index < 0 or index >= limit for index, limit in zip(selection, (adc.shape[0], adc.shape[1], adc.shape[3]))):
        figure.add_annotation(text="Selected frame, chirp or channel is unavailable in this ADC.", showarrow=False)
        figure.update_layout(title=title, xaxis_visible=False, yaxis_visible=False)
        return _style_figure(figure, plot_template=plot_template)
    trace = adc[selection[0], selection[1], :, selection[2]]
    figure.add_trace(
        go.Scatter(
            y=trace.real,
            mode="lines",
            name="I",
            line={"color": palette["blue"]},
        )
    )
    figure.add_trace(
        go.Scatter(
            y=trace.imag,
            mode="lines",
            name="Q",
            line={"color": palette["orange"]},
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="ADC sample",
        yaxis_title="Amplitude",
        height=310,
        margin={"l": 60, "r": 20, "t": 50, "b": 55},
    )
    return _style_figure(figure, plot_template=plot_template)


def _range_time_figure(products, go, *, title: str, plot_template: str):
    power = np.asarray(products.range_time_power)
    slow_time = np.arange(power.shape[0])
    slow_time_title = "Slow-time index"
    adc_times_s = getattr(products, "adc_times_s", None)
    if adc_times_s is not None:
        candidate_times = np.asarray(adc_times_s, dtype=float).reshape(-1)
        if candidate_times.size == power.shape[0]:
            slow_time = candidate_times
            slow_time_title = "Time [s]"
    figure = go.Figure(
        go.Heatmap(
            z=_relative_db(power),
            x=products.range_time_ranges_m,
            y=slow_time,
            colorscale="Viridis",
            zmin=-60,
            zmax=0,
            colorbar={"title": "dB"},
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Range [m]",
        yaxis_title=slow_time_title,
        height=360,
        margin={"l": 65, "r": 20, "t": 50, "b": 55},
    )
    return _style_figure(figure, plot_template=plot_template)


def _range_doppler_figure(
    products,
    go,
    *,
    title: str,
    plot_template: str,
    frame_index: int = 0,
    source_frame_index: int | None = None,
):
    power_frames = np.asarray(products.range_doppler_power)
    frame_index = int(frame_index)
    if frame_index < 0 or frame_index >= power_frames.shape[0]:
        raise ValueError(
            f"frame_index must be in [0, {power_frames.shape[0] - 1}]"
        )
    power = power_frames[frame_index]
    frame_title = title
    if source_frame_index is not None:
        frame_title = (
            f"{title} · simulated frame {frame_index} · "
            f"source motion frame {int(source_frame_index)}"
        )
    figure = go.Figure(
        go.Heatmap(
            z=_relative_db(power),
            x=products.range_doppler_ranges_m,
            y=products.velocities_mps,
            colorscale="Magma",
            zmin=-60,
            zmax=0,
            colorbar={"title": "dB"},
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title=frame_title,
        xaxis_title="Range [m]",
        yaxis_title="Velocity [m/s, +away]",
        height=360,
        margin={"l": 65, "r": 20, "t": 50, "b": 55},
    )
    return _style_figure(figure, plot_template=plot_template)


def _bundle_primary_label(loaded) -> str:
    """Describe a bundle's primary ADC without assuming every simulation is PO."""

    if loaded.data_origin == "measurement":
        return "Measured"
    descriptor = getattr(getattr(loaded, "bundle", None), "descriptor", None)
    if not isinstance(descriptor, dict):
        descriptor = {}
    artifacts = descriptor.get("adcs", {})
    primary_key = str(descriptor.get("primary_adc", ""))
    primary_artifact = (
        artifacts.get(primary_key, {}) if isinstance(artifacts, dict) else {}
    )
    if not isinstance(primary_artifact, dict):
        primary_artifact = {}
    simulation_mode = str(
        primary_artifact.get(
            "simulation_mode",
            descriptor.get("primary_simulation_mode", ""),
        )
    )
    if simulation_mode in _HUMAN_ROOM_MODE_LABELS:
        return f"Primary {_HUMAN_ROOM_MODE_LABELS[simulation_mode]} simulation"
    primary_solver = str(
        primary_artifact.get(
            "solver",
            descriptor.get("primary_solver", ""),
        )
    ).casefold()
    if primary_solver in {"rt", "po"}:
        return f"Primary {primary_solver.upper()} simulation"
    return "Primary simulation"


def _bundle_scene_path(bundle) -> Path | None:
    """Resolve the portable scene declared directly or through environment."""

    if bundle.scene_path is not None:
        return Path(bundle.scene_path)
    environment = bundle.environment or {}
    raw_path = environment.get("scene_path") or environment.get("scene_file")
    geometry = environment.get("geometry_outputs")
    if raw_path is None and isinstance(geometry, dict):
        raw_path = geometry.get("scene_path")
    if not raw_path:
        return None
    root = Path(bundle.root).resolve()
    candidate = Path(str(raw_path))
    candidate = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _bundle_scene_boxes(scene_path: Path | None) -> tuple[dict, ...]:
    """Extract lightweight cube geometry from a Mitsuba scene document."""

    if scene_path is None:
        return ()
    try:
        if scene_path.stat().st_size > _MAX_SCENE_XML_UPLOAD_BYTES:
            return ()
        return _scene_xml_boxes(scene_path.read_bytes())
    except OSError:
        return ()


def _scene_xml_boxes(scene_xml: bytes) -> tuple[dict, ...]:
    """Extract axis-aligned cubes for the room preview and floor check."""

    if len(scene_xml) > _MAX_SCENE_XML_UPLOAD_BYTES:
        return ()
    try:
        root = ET.fromstring(scene_xml)
    except ET.ParseError:
        return ()

    def local_name(element) -> str:
        return str(element.tag).rsplit("}", maxsplit=1)[-1]

    def vector(element, default) -> np.ndarray:
        if "value" in element.attrib:
            value = float(element.attrib["value"])
            return np.full(3, value, dtype=float)
        return np.asarray(
            [
                float(element.attrib.get(axis, default[index]))
                for index, axis in enumerate(("x", "y", "z"))
            ],
            dtype=float,
        )

    material_types = {}
    for material in root.iter():
        if local_name(material) != "bsdf" or not material.attrib.get("id"):
            continue
        material_type = str(material.attrib.get("type", "Unspecified"))
        for parameter in material:
            if (
                local_name(parameter) == "string"
                and parameter.attrib.get("name") == "type"
                and parameter.attrib.get("value")
            ):
                material_type = str(parameter.attrib["value"])
                break
        material_types[str(material.attrib["id"])] = material_type

    boxes = []
    for shape_index, shape in enumerate(root.iter()):
        if local_name(shape) != "shape" or shape.attrib.get("type") != "cube":
            continue
        if any(
            local_name(operation) not in {"scale", "translate"}
            for transform in shape
            if local_name(transform) == "transform"
            for operation in transform
        ):
            continue
        try:
            scale = np.ones(3, dtype=float)
            translate = np.zeros(3, dtype=float)
            for transform in shape.iter():
                name = local_name(transform)
                if name == "scale":
                    factor = vector(transform, (1.0, 1.0, 1.0))
                    scale *= factor
                    translate *= factor
                elif name == "translate":
                    translate += vector(transform, (0.0, 0.0, 0.0))
        except (TypeError, ValueError):
            continue
        if not np.all(np.isfinite(scale)) or not np.all(np.isfinite(translate)):
            continue
        material_id = next(
            (
                str(reference.attrib["id"])
                for reference in shape.iter()
                if local_name(reference) == "ref"
                and reference.attrib.get("name") == "bsdf"
                and reference.attrib.get("id")
            ),
            None,
        )
        boxes.append(
            {
                "id": shape.attrib.get("id", f"scene_cube_{shape_index + 1}"),
                "material": material_types.get(
                    material_id,
                    material_id or "Unspecified",
                ),
                "material_id": material_id,
                "scale": tuple(float(value) for value in np.abs(scale)),
                "translate": tuple(float(value) for value in translate),
            }
        )
    return tuple(boxes)


def _bundle_mesh_sequence(
    bundle,
    *,
    smpl_model_dir: str | Path | None = None,
) -> tuple[MeshSequence | None, Path | None, str | None]:
    """Load or evaluate the bundle target at its primary frame time."""

    mesh_path = bundle.evaluated_motion_path or bundle.target_mesh_path
    source_kind = None
    if mesh_path is not None:
        path = Path(mesh_path)
        arrays = preflight_npz(path)
        require_array(
            arrays,
            "vertices",
            allowed_ndim=(2, 3),
            dtype_kinds=frozenset({"i", "u", "f"}),
            trailing_shape=(3,),
            max_elements=120_000_000,
        )
        require_array(
            arrays,
            "faces",
            allowed_ndim=(2,),
            dtype_kinds=frozenset({"i", "u"}),
            trailing_shape=(3,),
            max_elements=60_000_000,
        )
        with np.load(path, allow_pickle=False) as archive:
            vertices = np.asarray(archive["vertices"], dtype=np.float32)
            faces = np.asarray(archive["faces"], dtype=np.uint32)
            if vertices.ndim == 2:
                vertices = vertices[None, ...]
            times = (
                np.asarray(archive["times"], dtype=float)
                if "times" in archive.files
                else np.arange(vertices.shape[0], dtype=float)
            )
        sequence = MeshSequence(vertices=vertices, faces=faces, times=times)
        source_kind = (
            "evaluated motion"
            if bundle.evaluated_motion_path is not None
            else "static target"
        )
    elif bundle.motion_path is not None:
        path = Path(bundle.motion_path)
        loaded_motion = load_amass_motion(
            path,
            filename=path.name,
            smpl_model_dir=smpl_model_dir,
        )
        sequence = loaded_motion.sequence
        environment = bundle.environment or {}
        if "human_position_m" in environment:
            placed = prepare_human_room_preview(
                loaded_motion,
                human_position_m=environment["human_position_m"],
                human_yaw_deg=float(environment.get("human_yaw_deg", 0.0)),
            )
            sequence = placed.mesh_sequence
        source_kind = "AMASS parameters"
    else:
        return None, None, None

    primary_time = float(bundle.motion_times_s[0])
    primary_vertices = np.asarray(
        sequence.vertices_at(primary_time),
        dtype=np.float32,
    )
    primary = MeshSequence(
        vertices=primary_vertices[None, ...],
        faces=np.asarray(sequence.faces, dtype=np.uint32),
        times=np.asarray([0.0]),
    )
    return primary, path, source_kind


def _bundle_target_material(bundle, mesh_source: str | None) -> str:
    """Resolve the displayed target material from portable bundle metadata."""

    if mesh_source in {"evaluated motion", "AMASS parameters"}:
        return "Human skin"
    descriptor = bundle.descriptor if isinstance(bundle.descriptor, dict) else {}
    target = descriptor.get("target", {})
    if isinstance(target, dict) and target.get("material"):
        return str(target["material"])
    environment = bundle.environment or {}
    if isinstance(environment, dict) and environment.get("target_material"):
        return str(environment["target_material"])
    manifests = descriptor.get("manifests", {})
    manifest_path = manifests.get("primary") if isinstance(manifests, dict) else None
    if manifest_path:
        try:
            manifest = json.loads(
                (Path(bundle.root) / str(manifest_path)).read_text(
                    encoding="utf-8"
                )
            )
            material = manifest.get("scene", {}).get("parameters", {}).get(
                "material"
            )
            if material:
                return str(material)
        except (AttributeError, json.JSONDecodeError, OSError, TypeError):
            pass
    summary_path = Path(bundle.root) / "bundle_summary.json"
    if summary_path.is_file():
        try:
            material = json.loads(summary_path.read_text(encoding="utf-8")).get(
                "material"
            )
            if material:
                return str(material)
        except (AttributeError, json.JSONDecodeError, OSError, TypeError):
            pass
    return "Unspecified"


def _bundle_scene_figure(
    loaded,
    go,
    *,
    smpl_model_dir: str | Path | None = None,
    plot_template: str,
):
    """Render bundle-declared geometry without requiring a fresh simulation."""

    bundle = loaded.bundle
    scene_path = _bundle_scene_path(bundle)
    room_boxes = _bundle_scene_boxes(scene_path)
    mesh_error = None
    try:
        sequence, mesh_path, mesh_source = _bundle_mesh_sequence(
            bundle,
            smpl_model_dir=smpl_model_dir,
        )
    except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        sequence, mesh_path, mesh_source = None, None, None
        mesh_error = f"{type(exc).__name__}: {exc}"
    preview = None
    if sequence is not None:
        preview = HumanRoomPreview(
            room_preset="bundle",
            room_boxes=room_boxes,
            mesh_sequence=sequence,
            human_position_m=np.zeros(3, dtype=float),
            human_yaw_deg=0.0,
            source_name=mesh_path.name,
            scene_source_name=(
                scene_path.name if scene_path is not None else loaded.bundle_id
            ),
        )
    orientation_deg = np.rad2deg(
        np.asarray(bundle.sensor_config.orientation, dtype=float)
    )
    figure = _human_room_scene_figure(
        preview,
        go,
        frame_index=0,
        radar_orientation_deg=orientation_deg,
        room_boxes_override=room_boxes,
        title_override=f"Primary bundle scene · {loaded.bundle_id}",
        plot_template=plot_template,
    )
    if mesh_path is not None:
        trace_label = (
            "Bundle human"
            if mesh_source in {"evaluated motion", "AMASS parameters"}
            else "Bundle target"
        )
        target_material = _bundle_target_material(bundle, mesh_source)
        for trace in figure.data:
            if trace.name == "Human mesh":
                trace.name = trace_label
                trace.hovertemplate = (
                    f"Object: {trace_label} facet %{{pointNumber}}<br>"
                    f"Material: {target_material}<extra></extra>"
                )
    figure.update_layout(
        meta={
            "bundle_mesh_source": mesh_source,
            "bundle_mesh_error": mesh_error,
            "bundle_has_target_preview": sequence is not None,
        }
    )
    return figure


def _bundle_range_profile_figure(
    products,
    go,
    *,
    title: str,
    plot_template: str,
):
    """Plot a single bundle ADC range profile on a relative-power scale."""

    palette = _plot_theme_tokens(plot_template)
    figure = go.Figure(
        go.Scatter(
            x=np.asarray(products.range_time_ranges_m, dtype=float),
            y=_relative_db(products.range_profile_power),
            mode="lines",
            line={"color": palette["blue"], "width": 2.5},
            name="Range profile",
            hovertemplate=(
                "Range %{x:.3f} m<br>Relative power %{y:.1f} dB<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Range [m]",
        yaxis_title="Relative power [dB]",
        yaxis_range=[-80.0, 0.0],
        height=330,
        margin={"l": 60, "r": 20, "t": 50, "b": 55},
    )
    return _style_figure(figure, plot_template=plot_template)


def _bundle_range_profile_comparison_figure(
    primary_label,
    primary_products,
    candidates,
    go,
    *,
    plot_template: str,
):
    """Overlay primary and candidate range profiles on shared axes."""

    palette = _plot_theme_tokens(plot_template)
    colors = (
        palette["blue"],
        palette["orange"],
        "#2dd4bf",
        "#c084fc",
        "#facc15",
        "#fb7185",
    )
    figure = go.Figure()
    series = [(primary_label, primary_products), *list(candidates)]
    for index, (label, products) in enumerate(series):
        figure.add_trace(
            go.Scatter(
                x=np.asarray(products.range_time_ranges_m, dtype=float),
                y=_relative_db(products.range_profile_power),
                mode="lines",
                line={"color": colors[index % len(colors)], "width": 2.5},
                name=str(label),
                hovertemplate=(
                    f"{label}<br>Range %{{x:.3f}} m<br>"
                    "Relative power %{y:.1f} dB<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        template="plotly_dark",
        title=(
            "Range profile comparison"
            if candidates
            else f"{primary_label} range profile"
        ),
        xaxis_title="Range [m]",
        yaxis_title="Relative power [dB]",
        yaxis_range=[-80.0, 0.0],
        height=350,
        margin={"l": 60, "r": 20, "t": 50, "b": 55},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return _style_figure(figure, plot_template=plot_template)


def _human_room_results_view(
    mode_results,
    pn,
    go,
    *,
    plot_template: str,
):
    """Build product-first views that compare every completed solver mode."""

    results = [
        mode_results[mode]
        for mode in HUMAN_ROOM_SIMULATION_MODES
        if mode in mode_results
    ]
    if not results:
        raise ValueError("at least one completed simulation mode is required")

    palette = _plot_theme_tokens(plot_template)
    colors = (
        palette["blue"],
        palette["orange"],
        palette["green"],
        palette["purple"],
    )
    lower_db, upper_db = _absolute_db_limits(
        [np.asarray(result.range_profiles["total"]) for result in results],
        dynamic_range_db=100.0,
        headroom_db=3.0,
    )
    profile_figure = go.Figure()
    for color, result in zip(colors, results):
        mode_label = _HUMAN_ROOM_MODE_LABELS.get(
            result.simulation_mode,
            result.simulation_mode.replace("_", " ").title(),
        )
        profile_figure.add_trace(
            go.Scatter(
                x=result.ranges_m,
                y=_absolute_power_db(result.range_profiles["total"]),
                mode="lines",
                line={"color": color, "width": 2.5, "dash": "solid"},
                name=mode_label,
            )
        )
    profile_figure.update_layout(
        template="plotly_dark",
        title="Range profile comparison",
        xaxis_title="Range [m]",
        yaxis_title="Power [dB, simulator units]",
        yaxis_range=[lower_db, upper_db],
        height=380,
        margin={"l": 60, "r": 20, "t": 52, "b": 55},
    )
    profile_figure = _style_figure(
        profile_figure,
        plot_template=plot_template,
    )
    profile = pn.pane.Plotly(
        go.Figure(profile_figure),
        height=380,
    )
    hybrid_component_pane = None
    hybrid_component_figure = None
    hybrid_result = next(
        (
            result
            for result in results
            if result.simulation_mode == "hybrid_po"
        ),
        None,
    )
    if hybrid_result is not None:
        hybrid_component_figure = _hybrid_po_component_profile_figure(
            hybrid_result,
            go,
            plot_template=plot_template,
        )
        if hybrid_component_figure is not None:
            hybrid_component_pane = pn.pane.Plotly(
                go.Figure(hybrid_component_figure),
                height=380,
            )
    profile_view = (
        pn.Column(profile, hybrid_component_pane)
        if hybrid_component_pane is not None
        else profile
    )

    def mode_label(result):
        return _HUMAN_ROOM_MODE_LABELS.get(
            result.simulation_mode,
            result.simulation_mode.replace("_", " ").title(),
        )

    range_time_panes = [
        pn.pane.Plotly(
            _range_time_figure(
                result,
                go,
                title=f"{mode_label(result)} range-time map",
                plot_template=plot_template,
            ),
            height=380,
        )
        for result in results
    ]
    range_doppler_frame_count = min(
        int(result.range_doppler_power.shape[0]) for result in results
    )
    range_doppler_frame = pn.widgets.IntSlider(
        label=f"Range-Doppler frame index (0–{max(range_doppler_frame_count - 1, 0)})",
        start=0,
        end=max(range_doppler_frame_count - 1, 0),
        value=0,
        step=1,
        disabled=range_doppler_frame_count <= 1,
    )
    previous_range_doppler_frame = pn.widgets.Button(
        label="Previous frame",
        icon="chevron-left",
        color="light",
        disabled=True,
    )
    next_range_doppler_frame = pn.widgets.Button(
        label="Next frame",
        icon="chevron-right",
        color="light",
        disabled=range_doppler_frame_count <= 1,
    )
    range_doppler_frame_status = pn.pane.Markdown(
        f"**Displayed frame index:** 0 / {max(range_doppler_frame_count - 1, 0)}"
    )
    begin_frames = {
        result.simulation_mode: int(
            result.manifest.scene.parameters["motion_begin_frame_index"]
        )
        for result in results
    }

    def range_doppler_figure(result, frame_index: int):
        return _range_doppler_figure(
            result,
            go,
            title=f"{mode_label(result)} range-Doppler map",
            frame_index=int(frame_index),
            source_frame_index=(
                begin_frames[result.simulation_mode] + int(frame_index)
            ),
            plot_template=plot_template,
        )

    range_doppler_panes = [
        pn.pane.Plotly(range_doppler_figure(result, 0), height=380)
        for result in results
    ]
    product_tabs = pn.Tabs(
        ("Range profile", profile_view),
        ("Range-time", pn.Column(*range_time_panes)),
        (
            "Range-Doppler",
            pn.Column(
                pn.Row(
                    previous_range_doppler_frame,
                    next_range_doppler_frame,
                ),
                range_doppler_frame,
                range_doppler_frame_status,
                *range_doppler_panes,
            ),
        ),
        dynamic=True,
        sizing_mode="stretch_width",
    )

    def reset_plot(pane, figure):
        pane.relayout_data = {}
        pane.viewport = None
        pane.object = figure

    def reset_range_doppler(display_frame: int):
        frame_index = int(display_frame)
        previous_range_doppler_frame.disabled = display_frame <= 0
        next_range_doppler_frame.disabled = (
            display_frame >= range_doppler_frame_count - 1
        )
        range_doppler_frame_status.object = (
            f"**Displayed frame index:** {display_frame} / "
            f"{max(range_doppler_frame_count - 1, 0)}"
        )
        for result, pane in zip(results, range_doppler_panes):
            reset_plot(pane, range_doppler_figure(result, frame_index))

    range_doppler_frame.param.watch(
        lambda event: reset_range_doppler(int(event.new)), "value"
    )

    def step_range_doppler_frame(offset: int):
        range_doppler_frame.value = int(
            np.clip(
                int(range_doppler_frame.value) + int(offset),
                0,
                max(range_doppler_frame_count - 1, 0),
            )
        )

    previous_range_doppler_frame.on_click(
        lambda _event: step_range_doppler_frame(-1)
    )
    next_range_doppler_frame.on_click(
        lambda _event: step_range_doppler_frame(1)
    )

    def reset_active_product(event):
        if int(event.new) == 0:
            reset_plot(profile, go.Figure(profile_figure))
            if hybrid_component_pane is not None:
                reset_plot(
                    hybrid_component_pane,
                    go.Figure(hybrid_component_figure),
                )
        elif int(event.new) == 1:
            for result, pane in zip(results, range_time_panes):
                reset_plot(
                    pane,
                    _range_time_figure(
                        result,
                        go,
                        title=f"{mode_label(result)} range-time map",
                        plot_template=plot_template,
                    ),
                )
        else:
            reset_range_doppler(int(range_doppler_frame.value))

    product_tabs.param.watch(reset_active_product, "active")
    metadata_view = {}
    refinement_notes = []
    for result in results:
        metadata = result.metadata
        metadata_view[mode_label(result)] = {
            "simulation_mode": result.simulation_mode,
            "manifest_sha256": result.manifest.fingerprint,
            "adc_shape": list(result.adc.shape),
            "components": sorted(result.components),
            "runtime_s": result.runtime_s,
            "motion_frame_window": {
                "begin_frame_index": begin_frames[result.simulation_mode],
                "end_frame_index": int(
                    result.manifest.scene.parameters["motion_end_frame_index"]
                ),
            },
            "path_count_max": (
                int(np.max(metadata.path_counts))
                if metadata.path_counts is not None
                else None
            ),
            "visible_face_count_max": (
                int(np.max(metadata.visible_face_counts))
                if metadata.visible_face_counts is not None
                else None
            ),
            "po_calibration": metadata.po_calibration,
            "runtime_profile_s": metadata.runtime_profile_s,
        }
        for event in getattr(metadata, "events", None) or []:
            if event.get("reason") != "adaptive_po_refinement":
                continue
            metadata_view[mode_label(result)]["po_refinement"] = event
            if event["limited_passes"]:
                refinement_notes.append(
                    f"**{mode_label(result)} — PO refinement limit:** "
                    f"up to {event['max_unresolved_parent_faces']:,} facets remained above a tolerance "
                    "when subdivision reached its limit. Raise **PO maximum refinement depth** "
                    "or **PO subfaces per parent (0 = unlimited)** in **Settings → Solver settings**."
                )
    _fit_active_tab_height(product_tabs)
    result_sections = [product_tabs, *[pn.pane.Alert(note, alert_type="warning") for note in refinement_notes]]
    rt_segment_figure = _human_room_rt_segment_histogram(
        results,
        go,
        plot_template=plot_template,
    )
    if rt_segment_figure is not None:
        result_sections.extend(
            (
                pn.pane.Markdown("### RT path segments"),
                pn.pane.Plotly(rt_segment_figure, height=340),
            )
        )
    summary = [
        "| Mode | Runtime | RT paths (max) | PO visible facets (max) | Calibration |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    breakdown = []
    for label, details in metadata_view.items():
        calibration = details["po_calibration"] or {}
        calibration_label = str(calibration.get("mode", "applied")) if calibration.get("applied") else "Uncalibrated"
        paths = details["path_count_max"]
        facets = details["visible_face_count_max"]
        summary.append(
            f"| {label} | {details['runtime_s']:.3g} s | "
            f"{paths if paths is not None else '—'} | "
            f"{facets if facets is not None else '—'} | {calibration_label} |"
        )
        stages = details["runtime_profile_s"] or {}
        stage_table = _runtime_stage_table(stages)
        if stage_table:
            breakdown.append(f"**{label} measured stages**\n\n{stage_table}")
    result_sections.extend(
        (
            pn.pane.Markdown("### Run metadata"),
            pn.pane.Markdown("\n".join(summary)),
            pn.pane.Markdown("\n\n".join(breakdown)),
            pn.Card(
                pn.pane.JSON(metadata_view, depth=2, height=380,
                             stylesheets=[_comparison_json_stylesheet(plot_template)]),
                title="Advanced · Raw run metadata", collapsed=True,
                sizing_mode="stretch_width",
            ),
        )
    )
    view = pn.Column(*result_sections, sizing_mode="stretch_width")
    _readable_markdown(view, pn, plot_template)
    _style_secondary_buttons(view, pn, plot_template)
    return view


def _human_room_result_mode_view(result, pn, go, *, plot_template: str):
    """Backward-compatible single-mode wrapper around the product-first view."""

    return _human_room_results_view(
        {result.simulation_mode: result},
        pn,
        go,
        plot_template=plot_template,
    )


def _comparison_figure(
    comparison,
    make_subplots,
    go,
    *,
    product: str,
    plot_template: str,
):
    if product == "range_time":
        primary = comparison.primary.range_time_power
        candidate = comparison.candidate.range_time_power
        x = comparison.primary.range_time_ranges_m
        y = np.arange(primary.shape[0])
        y_title = "Slow-time index"
        colorscale = "Viridis"
        title = "Range-time"
    else:
        primary = comparison.primary.range_doppler_power[0]
        candidate = comparison.candidate.range_doppler_power[0]
        x = comparison.primary.range_doppler_ranges_m
        y = comparison.primary.velocities_mps
        y_title = "Velocity [m/s, +away]"
        colorscale = "Magma"
        title = "Range-Doppler"
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Bundle primary", "Comparison ADC"),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    for column, values in enumerate((primary, candidate), start=1):
        figure.add_trace(
            go.Heatmap(
                z=_relative_db(values),
                x=x,
                y=y,
                colorscale=colorscale,
                zmin=-60,
                zmax=0,
                showscale=column == 2,
                colorbar={"title": "dB"} if column == 2 else None,
            ),
            row=1,
            col=column,
        )
    figure.update_xaxes(title_text="Range [m]", row=1, col=1)
    figure.update_xaxes(title_text="Range [m]", row=1, col=2)
    figure.update_yaxes(title_text=y_title, row=1, col=1)
    figure.update_layout(
        template="plotly_dark",
        title=f"{title} comparison",
        height=410,
        margin={"l": 65, "r": 20, "t": 70, "b": 55},
    )
    return _style_figure(figure, plot_template=plot_template)


def _npz_bytes(**arrays) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _json_bytes(value) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _obj_bytes(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    lines = ["# HERMES transformed static target"]
    lines.extend(
        f"v {float(x):.9g} {float(y):.9g} {float(z):.9g}"
        for x, y, z in np.asarray(vertices)
    )
    lines.extend(
        f"f {int(i) + 1} {int(j) + 1} {int(k) + 1}"
        for i, j, k in np.asarray(faces)
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _physics_export(result) -> BytesIO:
    """Build a loadable HERMES bundle for a static-target simulation."""

    descriptor = {
        "schema_version": 1,
        "profile": "hermes",
        "primary_adc": "primary",
        "adcs": {
            "primary": {
                "path": "radar_adc.npz",
                "origin": "simulation",
                "solver": "po",
            },
            "rt": {
                "path": "simulated_adc_rt.npz",
                "origin": "simulation",
                "solver": "rt",
            },
            "po": {
                "path": "simulated_adc_po.npz",
                "origin": "simulation",
                "solver": "po",
            },
        },
        "primary_solver": "po",
        "target": {
            "mesh": "target_mesh.npz",
            "obj": "target.obj",
        },
        "scene": {"file": "scene.xml", "assets": ["target.obj"]},
        "environment": "environment.json",
        "manifests": {"primary": "experiment_manifest.json"},
        "diagnostics": {"solver": "solver_diagnostics.npz"},
    }
    sensor = dict(result.sensor_parameters)
    sensor.update(
        {
            "name": "radar",
            "virtual_channel_order": "tx_major",
            "metadata": {
                "generated_by": "HERMES GUI",
                "manifest_sha256": result.manifest.fingerprint,
            },
        }
    )
    frames = {
        "frames": [
            {
                "bundle_index": 0,
                "motion_time_s": 0.0,
                "radar_frame_id": "simulation-frame-0",
                "metadata": {"static_target": True},
            }
        ]
    }
    environment = {
        "scene_path": "scene.xml",
        "coordinate_system": (
            "right-handed; world +x is the unrotated radar boresight; "
            "current radar pose is in sensor.json"
        ),
        "target_geometry": "target_mesh.npz",
        "target_anchor_m": [
            float(value) for value in _target_position(result)
        ],
    }
    summary = {
        "profile": "hermes",
        "target_type": result.target_type,
        "target_position_m": [
            float(value) for value in _target_position(result)
        ],
        "material": result.material_name,
        "adc_shape": [int(value) for value in result.po_adc.shape],
        "rt_path_count": int(result.rt_path_count),
        "runtime_s": {
            "rt": float(result.rt_runtime_s),
            "po": float(result.po_runtime_s),
        },
        "amplitude_calibration": "none",
        "manifest_sha256": result.manifest.fingerprint,
    }
    adc_times = np.asarray(result.adc_times_s, dtype=float)
    primary_adc = _npz_bytes(adc=result.po_adc, times=adc_times)
    po_adc = _npz_bytes(adc=result.po_adc, times=adc_times)
    rt_adc = _npz_bytes(adc=result.rt_adc, times=adc_times)
    target_mesh = _npz_bytes(
        vertices=result.vertices,
        faces=result.faces,
        times=np.asarray([0.0]),
    )
    diagnostics = _npz_bytes(
        face_power=result.face_power,
        face_visibility=result.face_visibility,
        rt_face_hit_counts=result.rt_face_hit_counts,
        ranges_m=result.ranges_m,
        po_range_profile_power=result.po_range_profile_power,
        rt_range_profile_power=result.rt_range_profile_power,
    )
    relative_permittivity = float(
        result.target_parameters["relative_permittivity"]
    )
    conductivity = float(result.target_parameters["conductivity_s_per_m"])
    thickness = float(result.target_parameters["thickness_m"])
    scattering = float(
        result.target_parameters["rt_diffuse_reflection_coefficient"]
    )
    scene_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<scene version="2.1.0">\n'
        '  <bsdf type="radio-material" id="target-material">\n'
        f'    <float name="relative_permittivity" value="{relative_permittivity:.9g}"/>\n'
        f'    <float name="conductivity" value="{conductivity:.9g}"/>\n'
        f'    <float name="thickness" value="{thickness:.9g}"/>\n'
        f'    <float name="scattering_coefficient" value="{scattering:.9g}"/>\n'
        '    <float name="xpd_coefficient" value="0"/>\n'
        "  </bsdf>\n"
        '  <shape type="obj" id="target">\n'
        '    <string name="filename" value="target.obj"/>\n'
        '    <boolean name="face_normals" value="true"/>\n'
        '    <ref name="bsdf" id="target-material"/>\n'
        "  </shape>\n"
        "</scene>\n"
    ).encode("utf-8")
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("bundle.json", _json_bytes(descriptor))
        bundle.writestr("sensor.json", _json_bytes(sensor))
        bundle.writestr("frames.json", _json_bytes(frames))
        bundle.writestr("radar_adc.npz", primary_adc)
        bundle.writestr("simulated_adc_po.npz", po_adc)
        bundle.writestr("simulated_adc_rt.npz", rt_adc)
        bundle.writestr("target_mesh.npz", target_mesh)
        bundle.writestr("target.obj", _obj_bytes(result.vertices, result.faces))
        bundle.writestr("scene.xml", scene_xml)
        bundle.writestr("environment.json", _json_bytes(environment))
        bundle.writestr(
            "experiment_manifest.json",
            _json_bytes(result.manifest.to_dict()),
        )
        bundle.writestr("solver_diagnostics.npz", diagnostics)
        bundle.writestr("bundle_summary.json", _json_bytes(summary))
        bundle.writestr(
            "README.txt",
            "HERMES static-target bundle\n"
            "radar_adc.npz is the primary PO raw ADC cube. RT and PO raw ADC "
            "cubes are also stored separately for comparison.\n"
            "No measured data or licensed body model is included.\n"
            f"Manifest SHA-256: {result.manifest.fingerprint}\n",
        )
    archive.seek(0)
    return archive


def _human_room_scene_xml(result) -> bytes:
    """Serialize the uploaded scene or the prepared room for a bundle."""

    custom_scene = getattr(result, "scene_xml", None)
    if custom_scene is None:
        custom_scene = getattr(result.preview, "scene_xml", None)
    if custom_scene is not None:
        return bytes(custom_scene)

    root = ET.fromstring(
        bedroom_scene_xml_for_boxes(result.preview.room_boxes)
    )
    human_material = ET.SubElement(
        root, "bsdf", {"type": "radio-material", "id": "human-material"}
    )
    for name, value in (
        ("relative_permittivity", "38"),
        ("conductivity", "1.5"),
        ("thickness", "0.01"),
    ):
        ET.SubElement(
            human_material, "float", {"name": name, "value": value}
        )
    human_shape = ET.SubElement(
        root, "shape", {"type": "obj", "id": "human-frame-0"}
    )
    ET.SubElement(
        human_shape,
        "string",
        {"name": "filename", "value": "target.obj"},
    )
    ET.SubElement(
        human_shape,
        "boolean",
        {"name": "face_normals", "value": "true"},
    )
    ET.SubElement(
        human_shape, "ref", {"name": "bsdf", "id": "human-material"}
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _human_room_export(result, *, mode_results=None) -> BytesIO:
    """Build a loadable bundle containing one or more room solver modes."""

    primary_mode = getattr(result, "simulation_mode", "hybrid_po")
    mode_results = dict(mode_results or {primary_mode: result})
    mode_results.setdefault(primary_mode, result)
    components = dict(result.components)
    rt_result = (
        mode_results.get("coherent_rt")
        or mode_results.get("full_rt")
    )
    po_result = mode_results.get("human_only_po")
    rt_adc = np.asarray(
        rt_result.adc
        if rt_result is not None
        else components.get(
            "static_environment_blocked",
            result.adc
            if primary_mode in ("full_rt", "coherent_rt")
            else np.zeros_like(result.adc),
        )
    )
    po_adc = np.asarray(
        po_result.adc
        if po_result is not None
        else components.get(
            "human_po",
            result.adc
            if primary_mode == "human_only_po"
            else np.zeros_like(result.adc),
        )
    )
    additional_simulations = {
        mode: f"simulated_adc_{mode}.npz"
        for mode in mode_results
    }
    primary_solver = (
        "rt" if primary_mode in ("full_rt", "coherent_rt") else "po"
    )
    adc_artifacts = {
        "primary": {
            "path": "radar_adc.npz",
            "origin": "simulation",
            "solver": primary_solver,
            "simulation_mode": primary_mode,
        },
        "rt": {
            "path": "simulated_adc_rt.npz",
            "origin": "simulation",
            "solver": "rt",
        },
        "po": {
            "path": "simulated_adc_po.npz",
            "origin": "simulation",
            "solver": "po",
        },
        **{
            mode: {
                "path": path,
                "origin": "simulation",
                "solver": (
                    "rt" if mode in ("full_rt", "coherent_rt") else "po"
                ),
                "simulation_mode": mode,
            }
            for mode, path in additional_simulations.items()
        },
    }
    descriptor = {
        "schema_version": 1,
        "profile": "hermes",
        "primary_adc": "primary",
        "adcs": adc_artifacts,
        "primary_solver": primary_solver,
        "target": {"mesh": "target_mesh.npz", "obj": "target.obj"},
        "motion": {"evaluated_mesh": "human_motion.npz"},
        "scene": {
            "file": "scene.xml",
            "assets": ["target.obj", "human_motion.npz"],
        },
        "environment": "environment.json",
        "manifests": {
            "primary": "experiment_manifest.json",
            **{
                mode: f"experiment_manifest_{mode}.json"
                for mode in mode_results
            },
        },
        "diagnostics": {"solver": "solver_diagnostics.npz"},
        "primary_simulation_mode": primary_mode,
    }
    motion_payload = getattr(result.preview, "motion_payload", None)
    if motion_payload is not None:
        descriptor["motion"]["parameters"] = "amass_sequence.npz"
    radar_parameters = dict(result.manifest.radar.parameters)
    scene_parameters = dict(result.manifest.scene.parameters)
    begin_frame_index = int(
        scene_parameters.get("motion_begin_frame_index", 0)
    )
    end_frame_index = int(
        scene_parameters.get(
            "motion_end_frame_index",
            begin_frame_index + int(result.adc.shape[0]) - 1,
        )
    )
    descriptor["motion"]["frame_window"] = {
        "begin_frame_index": begin_frame_index,
        "end_frame_index": end_frame_index,
    }
    tdm = bool(radar_parameters["tdm_enabled"])
    tx_indices = list(radar_parameters["tx_indices_zero_based"])
    rx_indices = list(radar_parameters["rx_indices_zero_based"])
    sensor = {
        "name": "human-room-radar",
        "board_model": result.manifest.radar.board_model,
        "position": [0.0, 0.0, 0.0],
        "orientation": result.radar_orientation_rad.tolist(),
        "orientation_deg": radar_parameters["orientation_deg"],
        "pattern_mode": radar_parameters["antenna_pattern_mode"],
        "cosine_3db_beamwidth_deg": radar_parameters[
            "cosine_3db_beamwidth_deg"
        ],
        "tdm_enabled": tdm,
        "tx_indices_zero_based": tx_indices,
        "rx_indices_zero_based": rx_indices,
        "virtual_channel_order": "tx_major",
        "fmcw": {
            "carrier_frequency_hz": radar_parameters[
                "carrier_frequency_hz"
            ],
            "slope_hz_per_s": radar_parameters["slope_hz_per_s"],
            "chirp_duration_s": radar_parameters["chirp_duration_s"],
            "chirp_repetition_time_s": radar_parameters[
                "chirp_repetition_time_s"
            ],
            "sampling_frequency_hz": radar_parameters[
                "sampling_frequency_hz"
            ],
            "adc_mode": radar_parameters.get("adc_mode", "complex1x"),
            "if_bandwidth_hz": radar_parameters.get("if_bandwidth_hz"),
            "num_adc_samples": radar_parameters["num_adc_samples"],
            "num_chirps_per_frame": radar_parameters[
                "num_chirps_per_frame"
            ],
            "frame_period_s": radar_parameters["frame_period_s"],
            "num_tx": len(tx_indices),
            "tdm_enabled": tdm,
        },
        "metadata": {
            "generated_by": "HERMES GUI",
            "scenario": "human-in-room",
            "primary_simulation_mode": primary_mode,
            "manifest_sha256": result.manifest.fingerprint,
        },
    }
    source_motion_times = None
    if motion_payload is not None:
        timing_key = (
            "bundle_times" if "bundle_times" in motion_payload else "times"
        )
        source_motion_times = np.asarray(
            motion_payload[timing_key],
            dtype=float,
        )
    frames = {
        "frames": [
            {
                "bundle_index": frame,
                "motion_time_s": float(
                    result.adc_times_s[frame, 0]
                    if source_motion_times is None
                    else source_motion_times[begin_frame_index + frame]
                ),
                "source_motion_frame_index": begin_frame_index + frame,
                "radar_frame_id": f"simulation-frame-{frame}",
                "metadata": {"prepared_room": result.preview.room_preset},
            }
            for frame in range(result.adc.shape[0])
        ]
    }
    environment = {
        "scene_path": "scene.xml",
        "coordinate_system": (
            "right-handed radar-relative frame; radar phase center is "
            "(0, 0, 0), +x is boresight at zero orientation, +y is left, "
            "and +z is up"
        ),
        "scene_editable": False,
        "prepared_scene": (
            result.preview.room_preset
            if getattr(result.preview, "scene_xml", None) is None
            else None
        ),
        "scene_source": getattr(
            result.preview,
            "scene_source_name",
            result.preview.room_preset,
        ),
        "target_geometry": "target_mesh.npz",
        "motion_mesh": "human_motion.npz",
        "human_position_m": result.preview.human_position_m.tolist(),
        "human_yaw_deg": result.preview.human_yaw_deg,
        "motion_coordinate_settings": getattr(
            result.preview, "motion_coordinate_settings", None
        ),
        "motion_frame_window": {
            "begin_frame_index": begin_frame_index,
            "end_frame_index": end_frame_index,
        },
    }
    if motion_payload is not None:
        environment["motion_parameters"] = "amass_sequence.npz"
    summary = {
        "profile": "hermes",
        "scenario": "human-in-room",
        "scene_source": getattr(
            result.preview,
            "scene_source_name",
            result.preview.room_preset,
        ),
        "motion_source": result.preview.source_name,
        "motion_frame_window": {
            "begin_frame_index": begin_frame_index,
            "end_frame_index": end_frame_index,
        },
        "primary_simulation_mode": primary_mode,
        "simulation_modes": list(mode_results),
        "mode_runtime_s": {
            mode: float(mode_result.runtime_s)
            for mode, mode_result in mode_results.items()
        },
        "adc_shape": list(map(int, result.adc.shape)),
        "components": sorted(components),
        "runtime_s": float(result.runtime_s),
        "manifest_sha256": result.manifest.fingerprint,
        "public_release_note": (
            "The bundle contains evaluated motion geometry and, when "
            "available, source AMASS-like parameters. Redistribution rights "
            "remain the responsibility of the bundle creator."
        ),
    }
    diagnostic_arrays = {
        "adc_if_filter_json": np.asarray(json.dumps(
            getattr(result.metadata, "adc_if_filter", None))),
        "ranges_m": np.asarray(result.ranges_m),
        **{
            f"range_profile_{name}": np.asarray(values)
            for name, values in result.range_profiles.items()
        },
    }
    for name in (
        "range_time_power",
        "range_time_ranges_m",
        "range_doppler_power",
        "range_doppler_ranges_m",
        "velocities_mps",
    ):
        value = getattr(result, name, None)
        if value is not None:
            diagnostic_arrays[name] = np.asarray(value)
    for name in (
        "path_counts",
        "visible_face_counts",
        "effective_visible_area_fraction",
        "static_path_counts",
        "blocked_static_path_counts",
        "human_env_path_counts",
        "env_human_path_counts",
        "path_depth_histogram",
    ):
        value = getattr(result.metadata, name, None)
        if value is not None:
            diagnostic_arrays[name] = np.asarray(value)

    sequence = result.preview.mesh_sequence
    first_vertices = np.asarray(
        sequence.vertices_at(float(sequence.times[0]))
    )
    archive = BytesIO()
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as bundle:
        bundle.writestr("bundle.json", _json_bytes(descriptor))
        bundle.writestr("sensor.json", _json_bytes(sensor))
        bundle.writestr("frames.json", _json_bytes(frames))
        bundle.writestr(
            "radar_adc.npz",
            _npz_bytes(adc=result.adc, times=result.adc_times_s),
        )
        bundle.writestr(
            "simulated_adc_rt.npz",
            _npz_bytes(adc=rt_adc, times=result.adc_times_s),
        )
        bundle.writestr(
            "simulated_adc_po.npz",
            _npz_bytes(adc=po_adc, times=result.adc_times_s),
        )
        for mode, mode_result in mode_results.items():
            bundle.writestr(
                additional_simulations[mode],
                _npz_bytes(
                    adc=mode_result.adc,
                    times=mode_result.adc_times_s,
                ),
            )
            bundle.writestr(
                f"experiment_manifest_{mode}.json",
                _json_bytes(mode_result.manifest.to_dict()),
            )
        bundle.writestr(
            "target_mesh.npz",
            _npz_bytes(
                vertices=first_vertices,
                faces=sequence.faces,
                times=np.asarray([0.0]),
            ),
        )
        bundle.writestr(
            "human_motion.npz",
            _npz_bytes(
                vertices=sequence.vertices,
                faces=sequence.faces,
                times=sequence.times,
            ),
        )
        if motion_payload is not None:
            bundle.writestr(
                "amass_sequence.npz",
                _npz_bytes(**dict(motion_payload)),
            )
        bundle.writestr(
            "target.obj",
            _obj_bytes(first_vertices, sequence.faces),
        )
        bundle.writestr("scene.xml", _human_room_scene_xml(result))
        bundle.writestr("environment.json", _json_bytes(environment))
        bundle.writestr(
            "experiment_manifest.json",
            _json_bytes(result.manifest.to_dict()),
        )
        bundle.writestr(
            "solver_diagnostics.npz",
            _npz_bytes(**diagnostic_arrays),
        )
        bundle.writestr("bundle_summary.json", _json_bytes(summary))
        bundle.writestr(
            "README.txt",
            "HERMES prepared-room human simulation bundle\n"
            f"radar_adc.npz contains the {primary_mode} raw ADC cube. "
            "Each selected mode is also stored in a simulated_adc_<mode>.npz "
            "archive; simulated_adc_rt.npz and simulated_adc_po.npz retain "
            "the HERMES comparison aliases.\n"
            "human_motion.npz contains the evaluated mesh sequence. When "
            "present, amass_sequence.npz contains the source AMASS-like pose "
            "parameters; licensed SMPL model files are not embedded.\n"
            f"Manifest SHA-256: {result.manifest.fingerprint}\n",
        )
    archive.seek(0)
    return archive


def _source_fixture_options() -> dict[str, str]:
    public_root = Path(__file__).resolve().parents[3]
    fixture_root = public_root / "data" / "validation_bundles"
    options = {"Custom path": ""}
    for name in (
        "rtpose_seq10_frame0020",
        "mmradarpose_p1_an0_ac4_r0_frame0240",
    ):
        path = fixture_root / name
        if path.is_dir():
            options[name] = str(path)
    return options


def _default_human_mesh_path() -> Path | None:
    """Return the internal RT-Pose demo mesh in a full source checkout."""

    repository_root = Path(__file__).resolve().parents[4]
    candidate = (
        repository_root
        / "internal"
        / "data"
        / "benchmark_clips"
        / "real_radar"
        / "RT-pose"
        / "rtpose_seq186_frame0020"
        / "mesh_sequence.npz"
    )
    return candidate if candidate.is_file() else None


def _default_amass_motion_path() -> Path | None:
    """Return the redistributable AMASS-like GUI motion fixture."""

    public_root = Path(__file__).resolve().parents[3]
    candidate = (
        public_root
        / "data"
        / "AMASS"
        / "walking_poses_cmu_105_02.npz"
    )
    return candidate if candidate.is_file() else None


def _default_smpl_model_dir() -> Path:
    """Resolve the licensed model directory without exporting its path."""

    configured = os.environ.get("MMWAVE_SMPL_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    public_root = Path(__file__).resolve().parents[3]
    return public_root / "models" / "smpl_models"


def build_app(
    *,
    initial_bundle: str | None = None,
    remote_access: bool = False,
):
    """Build the HERMES app, restricting server paths for remote sessions."""

    pn, go, make_subplots = _gui_modules()
    remote_access = bool(remote_access)
    trusted_bundle_paths: set[str] = set()
    trusted_simulated_paths: set[str] = set()
    uploaded_bundle_paths: set[str] = set()
    uploaded_simulated_paths: set[str] = set()
    pn.extension(
        "plotly",
        "filedropper",
        notifications=True,
        sizing_mode="stretch_width",
    )
    if _APP_CSS not in pn.config.raw_css:
        pn.config.raw_css.append(_APP_CSS)
    plot_template = _plot_template(pn.state.session_args)
    dark_theme = plot_template == "plotly_dark"
    theme_scope_class = (
        "hermes-theme-dark" if dark_theme else "hermes-theme-light"
    )
    theme_button = pn.widgets.Button(
        label="",
        icon="sun" if dark_theme else "moon",
        icon_size="30px",
        description=(
            "Switch to light theme" if dark_theme else "Switch to dark theme"
        ),
        width=52,
        height=44,
        margin=(0, 4),
        css_classes=["hermes-theme-button", "hermes-icon-theme"],
    )
    settings_button = pn.widgets.Button(
        label="",
        icon="settings",
        icon_size="30px",
        description="Radar and solver settings",
        width=52,
        height=44,
        margin=(0, 4),
        css_classes=["hermes-theme-button", "hermes-icon-settings"],
    )
    gui_activity_state = {
        "static": False,
        "dynamic": False,
        "comparison": False,
        "bundle_load": False,
    }
    settings_dialog_state = {"editing": False, "values": {}, "phase_spans": {}}

    def set_gui_activity(name: str, active: bool) -> None:
        gui_activity_state[name] = bool(active)
        theme_button.disabled = any(gui_activity_state.values())
        settings_button.disabled = any(gui_activity_state.values())
        load_motion_preview.disabled = any(gui_activity_state.values())
        refresh_result_validity()
    po_integration_mode = pn.widgets.Select(
        label="PO facet integration",
        options={
            "Adaptive analytic parent facet": "parent_face_adaptive_analytic",
            "Adaptive parent-facet quadrature": "parent_face_quadrature",
            "Far-field analytic parent facet": (
                "parent_face_far_field_analytic"
            ),
            "Face centroid (fast approximation)": "face_centroid",
        },
        value=_SOLVER_SETTINGS_DEFAULTS["po_integration_mode"],
    )
    po_phase_span = pn.widgets.FloatInput(
        label="PO subdivision phase span [rad]",
        description="Maximum linear phase span Δφmax for adaptive analytic facets; legacy quadrature uses its sampled phase span.",
        value=_SOLVER_SETTINGS_DEFAULTS[
            "po_quadrature_phase_span_scale_rad"
        ],
        format="0.[000]",
        step=0.1,
        start=0.05,
        end=20.0,
    )
    po_curvature = pn.widgets.FloatInput(
        label="PO curvature tolerance [rad]",
        description="Split when the edge-midpoint path error δT = kc/3 × max|L(midpoint) − mean L(endpoints)| exceeds εφ.",
        value=_SOLVER_SETTINGS_DEFAULTS["po_curvature_error_rad"],
        format="0.[000000]", step=0.001, start=0.000001, end=10.0,
    )

    # Quadrature samples the phase while analytic facets integrate it, so a
    # span suited to one mode is not carried into the other.
    phase_span_by_mode = {po_integration_mode.value: po_phase_span.value}
    solver_settings_state = {"restoring": False}

    def on_po_integration_mode_change(_event=None):
        if (_event is not None and _event.old != _event.new
                and not solver_settings_state["restoring"]):
            phase_span_by_mode[_event.old] = po_phase_span.value
            po_phase_span.value = phase_span_by_mode.get(
                _event.new,
                default_phase_span_scale_rad(_event.new),
            )
        po_curvature.disabled = po_integration_mode.value != "parent_face_adaptive_analytic"
        po_phase_span.disabled = po_integration_mode.value not in {
            "parent_face_adaptive_analytic", "parent_face_quadrature",
        }

    po_integration_mode.param.watch(on_po_integration_mode_change, "value")
    on_po_integration_mode_change()
    po_refinement_depth = pn.widgets.IntInput(
        label="PO maximum refinement depth",
        value=_SOLVER_SETTINGS_DEFAULTS[
            "po_quadrature_max_refinement_depth"
        ],
        step=1,
        start=0,
        end=16,
    )
    po_subface_cap = pn.widgets.IntInput(
        label="PO subfaces per parent (0 = unlimited)",
        value=_SOLVER_SETTINGS_DEFAULTS[
            "po_quadrature_max_subfaces_per_parent"
        ],
        step=4,
        start=0,
        end=4096,
    )
    coupling_power_cutoff = pn.widgets.FloatInput(
        label="Image-source power cutoff [dB relative to strongest]",
        description=("Retain reflectors at or above this estimated power level toward the human motion volume. "
                     "-20 dB means at least 1% of the strongest candidate's estimated power."),
        value=_SOLVER_SETTINGS_DEFAULTS["coupling_power_threshold_db"],
        step=5., start=-120., end=0.,
    )
    coupling_max_count = pn.widgets.IntInput(
        label="Image-source maximum count",
        description="Keep at most this many of the strongest reflectors passing the power cutoff (maximum 4).",
        value=_SOLVER_SETTINGS_DEFAULTS["coupling_max_reflectors"],
        step=1, start=1, end=4,
    )
    rt_samples = pn.widgets.IntInput(
        label="RT ray samples per source",
        description=_RT_SOLVER_SETTING_HELP["rt_samples_per_source"],
        value=_SOLVER_SETTINGS_DEFAULTS["rt_samples_per_source"],
        step=500,
        start=100,
        end=2_000_000,
    )
    rt_path_cap = pn.widgets.IntInput(
        label="RT maximum paths per source",
        description=_RT_SOLVER_SETTING_HELP["rt_max_paths_per_source"],
        value=_SOLVER_SETTINGS_DEFAULTS["rt_max_paths_per_source"],
        step=500,
        start=100,
        end=2_000_000,
    )
    rt_depth = pn.widgets.IntInput(
        label="RT maximum path depth",
        description=_RT_SOLVER_SETTING_HELP["rt_max_depth"],
        value=_SOLVER_SETTINGS_DEFAULTS["rt_max_depth"],
        step=1,
        start=1,
        end=10,
    )
    apply_solver_settings = pn.widgets.Button(
        label="Apply",
        icon="check",
        color="primary",
        width=120,
        height=40,
        sizing_mode="fixed",
    )
    cancel_solver_settings = pn.widgets.Button(
        label="Cancel", color="light", width=110, height=40, sizing_mode="fixed",
    )
    rerun_after_apply = pn.widgets.Checkbox(
        label="Apply & rerun this page", value=False, align="center",
    )
    solver_setting_widgets = {
        "po_integration_mode": po_integration_mode,
        "po_quadrature_phase_span_scale_rad": po_phase_span,
        "po_curvature_error_rad": po_curvature,
        "po_quadrature_max_refinement_depth": po_refinement_depth,
        "po_quadrature_max_subfaces_per_parent": po_subface_cap,
        "coupling_power_threshold_db": coupling_power_cutoff,
        "coupling_max_reflectors": coupling_max_count,
        "rt_samples_per_source": rt_samples,
        "rt_max_paths_per_source": rt_path_cap,
        "rt_max_depth": rt_depth,
    }

    def current_solver_settings_payload() -> dict[str, object]:
        return {
            "schema_version": _SOLVER_SETTINGS_SCHEMA_VERSION,
            "po_defaults_version": _PO_DEFAULTS_VERSION,
            "po_phase_span_by_mode": {
                **phase_span_by_mode,
                po_integration_mode.value: po_phase_span.value,
            },
            "settings": {
                name: widget.value
                for name, widget in solver_setting_widgets.items()
            },
        }

    def restore_solver_settings(payload):
        restored = _validated_solver_settings_payload(payload)
        spans = _validated_po_phase_span_by_mode(payload)
        if not restored and not spans:
            return restored, False
        if payload.get("po_defaults_version") != _PO_DEFAULTS_VERSION:
            phase_span_by_mode.clear()
        mode = restored.get("po_integration_mode", po_integration_mode.value)
        if "po_integration_mode" in restored or mode in spans:
            restored.setdefault(
                "po_quadrature_phase_span_scale_rad",
                spans.get(mode, phase_span_by_mode.get(
                    mode, default_phase_span_scale_rad(mode))),
            )
        changed = False
        solver_settings_state["restoring"] = True
        try:
            for name, value in restored.items():
                widget = solver_setting_widgets[name]
                if widget.value != value:
                    # Keep normal widget events so browser models update.
                    widget.value = value
                    changed = True
        finally:
            solver_settings_state["restoring"] = False
        phase_span_by_mode.update(spans)
        # The active scalar also restores profiles saved before the map existed.
        phase_span_by_mode[po_integration_mode.value] = po_phase_span.value
        return restored, changed

    solver_settings_storage = _solver_settings_storage_component(pn)
    motion_axis_options = {
        f"{sign}{axis.upper()}": f"{sign}{axis}"
        for axis in "xyz"
        for sign in ("+", "-")
    }
    motion_axes_mode = pn.widgets.Select(
        label="Motion coordinate mapping",
        css_classes=["hermes-settings-wide"],
        options={
            "Use file mapping / keep stored axes": "file",
            "Specify source coordinate frame": "manual",
        },
        value="file",
    )
    motion_up_axis = pn.widgets.Select(
        label="Source up axis",
        options=motion_axis_options,
        value="+z",
    )
    motion_forward_axis = pn.widgets.Select(
        label="Source forward axis",
        options=motion_axis_options,
        value="+x",
        description="Fixed reference direction, not the subject's changing heading.",
    )
    motion_handedness = pn.widgets.Select(
        label="Source handedness",
        options={"Right-handed": "right", "Left-handed": "left"},
        value="right",
    )
    motion_length_unit = pn.widgets.Select(
        label="Evaluated mesh units",
        options={"Metres": "m", "Centimetres": "cm", "Millimetres": "mm"},
        value="m",
        description="SMPL-generated meshes normally use metres.",
    )
    motion_coordinate_summary = pn.pane.Markdown(css_classes=["hermes-callout"])
    motion_coordinate_widgets = {
        "motion_axes_mode": motion_axes_mode,
        "motion_up_axis": motion_up_axis,
        "motion_forward_axis": motion_forward_axis,
        "motion_handedness": motion_handedness,
        "motion_length_unit": motion_length_unit,
    }
    motion_coordinate_state = {"syncing": False}

    def current_motion_coordinate_arguments():
        if motion_axes_mode.value == "file":
            return {}
        return {
            "motion_source_up_axis": motion_up_axis.value,
            "motion_source_forward_axis": motion_forward_axis.value,
            "motion_source_handedness": motion_handedness.value,
            "motion_source_length_unit": motion_length_unit.value,
        }

    def sync_motion_coordinate_controls():
        motion_coordinate_state["syncing"] = True
        try:
            motion_forward_axis.options = {
                label: value
                for label, value in motion_axis_options.items()
                if value[-1] != motion_up_axis.value[-1]
            }
            manual = motion_axes_mode.value == "manual"
            for widget in (
                motion_up_axis, motion_forward_axis,
                motion_handedness, motion_length_unit,
            ):
                widget.disabled = not manual
            if not manual:
                motion_coordinate_summary.object = (
                    "File `output_axes` / `output_signs` are used when present. "
                    "Otherwise axes and scale stay as stored; no frame is guessed."
                )
                return
            transform = motion_source_to_radar_transform(
                motion_up_axis.value,
                motion_forward_axis.value,
                handedness=motion_handedness.value,
                length_unit=motion_length_unit.value,
            )
            axes = np.argmax(np.abs(transform), axis=1)
            signs = np.sign(transform[np.arange(3), axes])
            mapping = " · ".join(
                f"GUI {target.upper()} ← {'+' if sign > 0 else '-'}{'XYZ'[axis]} source"
                for target, axis, sign in zip("xyz", axes, signs)
            )
            motion_coordinate_summary.object = (
                f"**{mapping}**  \n"
                "Converted to metres, then human yaw and position are applied. "
                "This selection overrides any file axis mapping."
            )
        finally:
            motion_coordinate_state["syncing"] = False

    sync_motion_coordinate_controls()
    solver_controls_panel = pn.Column(
        _settings_note(
            pn,
            "These values control the numerical PO integration and RT search. "
            "Adaptive PO subdivision is temporary and is coherently integrated "
            "back onto each original facet. Solver values are saved in this "
            "browser and restored on the next visit.",
        ),
        _settings_section(
            pn, "Physical optics", "Static Target · Dynamic Scenes",
            _settings_grid(
                pn,
                po_integration_mode,
                po_phase_span,
                po_curvature,
                po_refinement_depth,
                po_subface_cap,
                columns=3,
            ),
        ),
        _settings_section(
            pn, "Ray tracing", "Static Target · Dynamic Scenes",
            _settings_grid(pn, rt_samples, rt_path_cap, rt_depth, columns=3),
        ),
        _settings_section(
            pn, "Image-source reflectors", "Dynamic Scenes · Hybrid PO",
            _settings_grid(pn, coupling_power_cutoff, coupling_max_count, columns=3),
        ),
        sizing_mode="stretch_width",
        margin=0,
    )
    motion_controls_panel = pn.Column(
        _settings_note(
            pn,
            "Map source coordinates to **+X forward, +Y left, +Z up**, in "
            "metres. Apply updates the loaded motion preview. Motion "
            "coordinates are kept with the current GUI session and recorded "
            "in exported runs.",
        ),
        _settings_section(
            pn, "Source coordinate frame", "Dynamic Scenes",
            _settings_grid(pn, motion_axes_mode, columns=4),
            _settings_grid(
                pn,
                motion_up_axis,
                motion_forward_axis,
                motion_handedness,
                motion_length_unit,
                columns=4,
            ),
            motion_coordinate_summary,
        ),
        sizing_mode="stretch_width",
        margin=0,
    )

    target_type = pn.widgets.Select(
        label="Target type",
        options={
            "Plate": "plate",
            "Trihedral corner reflector": "trihedral",
            "Human mesh": "human_mesh",
        },
        value="plate",
    )
    def compact_numeric(label, *, value, start, end, step, format="0.[0]", description=None):
        return pn.widgets.FloatInput(
            label=label, value=value, start=start, end=end, step=step,
            format=format, description=description,
            sizing_mode="stretch_width", min_width=0, margin=(5, 0),
        )

    def numeric_row(css_class, *widgets):
        return pn.Row(
            *widgets, sizing_mode="stretch_width", min_width=0,
            margin=(0, 10), styles={"gap": "10px"}, css_classes=[css_class],
        )

    target_range = compact_numeric(
        "X [m]", start=0.5, end=5.0, step=0.1, value=2.0,
        format="0.00", description="Range-forward position.",
    )
    target_y = compact_numeric(
        "Y [m]", start=-2.0, end=2.0, step=0.05, value=0.0,
        format="0.00", description="Lateral-left position.",
    )
    target_z = compact_numeric(
        "Z [m]", start=-1.5, end=2.5, step=0.05, value=0.0,
        format="0.00", description="Vertical-up position.",
    )
    plate_width = compact_numeric(
        "Width [m]", value=0.30, format="0.00", step=0.01, start=0.02, end=3.0,
    )
    plate_height = compact_numeric(
        "Height [m]", value=0.30, format="0.00", step=0.01, start=0.02, end=3.0,
    )
    corner_edge = pn.widgets.FloatInput(
        label="Corner edge length [m]",
        value=0.20,
        format="0.00",
        step=0.01,
        start=0.02,
        end=2.0,
        visible=False,
    )
    yaw = compact_numeric("Yaw [deg]", start=-75, end=75, step=1, value=0)
    pitch = compact_numeric("Pitch [deg]", start=-75, end=75, step=1, value=0)
    roll = compact_numeric("Roll [deg]", start=-180, end=180, step=1, value=0)
    radar_yaw = compact_numeric("Yaw [deg]", start=-180, end=180, step=1, value=0)
    radar_pitch = compact_numeric("Pitch [deg]", start=-90, end=90, step=1, value=0)
    radar_roll = compact_numeric("Roll [deg]", start=-180, end=180, step=1, value=0)
    dynamic_radar_yaw = compact_numeric("Yaw [deg]", start=-180, end=180, step=1, value=0)
    dynamic_radar_pitch = compact_numeric("Pitch [deg]", start=-90, end=90, step=1, value=0)
    dynamic_radar_roll = compact_numeric("Roll [deg]", start=-180, end=180, step=1, value=0)
    material = pn.widgets.Select(
        label="Material preset",
        options={
            "PEC · ideal PO, rough/diffuse RT": "pec",
            "Aluminum · smooth/specular": "aluminum",
            "Concrete · rough/diffuse": "concrete",
        },
        value=MATERIAL_PRESETS[0],
    )
    material_details = pn.pane.Markdown()

    def update_material_details():
        parameters = material_preset_parameters(material.value)
        coefficient = float(parameters["diffuse_coefficient"])
        if coefficient == 0.0:
            material_details.object = (
                "**RT diffuse scattering coefficient:** `0.00`  \n"
                "Diffuse RT paths are disabled for this smooth-surface preset. "
                "PO still uses its electromagnetic material model."
            )
        else:
            backscattering_lambda = float(
                parameters["backscattering_lambda"]
            )
            material_details.object = (
                f"**RT diffuse scattering coefficient:** `{coefficient:.2f}`  \n"
                "**Backscattering lobe mixture λ:** "
                f"`{backscattering_lambda:.2f}`  \n"
                "These RT rough-surface terms are independent of the PO "
                "electromagnetic material model."
            )

    update_material_details()
    human_upload = pn.widgets.FileInput(
        label="Human mesh (.obj or .npz)",
        accept=".obj,.npz",
        stylesheets=[_file_input_stylesheet(plot_template, compact=False)],
        description="Browser uploads are limited to 64 MiB.",
        visible=False,
    )
    human_diffuse = pn.widgets.FloatSlider(
        label="Human diffuse reflection coefficient (RT)",
        start=0.0,
        end=1.0,
        step=0.05,
        value=0.35,
        visible=False,
    )
    # Never let a remotely connected browser implicitly select or export a
    # participant-derived mesh from the private source checkout.
    default_human_mesh = None if remote_access else _default_human_mesh_path()
    human_source = pn.pane.Markdown(
        (
            "**Default mesh:** regenerated RT-Pose seq186 frame0020 "
            "`mesh_sequence.npz` (internal). Upload a different `.obj` or "
            "`.npz` above to override it."
            if default_human_mesh is not None
            else "**No default mesh is installed.** Upload an `.obj` or `.npz`."
        ),
        visible=False,
    )
    board = pn.widgets.Select(
        label="TI radar",
        options=list(available_ti_boards()),
        value="IWR6843AOPEVM",
    )
    board_spec = get_ti_board_spec(board.value)
    tdm_enabled = pn.widgets.Checkbox(
        label="Enable TDM-MIMO",
        value=False,
    )
    tx_antennas = pn.widgets.MultiChoice(
        label="Active Tx antennas",
        options={
            f"TX{index + 1}": index for index in range(board_spec.num_tx)
        },
        value=list(range(board_spec.num_tx)),
    )
    rx_antennas = pn.widgets.MultiChoice(
        label="Active Rx antennas",
        options={
            f"RX{index + 1}": index for index in range(board_spec.num_rx)
        },
        value=list(range(board_spec.num_rx)),
    )
    antenna_pattern = pn.widgets.Select(
        label="Antenna pattern",
        options={
            "Digitized TI profile": "digitized",
            "Omnidirectional": "none",
            "Cosine": "cosine",
        },
        value="cosine",
    )
    cosine_3db_beamwidth = pn.widgets.FloatInput(
        label="Cosine 3 dB beamwidth [deg]",
        value=60.0,
        step=1.0,
        start=1.0,
        end=179.0,
    )
    antenna_pattern_note = pn.pane.Markdown()
    carrier_frequency = pn.widgets.FloatInput(
        label="Carrier frequency [GHz]",
        value=_rounded_gui_float(board_spec.design_frequency_hz / 1e9),
        format="0.[000]",
        step=0.1,
        start=50.0,
        end=90.0,
    )
    chirp_slope = pn.widgets.FloatInput(
        label="Chirp slope [MHz/µs]",
        value=68.0,
        step=0.1,
        start=0.1,
        end=200.0,
    )
    chirp_duration = pn.widgets.FloatInput(
        label="Chirp duration [µs]",
        value=58.0,
        step=1.0,
        start=1.0,
        end=1000.0,
    )
    chirp_repetition = pn.widgets.FloatInput(
        label="Chirp repetition time [µs]",
        value=65.0,
        step=1.0,
        start=1.0,
        end=2000.0,
    )
    sampling_frequency = pn.widgets.FloatInput(
        label="ADC sampling rate [MS/s]",
        value=4.5,
        step=0.1,
        start=0.1,
        end=50.0,
    )
    adc_sampling_mode = pn.widgets.Select(
        label="ADC sampling mode", value="complex1x",
        options={"Complex 1x (0–fs)": "complex1x",
                 "Complex 2x (0–fs/2)": "complex2x", "Real (0–fs/2)": "real"},
    )
    adc_if_bandwidth = pn.widgets.FloatInput(
        label="IF bandwidth cap [MHz] (0 = board default)", value=0.,
        start=0., end=50., step=.1,
    )
    num_adc_samples = pn.widgets.IntInput(
        label="ADC samples per chirp",
        value=225,
        step=1,
        start=16,
        end=4096,
    )
    num_chirps = pn.widgets.IntInput(
        label="Chirps per frame",
        value=16,
        step=1,
        start=1,
        end=4096,
    )
    frame_period = pn.widgets.FloatInput(
        label="Frame period [ms]",
        value=50.0,
        step=1.0,
        start=1.0,
        end=10_000.0,
    )
    load_radar_defaults = pn.widgets.Button(
        label="Load default FMCW values",
        color="light",
        icon="refresh",
    )
    radar_summary = pn.pane.Markdown()
    ignored_radar_events: dict[int, int] = {}

    def update_antenna_pattern_controls():
        pattern_mode = antenna_pattern.value
        cosine_3db_beamwidth.visible = pattern_mode == "cosine"
        if pattern_mode == "cosine":
            antenna_pattern_note.object = (
                "The specified width is the full **power-pattern 3 dB "
                "beamwidth**, centered on radar local +x."
            )
        elif pattern_mode == "digitized":
            antenna_pattern_note.object = (
                "Uses the packaged per-channel TI pattern profile."
                if ti_digitized_pattern_asset_available()
                else (
                    "**Digitized profile unavailable:** install "
                    "`ti_digitized_patterns.npz` in the radar data package "
                    "before running this mode."
                )
            )
        else:
            antenna_pattern_note.object = (
                "Uses the board's scalar gain uniformly in every direction."
            )

    def update_radar_summary():
        selected_tx = [int(value) for value in tx_antennas.value]
        selected_rx = [int(value) for value in rx_antennas.value]
        active_tx_indices = selected_tx
        active_tx = len(active_tx_indices)
        active_rx = len(selected_rx)
        bandwidth_hz = (float(chirp_slope.value) * 1e12
                        * int(num_adc_samples.value)
                        / (float(sampling_frequency.value) * 1e6))
        range_resolution_m = 299_792_458.0 / (2.0 * bandwidth_hz)
        pattern_summary = {
            "digitized": "digitized TI profile",
            "none": "omnidirectional",
            "cosine": (
                f"cosine, {float(cosine_3db_beamwidth.value):.1f}° "
                "3 dB width"
            ),
        }[antenna_pattern.value]
        board_if = get_ti_board_spec(board.value).adc.get("if_bandwidth_hz")
        requested_if = float(adc_if_bandwidth.value)*1e6 or board_if
        effective_if = if_cutoff_hz(SimpleNamespace(
            adc_mode=adc_sampling_mode.value,
            sampling_frequency=float(sampling_frequency.value)*1e6,
            if_bandwidth_hz=min(requested_if, board_if) if board_if else requested_if))
        if_range_m = 299_792_458.*effective_if/(2.*abs(float(chirp_slope.value)*1e12))
        radar_summary.object = (
            f"**{active_tx} Tx × {active_rx} Rx · "
            f"{active_tx * active_rx} channels**  \n"
            f"Active `TX {[value + 1 for value in active_tx_indices]}` · "
            f"`RX {[value + 1 for value in selected_rx]}`  \n"
            f"Sampled bandwidth `{bandwidth_hz / 1e9:.3f} GHz` · range "
            f"resolution `{range_resolution_m * 100:.2f} cm`  \n"
            f"Usable IF `{effective_if/1e6:.3f} MHz` · maximum range `{if_range_m:.2f} m`  \n"
            f"Antenna `{pattern_summary}`"
        )

    def set_radar_defaults():
        selected_spec = get_ti_board_spec(board.value)
        antenna_defaults = _RADAR_ACTIVE_ANTENNA_DEFAULTS.get(
            selected_spec.key,
            {
                "tx": tuple(range(selected_spec.num_tx)),
                "rx": tuple(range(selected_spec.num_rx)),
            },
        )
        default_tx = list(antenna_defaults["tx"])
        default_rx = list(antenna_defaults["rx"])
        defaults = {
            carrier_frequency: _rounded_gui_float(
                selected_spec.design_frequency_hz / 1e9
            ),
            chirp_slope: 68.0,
            chirp_duration: 58.0,
            chirp_repetition: 65.0,
            sampling_frequency: 4.5,
            adc_sampling_mode: "complex1x",
            adc_if_bandwidth: 0.,
            num_adc_samples: 225,
            num_chirps: 16,
            frame_period: 50.0,
            tdm_enabled: False,
            tx_antennas: default_tx,
            rx_antennas: default_rx,
        }
        for selector, prefix, count, desired in (
            (tx_antennas, "TX", selected_spec.num_tx, default_tx),
            (rx_antennas, "RX", selected_spec.num_rx, default_rx),
        ):
            if list(selector.value) != desired:
                ignored_radar_events[id(selector)] = (
                    ignored_radar_events.get(id(selector), 0) + 1
                )
            selector.options = {
                f"{prefix}{index + 1}": index for index in range(count)
            }
        for widget, value in defaults.items():
            if widget.value != value:
                if widget not in (tx_antennas, rx_antennas):
                    ignored_radar_events[id(widget)] = (
                        ignored_radar_events.get(id(widget), 0) + 1
                    )
                widget.value = value
        update_radar_summary()

    update_antenna_pattern_controls()
    update_radar_summary()
    refine_physics = pn.widgets.Button(
        label="Recompute at higher fidelity",
        color="primary",
        icon="sparkles",
        width=250,
        height=40,
        sizing_mode="fixed",
        description=(
            "Increase persistent mesh density and PO visibility sampling."
        ),
    )
    stop_physics = pn.widgets.Button(
        label="Stop",
        color="light",
        icon="player-stop",
        width=105,
        height=40,
        sizing_mode="fixed",
        description="Stop the active static RT/PO run at a safe boundary.",
        disabled=True,
    )
    initial_result = run_static_target_experiment(
        target_type=target_type.value,
        yaw_deg=yaw.value,
        pitch_deg=pitch.value,
        roll_deg=roll.value,
        target_y_m=target_y.value,
        target_z_m=target_z.value,
        radar_yaw_deg=radar_yaw.value,
        radar_pitch_deg=radar_pitch.value,
        radar_roll_deg=radar_roll.value,
        range_m=target_range.value,
        width_m=plate_width.value,
        height_m=plate_height.value,
        material_name=material.value,
        fidelity="preview",
        board_model=board.value,
        carrier_frequency_hz=carrier_frequency.value * 1e9,
        slope_hz_per_s=chirp_slope.value * 1e12,
        chirp_duration_s=chirp_duration.value * 1e-6,
        chirp_repetition_time_s=chirp_repetition.value * 1e-6,
        sampling_frequency_hz=sampling_frequency.value * 1e6,
        adc_mode=adc_sampling_mode.value,
        if_bandwidth_hz=adc_if_bandwidth.value * 1e6 or None,
        num_adc_samples=num_adc_samples.value,
        num_chirps_per_frame=num_chirps.value,
        frame_period_s=frame_period.value * 1e-3,
        tdm_enabled=tdm_enabled.value,
        selected_tx_indices=tuple(tx_antennas.value),
        selected_rx_indices=tuple(rx_antennas.value),
        antenna_pattern_mode=antenna_pattern.value,
        cosine_3db_beamwidth_deg=cosine_3db_beamwidth.value,
        po_integration_mode=po_integration_mode.value,
        po_quadrature_phase_span_scale_rad=po_phase_span.value,
        po_curvature_error_rad=po_curvature.value,
        po_quadrature_max_refinement_depth=po_refinement_depth.value,
        po_quadrature_max_subfaces_per_parent=po_subface_cap.value,
        rt_samples_per_source=rt_samples.value,
        rt_max_paths_per_source=rt_path_cap.value,
        rt_max_depth=rt_depth.value,
    )
    physics_scene = pn.pane.Plotly(
        _physics_scene_figure(
            initial_result,
            go,
            plot_template=plot_template,
        ),
        height=500,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            # Plotly's "default" camera is its generic +x/+y/+z view, not this
            # application's saved radar-frame view. Keep only "last save",
            # which returns to the actual initial camera.
            "modeBarButtonsToRemove": ["resetCameraDefault3d"],
        },
    )
    scene_interaction = _scene_interaction_component(pn, physics_scene)
    physics_profile = pn.pane.Plotly(
        _range_profile_figure(
            initial_result,
            go,
            plot_template=plot_template,
        ),
        height=310,
    )
    initial_angle_products = _angle_fft_range_products(initial_result)
    initial_centroid_bin = int(initial_angle_products["centroid_range_bin"])
    angle_range_bins = pn.widgets.IntRangeSlider(
        label="Angle-FFT range bins (inclusive)",
        start=0,
        end=len(initial_angle_products["ranges_m"]) - 1,
        step=1,
        value=(initial_centroid_bin, initial_centroid_bin),
        sizing_mode="stretch_width",
    )
    angle_map = pn.pane.Plotly(
        _angle_map_figure(
            initial_angle_products,
            make_subplots,
            go,
            range_bins=angle_range_bins.value,
            plot_template=plot_template,
        ),
        height=420,
    )
    angle_status = pn.pane.Markdown()

    def update_angle_status(products, range_bins):
        ranges_m = np.asarray(products["ranges_m"], dtype=float)
        first_bin, last_bin = sorted(int(value) for value in range_bins)
        selected = (
            f"bin `{first_bin}` at `{ranges_m[first_bin]:.3f} m`"
            if first_bin == last_bin
            else (
                f"bins `{first_bin}–{last_bin}` at "
                f"`{ranges_m[first_bin]:.3f}–{ranges_m[last_bin]:.3f} m`"
            )
        )
        angle_status.object = (
            f"**Selected:** {selected} · target centroid "
            f"`{float(products['centroid_range_m']):.3f} m` "
            f"(nearest bin `{int(products['centroid_range_bin'])}`)"
        )

    update_angle_status(initial_angle_products, angle_range_bins.value)
    physics_status = pn.pane.Markdown(
        f"**LIVE RT + PO** · {initial_result.faces.shape[0]:,} facets · "
        f"{initial_result.rt_path_count:,} RT paths · "
        f"PO {initial_result.po_runtime_s:.3f} s · "
        f"RT {initial_result.rt_runtime_s:.3f} s",
        css_classes=["hermes-status"],
    )
    physics_download = pn.widgets.FileDownload(
        callback=lambda: _physics_export(initial_result),
        filename="hermes-static-target-bundle.zip",
        label="Export loadable bundle",
        color="light",
    )
    physics_state = {
        "result": initial_result,
        "refined": False,
        "revision": 0,
        "target_result_sequence": 0,
        "running": False,
        "pending_request": None,
        "cancel_event": None,
    }
    angle_state = {
        "products": initial_angle_products,
        "revision": 0,
    }
    diagnostic_top_k = pn.widgets.IntInput(
        label="Top RT paths",
        value=12,
        start=1,
        end=100,
        step=1,
        width=135,
    )
    show_rt_paths = pn.widgets.Toggle(
        label="Show",
        icon="route",
        color="primary",
        width=100,
        align="end",
    )
    diagnostics_status = pn.pane.Markdown(
        "RT path geometry is off. Enable it to run one diagnostic trace and "
        "compute the interaction-depth histogram; ADC synthesis is not rerun.",
        css_classes=["hermes-status"],
    )
    diagnostics_depth = pn.pane.Plotly(height=300, visible=False)
    diagnostics_summary = pn.pane.Markdown(
        f"**RT valid link-paths:** `{initial_result.rt_path_count:,}` · "
        f"**PO visible facets:** `{initial_result.visible_face_count:,}`"
    )
    diagnostics_state = {
        "revision": 0,
        "result": None,
        "top_k": None,
        "cancel_event": None,
    }

    def invalidate_physics_result() -> None:
        """Fail closed whenever solver inputs no longer match the result."""

        physics_state["result"] = None
        physics_state["refined"] = False
        gui_session_state["static_cacheable"] = False
        physics_download.callback = lambda: (
            _physics_export(physics_state["result"])
            if physics_state["result"] is not None else BytesIO()
        )
        physics_download.disabled = True
        diagnostics_state["revision"] += 1
        diagnostics_state["result"] = None
        diagnostics_state["top_k"] = None
        diagnostic_cancel = diagnostics_state.get("cancel_event")
        if diagnostic_cancel is not None:
            diagnostic_cancel.set()
        diagnostics_state["cancel_event"] = None
        import param

        with param.parameterized.discard_events(show_rt_paths):
            show_rt_paths.value = False
        show_rt_paths.disabled = True
        diagnostic_top_k.disabled = True
        diagnostics_depth.visible = False
        diagnostics_summary.object = "**No current solver result.**"
        diagnostics_status.object = (
            "RT path geometry is unavailable until the current static "
            "simulation finishes."
        )

    def present_physics_result(
        result,
        *,
        refined: bool,
        preserve_scene_frame: bool,
        geometry_already_previewed: bool,
        result_settings=None,
    ):
        had_diagnostics_overlay = diagnostics_state["result"] is not None
        physics_state["result"] = result
        physics_state["refined"] = bool(refined)
        gui_session_state["static_cacheable"] = True
        if preserve_scene_frame and not had_diagnostics_overlay:
            physics_state["target_result_sequence"] += 1
            scene_interaction.target_result = _target_trace_update(
                result,
                sequence=physics_state["target_result_sequence"],
                include_geometry=not geometry_already_previewed,
            )
        else:
            physics_scene.object = _physics_scene_figure(
                result,
                go,
                plot_template=plot_template,
            )
        physics_profile.object = _range_profile_figure(
            result,
            go,
            plot_template=plot_template,
        )
        angle_state["revision"] += 1
        angle_products = _angle_fft_range_products(result)
        angle_state["products"] = angle_products
        centroid_bin = int(angle_products["centroid_range_bin"])
        import param

        with param.parameterized.discard_events(angle_range_bins):
            angle_range_bins.start = 0
            angle_range_bins.end = len(angle_products["ranges_m"]) - 1
            angle_range_bins.value = (centroid_bin, centroid_bin)
        angle_map.object = _angle_map_figure(
            angle_products,
            make_subplots,
            go,
            range_bins=angle_range_bins.value,
            plot_template=plot_template,
        )
        update_angle_status(angle_products, angle_range_bins.value)
        physics_download.callback = lambda: (
            _physics_export(physics_state["result"])
            if physics_state["result"] is not None else BytesIO()
        )
        physics_download.disabled = False
        show_rt_paths.disabled = False
        diagnostic_top_k.disabled = False
        tier = "HIGHER FIDELITY" if refined else "LIVE RT + PO"
        physics_status.object = (
            f"**{tier}** · {result.faces.shape[0]:,} facets · "
            f"{result.visible_face_count:,} PO-visible · "
            f"{result.rt_path_count:,} RT paths · "
            f"PO {result.po_runtime_s:.3f} s · RT {result.rt_runtime_s:.3f} s · "
            "uncalibrated amplitudes · manifest "
            f"`{result.manifest.fingerprint[:12]}`"
        )
        diagnostics_state["revision"] += 1
        diagnostics_state["result"] = None
        diagnostics_state["top_k"] = None
        import param

        with param.parameterized.discard_events(show_rt_paths):
            show_rt_paths.value = False
        diagnostics_depth.visible = False
        diagnostics_summary.object = (
            f"**RT valid link-paths:** `{result.rt_path_count:,}` · "
            f"**PO visible facets:** `{result.visible_face_count:,}`"
        )
        diagnostics_status.object = (
            "RT path geometry is off. Enable it to inspect the current "
            "simulated state."
        )
        remember_result_settings("static", result_settings)
        persist_gui_session()

    def current_physics_inputs() -> dict:
        selected_human_mesh = human_upload.value
        selected_human_filename = human_upload.filename
        if selected_human_mesh:
            selected_human_mesh = _validated_upload_payload(
                selected_human_mesh,
                label="human mesh",
                max_bytes=_MAX_STATIC_MESH_UPLOAD_BYTES,
            )
            if Path(selected_human_filename or "").suffix.casefold() == ".npz":
                _preflight_gui_mesh_npz(selected_human_mesh)
            elif (
                Path(selected_human_filename or "").suffix.casefold()
                == ".obj"
            ):
                _preflight_gui_mesh_obj(selected_human_mesh)
        if (
            target_type.value == "human_mesh"
            and not selected_human_mesh
            and default_human_mesh is not None
        ):
            selected_human_mesh = default_human_mesh
            selected_human_filename = default_human_mesh.name
        return {
            "target_type": target_type.value,
            "range_m": target_range.value,
            "target_y_m": target_y.value,
            "target_z_m": target_z.value,
            "width_m": plate_width.value,
            "height_m": plate_height.value,
            "corner_edge_m": corner_edge.value,
            "yaw_deg": yaw.value,
            "pitch_deg": pitch.value,
            "roll_deg": roll.value,
            "radar_yaw_deg": radar_yaw.value,
            "radar_pitch_deg": radar_pitch.value,
            "radar_roll_deg": radar_roll.value,
            "material_name": material.value,
            "human_mesh": selected_human_mesh,
            "human_mesh_filename": selected_human_filename,
            "diffuse_reflection_coefficient": human_diffuse.value,
            "board_model": board.value,
            "carrier_frequency_hz": carrier_frequency.value * 1e9,
            "slope_hz_per_s": chirp_slope.value * 1e12,
            "chirp_duration_s": chirp_duration.value * 1e-6,
            "chirp_repetition_time_s": chirp_repetition.value * 1e-6,
            "sampling_frequency_hz": sampling_frequency.value * 1e6,
            "adc_mode": adc_sampling_mode.value,
            "if_bandwidth_hz": adc_if_bandwidth.value * 1e6 or None,
            "num_adc_samples": num_adc_samples.value,
            "num_chirps_per_frame": num_chirps.value,
            "frame_period_s": frame_period.value * 1e-3,
            "tdm_enabled": tdm_enabled.value,
            "selected_tx_indices": tuple(tx_antennas.value),
            "selected_rx_indices": tuple(rx_antennas.value),
            "antenna_pattern_mode": antenna_pattern.value,
            "cosine_3db_beamwidth_deg": cosine_3db_beamwidth.value,
            "po_integration_mode": po_integration_mode.value,
            "po_quadrature_phase_span_scale_rad": po_phase_span.value,
            "po_curvature_error_rad": po_curvature.value,
            "po_quadrature_max_refinement_depth": po_refinement_depth.value,
            "po_quadrature_max_subfaces_per_parent": po_subface_cap.value,
            "rt_samples_per_source": rt_samples.value,
            "rt_max_paths_per_source": rt_path_cap.value,
            "rt_max_depth": rt_depth.value,
        }

    async def update_physics(
        *,
        fidelity: str,
        preserve_scene_frame: bool = False,
        geometry_already_previewed: bool = False,
        force: bool = False,
    ):
        if gui_session_state["restoring"]:
            return
        if not force and fidelity == "preview" and settings_match_result("static"):
            physics_state["revision"] += 1
            physics_state["pending_request"] = None
            cancel_event = physics_state.get("cancel_event")
            if cancel_event is not None:
                cancel_event.set()
            refresh_result_validity()
            return
        physics_state["revision"] += 1
        revision = physics_state["revision"]
        request = {
            "revision": revision,
            "fidelity": fidelity,
            "refined": fidelity == "high_fidelity",
            "preserve_scene_frame": bool(preserve_scene_frame),
            "geometry_already_previewed": bool(geometry_already_previewed),
            "inputs": current_physics_inputs(),
            "settings": capture_result_settings("static"),
        }
        physics_state["pending_request"] = request
        invalidate_physics_result()
        if (
            target_type.value == "human_mesh"
            and not human_upload.value
            and default_human_mesh is None
        ):
            physics_state["pending_request"] = None
            cancel_event = physics_state.get("cancel_event")
            if cancel_event is not None:
                cancel_event.set()
            physics_status.object = (
                "**Human mesh selected.** Upload a pickle-free `.obj` or `.npz` "
                "mesh to run RT and PO."
            )
            return
        if physics_state["running"]:
            cancel_event = physics_state.get("cancel_event")
            if cancel_event is not None:
                cancel_event.set()
            physics_status.object = (
                "**UPDATING** · The active run is stopping; the latest "
                "control state is queued."
            )
            return

        physics_state["running"] = True
        set_gui_activity("static", True)
        refine_physics.disabled = True
        stop_physics.disabled = False
        stop_physics.color = "danger"
        physics_scene.loading = True
        physics_profile.loading = True
        angle_map.loading = True
        try:
            while physics_state["pending_request"] is not None:
                active_request = physics_state["pending_request"]
                physics_state["pending_request"] = None
                cancel_event = threading.Event()
                physics_state["cancel_event"] = cancel_event
                physics_status.object = (
                    "Running higher-density RT and PO..."
                    if active_request["refined"]
                    else "Updating RT and PO from the current controls..."
                )
                try:
                    result = await asyncio.to_thread(
                        run_static_target_experiment,
                        fidelity=active_request["fidelity"],
                        cancel_check=cancel_event.is_set,
                        **active_request["inputs"],
                    )
                except InterruptedError:
                    if (
                        physics_state["pending_request"] is None
                        and active_request["revision"]
                        == physics_state["revision"]
                    ):
                        physics_status.object = (
                            "**STOPPED** · The static simulation was "
                            "terminated at a safe solver boundary."
                        )
                    continue
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    if (
                        active_request["revision"]
                        == physics_state["revision"]
                    ):
                        physics_status.object = (
                            "**Update failed:** "
                            f"{_safe_markdown_code(f'{type(exc).__name__}: {exc}')}"
                        )
                        pn.state.notifications.error(str(exc), duration=8000)
                    continue
                if (
                    active_request["revision"] == physics_state["revision"]
                    and physics_state["pending_request"] is None
                    and active_request["settings"]["fingerprint"]
                    == capture_result_settings("static")["fingerprint"]
                ):
                    present_physics_result(
                        result,
                        refined=active_request["refined"],
                        preserve_scene_frame=active_request[
                            "preserve_scene_frame"
                        ],
                        geometry_already_previewed=active_request[
                            "geometry_already_previewed"
                        ],
                        result_settings=active_request["settings"],
                    )
        finally:
            physics_state["running"] = False
            physics_state["cancel_event"] = None
            refine_physics.disabled = False
            stop_physics.disabled = True
            stop_physics.color = "light"
            set_gui_activity("static", False)
            physics_download.disabled = physics_state["result"] is None
            physics_scene.loading = False
            physics_profile.loading = False
            angle_map.loading = False
            refresh_result_validity()

    async def on_refine_physics(_event):
        await update_physics(fidelity="high_fidelity")

    def on_stop_physics(_event):
        physics_state["pending_request"] = None
        cancel_event = physics_state.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        diagnostic_cancel = diagnostics_state.get("cancel_event")
        if diagnostic_cancel is not None:
            diagnostic_cancel.set()
        stop_physics.disabled = True
        if cancel_event is not None:
            physics_status.object = (
                "**STOPPING** · Waiting for the current RT/PO kernel to "
                "reach a safe boundary."
            )
        elif diagnostic_cancel is not None:
            diagnostics_status.object = (
                "**STOPPING** · Waiting for RT path extraction to reach a "
                "safe boundary."
            )

    async def compute_rt_path_overlay():
        if physics_state["result"] is None:
            diagnostics_status.object = (
                "Run the static simulation before requesting RT paths."
            )
            return
        diagnostics_state["revision"] += 1
        revision = diagnostics_state["revision"]
        source_result = physics_state["result"]
        requested_top_k = int(diagnostic_top_k.value)
        cancel_event = threading.Event()
        diagnostics_state["cancel_event"] = cancel_event
        set_gui_activity("static", True)
        show_rt_paths.disabled = True
        diagnostic_top_k.disabled = True
        stop_physics.disabled = False
        stop_physics.color = "danger"
        physics_scene.loading = True
        diagnostics_depth.loading = True
        diagnostics_status.object = (
            "Tracing the current state and extracting compact RT/PO "
            "diagnostics..."
        )
        try:
            diagnostic_result = await asyncio.to_thread(
                run_static_target_diagnostics,
                source_result,
                top_k=requested_top_k,
                rt_samples_per_source=int(rt_samples.value),
                rt_max_paths_per_source=int(rt_path_cap.value),
                rt_max_depth=int(rt_depth.value),
                cancel_check=cancel_event.is_set,
            )
            if (
                revision == diagnostics_state["revision"]
                and physics_state["result"] is source_result
                and requested_top_k == int(diagnostic_top_k.value)
            ):
                diagnostics_state["result"] = diagnostic_result
                diagnostics_state["top_k"] = requested_top_k
                physics_scene.object = _solver_diagnostics_scene_figure(
                    source_result,
                    diagnostic_result,
                    go,
                    plot_template=plot_template,
                )
                diagnostics_depth.object = _path_depth_figure(
                    diagnostic_result,
                    go,
                    plot_template=plot_template,
                )
                diagnostics_summary.object = _solver_diagnostics_summary(
                    diagnostic_result
                )
                diagnostics_depth.visible = True
                diagnostics_status.object = (
                    f"**READY** · "
                    f"{diagnostic_result.rt.selected_path_indices.size:,} "
                    "target-touching RT paths overlaid in **RT/PO results**"
                )
                persist_gui_session()
        except InterruptedError:
            if revision == diagnostics_state["revision"]:
                diagnostics_status.object = (
                    "**STOPPED** · RT path extraction was cancelled."
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if revision == diagnostics_state["revision"]:
                diagnostics_status.object = (
                    "**Diagnostic trace failed:** "
                    f"{_safe_markdown_code(f'{type(exc).__name__}: {exc}')}"
                )
                pn.state.notifications.error(str(exc), duration=8000)
                import param

                with param.parameterized.discard_events(show_rt_paths):
                    show_rt_paths.value = False
        finally:
            if diagnostics_state.get("cancel_event") is cancel_event:
                diagnostics_state["cancel_event"] = None
            if revision == diagnostics_state["revision"]:
                has_result = physics_state["result"] is not None
                show_rt_paths.disabled = not has_result
                diagnostic_top_k.disabled = not has_result
                stop_physics.disabled = not physics_state["running"]
                stop_physics.color = "danger" if physics_state["running"] else "light"
                physics_scene.loading = False
                diagnostics_depth.loading = False
            set_gui_activity("static", bool(physics_state["running"]))

    async def on_show_rt_paths(event):
        if physics_state["result"] is None:
            return
        if not bool(event.new):
            physics_scene.object = _physics_scene_figure(
                physics_state["result"],
                go,
                plot_template=plot_template,
            )
            diagnostics_status.object = (
                "RT path geometry is off. The latest path counts and "
                "interaction-depth histogram remain below."
                if diagnostics_state["result"] is not None
                else (
                    "RT path geometry is off. Enable it to inspect the "
                    "current simulated state."
                )
            )
            persist_gui_session()
            return
        if (
            diagnostics_state["result"] is not None
            and diagnostics_state["top_k"] == int(diagnostic_top_k.value)
        ):
            physics_scene.object = _solver_diagnostics_scene_figure(
                physics_state["result"],
                diagnostics_state["result"],
                go,
                plot_template=plot_template,
            )
            diagnostics_status.object = (
                f"**READY** · "
                f"{diagnostics_state['result'].rt.selected_path_indices.size:,} "
                "target-touching RT paths overlaid"
            )
            persist_gui_session()
            return
        await compute_rt_path_overlay()

    async def on_diagnostic_top_k_change(_event):
        if show_rt_paths.value:
            await compute_rt_path_overlay()

    ignored_physics_events: dict[int, int] = {}

    async def on_physics_control_change(event):
        changed_widget = getattr(event, "obj", None)
        ignored_count = ignored_physics_events.get(id(changed_widget), 0)
        if ignored_count:
            if ignored_count == 1:
                ignored_physics_events.pop(id(changed_widget), None)
            else:
                ignored_physics_events[id(changed_widget)] = ignored_count - 1
            return
        if changed_widget in (plate_width, plate_height, corner_edge):
            normalized = _rounded_target_size(changed_widget.value)
            if normalized != changed_widget.value:
                import param

                with param.parameterized.discard_events(changed_widget):
                    changed_widget.value = normalized
        await update_physics(
            fidelity="preview",
            preserve_scene_frame=changed_widget in (yaw, pitch, roll),
        )

    async def on_target_gesture(event):
        gesture = getattr(event, "new", None)
        if not isinstance(gesture, dict):
            return
        try:
            new_yaw = float(gesture["yaw_deg"])
            new_pitch = float(gesture["pitch_deg"])
        except (KeyError, TypeError, ValueError):
            return
        if not np.isfinite(new_yaw) or not np.isfinite(new_pitch):
            return
        new_yaw = float(np.clip(new_yaw, yaw.start, yaw.end))
        new_pitch = float(np.clip(new_pitch, pitch.start, pitch.end))
        yaw_delta = abs(new_yaw - float(yaw.value))
        pitch_delta = abs(new_pitch - float(pitch.value))
        if yaw_delta < 0.05 and pitch_delta < 0.05:
            return
        # Synchronize visible numeric fields while the gesture owns one solve.
        # Count the value events so delayed async watchers also skip this solve.
        for widget, value in ((yaw, round(new_yaw, 1)), (pitch, round(new_pitch, 1))):
            if widget.value != value:
                ignored_physics_events[id(widget)] = (
                    ignored_physics_events.get(id(widget), 0) + 1
                )
                widget.value = value
        await update_physics(
            fidelity="preview",
            preserve_scene_frame=True,
            geometry_already_previewed=True,
        )

    async def on_angle_range_change(_event):
        angle_state["revision"] += 1
        revision = angle_state["revision"]
        selected_bins = tuple(int(value) for value in angle_range_bins.value)
        products = angle_state["products"]
        angle_map.loading = True
        try:
            figure = await asyncio.to_thread(
                _angle_map_figure,
                products,
                make_subplots,
                go,
                range_bins=selected_bins,
                plot_template=plot_template,
            )
            if revision == angle_state["revision"]:
                angle_map.object = figure
                update_angle_status(products, selected_bins)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if revision == angle_state["revision"]:
                pn.state.notifications.error(str(exc), duration=8000)
        finally:
            if revision == angle_state["revision"]:
                angle_map.loading = False

    async def on_target_type_change(_event):
        is_plate = target_type.value == "plate"
        is_corner = target_type.value == "trihedral"
        is_human = target_type.value == "human_mesh"
        plate_width.visible = is_plate
        plate_height.visible = is_plate
        corner_edge.visible = is_corner
        material.visible = not is_human
        material_details.visible = not is_human
        human_upload.visible = is_human
        human_diffuse.visible = is_human
        human_source.visible = is_human
        await update_physics(fidelity="preview")

    def on_board_selection_change(_event):
        if gui_session_state["restoring"]:
            return
        set_radar_defaults()

    async def on_board_change(_event):
        if gui_session_state["restoring"] or settings_dialog_state["editing"]:
            return
        refresh_result_validity()

    async def on_load_radar_defaults(_event):
        set_radar_defaults()

    async def on_radar_control_change(event):
        if gui_session_state["restoring"]:
            return
        changed_widget = getattr(event, "obj", None)
        ignored_count = ignored_radar_events.get(id(changed_widget), 0)
        if ignored_count:
            if ignored_count == 1:
                ignored_radar_events.pop(id(changed_widget), None)
            else:
                ignored_radar_events[id(changed_widget)] = ignored_count - 1
            update_radar_summary()
            return
        if changed_widget in (tx_antennas, rx_antennas):
            if not changed_widget.value:
                import param

                with param.parameterized.discard_events(changed_widget):
                    changed_widget.value = [0]
                pn.state.notifications.warning(
                    "At least one antenna must remain active.",
                    duration=4000,
                )
        if isinstance(changed_widget, pn.widgets.FloatInput):
            normalized = _rounded_gui_float(changed_widget.value)
            if normalized != changed_widget.value:
                import param

                with param.parameterized.discard_events(changed_widget):
                    changed_widget.value = normalized
        update_radar_summary()
        if not settings_dialog_state["editing"]:
            refresh_result_validity()

    async def on_material_change(event):
        update_material_details()
        await on_physics_control_change(event)

    async def on_antenna_pattern_change(event):
        update_antenna_pattern_controls()
        await on_radar_control_change(event)

    refine_physics.on_click(on_refine_physics)
    stop_physics.on_click(on_stop_physics)
    show_rt_paths.param.watch(on_show_rt_paths, "value")
    diagnostic_top_k.param.watch(on_diagnostic_top_k_change, "value")
    load_radar_defaults.on_click(on_load_radar_defaults)
    target_type.param.watch(on_target_type_change, "value")
    scene_interaction.param.watch(on_target_gesture, "target_gesture")
    angle_range_bins.param.watch(on_angle_range_change, "value_throttled")
    for numeric in (
        target_range,
        target_y,
        target_z,
        yaw,
        pitch,
        roll,
        radar_yaw,
        radar_pitch,
        radar_roll,
        plate_width, plate_height, corner_edge,
    ):
        numeric.param.watch(on_physics_control_change, "value")
    human_diffuse.param.watch(on_physics_control_change, "value_throttled")
    material.param.watch(on_material_change, "value")
    human_upload.param.watch(on_physics_control_change, "value")
    board.param.watch(on_board_selection_change, "value")
    board.param.watch(on_board_change, "value")
    tdm_enabled.param.watch(on_radar_control_change, "value")
    tx_antennas.param.watch(on_radar_control_change, "value")
    rx_antennas.param.watch(on_radar_control_change, "value")
    antenna_pattern.param.watch(on_antenna_pattern_change, "value")
    cosine_3db_beamwidth.param.watch(on_radar_control_change, "value")
    for numeric in (
        carrier_frequency,
        chirp_slope,
        chirp_duration,
        chirp_repetition,
        sampling_frequency,
        adc_sampling_mode,
        adc_if_bandwidth,
        num_adc_samples,
        num_chirps,
        frame_period,
    ):
        numeric.param.watch(on_radar_control_change, "value")

    radar_used_by = "Static Target · Dynamic Scenes"
    radar_controls = pn.Row(
        pn.Column(
            _settings_note(
                pn,
                "Hardware, antenna, and waveform settings apply to subsequent "
                "static and dynamic runs. Each workflow keeps its own radar "
                "orientation.",
            ),
            _settings_section(
                pn, "Hardware and antennas", radar_used_by,
                _settings_grid(
                    pn,
                    board,
                    tx_antennas,
                    rx_antennas,
                    antenna_pattern,
                    cosine_3db_beamwidth,
                    tdm_enabled,
                    columns=3,
                ),
                antenna_pattern_note,
            ),
            _settings_section(
                pn, "Chirp", radar_used_by,
                _settings_grid(
                    pn,
                    carrier_frequency,
                    chirp_slope,
                    chirp_duration,
                    chirp_repetition,
                ),
            ),
            _settings_section(
                pn, "Sampling", radar_used_by,
                _settings_grid(
                    pn,
                    sampling_frequency,
                    adc_sampling_mode,
                    adc_if_bandwidth,
                    num_adc_samples,
                ),
            ),
            _settings_section(
                pn, "Frame", radar_used_by,
                _settings_grid(pn, num_chirps, frame_period),
            ),
            sizing_mode="stretch_width",
            min_width=0,
            margin=0,
        ),
        # The derived values stay in view while the waveform is edited.
        pn.Column(
            pn.pane.HTML(
                '<div class="hermes-group-title" role="heading" '
                'aria-level="3">Derived</div>',
                sizing_mode="stretch_width",
                margin=(0, 10),
            ),
            radar_summary,
            load_radar_defaults,
            sizing_mode="stretch_width",
            min_width=260,
            max_width=260,
            margin=0,
            css_classes=["hermes-settings-readout"],
        ),
        sizing_mode="stretch_width",
        margin=0,
        css_classes=["hermes-settings-split"],
    )
    settings_modal = pn.Column(
        pn.pane.HTML(
            '<div class="hermes-settings-title" role="heading" aria-level="2">'
            "Settings</div>"
            '<p class="hermes-settings-sub">Edits take effect together when '
            "you Apply. Bundle Comparison uses the radar and solver settings "
            "stored in its primary bundle.</p>",
            sizing_mode="stretch_width",
            margin=(0, 10),
        ),
        pn.Tabs(
            ("Radar configuration", radar_controls),
            ("Solver settings", solver_controls_panel),
            ("Motion coordinates", motion_controls_panel),
            dynamic=False,
            sizing_mode="stretch_width",
        ),
        pn.Row(
            rerun_after_apply,
            pn.Spacer(sizing_mode="stretch_width"),
            cancel_solver_settings,
            apply_solver_settings,
            sizing_mode="stretch_width",
            margin=0,
            css_classes=["hermes-settings-footer"],
        ),
        min_width=0,
        max_width=980,
        sizing_mode="stretch_width",
        css_classes=[theme_scope_class, "hermes-settings"],
    )
    static_result_banner = pn.Column(visible=False, name="Static result settings")
    physics_controls = pn.Column(
        _rail_nav_component(pn, (
            ("target", "Target"),
            ("orientation", "Orientation"),
            ("material", "Material"),
            ("radar", "Radar"),
        )),
        _rail_group(
            pn, "target", "Target",
            target_type,
            numeric_row("hermes-static-position", target_range, target_y, target_z),
            numeric_row("hermes-static-size", plate_width, plate_height),
            corner_edge,
        ),
        _rail_group(
            pn, "orientation", "Target orientation",
            numeric_row("hermes-static-orientation", yaw, pitch, roll),
        ),
        _rail_group(
            pn, "material", "Material and surface",
            material,
            material_details,
            human_upload,
            human_diffuse,
            human_source,
        ),
        _rail_group(
            pn, "radar", "Radar orientation",
            numeric_row("hermes-static-radar-orientation", radar_yaw, radar_pitch, radar_roll),
        ),
        _rail_group(pn, "export", "Export", physics_download),
        sizing_mode="stretch_width",
        min_width=380,
        max_width=380,
        margin=0,
        css_classes=["hermes-rail", "hermes-scene-controls", "hermes-static-controls"],
    )
    physics_workspace = pn.Column(
        pn.Column(
            pn.Row(
                diagnostic_top_k,
                show_rt_paths,
                sizing_mode="stretch_width",
            ),
            diagnostics_status,
            scene_interaction,
            pn.pane.Markdown(
                "**Mouse:** orbit camera · **Shift + drag:** rotate "
                "target yaw/pitch · **Wheel:** zoom · **Roll:** use "
                "the numeric field. Shift-drag previews geometry only; one "
                "RT/PO simulation runs when the mouse is released.",
                css_classes=["hermes-compact-hint"],
            ),
            sizing_mode="stretch_width",
            css_classes=["hermes-card"],
        ),
        pn.Column(
            physics_profile,
            sizing_mode="stretch_width",
            css_classes=["hermes-card"],
        ),
        pn.Column(
            angle_range_bins,
            angle_status,
            angle_map,
            sizing_mode="stretch_width",
            css_classes=["hermes-card"],
        ),
        pn.Column(
            pn.pane.HTML(
                '<div class="hermes-card-title" role="heading" aria-level="3">'
                "Solver summary</div>",
                sizing_mode="stretch_width",
                margin=(0, 10),
            ),
            diagnostics_summary,
            diagnostics_depth,
            sizing_mode="stretch_width",
            css_classes=["hermes-card"],
        ),
        sizing_mode="stretch_width",
        margin=0,
        styles={"gap": "16px"},
        css_classes=["hermes-workspace"],
    )
    physics_tab = pn.Row(
        physics_controls,
        pn.Column(
            _run_bar(
                pn,
                title="Static Target",
                subtitle="RT and PO for a plate, corner reflector or human mesh.",
                help_text=(
                    "Compare ray tracing and physical optics for one static "
                    "object. Radar hardware, waveform, antennas and solver "
                    "settings come from Settings (gear icon); the radar "
                    "orientation is set in the control rail. Scene frame: +x "
                    "is initial range-forward, +y is left, and +z is up. Green "
                    "shows radar boresight; blue shows radar local +z."
                ),
                status=physics_status,
                actions=(stop_physics, refine_physics),
                banner=static_result_banner,
            ),
            physics_workspace,
            sizing_mode="stretch_width",
            min_width=0,
            margin=0,
            styles={"gap": "16px"},
            css_classes=["hermes-work"],
        ),
        sizing_mode="stretch_width",
        css_classes=["hermes-page"],
    )

    human_room_preset = pn.widgets.Select(
        label="Prepared scene fallback",
        options={"Tutorial bedroom": "tutorial_bedroom"},
        value="tutorial_bedroom",
    )
    human_room_scene_upload = pn.widgets.FileInput(
        label="Static environment scene (.xml, optional)",
        accept=".xml",
        stylesheets=[_file_input_stylesheet(plot_template, compact=False)],
        description=(
            "Disabled for remotely bound sessions because scene XML can "
            "reference server-side assets. Local uploads are limited to 2 MiB."
        ),
        disabled=remote_access,
    )
    default_amass_motion = _default_amass_motion_path()
    default_model_dir = _default_smpl_model_dir()
    human_room_upload_help = (
        "Upload a pickle-free motion archive with vertices, faces, and times, "
        "or AMASS-like poses, trans, betas, and timing. Supplied meshes are "
        "used directly; pose-only archives need the configured SMPL model. "
        f"Browser uploads are limited to {_MAX_MOTION_UPLOAD_BYTES // _MIB} MiB."
    )
    human_room_upload = pn.widgets.FileInput(
        label="AMASS-like human motion (.npz)",
        accept=".npz",
        stylesheets=[_file_input_stylesheet(plot_template)],
        description=human_room_upload_help,
        width=132,
        height=40,
        sizing_mode="fixed",
    )
    human_room_upload_label = pn.Row(
        pn.pane.Markdown(
            "**AMASS-like human motion (.npz)**",
            margin=0,
            styles={"white-space": "nowrap", "width": "max-content", "flex": "0 0 auto"},
            stylesheets=[
                ":host {width:max-content !important; padding:0 !important;} "
                "p {margin:0 !important;}"
            ],
        ),
        pn.widgets.TooltipIcon(value=human_room_upload_help, margin=0),
        sizing_mode="stretch_width",
        styles={"align-items": "center", "justify-content": "flex-start", "gap": "4px"},
        margin=(5, 16),
    )
    human_room_upload_filename = pn.widgets.StaticText(
        value=(
            f"Bundled default: {default_amass_motion.name}"
            if default_amass_motion is not None
            else "No file selected"
        ),
        align="end",
        height=42,
        sizing_mode="stretch_width",
        styles={
            "background": "transparent",
            "border": "0",
            "overflow": "hidden",
            "padding": "8px 4px",
            "text-overflow": "ellipsis",
            "white-space": "nowrap",
        },
    )
    human_room_upload_control = pn.Column(
        human_room_upload_label,
        pn.Row(
            human_room_upload,
            human_room_upload_filename,
            sizing_mode="stretch_width",
            styles={"align-items": "center", "gap": "8px"},
        ),
        sizing_mode="stretch_width",
        margin=0,
    )
    human_room_smpl_model_dir = pn.widgets.TextInput(
        label="Licensed SMPL model directory",
        value=str(default_model_dir),
        placeholder="/path/to/smpl_models",
        description=(
            "In remote mode this server-side directory is fixed at launch "
            "from MMWAVE_SMPL_MODEL_DIR and cannot be changed by a browser."
        ),
        disabled=remote_access,
    )
    load_motion_preview = pn.widgets.Button(
        label="Load default motion", color="light",
        visible=default_amass_motion is not None,
        description="Load the bundled motion. Uploaded motion previews automatically.",
    )
    human_room_x = compact_numeric(
        "X [m]", start=0.5, end=2.8, step=0.01, format="0.00", value=1.6,
        description="Range-forward position.",
    )
    human_room_y = compact_numeric(
        "Y [m]", start=-2.0, end=2.0, step=0.01, format="0.00", value=0.0,
        description="Lateral-left position.",
    )
    human_room_z = compact_numeric(
        "Z [m]", start=-0.8, end=1.5, step=0.01, format="0.00", value=0.0,
        description="Height relative to the loaded motion.",
    )
    human_room_yaw = compact_numeric(
        "Yaw [deg]", start=-180, end=180, step=1, value=0,
    )
    human_room_diffuse = compact_numeric(
        "Diffuse (RT)", start=0.0, end=1.0, step=0.05, value=0.35, format="0.00",
        description="Human diffuse reflection coefficient for ray tracing.",
    )
    human_room_frame = pn.widgets.IntInput(
        label="Displayed frame (max 0)",
        start=0,
        end=0,
        step=1,
        value=0,
        disabled=True,
        width=190,
    )
    human_room_player = pn.widgets.Player(
        label="Motion playback",
        start=0,
        end=0,
        value=0,
        step=1,
        interval=100,
        loop_policy="loop",
        show_loop_controls=False,
        visible_buttons=[],
        disabled=True,
        width=None,
        sizing_mode="stretch_width",
    )
    human_room_previous_frame = pn.widgets.Button(
        label="Prev",
        icon="player-skip-back",
        color="light",
        disabled=True,
        width=40,
        height=38,
    )
    human_room_reverse = pn.widgets.Button(
        label="Reverse",
        icon="player-play",
        color="light",
        disabled=True,
        width=40,
        height=38,
    )
    human_room_fast_reverse = pn.widgets.Button(
        label="Fast reverse",
        icon="player-track-prev",
        color="light",
        disabled=True,
        width=40,
        height=38,
    )
    human_room_play_pause = pn.widgets.Button(
        label="Play",
        icon="player-play",
        color="light",
        disabled=True,
        width=40,
        height=38,
    )
    human_room_fast_forward = pn.widgets.Button(
        label="Fast forward",
        icon="player-track-next",
        color="light",
        disabled=True,
        width=40,
        height=38,
    )
    human_room_next_frame = pn.widgets.Button(
        label="Next",
        icon="player-skip-forward",
        color="light",
        disabled=True,
        width=40,
        height=38,
    )
    human_room_transport_buttons = (
        human_room_previous_frame,
        human_room_fast_reverse,
        human_room_reverse,
        human_room_play_pause,
        human_room_fast_forward,
        human_room_next_frame,
    )
    for button in human_room_transport_buttons:
        button.description = button.label
        button.margin = 0
        button.stylesheets = [
            ".bk-btn {font-size:0 !important;} "
            ".bk-TablerIcon {font-size:18px !important; width:18px; height:18px;}"
        ]
    human_room_reverse.stylesheets += [".bk-TablerIcon {transform:rotate(180deg);}"]
    human_room_frame_counter = pn.widgets.StaticText(value="Frame — / —", width=140)
    human_room_playback_controls = pn.Row(
        pn.Spacer(sizing_mode="stretch_width"),
        *human_room_transport_buttons,
        human_room_frame_counter,
        pn.Spacer(sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
        styles={"align-items": "center", "gap": "6px"},
    )
    human_room_begin_frame = pn.widgets.IntInput(
        label="Begin frame (inclusive)",
        value=0,
        start=0,
        end=0,
        step=1,
        disabled=True,
    )
    human_room_end_frame = pn.widgets.IntInput(
        label="End frame (max 0)",
        value=0,
        start=0,
        end=0,
        step=1,
        disabled=True,
    )
    human_room_mode_checkboxes = {
        mode: pn.widgets.Checkbox(
            label=_HUMAN_ROOM_MODE_LABELS[mode],
            value=mode == "hybrid_po",
            margin=0,
            styles={"width": "max-content", "flex": "0 0 auto"},
        )
        for mode in HUMAN_ROOM_SIMULATION_MODES
    }
    human_room_mode_options = [
        pn.Row(
            human_room_mode_checkboxes[mode],
            pn.widgets.TooltipIcon(value=_HUMAN_ROOM_MODE_HELP[mode], margin=0),
            sizing_mode="stretch_width", min_width=0, margin=0,
            styles={"align-items": "center", "justify-content": "flex-start", "gap": "4px"},
        )
        for mode in HUMAN_ROOM_SIMULATION_MODES
    ]
    human_room_mode_controls = pn.Column(
        *(pn.Row(*human_room_mode_options[index:index + 2],
                 sizing_mode="stretch_width", margin=(5, 10), styles={"gap": "12px"})
          for index in range(0, len(human_room_mode_options), 2)),
        sizing_mode="stretch_width",
        margin=0,
    )
    run_human_room = pn.widgets.Button(
        label="Start",
        icon="player-play",
        color="primary",
        width=120,
        height=40,
        sizing_mode="fixed",
        disabled=True,
    )
    stop_human_room = pn.widgets.Button(
        label="Stop",
        icon="player-stop",
        color="light",
        width=105,
        height=40,
        sizing_mode="fixed",
        disabled=True,
    )
    human_room_download = pn.widgets.FileDownload(
        callback=lambda: BytesIO(),
        filename="hermes-human-room-bundle.zip",
        label="Export loadable bundle",
        color="light",
        disabled=True,
    )
    human_room_status = pn.pane.Markdown(
        "Load human motion, verify its placement, then start the "
        "selected simulation modes.",
        css_classes=["hermes-status"],
    )
    human_room_results = pn.Column(
        pn.pane.Markdown("Start one or more simulation modes to view results."),
        sizing_mode="stretch_width",
    )
    human_room_diagnostic_top_k = pn.widgets.IntInput(
        label="Top RT rays",
        value=12,
        start=1,
        end=50,
        step=1,
        width=135,
    )
    run_human_room_diagnostics_button = pn.widgets.Button(
        label="Show",
        icon="route",
        color="light",
        disabled=True,
        width=100,
        align="end",
    )
    human_room_diagnostics_status = pn.pane.Markdown(
        "Run a simulation to enable one-frame solver overlays.",
        css_classes=["hermes-status"],
    )
    human_room_state = {
        "preview": None,
        "result": None,
        "results": {},
        "revision": 0,
        "loaded_motion": None,
        "motion_cache_key": None,
        "preview_revision": 0,
        "cancel_event": None,
        "running": False,
        "diagnostics": None,
        "diagnostic_revision": 0,
        "diagnostic_cancel_event": None,
    }
    human_room_playback_state = {
        "syncing": False,
        "normal_step": 1,
        "fast_step": 4,
    }
    human_room_window_state = {"syncing": False}

    def show_human_room_playing_state(playing: bool):
        human_room_play_pause.label = "Pause" if playing else "Play"
        human_room_play_pause.description = human_room_play_pause.label
        human_room_play_pause.icon = (
            "player-pause" if playing else "player-play"
        )

    def configure_human_room_frame_controls(
        *,
        last_frame: int,
        selected_frame: int,
        disabled: bool,
        interval_ms: int,
        normal_step: int,
        fast_step: int,
    ):
        human_room_playback_state["syncing"] = True
        try:
            human_room_frame.end = int(last_frame)
            human_room_frame.label = f"Displayed frame (max {int(last_frame)})"
            human_room_frame.value = int(selected_frame)
            human_room_frame_counter.value = (
                f"Frame {int(selected_frame)} / {int(last_frame)}"
                if human_room_state["preview"] is not None else "Frame — / —"
            )
            human_room_frame.disabled = bool(disabled)
            human_room_player.direction = 0
            human_room_player.end = int(last_frame)
            human_room_player.value = int(selected_frame)
            human_room_player.step = max(int(normal_step), 1)
            human_room_player.interval = int(interval_ms)
            human_room_player.disabled = bool(disabled)
            for button in human_room_transport_buttons:
                button.disabled = bool(disabled)
            human_room_playback_state["normal_step"] = max(
                int(normal_step),
                1,
            )
            human_room_playback_state["fast_step"] = max(int(fast_step), 1)
            show_human_room_playing_state(False)
        finally:
            human_room_playback_state["syncing"] = False

    def configure_human_room_window_controls(
        *,
        last_frame: int,
        reset_window: bool,
        disabled: bool,
    ):
        human_room_window_state["syncing"] = True
        try:
            last_frame = int(last_frame)
            previous_begin = int(human_room_begin_frame.value)
            previous_end = int(human_room_end_frame.value)
            human_room_begin_frame.end = last_frame
            human_room_end_frame.end = last_frame
            human_room_end_frame.label = f"End frame (max {last_frame})"
            begin_frame, end_frame = _human_room_window_values(
                last_frame=last_frame,
                previous_begin=previous_begin,
                previous_end=previous_end,
                reset_window=reset_window,
            )
            human_room_begin_frame.value = begin_frame
            human_room_end_frame.value = end_frame
            human_room_begin_frame.disabled = bool(disabled)
            human_room_end_frame.disabled = bool(disabled)
        finally:
            human_room_window_state["syncing"] = False

    def current_human_room_source():
        if human_room_upload.value:
            payload = _validated_upload_payload(
                human_room_upload.value,
                label="human motion",
                max_bytes=_MAX_MOTION_UPLOAD_BYTES,
            )
            _preflight_gui_motion_npz(payload)
            return payload, human_room_upload.filename
        model_dir = _effective_smpl_model_dir(
            human_room_smpl_model_dir.value,
            default_model_dir,
            remote_access=remote_access,
        )
        if (
            default_amass_motion is not None
            and Path(model_dir).expanduser().is_dir()
        ):
            return default_amass_motion, default_amass_motion.name
        return None, None

    def on_human_room_upload_filename(event):
        human_room_upload_filename.value = (
            str(event.new)
            if event.new
            else (
                f"Bundled default: {default_amass_motion.name}"
                if default_amass_motion is not None
                else "No file selected"
            )
        )

    def current_human_room_scene():
        if human_room_scene_upload.value and not remote_access:
            return (
                _validated_upload_payload(
                    human_room_scene_upload.value,
                    label="scene XML",
                    max_bytes=_MAX_SCENE_XML_UPLOAD_BYTES,
                ),
                human_room_scene_upload.filename,
            )
        return None, None

    def current_room_boxes_override():
        scene_source, _scene_filename = current_human_room_scene()
        return _scene_xml_boxes(scene_source) if scene_source is not None else None

    def current_motion_cache_key(source, filename):
        if isinstance(source, (bytes, bytearray)):
            source_identity = hashlib.sha256(bytes(source)).hexdigest()
        else:
            source_path = Path(source)
            source_identity = (
                str(source_path.resolve()),
                source_path.stat().st_mtime_ns,
                source_path.stat().st_size,
            )
        return (
            filename,
            source_identity,
            str(
                Path(
                    _effective_smpl_model_dir(
                        human_room_smpl_model_dir.value,
                        default_model_dir,
                        remote_access=remote_access,
                    )
                )
                .expanduser()
                .resolve()
            ),
        )

    def current_human_position():
        return (
            float(human_room_x.value),
            float(human_room_y.value),
            float(human_room_z.value),
        )

    def current_dynamic_radar_orientation():
        return (
            float(dynamic_radar_yaw.value),
            float(dynamic_radar_pitch.value),
            float(dynamic_radar_roll.value),
        )

    def mark_human_room_stale(message: str):
        if gui_session_state["restoring"] or gui_session_state["restoring_uploads"]:
            return
        if settings_match_result("dynamic"):
            refresh_result_validity()
            return
        human_room_state["result"] = None
        human_room_state["results"] = {}
        human_room_state["diagnostics"] = None
        human_room_state["diagnostic_revision"] += 1
        diagnostic_cancel = human_room_state.get("diagnostic_cancel_event")
        if diagnostic_cancel is not None:
            diagnostic_cancel.set()
        human_room_state["diagnostic_cancel_event"] = None
        gui_session_state["dynamic_cacheable"] = False
        human_room_download.disabled = True
        run_human_room_diagnostics_button.disabled = True
        human_room_diagnostics_status.object = (
            "Run a simulation to enable one-frame solver overlays."
        )
        if result_records["dynamic"] is None:
            human_room_results.objects = [pn.pane.Markdown(message)]
        persist_gui_session()

    async def refresh_human_room_preview(*, mark_stale: bool = True):
        if mark_stale and gui_session_state["restoring"]:
            return
        human_room_state["preview_revision"] += 1
        preview_revision = human_room_state["preview_revision"]
        source, filename = current_human_room_source()
        scene_source, scene_filename = current_human_room_scene()
        human_room_preset.disabled = scene_source is not None
        if source is None:
            if scene_source is not None:
                await asyncio.to_thread(
                    load_scene_xml,
                    scene_source,
                    filename=scene_filename,
                )
                if preview_revision != human_room_state["preview_revision"]:
                    return
            human_room_state["preview"] = None
            human_room_state["loaded_motion"] = None
            human_room_state["motion_cache_key"] = None
            run_human_room.disabled = True
            configure_human_room_frame_controls(
                last_frame=0,
                selected_frame=0,
                disabled=True,
                interval_ms=100,
                normal_step=1,
                fast_step=1,
            )
            configure_human_room_window_controls(
                last_frame=0,
                reset_window=True,
                disabled=True,
            )
            human_room_scene.object = _human_room_scene_figure(
                None,
                go,
                frame_index=0,
                radar_orientation_deg=current_dynamic_radar_orientation(),
                room_boxes_override=current_room_boxes_override(),
                plot_template=plot_template,
            )
            if mark_stale:
                mark_human_room_stale(
                    "Motion input is unavailable; load a valid sequence and "
                    "run the simulation again."
                )
            human_room_status.object = (
                "**Human motion required.** Upload a pickle-free `.npz`. "
                "Pose-only archives also need a licensed SMPL model directory."
            )
            return
        cache_key = current_motion_cache_key(source, filename)
        motion_changed = cache_key != human_room_state["motion_cache_key"]
        if motion_changed:
            human_room_status.object = (
                "Loading human motion from "
                f"{_safe_markdown_code(filename)}..."
            )
            loaded_motion = await asyncio.to_thread(
                load_amass_motion,
                source,
                filename=filename,
                smpl_model_dir=_effective_smpl_model_dir(
                    human_room_smpl_model_dir.value,
                    default_model_dir,
                    remote_access=remote_access,
                ),
            )
            if preview_revision != human_room_state["preview_revision"]:
                return
            human_room_state["loaded_motion"] = loaded_motion
            human_room_state["motion_cache_key"] = cache_key
        preview = await asyncio.to_thread(
            prepare_human_room_preview,
            human_room_state["loaded_motion"],
            room_preset=human_room_preset.value,
            scene_xml=scene_source,
            scene_xml_filename=scene_filename,
            human_position_m=current_human_position(),
            human_yaw_deg=float(human_room_yaw.value),
            **current_motion_coordinate_arguments(),
        )
        if preview_revision != human_room_state["preview_revision"]:
            return
        human_room_state["preview"] = preview
        run_human_room.disabled = False
        frame_count = len(preview.mesh_sequence.times)
        last_frame = max(frame_count - 1, 0)
        selected_frame = min(
            int(human_room_frame.value),
            last_frame,
        )
        configure_human_room_frame_controls(
            last_frame=last_frame,
            selected_frame=selected_frame,
            disabled=frame_count <= 1,
            interval_ms=_motion_playback_interval_ms(
                preview.mesh_sequence.times
            ),
            normal_step=_motion_playback_step(preview.mesh_sequence.times),
            fast_step=_motion_fast_playback_step(
                preview.mesh_sequence.times
            ),
        )
        configure_human_room_window_controls(
            last_frame=last_frame,
            reset_window=motion_changed,
            disabled=False,
        )
        human_room_scene.object = _human_room_scene_figure(
            preview,
            go,
            frame_index=selected_frame,
            radar_orientation_deg=current_dynamic_radar_orientation(),
            room_boxes_override=current_room_boxes_override(),
            plot_template=plot_template,
        )
        if mark_stale:
            mark_human_room_stale(
                "Scene, motion, or radar configuration changed; "
                "run the simulation again."
            )
        if human_room_state["result"] is None:
            human_room_status.object = (
                f"**READY TO RUN** · "
                f"{_safe_markdown_code(preview.source_name)} · "
                f"{frame_count:,} motion frame(s) · "
                f"{preview.mesh_sequence.vertex_count:,} vertices · "
                f"{preview.mesh_sequence.face_count:,} facets · scene "
                f"{_safe_markdown_code(preview.scene_source_name)}"
            )
        refresh_result_validity()

    initial_human_preview = None
    human_room_scene = pn.pane.Plotly(
        _human_room_scene_figure(
            initial_human_preview,
            go,
            frame_index=0,
            radar_orientation_deg=current_dynamic_radar_orientation(),
            plot_template=plot_template,
        ),
        height=570,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["resetCameraDefault3d"],
        },
    )
    human_room_floor_status = pn.pane.Markdown(
        "Load human motion to check floor placement.",
        css_classes=["hermes-status"],
    )

    def update_human_room_floor_status(_event=None):
        human_room_floor_status.object = next(
            (
                trace.meta["floor_status"]
                for trace in human_room_scene.object.data
                if trace.name == "Below-floor vertices"
            ),
            "Load human motion to check floor placement.",
        )

    human_room_scene.param.watch(update_human_room_floor_status, "object")
    update_human_room_floor_status()
    human_room_scene_interaction = _dynamic_scene_interaction_component(
        pn,
        human_room_scene,
    )
    if initial_human_preview is not None:
        last_frame = max(
            len(initial_human_preview.mesh_sequence.times) - 1,
            0,
        )
        configure_human_room_frame_controls(
            last_frame=last_frame,
            selected_frame=0,
            disabled=last_frame == 0,
            interval_ms=_motion_playback_interval_ms(
                initial_human_preview.mesh_sequence.times
            ),
            normal_step=_motion_playback_step(
                initial_human_preview.mesh_sequence.times
            ),
            fast_step=_motion_fast_playback_step(
                initial_human_preview.mesh_sequence.times
            ),
        )
        configure_human_room_window_controls(
            last_frame=last_frame,
            reset_window=True,
            disabled=False,
        )

    async def on_human_room_controls(_event):
        expected_revision = human_room_state["preview_revision"] + 1
        try:
            await refresh_human_room_preview()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if expected_revision != human_room_state["preview_revision"]:
                return
            human_room_state["preview"] = None
            human_room_state["loaded_motion"] = None
            human_room_state["motion_cache_key"] = None
            mark_human_room_stale(
                "Dynamic inputs could not be loaded; correct them and run "
                "the simulation again."
            )
            human_room_status.object = (
                f"**Dynamic input load failed:** "
                f"{_safe_markdown_code(f'{type(exc).__name__}: {exc}')}"
            )
            pn.state.notifications.error(str(exc), duration=8000)

    async def on_motion_coordinate_change(event):
        if motion_coordinate_state["syncing"]:
            return
        sync_motion_coordinate_controls()
        if gui_session_state["restoring"] or settings_dialog_state["editing"]:
            return
        await on_human_room_controls(event)

    def diagnostics_for_source_frame(frame_index: int):
        diagnostics = human_room_state.get("diagnostics")
        if (
            diagnostics is not None
            and int(diagnostics.source_frame_index) == int(frame_index)
        ):
            return diagnostics
        return None

    def display_human_room_frame(frame_index: int):
        human_room_frame_counter.value = f"Frame {int(frame_index)} / {int(human_room_frame.end)}"
        preview = human_room_state["preview"]
        frame_diagnostics = diagnostics_for_source_frame(frame_index)
        updated_in_place = frame_diagnostics is None and (
            _update_human_room_scene_frame(
                human_room_scene.object,
                preview,
                frame_index=int(frame_index),
                update_title=not bool(human_room_player.direction),
            )
        )
        if not updated_in_place:
            human_room_scene.object = _human_room_scene_figure(
                preview,
                go,
                frame_index=int(frame_index),
                radar_orientation_deg=current_dynamic_radar_orientation(),
                room_boxes_override=current_room_boxes_override(),
                diagnostics=frame_diagnostics,
                plot_template=plot_template,
            )
        update_human_room_floor_status()
        diagnostics = human_room_state.get("diagnostics")
        if (
            diagnostics is not None
            and int(diagnostics.source_frame_index) != int(frame_index)
        ):
            human_room_diagnostics_status.object = (
                f"Solver overlays are retained for source frame "
                f"{int(diagnostics.source_frame_index)}; return to that frame "
                "or recompute them for the displayed frame."
            )

    def on_human_room_frame(event):
        if human_room_playback_state["syncing"]:
            return
        frame_index = int(
            np.clip(
                int(event.new),
                int(human_room_frame.start),
                int(human_room_frame.end),
            )
        )
        human_room_playback_state["syncing"] = True
        try:
            human_room_frame.value = frame_index
            human_room_player.direction = 0
            human_room_player.step = int(
                human_room_playback_state["normal_step"]
            )
            human_room_player.value = frame_index
            show_human_room_playing_state(False)
        finally:
            human_room_playback_state["syncing"] = False
        display_human_room_frame(frame_index)

    def on_human_room_player(event):
        if human_room_playback_state["syncing"]:
            return
        frame_index = int(event.new)
        human_room_playback_state["syncing"] = True
        try:
            human_room_frame.value = frame_index
        finally:
            human_room_playback_state["syncing"] = False
        display_human_room_frame(frame_index)

    def on_human_room_player_direction(_event):
        if human_room_playback_state["syncing"]:
            return
        human_room_playback_state["syncing"] = True
        try:
            human_room_player.step = int(
                human_room_playback_state["normal_step"]
            )
            show_human_room_playing_state(
                bool(human_room_player.direction)
            )
        finally:
            human_room_playback_state["syncing"] = False

    def start_fast_human_room_playback(direction: int):
        if human_room_player.disabled:
            return
        human_room_playback_state["syncing"] = True
        try:
            human_room_player.step = int(
                human_room_playback_state["fast_step"]
            )
            human_room_player.direction = int(np.sign(direction))
            show_human_room_playing_state(True)
        finally:
            human_room_playback_state["syncing"] = False

    def start_human_room_playback(direction: int):
        if human_room_player.disabled:
            return
        human_room_playback_state["syncing"] = True
        try:
            human_room_player.step = int(
                human_room_playback_state["normal_step"]
            )
            human_room_player.direction = int(np.sign(direction))
            show_human_room_playing_state(True)
        finally:
            human_room_playback_state["syncing"] = False

    def pause_human_room_playback():
        human_room_playback_state["syncing"] = True
        try:
            human_room_player.direction = 0
            human_room_player.step = int(
                human_room_playback_state["normal_step"]
            )
            show_human_room_playing_state(False)
        finally:
            human_room_playback_state["syncing"] = False
        preview = human_room_state.get("preview")
        figure = human_room_scene.object
        if preview is not None and figure is not None:
            figure.plotly_relayout(
                {
                    "title.text": _human_room_scene_title(
                        preview,
                        int(human_room_player.value),
                    )
                }
            )

    def step_human_room_frame(offset: int):
        if human_room_player.disabled:
            return
        pause_human_room_playback()
        next_frame = int(
            np.clip(
                int(human_room_player.value) + int(offset),
                int(human_room_player.start),
                int(human_room_player.end),
            )
        )
        human_room_player.value = next_frame

    def on_human_room_previous_frame(_event):
        step_human_room_frame(-1)

    def on_human_room_reverse(_event):
        start_human_room_playback(-1)

    def on_human_room_fast_reverse(_event):
        start_fast_human_room_playback(-1)

    def on_human_room_fast_forward(_event):
        start_fast_human_room_playback(1)

    def on_human_room_play_pause(_event):
        if human_room_player.direction:
            pause_human_room_playback()
        else:
            start_human_room_playback(1)

    def on_human_room_next_frame(_event):
        step_human_room_frame(1)

    def on_dynamic_radar_preview_change(_event):
        if gui_session_state["restoring"] or settings_dialog_state["editing"]:
            return
        preview = human_room_state["preview"]
        human_room_scene.object = _human_room_scene_figure(
            preview,
            go,
            frame_index=int(human_room_frame.value),
            radar_orientation_deg=current_dynamic_radar_orientation(),
            room_boxes_override=current_room_boxes_override(),
            plot_template=plot_template,
        )
        mark_human_room_stale(
            "Radar configuration changed; start the simulation again."
        )

    def on_human_run_option_change(_event):
        mark_human_room_stale(
            "Simulation options changed; start the simulation again."
        )

    def on_human_room_window_change(event):
        if human_room_window_state["syncing"]:
            return
        human_room_window_state["syncing"] = True
        try:
            begin_frame = int(human_room_begin_frame.value)
            end_frame = int(human_room_end_frame.value)
            if begin_frame > end_frame:
                if event.obj is human_room_begin_frame:
                    human_room_end_frame.value = begin_frame
                else:
                    human_room_begin_frame.value = end_frame
        finally:
            human_room_window_state["syncing"] = False
        on_human_run_option_change(event)

    human_room_run_input_widgets = (
        settings_button,
        apply_solver_settings,
        *solver_setting_widgets.values(),
        *motion_coordinate_widgets.values(),
        board,
        tdm_enabled,
        tx_antennas,
        rx_antennas,
        antenna_pattern,
        cosine_3db_beamwidth,
        carrier_frequency,
        chirp_slope,
        chirp_duration,
        chirp_repetition,
        sampling_frequency,
        adc_sampling_mode,
        adc_if_bandwidth,
        num_adc_samples,
        num_chirps,
        frame_period,
        human_room_preset,
        human_room_scene_upload,
        human_room_upload,
        human_room_smpl_model_dir,
        human_room_x,
        human_room_y,
        human_room_z,
        human_room_yaw,
        human_room_diffuse,
        human_room_begin_frame,
        human_room_end_frame,
        *human_room_mode_checkboxes.values(),
        dynamic_radar_yaw,
        dynamic_radar_pitch,
        dynamic_radar_roll,
    )

    async def on_run_human_room(_event):
        if human_room_state["running"] or gui_activity_state["dynamic"]:
            return
        source, filename = current_human_room_source()
        if source is None:
            human_room_status.object = (
                "**Human motion required.** Upload an `.npz` first."
            )
            return
        selected_modes = tuple(
            mode
            for mode in HUMAN_ROOM_SIMULATION_MODES
            if human_room_mode_checkboxes[mode].value
        )
        if not selected_modes:
            human_room_status.object = (
                "**Simulation mode required.** Select at least one mode."
            )
            return
        scene_source, scene_filename = current_human_room_scene()
        run_settings = capture_result_settings("dynamic")
        human_room_state["revision"] += 1
        revision = human_room_state["revision"]
        mark_human_room_stale(
            "Simulation is in progress; completed products will appear here."
        )
        cancel_event = threading.Event()
        human_room_state["cancel_event"] = cancel_event
        human_room_state["running"] = True
        human_room_state["diagnostics"] = None
        run_human_room_diagnostics_button.disabled = True
        human_room_diagnostics_status.object = (
            "Solver overlays will be available after the simulation finishes."
        )
        run_human_room.disabled = True
        stop_human_room.disabled = False
        stop_human_room.color = "danger"
        set_gui_activity("dynamic", True)
        human_room_scene.loading = True
        human_room_results.loading = True
        completed_results = {}
        progress_events = queue.SimpleQueue()
        selected_frame_count = (
            int(human_room_end_frame.value)
            - int(human_room_begin_frame.value)
            + 1
        )

        def publish_progress(event):
            # Facet callbacks remain available to the experiment diagnostics;
            # the running display advances only with solver/frame/chirp events.
            if "stage" not in event:
                progress_events.put(dict(event))

        def refresh_progress_status():
            latest = None
            while True:
                try:
                    latest = progress_events.get_nowait()
                except queue.Empty:
                    break
            if latest is None:
                return
            status = _human_room_running_status(latest)
            if human_room_status.object != status:
                human_room_status.object = status

        human_room_status.object = (
            "Running "
            + ", ".join(_HUMAN_ROOM_MODE_LABELS[mode] for mode in selected_modes)
            + "..."
        )
        run_input_states = _lock_widget_disabled_states(
            human_room_run_input_widgets
        )
        try:
            cache_key = current_motion_cache_key(source, filename)
            if cache_key != human_room_state["motion_cache_key"]:
                human_room_state["loaded_motion"] = await asyncio.to_thread(
                    load_amass_motion,
                    source,
                    filename=filename,
                    smpl_model_dir=_effective_smpl_model_dir(
                        human_room_smpl_model_dir.value,
                        default_model_dir,
                        remote_access=remote_access,
                    ),
                )
                human_room_state["motion_cache_key"] = cache_key
            run_arguments = dict(
                human_motion=human_room_state["loaded_motion"],
                human_motion_filename=filename,
                smpl_model_dir=_effective_smpl_model_dir(
                    human_room_smpl_model_dir.value,
                    default_model_dir,
                    remote_access=remote_access,
                ),
                room_preset=human_room_preset.value,
                scene_xml=scene_source,
                scene_xml_filename=scene_filename,
                human_position_m=current_human_position(),
                human_yaw_deg=float(human_room_yaw.value),
                **current_motion_coordinate_arguments(),
                radar_orientation_deg=current_dynamic_radar_orientation(),
                board_model=board.value,
                carrier_frequency_hz=float(carrier_frequency.value) * 1e9,
                slope_hz_per_s=float(chirp_slope.value) * 1e12,
                chirp_duration_s=float(chirp_duration.value) * 1e-6,
                chirp_repetition_time_s=float(chirp_repetition.value) * 1e-6,
                sampling_frequency_hz=float(sampling_frequency.value) * 1e6,
                adc_mode=adc_sampling_mode.value,
                if_bandwidth_hz=adc_if_bandwidth.value * 1e6 or None,
                num_adc_samples=int(num_adc_samples.value),
                num_chirps_per_frame=int(num_chirps.value),
                frame_period_s=float(frame_period.value) * 1e-3,
                tdm_enabled=bool(tdm_enabled.value),
                selected_tx_indices=tuple(tx_antennas.value),
                selected_rx_indices=tuple(rx_antennas.value),
                antenna_pattern_mode=antenna_pattern.value,
                cosine_3db_beamwidth_deg=float(
                    cosine_3db_beamwidth.value
                ),
                diffuse_reflection_coefficient=float(
                    human_room_diffuse.value
                ),
                begin_frame_index=int(human_room_begin_frame.value),
                end_frame_index=int(human_room_end_frame.value),
                coupling_enabled=True,
                coupling_power_threshold_db=float(coupling_power_cutoff.value),
                coupling_max_reflectors=int(coupling_max_count.value),
                rt_samples_per_source=int(rt_samples.value),
                rt_max_paths_per_source=int(rt_path_cap.value),
                rt_max_depth=int(rt_depth.value),
                po_visibility_samples_per_face=1,
                po_integration_mode=po_integration_mode.value,
                po_quadrature_phase_span_scale_rad=float(
                    po_phase_span.value
                ),
                po_curvature_error_rad=float(po_curvature.value),
                po_quadrature_max_refinement_depth=int(
                    po_refinement_depth.value
                ),
                po_quadrature_max_subfaces_per_parent=int(
                    po_subface_cap.value
                ),
                compute_backend="auto",
                cancel_check=cancel_event.is_set,
                progress_callback=publish_progress,
            )

            def run_selected_modes():
                for mode in selected_modes:
                    if cancel_event.is_set():
                        raise InterruptedError("Simulation cancelled by user")
                    publish_progress(
                        {
                            "simulation_mode": mode,
                            "frame_index": 0,
                            "num_frames": selected_frame_count,
                        }
                    )
                    completed_results[mode] = run_human_room_experiment(
                        simulation_mode=mode,
                        **run_arguments,
                    )
                return completed_results

            worker_task = asyncio.create_task(
                asyncio.to_thread(run_selected_modes)
            )
            while not worker_task.done():
                refresh_progress_status()
                await asyncio.sleep(0.15)
            refresh_progress_status()
            results = await worker_task
            if revision == human_room_state["revision"]:
                primary_mode = (
                    "hybrid_po"
                    if "hybrid_po" in results
                    else selected_modes[0]
                )
                result = results[primary_mode]
                human_room_state["result"] = result
                human_room_state["results"] = results
                gui_session_state["dynamic_cacheable"] = True
                human_room_results.objects = [
                    _human_room_results_view(
                        results,
                        pn,
                        go,
                        plot_template=plot_template,
                    )
                ]
                runtime_summary = " · ".join(
                    f"{_HUMAN_ROOM_MODE_LABELS[mode]} "
                    f"{results[mode].runtime_s:.2f} s"
                    for mode in selected_modes
                )
                human_room_status.object = (
                    f"**SIMULATION READY** · {len(results)} mode(s) · "
                    f"ADC `{tuple(result.adc.shape)}` · {runtime_summary}"
                )
                remember_result_settings("dynamic", run_settings)
                human_room_download.callback = lambda: _human_room_export(
                    human_room_state["result"],
                    mode_results=human_room_state["results"],
                ) if human_room_state["result"] is not None else BytesIO()
                human_room_download.disabled = False
                run_human_room_diagnostics_button.disabled = False
                human_room_diagnostics_status.object = (
                    "Choose a displayed source frame within the simulated "
                    "window, then show solver overlays."
                )
                persist_gui_session()
        except InterruptedError:
            if revision == human_room_state["revision"]:
                if completed_results:
                    primary_mode = (
                        "hybrid_po"
                        if "hybrid_po" in completed_results
                        else next(iter(completed_results))
                    )
                    human_room_state["result"] = completed_results[primary_mode]
                    human_room_state["results"] = dict(completed_results)
                    gui_session_state["dynamic_cacheable"] = True
                    human_room_results.objects = [
                        _human_room_results_view(
                            completed_results,
                            pn,
                            go,
                            plot_template=plot_template,
                        )
                    ]
                    human_room_download.callback = lambda: _human_room_export(
                        human_room_state["result"],
                        mode_results=human_room_state["results"],
                    ) if human_room_state["result"] is not None else BytesIO()
                    human_room_download.disabled = False
                    run_human_room_diagnostics_button.disabled = False
                    human_room_diagnostics_status.object = (
                        "Choose a displayed source frame within a retained "
                        "mode's simulated window, then show solver overlays."
                    )
                    retained = (
                        f" {len(completed_results)} completed mode result(s) "
                        "were retained."
                    )
                else:
                    retained = ""
                    if result_records["dynamic"] is None:
                        human_room_results.objects = [pn.pane.Markdown(
                            "The simulation was stopped before any mode "
                            "completed. Start again to generate results."
                        )]
                human_room_status.object = (
                    "**STOPPED** · The current simulation was terminated at "
                    f"the next safe solver boundary.{retained}"
                )
                if completed_results:
                    remember_result_settings("dynamic", run_settings)
                persist_gui_session()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if revision == human_room_state["revision"]:
                human_room_status.object = (
                    "**Simulation failed:** "
                    f"{_safe_markdown_code(f'{type(exc).__name__}: {exc}')}"
                )
                pn.state.notifications.error(str(exc), duration=8000)
                if result_records["dynamic"] is None:
                    human_room_results.objects = [pn.pane.Markdown("The simulation failed. Correct the settings and start again.")]
        finally:
            _restore_widget_disabled_states(run_input_states)
            if revision == human_room_state["revision"]:
                run_human_room.disabled = human_room_state["loaded_motion"] is None
                stop_human_room.disabled = True
                stop_human_room.color = "light"
                human_room_state["running"] = False
                set_gui_activity("dynamic", False)
                human_room_state["cancel_event"] = None
                human_room_scene.loading = False
                human_room_results.loading = False
                refresh_result_validity()

    def on_stop_human_room(_event):
        cancel_event = human_room_state.get("cancel_event")
        diagnostic_cancel = human_room_state.get("diagnostic_cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        if diagnostic_cancel is not None:
            diagnostic_cancel.set()
        if cancel_event is None and diagnostic_cancel is None:
            return
        stop_human_room.disabled = True
        human_room_status.object = (
            "**STOPPING** · Waiting for the current RT/PO kernel to reach a "
            "safe boundary."
        )

    async def on_run_human_room_diagnostics(_event):
        if human_room_state["running"] or gui_activity_state["dynamic"]:
            return
        results = human_room_state.get("results", {})
        if not results:
            human_room_diagnostics_status.object = (
                "Run a simulation before requesting solver overlays."
            )
            return
        source_frame = int(human_room_frame.value)
        representative = (
            results.get("hybrid_po")
            or results.get("full_rt")
            or results.get("coherent_rt")
            or next(iter(results.values()))
        )
        begin_frame = int(
            representative.manifest.scene.parameters[
                "motion_begin_frame_index"
            ]
        )
        end_frame = int(
            representative.manifest.scene.parameters[
                "motion_end_frame_index"
            ]
        )
        if source_frame < begin_frame or source_frame > end_frame:
            human_room_diagnostics_status.object = (
                f"Displayed source frame {source_frame} is outside the "
                f"simulated window {begin_frame}–{end_frame}."
            )
            return
        include_rt = bool(
            {"full_rt", "coherent_rt"}.intersection(results)
        )
        include_po = bool(
            {"human_only_po", "hybrid_po"}.intersection(results)
        )
        human_room_state["diagnostic_revision"] += 1
        diagnostic_revision = human_room_state["diagnostic_revision"]
        requested_top_k = int(human_room_diagnostic_top_k.value)
        result_identity = tuple(
            (mode, id(result)) for mode, result in results.items()
        )
        cancel_event = threading.Event()
        human_room_state["diagnostic_cancel_event"] = cancel_event
        run_human_room_diagnostics_button.disabled = True
        human_room_diagnostic_top_k.disabled = True
        human_room_frame.disabled = True
        run_human_room.disabled = True
        stop_human_room.disabled = False
        set_gui_activity("dynamic", True)
        human_room_scene.loading = True
        included = " and ".join(
            label
            for enabled, label in (
                (include_rt, "RT rays"),
                (include_po, "PO face power"),
            )
            if enabled
        )
        human_room_diagnostics_status.object = (
            f"Computing {included} for source frame {source_frame}..."
        )
        try:
            diagnostics = await asyncio.to_thread(
                run_human_room_diagnostics,
                representative,
                frame_index=source_frame - begin_frame,
                top_k=requested_top_k,
                include_rt=include_rt,
                include_po=include_po,
                cancel_check=cancel_event.is_set,
            )
            current_identity = tuple(
                (mode, id(result))
                for mode, result in human_room_state.get("results", {}).items()
            )
            if (
                diagnostic_revision == human_room_state["diagnostic_revision"]
                and result_identity == current_identity
                and source_frame == int(human_room_frame.value)
                and requested_top_k
                == int(human_room_diagnostic_top_k.value)
            ):
                human_room_state["diagnostics"] = diagnostics
                display_human_room_frame(source_frame)
                summary = []
                if diagnostics.rt is not None:
                    summary.append(
                        f"{diagnostics.rt.selected_path_indices.size} top RT "
                        "ray(s)"
                    )
                if diagnostics.po is not None:
                    summary.append(
                        f"{diagnostics.po.visible_face_count:,} visible PO "
                        "face(s)"
                    )
                human_room_diagnostics_status.object = (
                    f"**OVERLAYS READY** · source frame {source_frame} · "
                    + " · ".join(summary)
                )
                persist_gui_session()
        except InterruptedError:
            if diagnostic_revision == human_room_state["diagnostic_revision"]:
                human_room_diagnostics_status.object = (
                    "**STOPPED** · Solver overlay computation was cancelled."
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if diagnostic_revision == human_room_state["diagnostic_revision"]:
                human_room_diagnostics_status.object = (
                    f"**Overlay computation failed:** "
                    f"{_safe_markdown_code(f'{type(exc).__name__}: {exc}')}"
                )
                pn.state.notifications.error(str(exc), duration=8000)
        finally:
            human_room_state["diagnostic_cancel_event"] = None
            run_human_room_diagnostics_button.disabled = not bool(
                human_room_state.get("results")
            )
            human_room_diagnostic_top_k.disabled = False
            current_preview = human_room_state.get("preview")
            human_room_frame.disabled = (
                current_preview is None
                or len(current_preview.mesh_sequence.times) <= 1
            )
            run_human_room.disabled = False
            stop_human_room.disabled = True
            set_gui_activity("dynamic", False)
            human_room_scene.loading = False

    run_human_room.on_click(on_run_human_room)
    stop_human_room.on_click(on_stop_human_room)
    run_human_room_diagnostics_button.on_click(
        on_run_human_room_diagnostics
    )
    human_room_previous_frame.on_click(on_human_room_previous_frame)
    human_room_reverse.on_click(on_human_room_reverse)
    human_room_fast_reverse.on_click(on_human_room_fast_reverse)
    human_room_play_pause.on_click(on_human_room_play_pause)
    human_room_fast_forward.on_click(on_human_room_fast_forward)
    human_room_next_frame.on_click(on_human_room_next_frame)
    load_motion_preview.on_click(on_human_room_controls)
    human_room_upload.param.watch(on_human_room_controls, "value")
    human_room_upload.param.watch(
        on_human_room_upload_filename,
        "filename",
    )
    human_room_scene_upload.param.watch(on_human_room_controls, "value")
    human_room_smpl_model_dir.param.watch(on_human_room_controls, "value")
    human_room_preset.param.watch(on_human_room_controls, "value")
    for widget in motion_coordinate_widgets.values():
        widget.param.watch(on_motion_coordinate_change, "value")
    for numeric in (
        human_room_x,
        human_room_y,
        human_room_z,
        human_room_yaw,
        human_room_diffuse,
    ):
        numeric.param.watch(on_human_room_controls, "value")
    human_room_frame.param.watch(on_human_room_frame, "value")
    human_room_player.param.watch(on_human_room_player, "value")
    human_room_player.param.watch(
        on_human_room_player_direction,
        "direction",
    )
    human_room_begin_frame.param.watch(
        on_human_room_window_change,
        "value",
    )
    human_room_end_frame.param.watch(
        on_human_room_window_change,
        "value",
    )
    for mode_checkbox in human_room_mode_checkboxes.values():
        mode_checkbox.param.watch(on_human_run_option_change, "value")
    for widget, parameter in (
        (dynamic_radar_yaw, "value"),
        (dynamic_radar_pitch, "value"),
        (dynamic_radar_roll, "value"),
        (board, "value"),
        (tdm_enabled, "value"),
        (tx_antennas, "value"),
        (rx_antennas, "value"),
        (antenna_pattern, "value"),
        (cosine_3db_beamwidth, "value"),
        (carrier_frequency, "value"),
        (chirp_slope, "value"),
        (chirp_duration, "value"),
        (chirp_repetition, "value"),
        (sampling_frequency, "value"),
        (adc_sampling_mode, "value"),
        (adc_if_bandwidth, "value"),
        (num_adc_samples, "value"),
        (num_chirps, "value"),
        (frame_period, "value"),
    ):
        widget.param.watch(on_dynamic_radar_preview_change, parameter)

    human_room_controls = pn.Column(
        _rail_nav_component(pn, (
            ("motion", "Motion"),
            ("modes", "Modes"),
            ("human", "Human"),
            ("radar", "Radar"),
        )),
        _rail_group(
            pn, "motion", "Motion and scene",
            human_room_upload_control,
            load_motion_preview,
            human_room_preset,
            human_room_scene_upload,
            human_room_smpl_model_dir,
        ),
        _rail_group(
            pn, "modes", "Simulation modes",
            human_room_mode_controls,
            pn.Row(
                human_room_begin_frame,
                human_room_end_frame,
                sizing_mode="stretch_width",
            ),
        ),
        _rail_group(
            pn, "human", "Human placement",
            numeric_row("hermes-dynamic-position", human_room_x, human_room_y, human_room_z),
            numeric_row("hermes-dynamic-orientation", human_room_yaw, human_room_diffuse),
        ),
        _rail_group(
            pn, "radar", "Radar orientation",
            numeric_row("hermes-dynamic-radar-orientation", dynamic_radar_yaw, dynamic_radar_pitch, dynamic_radar_roll),
        ),
        _rail_group(pn, "export", "Export", human_room_download),
        sizing_mode="stretch_width",
        min_width=380,
        max_width=380,
        margin=0,
        css_classes=["hermes-rail", "hermes-scene-controls", "hermes-dynamic-controls"],
    )
    human_room_view_tabs = pn.Tabs(
        (
            "Scene and motion",
            pn.Column(
                pn.Row(
                    human_room_frame,
                    human_room_diagnostic_top_k,
                    run_human_room_diagnostics_button,
                    sizing_mode="stretch_width",
                ),
                human_room_diagnostics_status,
                human_room_floor_status,
                human_room_scene_interaction,
                human_room_player,
                human_room_playback_controls,
            ),
        ),
        (
            "Simulation results",
            human_room_results,
        ),
        # Keep both panes mounted.  Completed products can arrive while the
        # result tab is already selected (including after a theme restore);
        # lazy mounting can otherwise leave the browser attached to the old
        # placeholder even though the server has replaced the result column.
        dynamic=False,
        active=0,
        sizing_mode="stretch_width",
    )
    dynamic_result_banner = pn.Column(visible=False, name="Dynamic result settings")
    human_room_tab = pn.Row(
        human_room_controls,
        pn.Column(
            _run_bar(
                pn,
                title="Dynamic Scenes",
                subtitle="Moving human in a prepared or uploaded room.",
                help_text=(
                    "Ray tracing models the static room and physical optics "
                    "models the moving human. Load motion, check its floor "
                    "placement in the scene, choose simulation modes and a "
                    "frame window, then Start. The radar is fixed at (0, 0, 0). "
                    "Orbit the camera to inspect the floor and human placement. "
                    "Axis-aligned cubes from uploaded XML are shown in the "
                    "scene; the solver uses the complete XML scene. Red markers "
                    "show human vertices below the floor surface."
                ),
                status=human_room_status,
                actions=(run_human_room, stop_human_room),
                banner=dynamic_result_banner,
            ),
            pn.Column(
                human_room_view_tabs,
                sizing_mode="stretch_width",
                margin=0,
                css_classes=["hermes-workspace"],
            ),
            sizing_mode="stretch_width",
            min_width=0,
            margin=0,
            styles={"gap": "16px"},
            css_classes=["hermes-work"],
        ),
        sizing_mode="stretch_width",
        css_classes=["hermes-page"],
    )

    fixture_options = _source_fixture_options()
    for fixture_path in fixture_options.values():
        if fixture_path:
            trusted_bundle_paths.add(_server_path_key(fixture_path))
    if initial_bundle:
        # ``initial_bundle`` is supplied by the server operator on the CLI,
        # not by a browser session.
        trusted_bundle_paths.add(_server_path_key(initial_bundle))
    default_fixture = next(
        (
            value
            for label, value in fixture_options.items()
            if label == "rtpose_seq10_frame0020"
        ),
        "",
    )
    fixture = pn.widgets.Select(
        label="Measurement benchmark",
        options=fixture_options,
        value="" if initial_bundle else default_fixture,
    )
    bundle_path = pn.widgets.TextInput(
        label="HERMES bundle directory or ZIP",
        value=str(initial_bundle or fixture.value),
        placeholder="/path/to/hermes_bundle or hermes-bundle.zip",
        description=(
            "Direct server-side path entry is disabled for remote sessions; "
            "use an included fixture or browser upload."
        ),
        disabled=remote_access,
    )
    measurement_dropper = pn.widgets.FileDropper(
        label="Drop primary bundle ZIP or folder",
        multiple=True,
        previews=[],
        visible=fixture.value == "",
        max_file_size="256MB",
        max_files=_MAX_DROPPED_FILES,
        max_total_file_size="512MB",
        sizing_mode="stretch_width",
    )
    custom_bundle_path = {"value": str(initial_bundle or "")}

    def select_measurement_source(event):
        custom = event.new == ""
        measurement_dropper.visible = custom
        bundle_path.value = (
            custom_bundle_path["value"] if custom else str(event.new)
        )

    fixture.param.watch(select_measurement_source, "value")
    background = pn.widgets.Checkbox(
        label="Background subtraction",
        value=False,
        stylesheets=[_checkbox_stylesheet(plot_template)],
    )
    clutter = pn.widgets.Select(
        label="Clutter removal",
        options=["none", "mean"],
        value="none",
    )
    load_bundle_button = pn.widgets.Button(
        label="Load and validate bundle",
        color="primary",
        icon="folder-open",
    )
    measurement_status = pn.pane.Markdown(
        "",
        min_height=76,
        margin=(2, 10),
        css_classes=["hermes-status", "hermes-bundle-status"],
    )
    bundle_step_one_marker = pn.pane.HTML(
        "",
        width=30,
        height=30,
        sizing_mode="fixed",
        margin=0,
    )
    bundle_step_two_marker = pn.pane.HTML(
        "",
        width=30,
        height=30,
        sizing_mode="fixed",
        margin=0,
    )
    bundle_step_one_label = pn.pane.HTML(
        "<strong>Step 1 · Load and validate</strong>",
        height=30,
        margin=0,
    )
    bundle_step_two_label = pn.pane.HTML(
        "<strong>Step 2 · Configure and compare</strong>",
        height=30,
        margin=0,
    )

    def bundle_step_heading(marker, label):
        return pn.Row(
            marker,
            label,
            sizing_mode="stretch_width",
            margin=(0, 10),
            styles={"align-items": "center", "gap": "8px"},
        )
    bundle_scene_status = pn.pane.Markdown(
        "Load the primary bundle to visualize its declared scene."
    )
    bundle_scene_plot = pn.pane.Plotly(_empty_figure(go, plot_template), height=570)
    primary_adc_plot = pn.pane.Plotly(
        _empty_figure(go, plot_template),
        height=330,
        sizing_mode="stretch_width",
    )
    adc_selectors = {
        name: pn.widgets.IntInput(label=f"ADC {name}", value=0, start=0, end=0,
                                  disabled=True, width=130)
        for name in ("frame", "chirp", "channel")
    }
    candidate_adc_results = pn.Column(
        sizing_mode="stretch_width",
        visible=False,
    )
    range_profile_plot = pn.pane.Plotly(
        _empty_figure(go, plot_template),
        height=350,
        sizing_mode="stretch_width",
    )
    primary_range_time_plot = pn.pane.Plotly(
        _empty_figure(go, plot_template),
        height=380,
        sizing_mode="stretch_width",
    )
    candidate_range_time_results = pn.Column(
        sizing_mode="stretch_width",
        visible=False,
    )
    primary_range_doppler_plot = pn.pane.Plotly(
        _empty_figure(go, plot_template),
        height=380,
        sizing_mode="stretch_width",
    )
    candidate_range_doppler_results = pn.Column(
        sizing_mode="stretch_width",
        visible=False,
    )
    comparison_source = pn.widgets.Select(
        label="Comparison source",
        options={
            "Run simulation from loaded bundle": "run",
            "Load previously saved ADC or bundle": "saved",
        },
        value="run",
    )
    comparison_mode_options = {
        "Human-only PO": "human_only_po",
        "Full RT": "rt_retrace",
        "Coherent RT": "rt_coherent_bank",
        "Hybrid PO": "hybrid_static_env_po",
    }
    comparison_simulation_modes = pn.widgets.MultiChoice(
        label="Bundle simulation modes",
        options=comparison_mode_options,
        value=["human_only_po"],
        placeholder="+ Add simulation mode",
    )
    comparison_channel_zero = pn.widgets.Checkbox(
        label="Channel 0 only",
        value=False,
        stylesheets=[_checkbox_stylesheet(plot_template)],
    )
    simulated_path = pn.widgets.TextInput(
        label="External simulated ADC or HERMES bundle",
        placeholder="/path/to/simulated_adc.npz, bundle ZIP, or directory",
        description=(
            "Direct server-side path entry is disabled for remote sessions; "
            "upload the candidate through the browser."
        ),
        disabled=remote_access,
    )
    simulated_dropper = pn.widgets.FileDropper(
        label="Drop simulated ADC, bundle ZIP, or folder",
        multiple=True,
        previews=[],
        visible=False,
        max_file_size="256MB",
        max_files=_MAX_DROPPED_FILES,
        max_total_file_size="512MB",
        sizing_mode="stretch_width",
    )
    compare_button = pn.widgets.Button(
        label="Compare ADC products",
        color="primary",
        icon="chart-histogram",
        disabled=True,
        height=46,
        sizing_mode="stretch_width",
        margin=0,
    )
    simulated_adc_download = pn.widgets.FileDownload(
        callback=lambda: BytesIO(),
        filename="simulated_adc.npz",
        label="\u200b",
        color="light",
        icon="download",
        description="Download simulated ADC",
        width=44,
        height=40,
        sizing_mode="fixed",
        margin=0,
        disabled=True,
        css_classes=["hermes-icon-download"],
    )
    comparison_dependency_hint = pn.pane.Markdown(
        "🔒 **Load and validate a primary bundle to enable this step.**",
        margin=(2, 0, 4, 0),
        css_classes=["hermes-compact-hint"],
        styles={
            "color": _plot_theme_tokens(plot_template)["foreground"],
            "opacity": "0.78",
        },
    )
    stop_simulation_confirmation = _confirmation_dialog_component(pn)
    comparison_status = pn.pane.Markdown(
        "",
        visible=False,
        css_classes=["hermes-status", "hermes-comparison-output"],
        styles={"color": _plot_theme_tokens(plot_template)["foreground"]},
    )
    metrics_pane = pn.pane.JSON(
        {"status": "No comparison."},
        depth=3,
        css_classes=["hermes-comparison-output"],
        styles={"color": _plot_theme_tokens(plot_template)["foreground"]},
        stylesheets=[_comparison_json_stylesheet(plot_template)],
    )
    measurement_state = {
        "bundle": None,
        "comparisons": {},
        "adc_primary_products": None,
        "adc_candidates": [],
        "configuring_adc": False,
        "comparison_kind": None,
        "comparison_cache_key": None,
        "comparison_signature": None,
        "comparison_revision": 0,
        "simulated_adcs": {},
        "cancel_event": None,
        "running": False,
        "action_running": False,
        "load_revision": 0,
        "loading": False,
        "configuring": False,
        "untrusted_bundle": False,
        "remote_resimulation_allowed": True,
        "drop_workspace": None,
        "drop_sequence": 0,
        "drop_tokens": {"measurement": 0, "simulated": 0},
    }
    bundle_palette = _plot_theme_tokens(plot_template)

    def refresh_adc_selection(_event=None):
        products = measurement_state["adc_primary_products"]
        if products is None or measurement_state["configuring_adc"]:
            return
        loaded = measurement_state["bundle"]
        selection = {f"{name}_index": int(widget.value) for name, widget in adc_selectors.items()}
        suffix = " / ".join(f"{name} {widget.value}" for name, widget in adc_selectors.items())
        primary_adc_plot.object = _adc_trace_figure(
            products, go, title=f"{_bundle_primary_label(loaded)} ADC · {suffix}",
            plot_template=plot_template, **selection,
        )
        candidate_adc_results.objects = [
            pn.pane.Plotly(_adc_trace_figure(
                candidate, go, title=f"{label} ADC · {suffix}",
                plot_template=plot_template, **selection,
            ), height=330, sizing_mode="stretch_width")
            for label, candidate in measurement_state["adc_candidates"]
        ]

    def configure_adc_selection(products):
        measurement_state["adc_primary_products"] = products
        measurement_state["configuring_adc"] = True
        try:
            shape = products.adc.shape
            for widget, limit in zip(adc_selectors.values(), (shape[0], shape[1], shape[3])):
                widget.end = max(int(limit) - 1, 0)
                widget.value = min(int(widget.value), widget.end)
                widget.disabled = False
        finally:
            measurement_state["configuring_adc"] = False
        refresh_adc_selection()

    for selector in adc_selectors.values():
        selector.param.watch(refresh_adc_selection, "value")

    def current_comparison_signature() -> tuple[object, ...]:
        loaded = measurement_state.get("bundle")
        return (
            getattr(loaded, "fingerprint", None),
            str(comparison_source.value),
            tuple(str(mode) for mode in comparison_simulation_modes.value),
            bool(comparison_channel_zero.value),
            bool(background.value),
            str(clutter.value),
            str(simulated_path.value or ""),
        )

    def bundle_step_marker(number: int, state: str) -> str:
        if state == "complete":
            content = "&#10003;"
            background = "#3ecf8e"
            border = "#3ecf8e"
            foreground = "#06141f"
            shadow = "none"
        elif state == "active":
            content = str(number)
            background = "transparent"
            border = bundle_palette["accent"]
            foreground = bundle_palette["accent"]
            shadow = f"0 0 0 4px {bundle_palette['accent']}24"
        else:
            content = str(number)
            background = "transparent"
            border = bundle_palette["axis"]
            foreground = bundle_palette["axis"]
            shadow = "none"
        return (
            '<div style="align-items:center; background:'
            f"{background}; border:2px solid {border}; border-radius:50%; "
            f"box-shadow:{shadow}; color:{foreground}; display:flex; "
            'font-family:ui-monospace,monospace; font-size:12px; '
            'font-weight:700; height:26px; justify-content:center; '
            f'width:26px;">{content}</div>'
        )

    def update_bundle_step_rail(stage: int) -> None:
        stage = int(np.clip(stage, 0, 2))
        first_state = "complete" if stage >= 1 else "active"
        second_state = (
            "complete" if stage >= 2 else "active" if stage == 1 else "idle"
        )
        bundle_step_one_marker.object = bundle_step_marker(1, first_state)
        bundle_step_two_marker.object = bundle_step_marker(2, second_state)
        bundle_step_one_label.styles = {
            "align-items": "center",
            "color": (
                bundle_palette["foreground"]
                if stage >= 0
                else bundle_palette["axis"]
            ),
            "display": "flex",
            "height": "30px",
        }
        bundle_step_two_label.styles = {
            "align-items": "center",
            "color": (
                bundle_palette["foreground"]
                if stage >= 1
                else bundle_palette["axis"]
            ),
            "display": "flex",
            "height": "30px",
        }

    def set_measurement_status(kind: str, title: str, detail: str) -> None:
        colors = {
            "idle": (bundle_palette["axis"], "rgba(100, 116, 139, 0.08)"),
            "info": (bundle_palette["accent"], "rgba(79, 163, 216, 0.10)"),
            "success": ("#3ecf8e", "rgba(62, 207, 142, 0.11)"),
            "error": ("#ef4444", "rgba(239, 68, 68, 0.10)"),
        }
        border, background_color = colors[kind]
        measurement_status.object = f"**{title}**  \n{detail}"
        measurement_status.styles = {
            "background": background_color,
            "border-left": f"3px solid {border}",
            "border-radius": "8px",
            "color": bundle_palette["foreground"],
            "padding": "12px 14px",
        }

    def set_comparison_status(
        kind: str,
        title: str,
        detail: str = "",
    ) -> None:
        colors = {
            "info": (bundle_palette["accent"], "rgba(79, 163, 216, 0.10)"),
            "success": ("#3ecf8e", "rgba(62, 207, 142, 0.11)"),
            "error": ("#ef4444", "rgba(239, 68, 68, 0.10)"),
        }
        border, background_color = colors[kind]
        comparison_status.object = (
            f"**{title}**" + (f"  \n{detail}" if detail else "")
        )
        comparison_status.styles = {
            "background": background_color,
            "border-left": f"3px solid {border}",
            "border-radius": "8px",
            "color": bundle_palette["foreground"],
            "padding": "12px 14px",
        }
        comparison_status.visible = True

    set_measurement_status(
        "idle",
        "No bundle loaded yet",
        "Choose an included fixture or custom bundle, then load and validate it.",
    )
    update_bundle_step_rail(0)
    candidate_product_results = (
        candidate_adc_results,
        candidate_range_time_results,
        candidate_range_doppler_results,
    )
    primary_product_plots = (
        primary_adc_plot,
        range_profile_plot,
        primary_range_time_plot,
        primary_range_doppler_plot,
    )

    def clear_candidate_products():
        result_records["comparison"] = None
        measurement_state["comparisons"] = {}
        measurement_state["comparison_kind"] = None
        measurement_state["comparison_cache_key"] = None
        measurement_state["comparison_signature"] = None
        measurement_state["simulated_adcs"] = {}
        measurement_state["adc_candidates"] = []
        metrics_pane.object = {"status": "No comparison."}
        for results in candidate_product_results:
            results.objects = []
            results.visible = False
        simulated_adc_download.callback = lambda: BytesIO()
        simulated_adc_download.filename = "simulated_adc.npz"
        simulated_adc_download.disabled = True
        update_bundle_step_rail(
            1 if measurement_state["bundle"] is not None else 0
        )

    def clear_primary_products():
        measurement_state["adc_primary_products"] = None
        for selector in adc_selectors.values():
            selector.disabled = True
        for pane in primary_product_plots:
            pane.object = _empty_figure(go, plot_template)
        bundle_scene_plot.object = _empty_figure(go, plot_template)
        bundle_scene_status.object = (
            "Load the primary bundle to visualize its declared scene."
        )
        clear_candidate_products()

    def invalidate_loaded_measurement(_event):
        measurement_state["load_revision"] += 1
        measurement_state["comparison_revision"] += 1
        cancel_event = measurement_state.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        measurement_state["bundle"] = None
        clear_primary_products()
        comparison_status.object = ""
        comparison_status.visible = False
        set_measurement_status(
            "idle",
            "Bundle selection changed",
            "Load and validate the selected bundle to enable comparison.",
        )
        update_bundle_step_rail(0)
        update_comparison_source_visibility()

    bundle_path.param.watch(invalidate_loaded_measurement, "value")

    def materialize_drop(kind, uploaded, path_input, status_pane):
        try:
            workspace = measurement_state["drop_workspace"]
            if workspace is None:
                workspace = TemporaryDirectory(prefix="hermes-gui-drops-")
                measurement_state["drop_workspace"] = workspace
            measurement_state["drop_sequence"] += 1
            destination = Path(workspace.name) / (
                f"{kind}-{measurement_state['drop_sequence']:04d}"
            )
            selected = _materialize_dropped_files(uploaded, destination)
            trusted_paths = (
                trusted_bundle_paths
                if kind == "measurement"
                else trusted_simulated_paths
            )
            selected_key = _server_path_key(selected)
            trusted_paths.add(selected_key)
            uploaded_paths = (
                uploaded_bundle_paths
                if kind == "measurement"
                else uploaded_simulated_paths
            )
            uploaded_paths.add(selected_key)
            path_input.value = str(selected)
            if kind == "measurement":
                custom_bundle_path["value"] = str(selected)
                set_measurement_status(
                    "info",
                    "Bundle upload ready",
                    f"{_safe_markdown_code(selected.name)} · "
                    f"{len(uploaded):,} file(s). "
                    "Load and validate it to continue.",
                )
            else:
                set_comparison_status(
                    "info",
                    "Saved comparison ready",
                    f"{_safe_markdown_code(selected.name)} · "
                    f"{len(uploaded):,} file(s).",
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if kind == "measurement":
                set_measurement_status(
                    "error",
                    "Bundle upload failed",
                    _safe_markdown_code(f"{type(exc).__name__}: {exc}"),
                )
            else:
                set_comparison_status(
                    "error",
                    "Saved comparison upload failed",
                    _safe_markdown_code(f"{type(exc).__name__}: {exc}"),
                )
            pn.state.notifications.error(str(exc), duration=8000)

    def bind_file_dropper(kind, dropper, path_input, status_pane):
        def receive_drop(event):
            if not event.new:
                return
            measurement_state["drop_tokens"][kind] += 1
            token = measurement_state["drop_tokens"][kind]
            uploaded = dict(event.new)
            def settle_folder_upload():
                if token != measurement_state["drop_tokens"][kind]:
                    return
                materialize_drop(kind, uploaded, path_input, status_pane)

            document = pn.state.curdoc
            if document is not None and document.session_context is not None:
                document.add_timeout_callback(settle_folder_upload, 250)
            else:
                materialize_drop(kind, uploaded, path_input, status_pane)

        dropper.param.watch(receive_drop, "value")

    bind_file_dropper(
        "measurement",
        measurement_dropper,
        bundle_path,
        measurement_status,
    )
    bind_file_dropper(
        "simulated",
        simulated_dropper,
        simulated_path,
        comparison_status,
    )

    async def on_load_bundle(_event):
        if measurement_state["loading"]:
            return
        measurement_state["loading"] = True
        measurement_state["load_revision"] += 1
        load_revision = measurement_state["load_revision"]
        requested_path = str(bundle_path.value)
        measurement_state["bundle"] = None
        measurement_state["untrusted_bundle"] = False
        measurement_state["remote_resimulation_allowed"] = True
        clear_primary_products()
        set_gui_activity("bundle_load", True)
        load_input_states = _lock_widget_disabled_states(
            (
                fixture,
                measurement_dropper,
                bundle_path,
                background,
                clutter,
            )
        )
        load_bundle_button.disabled = True
        compare_button.disabled = True
        set_measurement_status(
            "info",
            "Validating bundle",
            "Checking the bundle contract and computing scene and radar products...",
        )
        try:
            requested_path = _authorized_server_path(
                requested_path,
                remote_access=remote_access,
                trusted_paths=trusted_bundle_paths,
            )
            requested_key = _server_path_key(requested_path)
            untrusted_bundle = requested_key in uploaded_bundle_paths
            loaded = await asyncio.to_thread(
                load_bundle,
                requested_path,
                resource_limits=(
                    GUI_BUNDLE_RESOURCE_LIMITS
                    if untrusted_bundle
                    else None
                ),
            )
            if (
                load_revision != measurement_state["load_revision"]
                or requested_path != str(bundle_path.value)
            ):
                return
            products = await asyncio.to_thread(
                loaded.primary_products,
                background_subtraction=background.value,
                clutter_removal=clutter.value,
            )
            if (
                load_revision != measurement_state["load_revision"]
                or requested_path != str(bundle_path.value)
            ):
                return
            scene_figure = await asyncio.to_thread(
                _bundle_scene_figure,
                loaded,
                go,
                smpl_model_dir=(
                    _effective_smpl_model_dir(
                        human_room_smpl_model_dir.value,
                        default_model_dir,
                        remote_access=remote_access,
                    )
                    or None
                ),
                plot_template=plot_template,
            )
            if (
                load_revision != measurement_state["load_revision"]
                or requested_path != str(bundle_path.value)
            ):
                return
            remote_resimulation_allowed = not (
                remote_access and untrusted_bundle
            )
            measurement_state["bundle"] = loaded
            measurement_state["untrusted_bundle"] = untrusted_bundle
            measurement_state["remote_resimulation_allowed"] = (
                remote_resimulation_allowed
            )
            clear_candidate_products()
            source_options = {}
            if loaded.can_resimulate and remote_resimulation_allowed:
                source_options["Run simulation from loaded bundle"] = "run"
            source_options["Load previously saved ADC or bundle"] = "saved"
            for solver in loaded.available_simulations:
                source_options[
                    f"Previously saved {solver.upper()} ADC in loaded bundle"
                ] = solver
            measurement_state["configuring"] = True
            try:
                comparison_source.options = source_options
                if loaded.can_resimulate and remote_resimulation_allowed:
                    comparison_source.value = "run"
                elif "rt" in loaded.available_simulations:
                    comparison_source.value = "rt"
                elif loaded.available_simulations:
                    comparison_source.value = loaded.available_simulations[0]
                else:
                    comparison_source.value = "saved"
            finally:
                measurement_state["configuring"] = False
            primary_label = _bundle_primary_label(loaded)
            bundle_scene_plot.object = scene_figure
            scene_meta = dict(bundle_scene_plot.object.layout.meta or {})
            if (
                loaded.bundle.motion_path is not None
                and not scene_meta.get("bundle_has_target_preview")
            ):
                bundle_scene_status.object = (
                    "**Human preview unavailable:** "
                    f"{_safe_markdown_code(scene_meta.get('bundle_mesh_error', 'unknown error'))}"
                    "  \n"
                    "Configure the licensed SMPL model directory under "
                    "Dynamic Scenes, then reload this bundle."
                )
            else:
                mesh_source = scene_meta.get("bundle_mesh_source")
                preview_note = (
                    f" The target uses the bundle's **{mesh_source}** at "
                    "the primary frame time."
                    if mesh_source
                    else ""
                )
                bundle_scene_status.object = (
                    "The scene is shown from the primary bundle only; "
                    "candidate ADC comparisons do not replace its geometry."
                    + preview_note
                )
            primary_adc_plot.object = _adc_trace_figure(
                products,
                go,
                title=f"{primary_label} ADC · frame 0 / chirp 0 / channel 0",
                plot_template=plot_template,
            )
            configure_adc_selection(products)
            range_profile_plot.object = _bundle_range_profile_comparison_figure(
                primary_label,
                products,
                [],
                go,
                plot_template=plot_template,
            )
            primary_range_time_plot.object = _range_time_figure(
                products,
                go,
                title=f"{primary_label} range time",
                plot_template=plot_template,
            )
            primary_range_doppler_plot.object = _range_doppler_figure(
                products,
                go,
                title=f"{primary_label} range Doppler",
                plot_template=plot_template,
            )
            adc_shape = tuple(int(value) for value in loaded.primary_adc.shape)
            frame_count = adc_shape[0] if adc_shape else 0
            set_measurement_status(
                "success",
                "Bundle loaded and validated",
                f"{_safe_markdown_code(loaded.profile)} / "
                f"{_safe_markdown_code(loaded.data_origin)} · ready for "
                "simulation or saved-ADC comparison.  \n"
                f"Frames `{frame_count:,}` · ADC shape `{adc_shape}` · "
                f"fingerprint {_safe_markdown_code(loaded.fingerprint[:12])}",
            )
            update_bundle_step_rail(1)
            update_comparison_source_visibility()
            persist_gui_session()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if load_revision == measurement_state["load_revision"]:
                measurement_state["bundle"] = None
                clear_primary_products()
                set_measurement_status(
                    "error",
                    "Bundle validation failed",
                    _safe_markdown_code(f"{type(exc).__name__}: {exc}"),
                )
                update_bundle_step_rail(0)
                update_comparison_source_visibility()
                pn.state.notifications.error(str(exc), duration=8000)
        finally:
            _restore_widget_disabled_states(load_input_states)
            measurement_state["loading"] = False
            load_bundle_button.disabled = False
            set_gui_activity("bundle_load", False)
            update_comparison_source_visibility()

    def export_simulated_adc():
        simulated_adcs = measurement_state.get("simulated_adcs", {})
        loaded = measurement_state.get("bundle")
        if (
            not simulated_adcs
            or loaded is None
            or measurement_state.get("comparison_signature")
            != current_comparison_signature()
        ):
            return BytesIO()

        def adc_payload(simulated):
            arrays = {"adc": np.asarray(simulated)}
            times = loaded.bundle.times
            if (
                times is not None
                and np.asarray(times).shape == np.asarray(simulated).shape[:-2]
            ):
                arrays["times"] = np.asarray(times)
            return _npz_bytes(**arrays)

        if len(simulated_adcs) == 1:
            return BytesIO(adc_payload(next(iter(simulated_adcs.values()))))
        archive = BytesIO()
        with zipfile.ZipFile(
            archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as exported:
            for mode, simulated in simulated_adcs.items():
                exported.writestr(
                    f"simulated_adc_{mode}.npz",
                    adc_payload(simulated),
                )
        archive.seek(0)
        return archive

    def on_confirm_stop_simulation(_event):
        stop_simulation_confirmation.open = False
        cancel_event = measurement_state.get("cancel_event")
        if cancel_event is None:
            return
        cancel_event.set()
        compare_button.disabled = True
        compare_button.label = "Stopping simulation..."
        set_comparison_status(
            "info",
            "Stopping simulation",
            "Waiting for the next safe solver boundary.",
        )

    stop_simulation_confirmation.param.watch(
        on_confirm_stop_simulation,
        "confirmation_sequence",
    )

    def render_comparison_results(
        results,
        *,
        comparison_kind: str,
        configuration_signature: tuple[object, ...] | None = None,
        result_settings=None,
    ):
        loaded = measurement_state["bundle"]
        if loaded is None or not results:
            return
        comparisons = {
            key: comparison
            for key, _label, _heading, _detail, _adc, comparison in results
        }
        measurement_state["comparisons"] = comparisons
        measurement_state["comparison_kind"] = comparison_kind
        measurement_state["comparison_signature"] = (
            current_comparison_signature()
            if configuration_signature is None
            else configuration_signature
        )
        metrics = {
            detail: comparison.metrics.to_dict()
            for _key, _label, _heading, detail, _adc, comparison in results
        }
        metrics_pane.object = (
            next(iter(metrics.values())) if len(metrics) == 1 else metrics
        )
        primary_products = results[0][-1].primary
        primary_label = _bundle_primary_label(loaded)
        primary_adc_plot.object = _adc_trace_figure(
            primary_products,
            go,
            title=f"{primary_label} ADC · frame 0 / chirp 0 / channel 0",
            plot_template=plot_template,
        )
        measurement_state["adc_candidates"] = [
            (f"{heading} {detail}", comparison.candidate)
            for _key, _label, heading, detail, _adc, comparison in results
        ]
        configure_adc_selection(primary_products)
        range_profile_plot.object = _bundle_range_profile_comparison_figure(
            primary_label,
            primary_products,
            [
                (f"{heading} {detail}", comparison.candidate)
                for (
                    _key,
                    _label,
                    heading,
                    detail,
                    _adc,
                    comparison,
                ) in results
            ],
            go,
            plot_template=plot_template,
        )
        primary_range_time_plot.object = _range_time_figure(
            primary_products,
            go,
            title=f"{primary_label} range time",
            plot_template=plot_template,
        )
        primary_range_doppler_plot.object = _range_doppler_figure(
            primary_products,
            go,
            title=f"{primary_label} range Doppler",
            plot_template=plot_template,
        )
        candidate_range_time_results.objects = [
            pn.pane.Plotly(
                _range_time_figure(
                    comparison.candidate,
                    go,
                    title=f"{heading} range time · {detail}",
                    plot_template=plot_template,
                ),
                height=380,
                sizing_mode="stretch_width",
            )
            for _key, _label, heading, detail, _adc, comparison in results
        ]
        candidate_range_doppler_results.objects = [
            pn.pane.Plotly(
                _range_doppler_figure(
                    comparison.candidate,
                    go,
                    title=f"{heading} range Doppler · {detail}",
                    plot_template=plot_template,
                ),
                height=380,
                sizing_mode="stretch_width",
            )
            for _key, _label, heading, detail, _adc, comparison in results
        ]
        for candidate_results in candidate_product_results:
            candidate_results.visible = True
        if comparison_kind == "run":
            measurement_state["simulated_adcs"] = {
                key: adc
                for key, _label, _heading, _detail, adc, _comparison in results
            }
            simulated_adc_download.callback = export_simulated_adc
            result_modes = [result[0] for result in results]
            simulated_adc_download.filename = (
                f"simulated_adc_{result_modes[0]}.npz"
                if len(result_modes) == 1
                else "simulated_adc_results.zip"
            )
            simulated_adc_download.disabled = False
        candidate_labels = ", ".join(result[1] for result in results)
        set_comparison_status(
            "success",
            "Comparison complete",
            f"Primary bundle vs {_safe_markdown_text(candidate_labels)}.",
        )
        update_bundle_step_rail(2)
        remember_result_settings("comparison", result_settings)

    async def on_compare(_event):
        if measurement_state["running"]:
            stop_simulation_confirmation.open = True
            return
        if measurement_state["action_running"]:
            return
        loaded = measurement_state["bundle"]
        if loaded is None:
            return

        comparison_source_value = str(comparison_source.value)
        run_simulation = comparison_source_value == "run"
        if run_simulation and not measurement_state.get(
            "remote_resimulation_allowed",
            True,
        ):
            set_comparison_status(
                "error",
                "Remote resimulation disabled",
                "Bundles uploaded to a remotely bound GUI cannot execute "
                "scene references. Use a server-configured trusted bundle.",
            )
            return
        selected_modes = list(comparison_simulation_modes.value)
        if run_simulation and not selected_modes:
            set_comparison_status(
                "error",
                "Simulation mode required",
                "Select at least one mode to continue.",
            )
            return
        measurement_state["comparison_revision"] += 1
        comparison_revision = measurement_state["comparison_revision"]
        configuration_signature = current_comparison_signature()
        run_settings = capture_result_settings("comparison")
        background_value = bool(background.value)
        clutter_value = str(clutter.value)
        channel_zero_value = bool(comparison_channel_zero.value)
        simulated_path_value = str(simulated_path.value or "")
        measurement_state["action_running"] = True
        set_gui_activity("comparison", True)
        run_input_states = _lock_widget_disabled_states(
            (
                fixture,
                measurement_dropper,
                bundle_path,
                background,
                clutter,
                load_bundle_button,
                comparison_source,
                comparison_simulation_modes,
                comparison_channel_zero,
                simulated_dropper,
                simulated_path,
            )
        )
        cancel_event = None
        clear_candidate_products()
        if run_simulation:
            cancel_event = threading.Event()
            measurement_state["cancel_event"] = cancel_event
            measurement_state["running"] = True
            compare_button.disabled = False
            compare_button.label = "Stop simulation"
            compare_button.icon = "player-stop"
            compare_button.color = "danger"
        else:
            compare_button.disabled = True

        try:
            results = []
            if run_simulation:
                from validation.runner import simulate_bundle_adc

                mode_labels = {
                    mode: label
                    for label, mode in comparison_mode_options.items()
                }
                for index, mode in enumerate(selected_modes, start=1):
                    if cancel_event.is_set():
                        raise InterruptedError("Simulation cancelled by user")
                    mode_label = mode_labels[mode]
                    set_comparison_status(
                        "info",
                        f"Running {mode_label}",
                        f"Mode {index} of {len(selected_modes)}.",
                    )
                    worker_task = asyncio.create_task(
                        asyncio.to_thread(
                            simulate_bundle_adc,
                            loaded.bundle,
                            mobility_mode=mode,
                            smpl_model_dir=(
                                _effective_smpl_model_dir(
                                    human_room_smpl_model_dir.value,
                                    default_model_dir,
                                    remote_access=remote_access,
                                )
                                or None
                            ),
                            channel_zero_only=channel_zero_value,
                            cancel_check=cancel_event.is_set,
                        )
                    )
                    while not worker_task.done():
                        await asyncio.sleep(0.15)
                    simulated = await worker_task
                    if cancel_event.is_set():
                        raise InterruptedError("Simulation cancelled by user")
                    comparison = await asyncio.to_thread(
                        loaded.compare_adc,
                        simulated,
                        background_subtraction=background_value,
                        clutter_removal=clutter_value,
                        channel_indices=(
                            (0,) if channel_zero_value else None
                        ),
                    )
                    results.append(
                        (
                            mode,
                            f"new {mode_label} simulation",
                            "Simulated",
                            mode_label,
                            np.asarray(simulated),
                            comparison,
                        )
                    )
            elif comparison_source_value == "saved":
                set_comparison_status(
                    "info",
                    "Comparing saved ADC",
                    "Loading and processing the selected candidate.",
                )
                simulated_path_value = _authorized_server_path(
                    simulated_path_value,
                    remote_access=remote_access,
                    trusted_paths=trusted_simulated_paths,
                )
                candidate_path = Path(simulated_path_value).expanduser()
                untrusted_candidate = (
                    _server_path_key(candidate_path)
                    in uploaded_simulated_paths
                )
                if candidate_path.is_dir() or (
                    candidate_path.suffix.casefold() == ".zip"
                ):
                    candidate_bundle = await asyncio.to_thread(
                        load_bundle,
                        candidate_path,
                        resource_limits=(
                            GUI_BUNDLE_RESOURCE_LIMITS
                            if untrusted_candidate
                            else None
                        ),
                    )
                    simulated = candidate_bundle.primary_adc
                    candidate_label = (
                        f"external {candidate_bundle.data_origin} bundle"
                    )
                else:
                    simulated = await asyncio.to_thread(
                        load_simulated_adc,
                        candidate_path,
                        npz_limits=(
                            GUI_NPZ_LIMITS
                            if untrusted_candidate
                            else None
                        ),
                    )
                    candidate_label = "external ADC"
                comparison = await asyncio.to_thread(
                    loaded.compare_adc,
                    simulated,
                    background_subtraction=background_value,
                    clutter_removal=clutter_value,
                )
                results.append(
                    (
                        "saved",
                        candidate_label,
                        "Loaded",
                        candidate_label.title(),
                        np.asarray(simulated),
                        comparison,
                    )
                )
            else:
                set_comparison_status(
                    "info",
                    "Comparing bundled ADC",
                    "Loading and processing the selected solver result.",
                )
                simulated = await asyncio.to_thread(
                    loaded.load_bundled_simulation,
                    comparison_source_value,
                )
                candidate_label = f"bundled {comparison_source_value.upper()}"
                comparison = await asyncio.to_thread(
                    loaded.compare_adc,
                    simulated,
                    background_subtraction=background_value,
                    clutter_removal=clutter_value,
                )
                results.append(
                    (
                        comparison_source_value,
                        candidate_label,
                        "Loaded",
                        candidate_label,
                        np.asarray(simulated),
                        comparison,
                    )
                )

            comparison_kind = "run" if run_simulation else "saved"
            if (
                comparison_revision
                != measurement_state["comparison_revision"]
                or measurement_state["bundle"] is not loaded
                or configuration_signature != current_comparison_signature()
            ):
                return
            render_comparison_results(
                results,
                comparison_kind=comparison_kind,
                configuration_signature=configuration_signature,
                result_settings=run_settings,
            )
            channel_indices = (
                (0,) if run_simulation and channel_zero_value else None
            )
            measurement_state["comparison_cache_key"] = _cache_theme_comparison(
                bundle_fingerprint=loaded.fingerprint,
                comparison_kind=comparison_kind,
                candidates=[result[:5] for result in results],
                background_subtraction=background_value,
                clutter_removal=clutter_value,
                channel_indices=channel_indices,
                configuration_signature=configuration_signature,
                result_settings=run_settings,
            )
        except InterruptedError:
            set_comparison_status(
                "info",
                "Simulation stopped",
                "Stopped at the next safe solver boundary.",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            set_comparison_status(
                "error",
                "Comparison failed",
                _safe_markdown_code(f"{type(exc).__name__}: {exc}"),
            )
            pn.state.notifications.error(str(exc), duration=8000)
        finally:
            stop_simulation_confirmation.open = False
            _restore_widget_disabled_states(run_input_states)
            measurement_state["action_running"] = False
            set_gui_activity("comparison", False)
            if run_simulation:
                measurement_state["running"] = False
                measurement_state["cancel_event"] = None
            update_comparison_source_visibility()
            persist_gui_session()

    load_bundle_button.on_click(on_load_bundle)
    compare_button.on_click(on_compare)

    def update_comparison_source_visibility(event=None):
        del event
        value = comparison_source.value
        run_simulation = value == "run"
        load_saved = value == "saved"
        comparison_simulation_modes.visible = run_simulation
        comparison_channel_zero.visible = run_simulation
        simulated_dropper.visible = load_saved
        simulated_adc_download.visible = run_simulation
        simulated_adc_download.disabled = not (
            run_simulation
            and measurement_state["comparison_kind"] == "run"
            and bool(measurement_state["simulated_adcs"])
        )
        if measurement_state["running"]:
            comparison_dependency_hint.visible = False
            compare_button.disabled = False
            compare_button.label = "Stop simulation"
            compare_button.icon = "player-stop"
            compare_button.color = "danger"
            return
        compare_button.color = "primary"
        if run_simulation:
            compare_button.label = "Run simulation"
            compare_button.icon = "player-play"
        elif load_saved:
            compare_button.label = "Load saved ADC and compare"
            compare_button.icon = "folder-open"
        else:
            compare_button.label = "Compare saved bundled ADC"
            compare_button.icon = "chart-histogram"

        bundle_ready = measurement_state["bundle"] is not None
        action_ready = bundle_ready
        hint = ""
        if not bundle_ready:
            hint = "🔒 **Load and validate a primary bundle to enable this step.**"
        elif run_simulation and not comparison_simulation_modes.value:
            action_ready = False
            hint = "▣ **Select at least one simulation mode to continue.**"
        elif load_saved:
            saved_path = str(simulated_path.value).strip()
            saved_path_allowed = bool(saved_path)
            if saved_path_allowed and remote_access:
                try:
                    saved_path_allowed = (
                        _server_path_key(saved_path)
                        in trusted_simulated_paths
                    )
                except (OSError, RuntimeError, ValueError):
                    saved_path_allowed = False
            if not saved_path_allowed:
                action_ready = False
                hint = (
                    "📎 **Drop a saved ADC file or bundle to enable "
                    "comparison.**"
                )
        compare_button.disabled = not action_ready
        comparison_dependency_hint.object = hint
        comparison_dependency_hint.visible = bool(hint)

    comparison_source.param.watch(
        update_comparison_source_visibility,
        "value",
    )
    comparison_simulation_modes.param.watch(
        update_comparison_source_visibility,
        "value",
    )
    simulated_path.param.watch(
        update_comparison_source_visibility,
        "value",
    )

    def invalidate_comparison_configuration(event) -> None:
        update_comparison_source_visibility()
        if (
            measurement_state["configuring"]
            or gui_session_state["restoring"]
        ):
            return
        measurement_state["comparison_revision"] += 1
        cancel_event = measurement_state.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        if result_records["comparison"] is None:
            clear_candidate_products()
        refresh_result_validity()
        if measurement_state["bundle"] is not None:
            set_comparison_status(
                "info",
                "Comparison settings changed",
                "Run the comparison again to generate matching products.",
            )

    for comparison_widget in (
        background,
        clutter,
        comparison_source,
        comparison_simulation_modes,
        comparison_channel_zero,
        simulated_path,
    ):
        comparison_widget.param.watch(
            invalidate_comparison_configuration,
            "value",
        )
    update_comparison_source_visibility()
    measurement_controls = _rail_group(
        pn, "load", "Step 1 · Primary bundle",
        fixture,
        pn.Column(bundle_path, visible=False),
        measurement_dropper,
        pn.Row(background, clutter),
        load_bundle_button,
        measurement_status,
        heading=bundle_step_heading(bundle_step_one_marker, bundle_step_one_label),
    )
    channel_download_row = pn.Row(
        comparison_channel_zero,
        pn.Spacer(sizing_mode="stretch_width"),
        simulated_adc_download,
        sizing_mode="stretch_width",
        margin=0,
        styles={"align-items": "center"},
    )
    comparison_actions = pn.Column(
        channel_download_row,
        pn.layout.Divider(margin=(2, 0)),
        comparison_dependency_hint,
        compare_button,
        sizing_mode="stretch_width",
        margin=(2, 10),
    )
    comparison_controls = _rail_group(
        pn, "compare", "Step 2 · Bundle comparison",
        comparison_source,
        comparison_simulation_modes,
        pn.Column(simulated_path, visible=False),
        simulated_dropper,
        comparison_actions,
        pn.pane.Markdown("Radar and solver settings: **primary bundle**. The Settings dialog is used by Static Target and Dynamic Scenes."),
        heading=bundle_step_heading(bundle_step_two_marker, bundle_step_two_label),
    )
    measurement_view_tabs = pn.Tabs(
        (
            "Scene",
            pn.Column(
                bundle_scene_status,
                bundle_scene_plot,
            ),
        ),
        ("ADC", pn.Column(
            pn.Row(*adc_selectors.values()),
            pn.pane.Markdown("Frame, chirp and channel indices start at 0."),
            pn.Row(primary_adc_plot, candidate_adc_results),
        )),
        (
            "Range profile",
            range_profile_plot,
        ),
        (
            "Range time",
            pn.Row(primary_range_time_plot, candidate_range_time_results),
        ),
        (
            "Range Doppler",
            pn.Row(
                primary_range_doppler_plot,
                candidate_range_doppler_results,
            ),
        ),
        (
            "Report",
            pn.Column(
                pn.pane.Markdown("### Detailed comparison report"),
                metrics_pane,
                sizing_mode="stretch_width",
            ),
        ),
        # Eagerly mount every Plotly pane. With lazy children, a theme reload
        # can restore the active index before its newly themed pane exists,
        # leaving stale or blank plots until every tab is visited manually.
        dynamic=False,
        sizing_mode="stretch_width",
    )
    comparison_result_banner = pn.Column(
        visible=False, name="Comparison result settings",
    )
    measurement_rail = pn.Column(
        _rail_nav_component(pn, (
            ("load", "Step 1 · Load"),
            ("compare", "Step 2 · Compare"),
        )),
        measurement_controls,
        comparison_controls,
        sizing_mode="stretch_width",
        min_width=380,
        max_width=380,
        margin=0,
        css_classes=["hermes-rail", "hermes-bundle-panels"],
    )
    measurement_tab = pn.Row(
        measurement_rail,
        pn.Column(
            _run_bar(
                pn,
                title="Bundle Comparison",
                subtitle="Measured radar ADC against a bundle-configured simulation.",
                help_text=(
                    "Load a dataset-neutral HERMES directory or GUI-exported "
                    "ZIP, inspect its primary raw ADC, then run a "
                    "bundle-configured simulation or compare previously "
                    "saved ADC. Radar and solver settings come from the "
                    "primary bundle; the Settings dialog applies to Static "
                    "Target and Dynamic Scenes."
                ),
                status=comparison_status,
                banner=comparison_result_banner,
            ),
            pn.Column(
                measurement_view_tabs,
                sizing_mode="stretch_width",
                margin=0,
                css_classes=["hermes-workspace"],
            ),
            # Outside the isolated workspace so its backdrop covers the run bar.
            stop_simulation_confirmation,
            sizing_mode="stretch_width",
            min_width=0,
            margin=0,
            styles={"gap": "16px"},
            css_classes=["hermes-work"],
        ),
        sizing_mode="stretch_width",
        css_classes=["hermes-page"],
    )

    tabs = pn.Tabs(
        ("Static Target", physics_tab),
        ("Dynamic Scenes", human_room_tab),
        ("Bundle Comparison", measurement_tab),
        # Keep the three workflow roots mounted across the theme reload.  A
        # dynamic outer Tabs model can restore its active index before Panel
        # has attached the newly selected child to the light-theme document,
        # leaving the inactive workflows visibly blank.  The nested result
        # tabs remain dynamic, where lazy rendering provides the useful win.
        dynamic=False,
        css_classes=[theme_scope_class],
    )
    gui_session_storage = _gui_session_storage_component(pn)
    gui_session_storage.jscallback(
        args={
            "workflow_tabs": tabs,
            "dynamic_tabs": human_room_view_tabs,
            "measurement_tabs": measurement_view_tabs,
            "dynamic_full_rt": human_room_mode_checkboxes["full_rt"],
            "dynamic_coherent_rt": human_room_mode_checkboxes["coherent_rt"],
            "dynamic_human_only_po": human_room_mode_checkboxes[
                "human_only_po"
            ],
            "dynamic_hybrid_po": human_room_mode_checkboxes["hybrid_po"],
        },
        loaded_state="""
const restored = cb_obj.loaded_state
if (restored != null && typeof restored === "object") {
  const boundedIndex = (value, upper) => Math.max(
    0, Math.min(upper, Number.isFinite(Number(value)) ? Number(value) : 0)
  )
  workflow_tabs.active = boundedIndex(restored.top_tab, 2)
  dynamic_tabs.active = 0
  measurement_tabs.active = boundedIndex(restored.measurement_tab, 5)
  const modes = restored.dynamic_modes
  if (modes != null && typeof modes === "object") {
    if ("full_rt" in modes) dynamic_full_rt.active = Boolean(modes.full_rt)
    if ("coherent_rt" in modes) {
      dynamic_coherent_rt.active = Boolean(modes.coherent_rt)
    }
    if ("human_only_po" in modes) {
      dynamic_human_only_po.active = Boolean(modes.human_only_po)
    }
    if ("hybrid_po" in modes) {
      dynamic_hybrid_po.active = Boolean(modes.hybrid_po)
    }
  }
}
""",
    )
    next_theme = "default" if dark_theme else "dark"
    theme_button.jscallback(
        args={
            "workflow_tabs": tabs,
            "dynamic_tabs": human_room_view_tabs,
            "measurement_tabs": measurement_view_tabs,
            "session_bridge": gui_session_storage,
            "dynamic_full_rt": human_room_mode_checkboxes["full_rt"],
            "dynamic_coherent_rt": human_room_mode_checkboxes["coherent_rt"],
            "dynamic_human_only_po": human_room_mode_checkboxes[
                "human_only_po"
            ],
            "dynamic_hybrid_po": human_room_mode_checkboxes["hybrid_po"],
            "bundle_fixture": fixture,
            "bundle_path_input": bundle_path,
            "background_toggle": background,
            "clutter_selector": clutter,
            "comparison_source_selector": comparison_source,
            "comparison_modes_selector": comparison_simulation_modes,
            "channel_zero_toggle": comparison_channel_zero,
            "simulated_path_input": simulated_path,
            **{f"adc_{name}_selector": widget for name, widget in adc_selectors.items()},
        },
        clicks=f"""
try {{
  const raw = window.sessionStorage.getItem("{_GUI_SESSION_STORAGE_KEY}")
  const previous = raw === null ? {{}} : JSON.parse(raw)
  const bridged = (
    session_bridge.data != null
    && session_bridge.data.saved_state != null
    && typeof session_bridge.data.saved_state === "object"
  ) ? session_bridge.data.saved_state : {{}}
  const saved = {{
    ...previous,
    ...bridged,
    schema_version: 2,
    top_tab: workflow_tabs.active,
    dynamic_tab: dynamic_tabs.active,
    measurement_tab: measurement_tabs.active,
    adc_selection: {{
      frame: adc_frame_selector.value,
      chirp: adc_chirp_selector.value,
      channel: adc_channel_selector.value,
    }},
    dynamic_modes: {{
      full_rt: dynamic_full_rt.active,
      coherent_rt: dynamic_coherent_rt.active,
      human_only_po: dynamic_human_only_po.active,
      hybrid_po: dynamic_hybrid_po.active,
    }},
    gui_result_cache_key: (
      bridged.gui_result_cache_key
      ?? previous.gui_result_cache_key
      ?? ""
    ),
    bundle_fixture: bundle_fixture.value,
    bundle_path: bundle_path_input.value,
    background_subtraction: background_toggle.active,
    clutter_removal: clutter_selector.value,
    comparison_source: comparison_source_selector.value,
    comparison_simulation_modes: comparison_modes_selector.value.map(
      value => ({json.dumps(comparison_mode_options)})[value] ?? value
    ),
    comparison_channel_zero: channel_zero_toggle.active,
    simulated_path: simulated_path_input.value,
    save_sequence: Date.now(),
  }}
  window.sessionStorage.setItem(
    "{_GUI_SESSION_STORAGE_KEY}", JSON.stringify(saved)
  )
}} catch (error) {{
  console.warn("Unable to save HERMES navigation before theme switch", error)
}}
const url = new URL(window.location.href)
url.searchParams.set("theme", "{next_theme}")
window.location.assign(url.toString())
""",
    )
    template = pn.template.FastListTemplate(
        title="HERMES · Interactive mmWave Radar Simulation",
        favicon="/apple-touch-icon.png",
        accent_base_color=_plot_theme_tokens(plot_template)["accent"],
        header_background="#101b2d",
        header_accent_base_color="#38bdf8",
        corner_radius=8,
        shadow=True,
        theme="dark" if dark_theme else "default",
        theme_toggle=False,
        busy_indicator=None,
        header=[
            pn.Row(
                solver_settings_storage,
                gui_session_storage,
                _modal_events_component(pn),
                pn.Spacer(sizing_mode="stretch_width"),
                settings_button,
                theme_button,
                sizing_mode="stretch_width",
                margin=0,
            )
        ],
        modal=[settings_modal],
        main=[tabs],
        main_max_width="1600px",
        main_layout=None,
    )

    static_session_widgets = {
        "target_type": target_type,
        "target_range": target_range,
        "target_y": target_y,
        "target_z": target_z,
        "plate_width": plate_width,
        "plate_height": plate_height,
        "corner_edge": corner_edge,
        "target_yaw": yaw,
        "target_pitch": pitch,
        "target_roll": roll,
        "radar_yaw": radar_yaw,
        "radar_pitch": radar_pitch,
        "radar_roll": radar_roll,
        "material": material,
        "human_diffuse": human_diffuse,
        "angle_range_bins": angle_range_bins,
        "diagnostic_top_k": diagnostic_top_k,
        "show_rt_paths": show_rt_paths,
    }
    radar_session_widgets = {
        "board": board,
        "tdm_enabled": tdm_enabled,
        "tx_antennas": tx_antennas,
        "rx_antennas": rx_antennas,
        "antenna_pattern": antenna_pattern,
        "cosine_3db_beamwidth": cosine_3db_beamwidth,
        "carrier_frequency": carrier_frequency,
        "chirp_slope": chirp_slope,
        "chirp_duration": chirp_duration,
        "chirp_repetition": chirp_repetition,
        "sampling_frequency": sampling_frequency,
        "adc_mode": adc_sampling_mode,
        "if_bandwidth": adc_if_bandwidth,
        "num_adc_samples": num_adc_samples,
        "num_chirps": num_chirps,
        "frame_period": frame_period,
    }
    dynamic_session_widgets = {
        **motion_coordinate_widgets,
        "room_preset": human_room_preset,
        "smpl_model_dir": human_room_smpl_model_dir,
        "human_x": human_room_x,
        "human_y": human_room_y,
        "human_z": human_room_z,
        "human_yaw": human_room_yaw,
        "human_diffuse": human_room_diffuse,
        "radar_yaw": dynamic_radar_yaw,
        "radar_pitch": dynamic_radar_pitch,
        "radar_roll": dynamic_radar_roll,
        "diagnostic_top_k": human_room_diagnostic_top_k,
    }
    if remote_access:
        # Browser session storage must never override the operator-selected
        # directory used by the pickle-based SMPL dependency.
        dynamic_session_widgets.pop("smpl_model_dir")
    static_solver_session_widgets = {
        name: widget
        for name, widget in static_session_widgets.items()
        if name
        not in {"angle_range_bins", "diagnostic_top_k", "show_rt_paths"}
    }
    dynamic_solver_session_widgets = {
        name: widget
        for name, widget in dynamic_session_widgets.items()
        if name != "diagnostic_top_k"
    }

    gui_session_state = {
        "restoring": False,
        "restoring_uploads": False,
        "save_sequence": 0,
        "bundle_restore_task": None,
        "dynamic_restore_task": None,
        "restored_state": {},
        "result_signature": None,
        "result_cache_key": "",
        "static_cacheable": True,
        "dynamic_cacheable": False,
    }

    def session_value(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (tuple, list)):
            return [session_value(item) for item in value]
        return value

    def widget_values(widgets):
        return {
            name: session_value(widget.value)
            for name, widget in widgets.items()
        }

    def restore_widget_values(widgets, values):
        if not isinstance(values, dict):
            return
        for name, widget in widgets.items():
            if name not in values:
                continue
            value = values[name]
            if isinstance(widget, pn.widgets.IntRangeSlider):
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    continue
                value = tuple(int(item) for item in value)
            try:
                # Normal events synchronize the new browser models. Solver
                # handlers ignore restoration, including queued async events.
                widget.value = value
            except (TypeError, ValueError):
                continue

    # Keep the settings behind the last completed products even after edits.
    # These records live in the bounded server cache. Only their opaque cache
    # key is saved in browser session state.
    result_widgets = {
        "static": {
            **{f"target.{key}": widget
               for key, widget in static_solver_session_widgets.items()},
            **{f"radar.{key}": widget
               for key, widget in radar_session_widgets.items()},
            **{f"solver.{key}": widget
               for key, widget in solver_setting_widgets.items()
               if not key.startswith("coupling_")},
            "upload.human_mesh": human_upload,
        },
        "dynamic": {
            **{f"scene.{key}": widget
               for key, widget in dynamic_solver_session_widgets.items()},
            **{f"radar.{key}": widget
               for key, widget in radar_session_widgets.items()},
            **{f"solver.{key}": widget
               for key, widget in solver_setting_widgets.items()},
            **{f"mode.{key}": widget
               for key, widget in human_room_mode_checkboxes.items()},
            "window.begin": human_room_begin_frame,
            "window.end": human_room_end_frame,
            "upload.motion": human_room_upload,
            **({"upload.scene": human_room_scene_upload} if not remote_access else {}),
        },
        "comparison": {
            "comparison.source": comparison_source,
            "comparison.modes": comparison_simulation_modes,
            "comparison.channel_zero": comparison_channel_zero,
            "comparison.background": background,
            "comparison.clutter": clutter,
            "comparison.saved_path": simulated_path,
        },
    }
    upload_identities = {}

    def capture_result_settings(page):
        values = {}
        uploads = {}
        for key, widget in result_widgets[page].items():
            if isinstance(widget, pn.widgets.FileInput):
                payload = widget.value or b""
                previous = upload_identities.get(widget)
                if previous is None or previous[0] is not payload:
                    previous = (payload, hashlib.sha256(payload).hexdigest())
                    upload_identities[widget] = previous
                values[key] = [str(widget.filename or ""), previous[1]]
                uploads[key] = (widget.filename, widget.value)
            else:
                value = settings_dialog_state["values"].get(widget, widget.value) \
                    if settings_dialog_state["editing"] else widget.value
                values[key] = session_value(value)
        if page == "comparison":
            loaded = measurement_state.get("bundle")
            values["bundle"] = getattr(loaded, "fingerprint", None)
        return {
            "fingerprint": _result_settings_fingerprint(values),
            "values": values,
            "uploads": uploads,
        }

    result_records = {
        "static": {
            "settings": capture_result_settings("static"),
            "result": initial_result,
            "refined": False,
            "status": str(physics_status.object or ""),
        },
        "dynamic": None,
        "comparison": None,
    }
    result_banners = {
        "static": static_result_banner,
        "dynamic": dynamic_result_banner,
        "comparison": comparison_result_banner,
    }
    result_banner_text = {}
    result_rerun_buttons = {}
    result_revert_buttons = {}
    for page, banner in result_banners.items():
        result_banner_text[page] = pn.pane.Markdown(
            "", css_classes=["hermes-banner-text"],
        )
        result_rerun_buttons[page] = pn.widgets.Button(
            label="Rerun", color="primary", width=100,
            description="Run this page with the current settings.",
        )
        result_revert_buttons[page] = pn.widgets.Button(
            label="Revert", color="light", width=100,
            description="Restore the settings used for the displayed results.",
        )
        banner.objects = [
            result_banner_text[page],
            pn.Row(
                result_rerun_buttons[page], result_revert_buttons[page],
                css_classes=["hermes-banner-actions"],
            ),
        ]
        banner.css_classes = ["hermes-runbar-banner"]
        # Shown inside the run bar, whose left edge also turns amber.
        banner.styles = {
            "background": "rgba(214, 158, 46, 0.10)",
            "border-radius": "8px",
            "padding": "2px 6px 6px",
        }
        banner.margin = (2, 4, 4, 4)

    def settings_match_result(page):
        record = result_records[page]
        return bool(
            record and record["settings"]["fingerprint"]
            == capture_result_settings(page)["fingerprint"]
        )

    def refresh_result_validity():
        load_motion_preview.visible = (
            default_amass_motion is not None
            and not human_room_upload.value
            and human_room_state["loaded_motion"] is None
        )
        run_human_room.disabled = (
            human_room_state["loaded_motion"] is None
            or human_room_state["running"]
            or gui_activity_state["dynamic"]
        )
        for index, (page, title) in enumerate((
            ("static", "Static Target"),
            ("dynamic", "Dynamic Scenes"),
            ("comparison", "Bundle Comparison"),
        )):
            record = result_records[page]
            was_stale = result_banners[page].visible
            current = capture_result_settings(page)
            stale = bool(
                record and record["settings"]["fingerprint"] != current["fingerprint"]
            )
            if stale:
                previous_values = record["settings"]["values"]
                changed = [
                    key for key, value in current["values"].items()
                    if previous_values.get(key) != value
                ]
                labels = [
                    str(result_widgets[page][key].label or key)
                    if key in result_widgets[page] else "Primary bundle"
                    for key in changed
                ]
                result_banner_text[page].object = (
                    "**Results need a rerun.** Changed: "
                    + ", ".join(_safe_markdown_code(label) for label in labels)
                    + ". The plots show the last completed run."
                )
                if (page == "static" and not gui_activity_state["static"]
                        and (not was_stale or str(physics_status.object).startswith(
                            ("**LIVE", "**HIGHER FIDELITY")))):
                    physics_status.object = (
                        "**LAST RUN (out of date)** · The displayed plots use previous settings."
                    )
                elif (page == "dynamic" and not gui_activity_state["dynamic"]
                        and (not was_stale or str(human_room_status.object).startswith(
                            ("**SIMULATION READY", "**READY TO RUN")))):
                    human_room_status.object = (
                        "**LAST RUN (out of date)** · The displayed plots use previous settings."
                    )
            result_banners[page].visible = stale
            if was_stale and not stale and record and not any(gui_activity_state.values()):
                if page == "static":
                    physics_status.object = record.get("status", physics_status.object)
                elif page == "dynamic":
                    human_room_status.object = record.get("status", human_room_status.object)
            tab_title = title + (" •" if stale else "")
            if tabs._names[index] != tab_title:  # pylint: disable=protected-access
                tabs._names[index] = tab_title  # pylint: disable=protected-access
                # Panel reuses unchanged TabPanels without updating titles.
                # Keep the mounted workflow models and update only the title.
                for model, _parent in tabs._models.values():  # pylint: disable=protected-access
                    def update_title(model=model, index=index, title=tab_title):
                        from bokeh.models import CustomJS

                        panel = model.tabs[index]
                        if not panel.js_property_callbacks.get("change:title"):
                            panel.js_on_change("title", CustomJS(
                                args={"workflow_tabs": model},
                                code="workflow_tabs.properties.tabs.change.emit()",
                            ))
                        panel.title = title

                    document = model.document
                    if document is not None and document.session_context is not None:
                        document.add_next_tick_callback(update_title)
                    else:
                        update_title()
            if page == "static" and record:
                physics_state["result"] = None if stale else record["result"]
                physics_state["refined"] = record["refined"]
                gui_session_state["static_cacheable"] = not stale
                physics_download.disabled = stale or physics_state["running"]
                diagnostic_busy = diagnostics_state.get("cancel_event") is not None
                show_rt_paths.disabled = stale or physics_state["running"] or diagnostic_busy
                diagnostic_top_k.disabled = stale or physics_state["running"] or diagnostic_busy
            elif page == "dynamic" and record:
                human_room_state["results"] = {} if stale else record["results"]
                human_room_state["result"] = None if stale else record["result"]
                gui_session_state["dynamic_cacheable"] = not stale
                human_room_download.disabled = stale or human_room_state["running"]
                run_human_room_diagnostics_button.disabled = (
                    stale or human_room_state["running"]
                    or human_room_state.get("diagnostic_cancel_event") is not None
                )
            elif page == "comparison" and record:
                simulated_adc_download.disabled = stale or not measurement_state["simulated_adcs"]
            running = any(gui_activity_state.values())
            result_rerun_buttons[page].disabled = running
            result_revert_buttons[page].disabled = running

    def remember_result_settings(page, settings=None):
        settings = settings or capture_result_settings(page)
        if page == "static":
            result_records[page] = {
                "settings": settings,
                "result": physics_state["result"],
                "refined": physics_state["refined"],
                "status": str(physics_status.object or ""),
            }
        elif page == "dynamic":
            result_records[page] = {
                "settings": settings,
                "result": human_room_state["result"],
                "results": dict(human_room_state["results"]),
                "status": str(human_room_status.object or ""),
            }
        else:
            result_records[page] = {"settings": settings}

    async def rerun_page(page):
        if page == "static":
            record = result_records[page]
            fidelity = "high_fidelity" if record and record["refined"] else "preview"
            await update_physics(fidelity=fidelity, force=True)
        elif page == "dynamic":
            await on_run_human_room(None)
        else:
            await on_compare(None)

    async def revert_page_settings(page):
        record = result_records[page]
        if record is None:
            return
        gui_session_state["restoring"] = True
        try:
            settings = record["settings"]
            # Board options must exist before restoring its antenna indices.
            if page in {"static", "dynamic"}:
                board.value = settings["values"]["radar.board"]
                spec = get_ti_board_spec(board.value)
                tx_antennas.options = {f"TX{i + 1}": i for i in range(spec.num_tx)}
                rx_antennas.options = {f"RX{i + 1}": i for i in range(spec.num_rx)}
            for key, widget in result_widgets[page].items():
                if key in settings["uploads"]:
                    widget.filename, widget.value = settings["uploads"][key]
                else:
                    widget.value = settings["values"][key]
            update_radar_summary()
        finally:
            gui_session_state["restoring"] = False
        if page == "dynamic":
            await refresh_human_room_preview(mark_stale=False)
            human_room_status.object = record.get("status", human_room_status.object)
        persist_solver_settings()
        persist_gui_uploads()
        persist_gui_session()

    for page in result_banners:
        async def rerun(_event, page=page):
            await rerun_page(page)

        async def revert(_event, page=page):
            await revert_page_settings(page)

        result_rerun_buttons[page].on_click(rerun)
        result_revert_buttons[page].on_click(revert)

    def dynamic_result_view_state():
        result_tabs = next(
            (
                component
                for component in human_room_results.select(pn.Tabs)
                if component._names  # pylint: disable=protected-access
                == ["Range profile", "Range-time", "Range-Doppler"]
            ),
            None,
        )
        range_doppler_slider = next(
            (
                widget
                for widget in human_room_results.select(pn.widgets.IntSlider)
                if str(widget.label).startswith("Range-Doppler frame")
            ),
            None,
        )
        return {
            "product_tab": int(result_tabs.active) if result_tabs else 0,
            "range_doppler_frame_index": (
                int(range_doppler_slider.value)
                if range_doppler_slider is not None
                else 0
            ),
        }

    def cache_current_gui_results() -> str:
        static_record = result_records["static"]
        dynamic_record = result_records["dynamic"]
        static_result = static_record["result"] if static_record else None
        dynamic_results = dict(dynamic_record["results"]) if dynamic_record else {}
        signature = (
            id(static_result),
            bool(physics_state.get("refined")),
            id(diagnostics_state.get("result")),
            bool(show_rt_paths.value),
            int(diagnostic_top_k.value),
            id(human_room_state.get("preview")),
            id(human_room_state.get("loaded_motion")),
            tuple((mode, id(result)) for mode, result in dynamic_results.items()),
            id(human_room_state.get("diagnostics")),
            static_record["settings"]["fingerprint"] if static_record else None,
            dynamic_record["settings"]["fingerprint"] if dynamic_record else None,
        )
        if signature == gui_session_state["result_signature"]:
            return str(gui_session_state["result_cache_key"])
        cache_key = _cache_theme_gui_results(
            static_result=static_result,
            static_settings=static_record["settings"] if static_record else None,
            static_refined=bool(physics_state.get("refined")),
            static_status=static_record.get("status", "") if static_record else "",
            static_diagnostics=diagnostics_state.get("result"),
            static_diagnostics_status=str(diagnostics_status.object or ""),
            static_show_rt_paths=bool(show_rt_paths.value),
            static_diagnostic_top_k=int(diagnostic_top_k.value),
            dynamic_preview=human_room_state.get("preview"),
            dynamic_loaded_motion=human_room_state.get("loaded_motion"),
            dynamic_motion_cache_key=human_room_state.get("motion_cache_key"),
            dynamic_results=dynamic_results,
            dynamic_settings=dynamic_record["settings"] if dynamic_record else None,
            dynamic_diagnostics=human_room_state.get("diagnostics"),
            dynamic_status=dynamic_record.get("status", "") if dynamic_record else "",
            dynamic_diagnostics_status=str(
                human_room_diagnostics_status.object or ""
            ),
        )
        gui_session_state["result_signature"] = signature
        gui_session_state["result_cache_key"] = cache_key
        return cache_key

    def persist_gui_session(_event=None):
        if (gui_session_state["restoring"] or solver_settings_state["restoring"]
                or settings_dialog_state["editing"]):
            return
        refresh_result_validity()
        gui_session_state["save_sequence"] += 1
        uploads_present = bool(
            human_upload.value
            or human_room_upload.value
            or (human_room_scene_upload.value and not remote_access)
        )
        gui_session_storage.saved_state = {
            "schema_version": 2,
            "po_defaults_version": _PO_DEFAULTS_VERSION,
            "image_source_selection_version": _IMAGE_SOURCE_SELECTION_VERSION,
            "po_phase_span_by_mode": current_solver_settings_payload()[
                "po_phase_span_by_mode"
            ],
            "top_tab": int(tabs.active),
            "dynamic_tab": int(human_room_view_tabs.active),
            "measurement_tab": int(measurement_view_tabs.active),
            "adc_selection": {name: int(widget.value) for name, widget in adc_selectors.items()},
            "static_controls": widget_values(static_session_widgets),
            "radar_controls": widget_values(radar_session_widgets),
            "dynamic_controls": widget_values(dynamic_session_widgets),
            "solver_settings": widget_values(solver_setting_widgets),
            "dynamic_modes": {
                mode: bool(checkbox.value)
                for mode, checkbox in human_room_mode_checkboxes.items()
            },
            "dynamic_frame": int(human_room_frame.value),
            "dynamic_begin_frame": int(human_room_begin_frame.value),
            "dynamic_end_frame": int(human_room_end_frame.value),
            "dynamic_result_view": dynamic_result_view_state(),
            "has_uploads": uploads_present,
            "has_motion": bool(human_room_upload.value),
            "bundle_fixture": str(fixture.value or ""),
            "bundle_path": str(bundle_path.value or ""),
            "bundle_loaded": measurement_state["bundle"] is not None,
            "background_subtraction": bool(background.value),
            "clutter_removal": str(clutter.value),
            "comparison_source": str(comparison_source.value),
            "comparison_simulation_modes": list(
                comparison_simulation_modes.value
            ),
            "comparison_channel_zero": bool(comparison_channel_zero.value),
            "simulated_path": str(simulated_path.value or ""),
            "comparison_completed": bool(measurement_state["comparisons"]),
            "comparison_kind": measurement_state["comparison_kind"],
            "comparison_cache_key": measurement_state["comparison_cache_key"],
            "gui_result_cache_key": cache_current_gui_results(),
            "save_sequence": gui_session_state["save_sequence"],
        }

    def encoded_upload(widget, fallback_filename: str, max_bytes: int):
        payload = widget.value
        if not payload:
            return None
        try:
            payload = _validated_upload_payload(
                payload,
                label=str(widget.label or "file"),
                max_bytes=max_bytes,
            )
        except (TypeError, ValueError) as exc:
            pn.state.notifications.error(str(exc), duration=8000)
            return None
        return {
            "filename": str(widget.filename or fallback_filename),
            "payload": base64.b64encode(payload).decode("ascii"),
        }

    def persist_gui_uploads(*_events):
        if gui_session_state["restoring_uploads"] or gui_session_state["restoring"]:
            return
        changed_widgets = {
            getattr(event, "obj", None) for event in _events
        }
        if human_upload in changed_widgets:
            gui_session_state["static_cacheable"] = False
        if changed_widgets.intersection(
            {human_room_upload, human_room_scene_upload}
        ):
            gui_session_state["dynamic_cacheable"] = False
        uploads = {
            "static_mesh": encoded_upload(
                human_upload,
                "human_mesh.npz",
                _MAX_STATIC_MESH_UPLOAD_BYTES,
            ),
            "motion": encoded_upload(
                human_room_upload,
                "amass_motion.npz",
                _MAX_MOTION_UPLOAD_BYTES,
            ),
            "scene_xml": (
                None
                if remote_access
                else encoded_upload(
                    human_room_scene_upload,
                    "scene.xml",
                    _MAX_SCENE_XML_UPLOAD_BYTES,
                )
            ),
        }
        gui_session_storage.saved_uploads = {
            name: upload for name, upload in uploads.items() if upload is not None
        }
        persist_gui_session()

    def apply_pending_dynamic_frame_state(restored):
        preview = human_room_state.get("preview")
        if preview is None:
            return
        import param

        last_frame = max(len(preview.mesh_sequence.times) - 1, 0)
        selected_frame = int(
            np.clip(int(restored.get("dynamic_frame", 0)), 0, last_frame)
        )
        begin_frame = int(
            np.clip(int(restored.get("dynamic_begin_frame", 0)), 0, last_frame)
        )
        end_frame = int(
            np.clip(
                int(restored.get("dynamic_end_frame", begin_frame)),
                begin_frame,
                last_frame,
            )
        )
        configure_human_room_frame_controls(
            last_frame=last_frame,
            selected_frame=selected_frame,
            disabled=last_frame == 0,
            interval_ms=_motion_playback_interval_ms(preview.mesh_sequence.times),
            normal_step=_motion_playback_step(preview.mesh_sequence.times),
            fast_step=_motion_fast_playback_step(preview.mesh_sequence.times),
        )
        configure_human_room_window_controls(
            last_frame=last_frame,
            reset_window=False,
            disabled=False,
        )
        with param.parameterized.discard_events(human_room_begin_frame):
            human_room_begin_frame.value = begin_frame
        with param.parameterized.discard_events(human_room_end_frame):
            human_room_end_frame.value = end_frame
        display_human_room_frame(selected_frame)

    def restore_dynamic_result_view(restored):
        view_state = restored.get("dynamic_result_view", {})
        if not isinstance(view_state, dict):
            return
        import param

        result_tabs = next(
            (
                component
                for component in human_room_results.select(pn.Tabs)
                if component._names  # pylint: disable=protected-access
                == ["Range profile", "Range-time", "Range-Doppler"]
            ),
            None,
        )
        if result_tabs is not None:
            active = int(np.clip(int(view_state.get("product_tab", 0)), 0, 2))
            with param.parameterized.discard_events(result_tabs):
                result_tabs.active = active
        range_doppler_slider = next(
            (
                widget
                for widget in human_room_results.select(pn.widgets.IntSlider)
                if str(widget.label).startswith("Range-Doppler frame")
            ),
            None,
        )
        if range_doppler_slider is not None:
            frame = int(
                np.clip(
                    int(view_state.get("range_doppler_frame_index",
                                       int(view_state.get("range_doppler_frame", 1)) - 1)),
                    int(range_doppler_slider.start),
                    int(range_doppler_slider.end),
                )
            )
            range_doppler_slider.value = frame

    def restore_cached_gui_results(restored):
        cache_key = str(restored.get("gui_result_cache_key", "") or "")
        cached = _get_theme_gui_results(cache_key)
        if cached is None:
            return
        static_result = cached.get("static_result")
        if static_result is not None:
            gui_session_state["static_cacheable"] = True
            present_physics_result(
                static_result,
                refined=bool(cached.get("static_refined")),
                preserve_scene_frame=False,
                geometry_already_previewed=False,
                result_settings=cached.get("static_settings"),
            )
            physics_status.object = str(
                cached.get("static_status") or physics_status.object
            )
            result_records["static"]["status"] = physics_status.object
            static_diagnostics = cached.get("static_diagnostics")
            if static_diagnostics is not None:
                diagnostics_state["result"] = static_diagnostics
                diagnostics_state["top_k"] = int(
                    cached.get("static_diagnostic_top_k", diagnostic_top_k.value)
                )
                diagnostics_summary.object = _solver_diagnostics_summary(
                    static_diagnostics
                )
                diagnostics_depth.object = _path_depth_figure(
                    static_diagnostics,
                    go,
                    plot_template=plot_template,
                )
                diagnostics_depth.visible = True
                if bool(cached.get("static_show_rt_paths")):
                    physics_scene.object = _solver_diagnostics_scene_figure(
                        static_result,
                        static_diagnostics,
                        go,
                        plot_template=plot_template,
                    )
                diagnostics_status.object = str(
                    cached.get("static_diagnostics_status")
                    or diagnostics_status.object
                )

        preview = cached.get("dynamic_preview")
        if preview is not None:
            human_room_state["preview"] = preview
            human_room_state["loaded_motion"] = cached.get(
                "dynamic_loaded_motion"
            )
            human_room_state["motion_cache_key"] = cached.get(
                "dynamic_motion_cache_key"
            )
            apply_pending_dynamic_frame_state(restored)
        dynamic_results = dict(cached.get("dynamic_results") or {})
        selection_changed = (
            restored.get("image_source_selection_version") != _IMAGE_SOURCE_SELECTION_VERSION
        )
        if selection_changed:
            dynamic_results.pop("hybrid_po", None)
        if dynamic_results:
            gui_session_state["dynamic_cacheable"] = True
            primary_mode = (
                "hybrid_po"
                if "hybrid_po" in dynamic_results
                else next(iter(dynamic_results))
            )
            human_room_state["result"] = dynamic_results[primary_mode]
            human_room_state["results"] = dynamic_results
            remember_result_settings("dynamic", cached.get("dynamic_settings"))
            human_room_state["diagnostics"] = (
                None if selection_changed else cached.get("dynamic_diagnostics")
            )
            human_room_results.objects = [
                _human_room_results_view(
                    dynamic_results,
                    pn,
                    go,
                    plot_template=plot_template,
                )
            ]
            human_room_status.object = str(
                cached.get("dynamic_status") or "**SIMULATION READY**"
            )
            result_records["dynamic"]["status"] = human_room_status.object
            human_room_download.callback = lambda: _human_room_export(
                human_room_state["result"],
                mode_results=human_room_state["results"],
            ) if human_room_state["result"] is not None else BytesIO()
            human_room_download.disabled = False
            run_human_room_diagnostics_button.disabled = False
            human_room_diagnostics_status.object = str(
                cached.get("dynamic_diagnostics_status")
                or "Choose a displayed source frame, then show solver overlays."
            )
            if human_room_state["diagnostics"] is not None:
                display_human_room_frame(int(human_room_frame.value))
            restore_dynamic_result_view(restored)
        gui_session_state["result_cache_key"] = cache_key

    def restore_gui_session(event, *, page_ready=False):
        # After a theme reload the browser reports its saved state before
        # its document_ready event. Until then Panel leaves held layout
        # updates unsent while their new models are already in the document;
        # a browser message arriving in between makes Bokeh treat those
        # models as known, so restored result views would reach the browser
        # without their definitions. Restore once the page is connected.
        # (Panel reports ``loaded`` only after its onload callbacks finish,
        # so the deferred call says it is ready instead of asking again.)
        if not page_ready and not pn.state.loaded:
            pn.state.onload(lambda: restore_gui_session(event, page_ready=True))
            return
        restored = getattr(event, "new", None)
        if not isinstance(restored, dict) or restored.get("schema_version") not in {
            1,
            2,
        }:
            return
        gui_session_state["restored_state"] = dict(restored)
        gui_session_state["restoring"] = True
        try:
            import param

            for navigation, value, upper in (
                (tabs, restored.get("top_tab", 0), 2),
                # Always land on the scene; cached results remain in their tab.
                (human_room_view_tabs, 0, 1),
                (measurement_view_tabs, restored.get("measurement_tab", 0), 5),
            ):
                # These changes must emit Param events so Panel updates the
                # browser-side Bokeh models.  Suppressing the events changes
                # only the Python objects, leaving the newly themed document
                # on its default tabs.  ``restoring`` already prevents the
                # persistence watcher from writing an intermediate state.
                navigation.active = int(np.clip(int(value), 0, upper))

            radar_values = restored.get("radar_controls", {})
            if isinstance(radar_values, dict) and "board" in radar_values:
                restored_board = str(radar_values["board"])
                if restored_board in board.options:
                    board.value = restored_board
                    selected_spec = get_ti_board_spec(board.value)
                    tx_antennas.options = {
                        f"TX{index + 1}": index
                        for index in range(selected_spec.num_tx)
                    }
                    rx_antennas.options = {
                        f"RX{index + 1}": index
                        for index in range(selected_spec.num_rx)
                    }
            restore_widget_values(radar_session_widgets, radar_values)
            restore_widget_values(
                static_session_widgets,
                restored.get("static_controls", {}),
            )
            motion_forward_axis.options = motion_axis_options
            restore_widget_values(
                dynamic_session_widgets,
                restored.get("dynamic_controls", {}),
            )
            sync_motion_coordinate_controls()
            restored_solver_settings, _ = restore_solver_settings(
                {
                    "schema_version": _SOLVER_SETTINGS_SCHEMA_VERSION,
                    "po_defaults_version": restored.get("po_defaults_version"),
                    "po_phase_span_by_mode": restored.get("po_phase_span_by_mode"),
                    "settings": restored.get("solver_settings", {}),
                }
            )
            previous_solver_settings = restored.get("solver_settings", {})
            po_defaults_changed = (
                isinstance(previous_solver_settings, dict)
                and any(
                    name.startswith("po_")
                    and previous_solver_settings.get(name) != value
                    for name, value in restored_solver_settings.items()
                )
            )
            restored_dynamic_modes = restored.get("dynamic_modes", {})
            if isinstance(restored_dynamic_modes, dict):
                for mode, checkbox in human_room_mode_checkboxes.items():
                    if mode in restored_dynamic_modes:
                        # Let Panel propagate the value to the rendered
                        # checkbox.  Mode-change callbacks may clear the
                        # initial empty result pane here; cached products are
                        # restored immediately below after all modes settle.
                        checkbox.value = bool(restored_dynamic_modes[mode])

            is_plate = target_type.value == "plate"
            is_corner = target_type.value == "trihedral"
            is_human = target_type.value == "human_mesh"
            plate_width.visible = is_plate
            plate_height.visible = is_plate
            corner_edge.visible = is_corner
            material.visible = not is_human
            material_details.visible = not is_human
            human_upload.visible = is_human
            human_diffuse.visible = is_human
            human_source.visible = is_human
            update_material_details()
            update_antenna_pattern_controls()
            update_radar_summary()
            if not po_defaults_changed:
                restore_cached_gui_results(restored)

            has_bundle_state = any(
                key in restored
                for key in (
                    "bundle_fixture",
                    "bundle_path",
                    "comparison_source",
                )
            )
            if has_bundle_state:
                restored_fixture = str(restored.get("bundle_fixture", ""))
                fixture_values = set(fixture.options.values())
                fixture.value = (
                    restored_fixture
                    if restored_fixture in fixture_values
                    else ""
                )
                restored_bundle_path = str(restored.get("bundle_path", ""))
                if fixture.value:
                    restored_bundle_path = str(fixture.value)
                elif remote_access and restored_bundle_path:
                    try:
                        restored_bundle_path = _authorized_server_path(
                            restored_bundle_path,
                            remote_access=True,
                            trusted_paths=trusted_bundle_paths,
                        )
                    except (OSError, PermissionError, RuntimeError, ValueError):
                        restored_bundle_path = ""
                if fixture.value == "":
                    custom_bundle_path["value"] = restored_bundle_path
                bundle_path.value = restored_bundle_path
                background.value = bool(
                    restored.get("background_subtraction", False)
                )
                restored_clutter = str(restored.get("clutter_removal", "none"))
                clutter.value = (
                    restored_clutter
                    if restored_clutter in clutter.options
                    else "none"
                )
                restored_modes = restored.get(
                    "comparison_simulation_modes",
                    ["human_only_po"],
                )
                valid_modes = set(comparison_mode_options.values())
                if isinstance(restored_modes, list):
                    comparison_simulation_modes.value = [
                        str(mode)
                        for mode in restored_modes
                        if str(mode) in valid_modes
                    ]
                comparison_channel_zero.value = bool(
                    restored.get("comparison_channel_zero", False)
                )
                restored_simulated_path = str(
                    restored.get("simulated_path", "")
                )
                if remote_access and restored_simulated_path:
                    try:
                        restored_simulated_path = _authorized_server_path(
                            restored_simulated_path,
                            remote_access=True,
                            trusted_paths=trusted_simulated_paths,
                        )
                    except (OSError, PermissionError, RuntimeError, ValueError):
                        restored_simulated_path = ""
                simulated_path.value = restored_simulated_path
                restored_source = str(restored.get("comparison_source", "run"))
                if restored_source in comparison_source.options.values():
                    comparison_source.value = restored_source
                update_comparison_source_visibility()
        except (TypeError, ValueError):
            return
        finally:
            def finish_restoration():
                gui_session_state["restoring"] = False
                refresh_result_validity()
                persist_gui_session()

            document = pn.state.curdoc
            if document is not None and document.session_context is not None:
                document.add_next_tick_callback(finish_restoration)
            else:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    finish_restoration()
                else:
                    # Queued widget watchers must see the restoration guard.
                    loop.call_soon(finish_restoration)

        if restored.get("has_uploads") or restored.get("has_motion"):
            pass
        elif human_room_state.get("preview") is None:
            source, _filename = current_human_room_source()
            if source is not None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None:
                    gui_session_state["dynamic_restore_task"] = loop.create_task(
                        refresh_human_room_preview(mark_stale=False)
                    )

        if not restored.get("bundle_loaded") or not bundle_path.value:
            persist_gui_session()
            return

        async def restore_loaded_bundle():
            await on_load_bundle(None)
            if measurement_state["bundle"] is None:
                return
            for name, widget in adc_selectors.items():
                value = restored.get("adc_selection", {}).get(name, 0)
                widget.value = int(np.clip(int(value), 0, widget.end))
            restored_source = str(restored.get("comparison_source", "run"))
            if restored_source in comparison_source.options.values():
                comparison_source.value = restored_source
            update_comparison_source_visibility()
            comparison_cache_key = str(
                restored.get("comparison_cache_key", "") or ""
            )
            cached_comparison = _get_theme_comparison(
                comparison_cache_key,
                bundle_fingerprint=measurement_state["bundle"].fingerprint,
            )
            if bool(restored.get("comparison_completed")) and cached_comparison:
                set_comparison_status(
                    "info",
                    "Restoring comparison",
                    "Applying the selected theme.",
                )
                cached_results = []
                for candidate in cached_comparison["candidates"]:
                    key, label, heading, detail, adc = candidate
                    comparison = await asyncio.to_thread(
                        measurement_state["bundle"].compare_adc,
                        adc,
                        background_subtraction=cached_comparison[
                            "background_subtraction"
                        ],
                        clutter_removal=cached_comparison["clutter_removal"],
                        channel_indices=cached_comparison["channel_indices"],
                    )
                    cached_results.append(
                        (key, label, heading, detail, adc, comparison)
                    )
                measurement_state["comparison_cache_key"] = comparison_cache_key
                render_comparison_results(
                    cached_results,
                    comparison_kind=str(cached_comparison["comparison_kind"]),
                    configuration_signature=cached_comparison.get("configuration_signature"),
                    result_settings=cached_comparison.get("result_settings"),
                )
                persist_gui_session()
                return
            should_restore_saved_comparison = (
                bool(restored.get("comparison_completed"))
                and restored.get("comparison_kind") == "saved"
                and restored_source != "run"
                and (
                    restored_source != "saved"
                    or bool(simulated_path.value)
                )
            )
            if should_restore_saved_comparison:
                await on_compare(None)
            persist_gui_session()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        gui_session_state["bundle_restore_task"] = loop.create_task(
            restore_loaded_bundle()
        )

    def decoded_upload(restored, extensions, max_bytes):
        if not isinstance(restored, dict):
            return None
        filename = Path(str(restored.get("filename", ""))).name
        encoded = restored.get("payload")
        if not filename.casefold().endswith(tuple(extensions)) or not isinstance(
            encoded, str
        ):
            return None
        maximum_encoded_length = 4 * ((int(max_bytes) + 2) // 3)
        if len(encoded) > maximum_encoded_length:
            return None
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            return None
        if len(payload) > int(max_bytes):
            return None
        return (filename, payload) if payload else None

    def restore_gui_uploads(event, *, page_ready=False):
        # Same ordering as restore_gui_session: wait for the connected page.
        if not page_ready and not pn.state.loaded:
            pn.state.onload(lambda: restore_gui_uploads(event, page_ready=True))
            return
        restored = getattr(event, "new", None)
        if not isinstance(restored, dict):
            return
        import param

        upload_specs = [
            (
                "static_mesh",
                human_upload,
                (".obj", ".npz"),
                _MAX_STATIC_MESH_UPLOAD_BYTES,
            ),
            (
                "motion",
                human_room_upload,
                (".npz",),
                _MAX_MOTION_UPLOAD_BYTES,
            ),
        ]
        if not remote_access:
            upload_specs.append(
                (
                    "scene_xml",
                    human_room_scene_upload,
                    (".xml",),
                    _MAX_SCENE_XML_UPLOAD_BYTES,
                )
            )
        gui_session_state["restoring_uploads"] = True
        try:
            for name, widget, extensions, max_bytes in upload_specs:
                decoded = decoded_upload(
                    restored.get(name),
                    extensions,
                    max_bytes,
                )
                if decoded is None:
                    continue
                filename, payload = decoded
                with param.parameterized.discard_events(widget):
                    widget.filename = filename
                    widget.value = payload
            on_human_room_upload_filename(
                type("UploadEvent", (), {"new": human_room_upload.filename})()
            )
        finally:
            gui_session_state["restoring_uploads"] = False

        refresh_result_validity()

        if human_room_state.get("preview") is not None:
            persist_gui_session()
            return

        async def restore_dynamic_uploads():
            await refresh_human_room_preview(mark_stale=False)
            apply_pending_dynamic_frame_state(gui_session_state["restored_state"])
            persist_gui_session()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        gui_session_state["dynamic_restore_task"] = loop.create_task(
            restore_dynamic_uploads()
        )

    gui_session_storage.param.watch(
        restore_gui_session,
        "loaded_state",
    )
    gui_session_storage.param.watch(
        restore_gui_uploads,
        "loaded_uploads",
    )
    for navigation_tabs in (
        tabs,
        human_room_view_tabs,
        measurement_view_tabs,
    ):
        navigation_tabs.param.watch(persist_gui_session, "active")
    for widget in (
        *static_session_widgets.values(),
        *radar_session_widgets.values(),
        *dynamic_session_widgets.values(),
        human_room_frame,
        human_room_begin_frame,
        human_room_end_frame,
        *human_room_mode_checkboxes.values(),
        *adc_selectors.values(),
    ):
        widget.param.watch(persist_gui_session, "value")
    for comparison_widget, parameter_name in (
        (fixture, "value"),
        (bundle_path, "value"),
        (background, "value"),
        (clutter, "value"),
        (comparison_source, "value"),
        (comparison_simulation_modes, "value"),
        (comparison_channel_zero, "value"),
        (simulated_path, "value"),
    ):
        comparison_widget.param.watch(persist_gui_session, parameter_name)
    for upload_widget in (
        human_upload,
        human_room_upload,
        human_room_scene_upload,
    ):
        upload_widget.param.watch(
            persist_gui_uploads,
            ["value", "filename"],
        )
    persist_gui_uploads()
    persist_gui_session()

    def persist_solver_settings(_event=None):
        if solver_settings_state["restoring"] or settings_dialog_state["editing"]:
            return
        solver_settings_storage.saved_settings = (
            current_solver_settings_payload()
        )
        persist_gui_session(_event)

    async def on_solver_settings_loaded(event):
        if gui_session_state["restoring"]:
            return
        # The session snapshot owns settings on a theme reload. A later
        # localStorage bridge event must not replace that restored snapshot.
        if gui_session_state["restored_state"]:
            persist_solver_settings()
            refresh_result_validity()
            return
        restore_solver_settings(getattr(event, "new", None))

        # Rewrite partial or outdated payloads in the current complete schema.
        persist_solver_settings()
        # A storage restore is not an edit or a request to run a solver.
        refresh_result_validity()

    async def on_apply_solver_settings(_event):
        motion_changed = settings_dialog_state["editing"] and any(
            widget.value != settings_dialog_state["values"][widget]
            for widget in motion_coordinate_widgets.values()
        )
        settings_dialog_state["editing"] = False
        ignored_radar_events.clear()
        persist_solver_settings()
        template.close_modal()
        theme_button.disabled = any(gui_activity_state.values())
        if motion_changed and human_room_state["loaded_motion"] is not None:
            await on_human_room_controls(None)
        if rerun_after_apply.value:
            await rerun_page(("static", "dynamic", "comparison")[tabs.active])

    def on_open_settings(_event):
        settings_dialog_state["values"] = {
            widget: session_value(widget.value)
            for widget in (*radar_session_widgets.values(), *solver_setting_widgets.values(),
                           *motion_coordinate_widgets.values())
        }
        settings_dialog_state["phase_spans"] = dict(phase_span_by_mode)
        settings_dialog_state["editing"] = True
        rerun_after_apply.value = False
        theme_button.disabled = True
        template.open_modal()

    async def on_cancel_settings(_event):
        if not settings_dialog_state["editing"]:
            return
        gui_session_state["restoring"] = True
        solver_settings_state["restoring"] = True
        try:
            saved = settings_dialog_state["values"]
            board.value = saved[board]
            spec = get_ti_board_spec(board.value)
            tx_antennas.options = {f"TX{i + 1}": i for i in range(spec.num_tx)}
            rx_antennas.options = {f"RX{i + 1}": i for i in range(spec.num_rx)}
            for widget, value in saved.items():
                widget.value = value
            phase_span_by_mode.clear()
            phase_span_by_mode.update(settings_dialog_state["phase_spans"])
            update_antenna_pattern_controls()
            sync_motion_coordinate_controls()
            update_radar_summary()
            await asyncio.sleep(0)
        finally:
            solver_settings_state["restoring"] = False
            gui_session_state["restoring"] = False
            settings_dialog_state["editing"] = False
            ignored_radar_events.clear()
        template.close_modal()
        theme_button.disabled = any(gui_activity_state.values())
        persist_gui_session()

    settings_button.on_click(on_open_settings)
    cancel_solver_settings.on_click(on_cancel_settings)
    modal_events = next(c for c in template.header[0].select(pn.reactive.ReactiveHTML)
                        if "dismissed" in c.param)
    modal_events.param.watch(on_cancel_settings, "dismissed")

    for key, widget in {**radar_session_widgets, **solver_setting_widgets, **motion_coordinate_widgets}.items():
        used_by = ("Dynamic Scenes" if key in motion_coordinate_widgets
                   else "Dynamic Scenes (Hybrid PO)" if key.startswith("coupling_")
                   else "Static Target, Dynamic Scenes")
        if "description" in widget.param:
            widget.description = (widget.description or "") + f" Used by: {used_by}."

    solver_settings_storage.param.watch(
        on_solver_settings_loaded,
        "loaded_settings",
    )
    for solver_widget in solver_setting_widgets.values():
        solver_widget.param.watch(persist_solver_settings, "value")
    apply_solver_settings.on_click(on_apply_solver_settings)

    for workflow in (physics_tab, human_room_tab):
        for pane in workflow.select(pn.pane.Markdown):
            if "hermes-status" in pane.css_classes:
                pane.styles = {**pane.styles, "min-height": "0"}
                pane.margin = (3, 10)
    for controls in (
        physics_controls, human_room_controls, measurement_rail, settings_modal,
    ):
        _compact_controls(controls, pn)
    _readable_markdown(tabs, pn, plot_template)
    _readable_markdown(settings_modal, pn, plot_template)
    _style_secondary_buttons(tabs, pn, plot_template)
    _style_secondary_buttons(settings_modal, pn, plot_template)
    for tab_group in (*tabs.select(pn.Tabs), *settings_modal.select(pn.Tabs)):
        _fit_active_tab_height(tab_group)

    def cancel_session_work(_session_context) -> None:
        physics_state["revision"] += 1
        physics_state["pending_request"] = None
        human_room_state["revision"] += 1
        human_room_state["diagnostic_revision"] += 1
        measurement_state["load_revision"] += 1
        measurement_state["comparison_revision"] += 1
        for cancel_event in (
            physics_state.get("cancel_event"),
            diagnostics_state.get("cancel_event"),
            human_room_state.get("cancel_event"),
            human_room_state.get("diagnostic_cancel_event"),
            measurement_state.get("cancel_event"),
        ):
            if cancel_event is not None:
                cancel_event.set()
        for task_name in ("bundle_restore_task", "dynamic_restore_task"):
            task = gui_session_state.get(task_name)
            if task is not None and not task.done():
                task.cancel()
        drop_workspace = measurement_state.get("drop_workspace")
        if drop_workspace is not None:
            drop_workspace.cleanup()
            measurement_state["drop_workspace"] = None

    document = pn.state.curdoc
    if document is not None and document.session_context is not None:
        document.on_session_destroyed(cancel_session_work)
    return template


__all__ = ["build_app"]
