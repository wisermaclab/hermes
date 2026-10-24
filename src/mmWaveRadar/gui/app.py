# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

# Service hooks remain available to workflows, assembly and existing callers.
# pylint: disable=unused-import

"""Panel/Plotly application composed over framework-neutral HERMES services."""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
from functools import partial
import hashlib
import html
import inspect
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

from bundle_prepare.paths import portable_relative_path
from bundle_prepare import BundleWriter
from bundle_prepare.safe_npz import (
    GUI_NPZ_LIMITS,
    preflight_npz,
    require_array,
)

from mmWaveRadar.defaults import default_smpl_model_dir as _default_smpl_model_dir
from .interaction_scripts import (
    FIND_PLOT_SCRIPT as _FIND_PLOT_SCRIPT,
    TARGET_INTERACTION_SCRIPTS,
    DYNAMIC_INTERACTION_SCRIPTS,
)
from ..dsp import angle_map_fft, direction_cosine_valid_mask, range_fft
from ..experiments import (
    HumanRoomPreview,
    MATERIAL_PRESETS,
    load_amass_motion,
    load_scene_xml,
    material_preset_parameters,
    motion_source_to_radar_transform,
    prepare_human_room_preview,
    prepared_room_boxes,
    SOLVERS,
    radar_frame_count,
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
from ..naming import (
    ARRAY_MODELS,
    HYBRID_CALIBRATIONS,
    RT_HUMAN_BOUNCES,
    SOLVER_LABELS,
    parse_index_spec,
    saved_tx_schedule,
)
from ..radar import (
    FMCWConfig,
    RadarHardware,
    available_ti_boards,
    get_ti_board_spec,
    ti_digitized_pattern_asset_available,
)
from ..simulation.types import DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD, POConfig
from ..simulation import (
    azimuth_elevation_grids,
    calibrate_range_cube_channel_phase,
    virtual_snapshot_grid,
)
from ..simulation.physical_optics import default_phase_span_scale_rad
from ..radar.adc import if_cutoff_hz
from ..targets import MeshSequence
from ..tutorial_support import prepared_room_scene_xml_for_boxes

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
            f"{label} upload is {len(value):,} bytes; the limit is " f"{int(max_bytes):,} bytes"
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
            raise ValueError(f"OBJ line {line_number} exceeds the 4 MiB line-size limit")
        parts = raw_line.strip().split()
        if not parts or parts[0].startswith(b"#"):
            continue
        if parts[0] == b"v":
            vertices += 1
            if vertices > _MAX_GUI_OBJ_VERTICES:
                raise ValueError("human mesh OBJ exceeds the 2,000,000-vertex limit")
        elif parts[0] == b"f":
            if len(parts) < 4:
                raise ValueError(f"invalid OBJ geometry at line {line_number}")
            triangles += len(parts) - 3
            if triangles > _MAX_GUI_OBJ_TRIANGLES:
                raise ValueError("human mesh OBJ exceeds the 4,000,000-triangle limit")


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
        timing_key = next((key for key in ("bundle_times", "times") if key in arrays), None)
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
                raise ValueError(f"AMASS-like {name} must match the pose frame count")
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
            f"dropped upload contains {len(uploaded):,} files; the limit is " f"{int(max_files):,}"
        )
    prepared: list[tuple[PurePosixPath, bytes]] = []
    total_bytes = 0
    for raw_name, raw_payload in uploaded.items():
        name = str(raw_name)
        relative = portable_relative_path(name, label="unsafe dropped-file path")
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
    if len(top_level) == 1 and any(len(relative.parts) > 1 for relative, _target in written):
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


_PLOT_FONT_FAMILY = "Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"

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

_SOLVER_LABELS = dict(SOLVER_LABELS)
_ANTENNA_NUMBERING_HELP = (
    "HERMES numbers antennas from 0 everywhere (GUI, CLI, JSON, validation, "
    "Python). TI documentation numbers them from 1, so TI's TX1 is TX0 here."
)
_SOLVER_HELP = {
    "rt": (
        "Ray tracing of the whole scene. Retraces every RT refresh frames "
        "(0 = every chirp) and phase-updates paths in between."
    ),
    "po": (
        "Physical optics on the human only; environment returns and "
        "human-room coupling are ignored."
    ),
    "hybrid_po": (
        "RT for the static room and PO for the human, with blocking and "
        "human-room coupling. Amplitudes follow the Hybrid PO calibration."
    ),
}

_SOLVER_SETTINGS_SCHEMA_VERSION = 2
_SOLVER_SETTINGS_STORAGE_KEY = "hermes.solver-settings.v2"
_PO_DEFAULTS_VERSION = 4
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
    "array_model": "phase_center",
    "po_integration_mode": "parent_face_adaptive_analytic",
    "po_quadrature_phase_span_scale_rad": 6.0,
    "po_curvature_error_rad": 0.005,
    "po_quadrature_max_refinement_depth": 3,
    "po_quadrature_max_subfaces_per_parent": 64,
    "po_refresh_frames": 1,
    "po_refresh_on_phase_error_rad": DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD,
    **{
        name: getattr(POConfig(), name)
        for name in (
            "po_refresh_on_area_fraction",
            "po_refresh_on_normal_deg",
            "po_adaptive_visibility_sampling",
            "po_adaptive_visibility_edge_margin",
        )
    },
    "po_visibility_samples_per_face": 1,
    "po_visibility_refresh_frames": 1,
    "po_visibility_fade_frames": 0,
    "rt_samples_per_source": 100_000,
    "rt_max_paths_per_source": 5_000,
    "rt_max_depth": 3,
    "rt_human_bounces": "single",
    "rt_refresh_frames": 1,
    "rt_visibility_refresh_frames": 1,
    "rt_visibility_fade_frames": 0,
    "hybrid_coupling": True,
    "hybrid_reflector_cutoff_db": -20.0,
    "hybrid_max_reflectors": 4,
    "hybrid_calibration": "rt_first_pose",
}
# The headless entry point supplies the same public defaults used by JSON.
_SOLVER_SETTINGS_DEFAULTS.update(
    {
        name: parameter.default
        for name, parameter in inspect.signature(run_human_room_experiment).parameters.items()
        if name in _SOLVER_SETTINGS_DEFAULTS and parameter.default is not None
    }
)
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
    "po_refresh_frames": (0, 1000, True),
    "po_refresh_on_phase_error_rad": (0.0, 1000.0, False),
    "po_refresh_on_area_fraction": (0.0, 1000.0, False),
    "po_refresh_on_normal_deg": (0.0, 180.0, False),
    "po_visibility_samples_per_face": (1, 4096, True),
    "po_adaptive_visibility_edge_margin": (0.0, 1.0, False),
    "po_visibility_refresh_frames": (0, 1000, True),
    "po_visibility_fade_frames": (0, 100, True),
    "rt_samples_per_source": (100, 2_000_000, True),
    "rt_max_paths_per_source": (100, 2_000_000, True),
    "rt_max_depth": (1, 10, True),
    "rt_refresh_frames": (0, 1000, True),
    "rt_visibility_refresh_frames": (0, 1000, True),
    "rt_visibility_fade_frames": (0, 100, True),
    "hybrid_reflector_cutoff_db": (-120.0, 0.0, False),
    "hybrid_max_reflectors": (1, 4, True),
}
_SOLVER_SETTINGS_CHOICES = {
    "array_model": ARRAY_MODELS,
    "po_integration_mode": _PO_INTEGRATION_MODES,
    "rt_human_bounces": RT_HUMAN_BOUNCES,
    "hybrid_calibration": HYBRID_CALIBRATIONS,
}
_SOLVER_SETTINGS_BOOLEANS = ("hybrid_coupling", "po_adaptive_visibility_sampling")
# Free-text antenna groups such as "0-1; 2"; empty means one group.
_SOLVER_SETTINGS_GROUP_TEXT = ("array_tx_groups", "array_rx_groups")


def _array_group_arguments(model: str, tx_text: str, rx_text: str) -> dict:
    """Run arguments for the GUI's antenna groups; ignored unless Subarrays."""

    if model != "subarray":
        return {"array_tx_groups": None, "array_rx_groups": None}
    return {
        "array_tx_groups": _parse_array_groups(tx_text, label="Tx groups"),
        "array_rx_groups": _parse_array_groups(rx_text, label="Rx groups"),
    }


def _call_with_array_groups(run, /, **arguments):
    """Calls a run after turning GUI group text into antenna groups.

    Parsing inside the run lets malformed groups surface as a run error.
    """

    arguments.update(
        _array_group_arguments(
            arguments.get("array_model", "phase_center"),
            arguments.pop("array_tx_groups", ""),
            arguments.pop("array_rx_groups", ""),
        )
    )
    return run(**arguments)


def _parse_array_groups(text: str, *, label: str) -> list[list[int]] | None:
    """Parses GUI antenna groups such as ``"0-1; 2"``; empty means one group."""

    parts = [part for part in str(text).split(";") if part.strip()]
    if not parts:
        return None
    try:
        return [list(parse_index_spec(part)) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from None


def _human_room_running_status(progress: dict) -> str:
    """Describe the current solver pass, frame, and chirp."""

    mode = progress.get("solver")
    mode_label = _SOLVER_LABELS.get(mode, str(mode).replace("_", " ").title())
    parts = [f"**RUNNING** · {mode_label}"]
    solver_mode = progress.get("solver_mode")
    if solver_mode == "setup":
        return " · ".join([*parts, "Preparing"])
    if mode == "hybrid_po":
        pass_label = {
            "po": "Human PO",
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

    # Session restoration selects tabs in JavaScript before the Python active
    # watcher can respond. Reveal the selected child in that same browser
    # update; otherwise the selected tab can retain visible=False from startup
    # until another tab change reaches the server.
    tabs.jscallback(active="""
for (const [index, panel] of cb_obj.tabs.entries()) {
  panel.child.visible = index === cb_obj.active
}
""")

    def show_active(_event=None):
        for index, child in enumerate(tabs):
            child.visible = index == tabs.active

    tabs.param.watch(show_active, "active")
    show_active()


def _readable_markdown(root, pn, plot_template):
    tokens = _plot_theme_tokens(plot_template)
    css = (
        f"code {{color:{tokens['accent']} !important; background:transparent !important;}}"
        "h2 {font-size:27px !important;} h3 {font-size:20px !important;}"
    )
    for pane in root.select(pn.pane.Markdown):
        if css not in pane.stylesheets:
            pane.stylesheets = [*pane.stylesheets, css]
    for card in root.select(pn.Card):
        header = card._header  # pylint: disable=protected-access
        if header is not None:
            header.stylesheets = [
                "h3 {font-size:20px !important; line-height:1.35; margin:4px 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}"
            ]


def _style_secondary_buttons(root, pn, plot_template):
    dark = plot_template == "plotly_dark"
    background, foreground, border, hover = (
        ("#182536", "#e7eef8", "#64748b", "#26384d")
        if dark
        else ("#e2e8f0", "#172033", "#94a3b8", "#cbd5e1")
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
            value=help_text,
            width=22,
            height=22,
            sizing_mode="fixed",
            margin=0,
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
            values.append((str(key), seconds))
    if not values:
        return ""
    threshold = max(0.001, max(seconds for _, seconds in values) * 0.001)
    largest = sorted(
        (entry for entry in values if entry[1] >= threshold),
        key=lambda entry: (-entry[1], entry[0]),
    )[:10]
    if not largest:
        return ""
    rows = ["| Stage | Seconds |", "| --- | ---: |"]
    for key, seconds in largest:
        words = [
            word.upper() if word in {"rt", "po", "adc", "bvh", "fft"} else word
            for word in key.split("_")
        ]
        label = " ".join(words)
        label = label[0].upper() + label[1:]
        rows.append(f"| {_safe_markdown_text(label)} | {seconds:,.3f} |")
    rows.append(
        "\nLargest measured stages; overlapping timings are not additive. "
        "Entries below 1 ms or 0.1% of the largest stage are omitted. "
        "All stages are available under Advanced."
    )
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
    for name, choices in _SOLVER_SETTINGS_CHOICES.items():
        if settings.get(name) in choices:
            validated[name] = settings[name]
    for name in _SOLVER_SETTINGS_BOOLEANS:
        if isinstance(settings.get(name), bool):
            validated[name] = settings[name]
    for name in _SOLVER_SETTINGS_GROUP_TEXT:
        value = settings.get(name)
        if isinstance(value, str):
            try:
                _parse_array_groups(value, label=name)
            except ValueError:
                continue
            validated[name] = value

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
        validated.update(
            {
                name: value
                for name, value in _SOLVER_SETTINGS_DEFAULTS.items()
                if name.startswith("po_")
            }
        )
    return validated


def _validated_po_phase_span_by_mode(payload: object) -> dict[str, float]:
    """Return valid remembered spans from a current solver-settings payload."""

    if not isinstance(payload, dict):
        return {}
    if (
        payload.get("schema_version") != _SOLVER_SETTINGS_SCHEMA_VERSION
        or payload.get("po_defaults_version") != _PO_DEFAULTS_VERSION
    ):
        return {}
    spans = payload.get("po_phase_span_by_mode")
    if not isinstance(spans, dict):
        return {}
    lower, upper, _ = _SOLVER_SETTINGS_NUMERIC_BOUNDS["po_quadrature_phase_span_scale_rad"]
    validated = {}
    for mode, value in spans.items():
        if (
            mode not in _PO_INTEGRATION_MODES
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
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
    if any(trace.type in {"mesh3d", "scatter3d"} for trace in figure.data):
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
    figure.update_layout(
        xaxis_visible=False, yaxis_visible=False, margin={"l": 20, "r": 20, "t": 30, "b": 20}
    )
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


def _profile_channels(values, fallback) -> np.ndarray:
    """Accept per-channel profiles and older single-profile result objects."""

    power = np.asarray(fallback if values is None else values, dtype=float)
    return power[:, None] if power.ndim == 1 else power


def _mean_range_profile_figure(
    series,
    go,
    *,
    title: str,
    plot_template: str,
    relative: bool = False,
    frame_index: int = 0,
    height: int = 350,
):
    """Overlay channel-mean power for one frame on the original single axes."""

    figure = go.Figure()
    peak_db = -80.0
    for label, color, ranges, power in series:
        mean_power = power.mean(axis=-1)
        reference = max(float(np.max(mean_power)), np.finfo(float).tiny)
        # Keep one scale across frames while displaying only the selected one.
        peak_db = max(peak_db, 0.0 if relative else 10.0 * np.log10(reference))
        if mean_power.ndim == 2:
            mean_power = mean_power[int(frame_index)]
        db = _absolute_power_db(mean_power)
        if relative:
            db -= 10.0 * np.log10(reference)
        db = np.where((mean_power > 0.0) & np.isfinite(db) & (db >= -80.0), db, np.nan)
        figure.add_trace(
            go.Scatter(
                x=ranges,
                y=db,
                mode="lines",
                connectgaps=False,
                name=label,
                showlegend=True,
                line={"color": color, "width": 2.5, "dash": "solid"},
                hovertemplate=(
                    "Range %{x:.3f} m<br>Power %{y:.1f} dB<extra>%{fullData.name}</extra>"
                ),
            )
        )
    upper_db = 0.0 if relative else max(-70.0, 5.0 * np.ceil((peak_db + 3.0) / 5.0))
    figure.update_layout(
        title=title,
        xaxis_title="Range [m]",
        yaxis_title="Relative power [dB]" if relative else "Power [dB, simulator units]",
        yaxis_range=[-80.0, upper_db],
        height=height,
        margin={"l": 60, "r": 20, "t": 52, "b": 55},
    )
    if relative:
        figure.update_layout(legend={"orientation": "h", "yanchor": "bottom", "y": 1.02})
    return _style_figure(figure, plot_template=plot_template)


def _simulation_channel_labels(parameters, num_channels: int, *, board_model=None):
    """Recover physical Tx/Rx IDs in the simulated virtual-channel order."""

    board_model = board_model or parameters.get("board_model")
    if board_model:
        hardware = RadarHardware.from_ti_board(board_model, pattern_mode="none")
        tx_ids = parameters.get("tx_indices_zero_based", list(range(hardware.num_tx)))
        rx_ids = parameters.get("rx_indices_zero_based", list(range(hardware.num_rx)))
        hardware = hardware.subset_tx(tx_ids).subset_rx(rx_ids)
        if hardware.num_virtual_channels == num_channels:
            return tuple(
                f"TX{tx_ids[t]}/RX{rx_ids[r]}"
                for t, r in zip(hardware.virtual_tx_indices(), hardware.virtual_rx_indices())
            )
    return tuple(f"Channel {index}" for index in range(num_channels))


def _channel_range_profile_figure(
    power,
    ranges_m,
    go,
    *,
    labels,
    title: str,
    plot_template: str,
    frame_index: int = 0,
):
    """Overlay individual channel powers for one frame on one set of axes."""

    from plotly.colors import qualitative

    power = np.asarray(power, dtype=float)
    peak = float(np.max(np.where(np.isfinite(power), power, 0.0)))
    upper_db = max(-70.0, 5.0 * np.ceil((10 * np.log10(max(peak, 1e-30)) + 3.0) / 5.0))
    if power.ndim == 3:
        power = power[int(frame_index)]
    figure = go.Figure()
    colors = qualitative.Light24 if plot_template == "plotly_dark" else qualitative.Dark24
    for channel, label in enumerate(labels):
        values = power[:, channel]
        db = _absolute_power_db(values)
        db = np.where((values > 0.0) & np.isfinite(db) & (db >= -80.0), db, np.nan)
        figure.add_trace(
            go.Scatter(
                x=ranges_m,
                y=db,
                name=label,
                mode="lines",
                connectgaps=False,
                line={"width": 1.7, "color": colors[channel % len(colors)]},
                hovertemplate=(
                    "Range %{x:.3f} m<br>Power %{y:.1f} dB<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="Range [m]",
        yaxis_title="Power [dB, simulator units]",
        yaxis_range=[-80.0, upper_db],
        height=420,
        legend={"title": "Channel"},
        margin={"l": 60, "r": 20, "t": 52, "b": 55},
    )
    return _style_figure(figure, plot_template=plot_template)


def _static_channel_profile_figures(result, go, *, plot_template: str):
    figures = []
    for mode in ("po", "rt"):
        power = getattr(result, f"{mode}_channel_range_profile_power", None)
        if power is None:
            continue
        figures.append(
            _channel_range_profile_figure(
                power,
                result.ranges_m,
                go,
                labels=_simulation_channel_labels(result.sensor_parameters, power.shape[-1]),
                title=f"{_SOLVER_LABELS[mode]} · range profiles by channel",
                plot_template=plot_template,
            )
        )
    return figures


def _face_colors(face_power: np.ndarray) -> list[str]:
    from matplotlib import colormaps, colors

    power = np.asarray(face_power, dtype=float)
    if not np.any(power > 0.0):
        return ["#000000"] * power.size
    db = _relative_db(power, floor_db=-60.0)
    normalized = np.clip((db + 60.0) / 60.0, 0.0, 1.0)
    cmap = colormaps.get_cmap("inferno")
    return [
        colors.to_hex(cmap(float(level)), keep_alpha=False) if value > 0.0 else "#000000"
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
        _scripts = dict(TARGET_INTERACTION_SCRIPTS)

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
        _scripts = dict(DYNAMIC_INTERACTION_SCRIPTS)

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
    boresight_length = float(np.clip(0.22 * float(np.linalg.norm(pivot)), 0.18, 0.55))
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
    range_spans = {axis: bounds[1] - bounds[0] for axis, bounds in ranges.items()}
    largest_range_span = max(range_spans.values())
    # Plotly's data-aspect normalization preserves scene-box volume rather than
    # forcing its longest side to one. Reproduce that normalization explicitly
    # so the frame fills the viewport while remaining invariant under rotation.
    aspect_normalizer = float(
        np.cbrt(np.prod(np.asarray(list(range_spans.values()), dtype=float)))
    )
    aspect_ratio = {axis: axis_span / aspect_normalizer for axis, axis_span in range_spans.items()}
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
                "Radar local +z" f"<br>roll {np.rad2deg(radar_roll):.1f}°" "<extra></extra>"
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
        title=("PO facet contribution · " f"{result.target_type.replace('_', ' ').title()}"),
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
        legend=({"orientation": "h", "y": 0.99, "x": 0.01} if human_faces else None),
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


def _human_room_scene_title(
    preview, frame_index: int, *, motion_time_s: float | None = None
) -> str:
    """Return the dynamic-scene title for one preview pose."""

    if preview is None:
        return "Dynamic scene · upload human motion to place the subject"
    selected_frame = int(np.clip(frame_index, 0, len(preview.mesh_sequence.times) - 1))
    if motion_time_s is None:
        motion_time_s = float(preview.mesh_sequence.times[selected_frame])
    return f"{preview.scene_source_name} · {preview.source_name} · " f"t = {motion_time_s:.3f} s"


def _human_position_bounds(room_boxes) -> tuple[tuple[float | None, float | None], ...]:
    """Use the floor footprint and room height as manual placement limits."""

    if not room_boxes:
        return ((None, None),) * 3
    centers = np.asarray([box["translate"] for box in room_boxes], dtype=float)
    half_sizes = np.abs(np.asarray([box["scale"] for box in room_boxes], dtype=float))
    lower = np.min(centers - half_sizes, axis=0).tolist()
    upper = np.max(centers + half_sizes, axis=0).tolist()
    floors = [box for box in room_boxes if "floor" in str(box["id"]).lower()]
    if floors:
        floor_centers = np.asarray([box["translate"] for box in floors], dtype=float)
        floor_sizes = np.abs(np.asarray([box["scale"] for box in floors], dtype=float))
        lower[:2] = np.min(floor_centers[:, :2] - floor_sizes[:, :2], axis=0).tolist()
        upper[:2] = np.max(floor_centers[:, :2] + floor_sizes[:, :2], axis=0).tolist()
        lower[2] = float(np.min(floor_centers[:, 2] + floor_sizes[:, 2]))
        ceilings = [box for box in room_boxes if "ceiling" in str(box["id"]).lower()]
        walls = [box for box in room_boxes if "wall" in str(box["id"]).lower()]
        if ceilings:
            upper[2] = min(
                float(box["translate"][2]) - abs(float(box["scale"][2])) for box in ceilings
            )
        elif walls:
            upper[2] = max(
                float(box["translate"][2]) + abs(float(box["scale"][2])) for box in walls
            )
        else:
            # A floor alone does not declare the room's height.
            upper[2] = None
    return tuple((lo, hi if hi is None or hi > lo else None) for lo, hi in zip(lower, upper))


def _human_floor_overlay_data(vertices, room_boxes) -> dict:
    """Mark human vertices below a floor's top surface within its footprint."""

    vertices = np.asarray(vertices, dtype=float)
    clearance = np.full(len(vertices), np.nan, dtype=float)
    floor_boxes = [box for box in room_boxes if "floor" in str(box["id"]).lower()]
    for box in floor_boxes:
        center = np.asarray(box["translate"], dtype=float)
        half_size = np.abs(np.asarray(box["scale"], dtype=float))
        over_floor = np.all(
            np.abs(vertices[:, :2] - center[:2]) <= half_size[:2] + 1e-6,
            axis=1,
        )
        floor_top = center[2] + half_size[2]
        clearance[over_floor] = np.fmin(clearance[over_floor], vertices[over_floor, 2] - floor_top)

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
    motion_time_s: float | None = None,
) -> bool:
    """Restyle the human mesh and floor markers during motion playback.

    Rebuilding the complete Plotly room for every player tick can make browser
    input outrun server rendering.  A Plotly restyle sends only the moving
    vertices and floor markers, preserving the room, radar, and current camera.
    """

    if figure is None or preview is None:
        return False
    human_trace_indices = [
        index for index, trace in enumerate(figure.data) if trace.name == "Human mesh"
    ]
    if len(human_trace_indices) != 1 or any(
        trace.name == "Top target-touching RT rays" for trace in figure.data
    ):
        return False
    sequence = preview.mesh_sequence
    selected_frame = int(np.clip(frame_index, 0, len(sequence.times) - 1))
    if motion_time_s is None:
        motion_time_s = float(sequence.times[selected_frame])
    vertices = np.asarray(
        sequence.vertices_at(float(motion_time_s)),
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
            overlay_data = _human_floor_overlay_data(vertices, overlay.meta["floor_boxes"])
            for key, value in overlay_data.items():
                updates.setdefault(key, [getattr(trace, key)]).append(value)
            trace_indices.append(index)
            break
    # The camera-interaction guard retains only the latest restyle, so the
    # mesh and floor markers must travel together in a single frame update.
    figure.plotly_restyle(updates, trace_indices)
    if update_title:
        figure.plotly_relayout(
            {
                "title.text": _human_room_scene_title(
                    preview,
                    selected_frame,
                    motion_time_s=motion_time_s,
                )
            }
        )
    return True


def _human_room_window_limit(duration_s: float, frame_period_s: float) -> float:
    """Latest selectable motion time [s]; a single static pose allows one frame."""

    duration_s = max(float(duration_s), 0.0)
    return duration_s if duration_s > 0.0 else float(frame_period_s)


def _human_room_window_values(
    *,
    duration_s: float,
    frame_period_s: float,
    previous_start: float,
    previous_end: float,
    reset_window: bool,
) -> tuple[float, float]:
    """Return a safe motion time window [s] for the GUI controls.

    Newly loaded motion starts with one radar frame from its first pose.
    """

    limit = _human_room_window_limit(duration_s, frame_period_s)
    if reset_window:
        return 0.0, min(float(frame_period_s), limit)
    start_s = min(max(float(previous_start), 0.0), limit)
    end_s = min(max(float(previous_end), start_s), limit)
    return start_s, end_s


def _nearest_pose_index(times, time_s: float) -> int:
    """Index of the preview pose closest to one motion time [s]."""

    times = np.asarray(times, dtype=float)
    return int(np.argmin(np.abs(times - float(time_s))))


def _radar_frame_motion_time(result, frame_index: int) -> float:
    """Motion time [s] at the start of one simulated radar frame."""

    return float(result.manifest.scene.parameters["motion_start_s"]) + int(frame_index) * float(
        result.manifest.radar.parameters["frame_period_s"]
    )


def _radar_frame_for_motion_time(result, time_s: float) -> int | None:
    """Simulated radar frame nearest one motion time, or None outside the run."""

    start_s = float(result.manifest.scene.parameters["motion_start_s"])
    period_s = float(result.manifest.radar.parameters["frame_period_s"])
    frame = int(np.floor((float(time_s) - start_s) / period_s + 0.5))
    if frame < 0 or frame >= int(np.asarray(result.adc).shape[0]):
        return None
    return frame


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
    motion_time_s: float | None = None,
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
        room_boxes = prepared_room_boxes("empty_room")
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
        material_name = str(box.get("material_display", box.get("material", "Unspecified")))
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
                    0.18 if "wall" in str(box["id"]) or "floor" in str(box["id"]).lower() else 0.45
                ),
                flatshading=True,
                name=object_name,
                showlegend=False,
                hovertemplate=(
                    f"Object: {object_name}<br>Material: {material_name}" "<extra></extra>"
                ),
            )
        )
    human_vertices = None
    if preview is not None:
        sequence = preview.mesh_sequence
        selected_frame = int(np.clip(frame_index, 0, len(sequence.times) - 1))
        if motion_time_s is None:
            motion_time_s = float(sequence.times[selected_frame])
        vertices = np.asarray(
            sequence.vertices_at(float(motion_time_s)),
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
                    "Object: Human facet %{pointNumber}<br>Material: Human skin<extra></extra>"
                ),
                lighting=(
                    _PO_POWER_LIGHTING
                    if has_po_power
                    else {
                        "ambient": 0.68,
                        "diffuse": 0.82,
                        "specular": 0.08,
                        "roughness": 0.86,
                    }
                ),
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
                    "Floor penetration: %{customdata:.2f} cm<br>z: %{z:.3f} m<extra></extra>"
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
    orientation = np.deg2rad(np.asarray(radar_orientation_deg, dtype=float))
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
        else _human_room_scene_title(preview, frame_index, motion_time_s=motion_time_s)
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


def _human_room_profile_figure(result, go, *, plot_template: str, frame_index: int = 0):
    """Plot one room-simulation mode on the original shared axes."""

    palette = _plot_theme_tokens(plot_template)
    solver = getattr(result, "solver", "hybrid_po")
    mode_label = _SOLVER_LABELS.get(solver, solver)
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
    channel_profiles = getattr(result, "channel_range_profiles", None) or {}
    series = [
        (
            labels.get(name, name.replace("_", " ").title()),
            color,
            result.ranges_m,
            _profile_channels(channel_profiles.get(name), values),
        )
        for color, (name, values) in zip(colors, result.range_profiles.items())
    ]
    return _mean_range_profile_figure(
        series,
        go,
        title=f"{mode_label} range profile",
        plot_template=plot_template,
        frame_index=frame_index,
        height=360,
    )


def _hybrid_po_component_profile_figure(result, go, *, plot_template: str, frame_index: int = 0):
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
    channel_profiles = getattr(result, "channel_range_profiles", None) or {}
    series = [
        (label, color, result.ranges_m, _profile_channels(channel_profiles.get(name), values))
        for name, label, color, values in available
    ]
    return _mean_range_profile_figure(
        series,
        go,
        title="Hybrid PO component range profiles",
        plot_template=plot_template,
        frame_index=frame_index,
        height=380,
    )


def _human_room_rt_segment_histogram(results, go, *, plot_template: str):
    """Aggregate RT path interaction depths as path-segment histograms."""

    palette = _plot_theme_tokens(plot_template)
    colors = (palette["blue"], palette["orange"])
    series = []
    for result in results:
        if result.solver != "rt":
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
        mode_label = _SOLVER_LABELS.get(result.solver, result.solver)
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
    po = _profile_channels(
        getattr(result, "po_channel_range_profile_power", None),
        result.po_range_profile_power,
    )
    rt = _profile_channels(
        getattr(result, "rt_channel_range_profile_power", None),
        result.rt_range_profile_power,
    )
    return _mean_range_profile_figure(
        [
            ("Physical optics (PO)", palette["blue"], result.ranges_m, po),
            ("Ray tracing (RT)", palette["orange"], result.ranges_m, rt),
        ],
        go,
        title="RT and PO range profiles · absolute simulated power",
        plot_template=plot_template,
        height=330,
    )


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
        tx_schedule=saved_tx_schedule(result.sensor_parameters),
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
        products[f"{solver}_range_cube"] = calibrate_range_cube_channel_phase(range_cube, hardware)
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
    products["centroid_range_bin"] = int(np.argmin(np.abs(common_ranges - centroid_range_m)))
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
            reduced = np.sum(power, axis=tuple(range(power.ndim - 1)))[None, :]
            chunk_u_axis = np.asarray(angle["u"], dtype=float)
            chunk_v_axis = np.asarray([0.0])
        elif "v" in angle:
            spatial_dimensions = 1
            reduced = np.sum(power, axis=tuple(range(power.ndim - 1)))[:, None]
            chunk_u_axis = np.asarray([0.0])
            chunk_v_axis = np.asarray(angle["v"], dtype=float)
        else:
            raise ValueError(f"{hardware.name} does not have an angle-resolving aperture")
        accumulated = reduced if accumulated is None else accumulated + reduced
        snapshot_count += int(np.prod(power.shape[: power.ndim - spatial_dimensions]))
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
    azimuth_deg, elevation_deg = azimuth_elevation_grids({"u": u_axis, "v": v_axis})
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


def _adc_trace_figure(
    products,
    go,
    *,
    title: str,
    plot_template: str,
    frame_index: int = 0,
    chirp_index: int = 0,
    channel_index: int = 0,
):
    palette = _plot_theme_tokens(plot_template)
    adc = np.asarray(products.adc)
    figure = go.Figure()
    selection = (int(frame_index), int(chirp_index), int(channel_index))
    if any(
        index < 0 or index >= limit
        for index, limit in zip(selection, (adc.shape[0], adc.shape[1], adc.shape[3]))
    ):
        figure.add_annotation(
            text="Selected frame, chirp or channel is unavailable in this ADC.", showarrow=False
        )
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
    motion_time_s: float | None = None,
):
    power_frames = np.asarray(products.range_doppler_power)
    frame_index = int(frame_index)
    if frame_index < 0 or frame_index >= power_frames.shape[0]:
        raise ValueError(f"frame_index must be in [0, {power_frames.shape[0] - 1}]")
    power = power_frames[frame_index]
    frame_title = title
    if motion_time_s is not None:
        frame_title = f"{title} · radar frame {frame_index} · " f"t = {float(motion_time_s):.3f} s"
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
    primary_artifact = artifacts.get(primary_key, {}) if isinstance(artifacts, dict) else {}
    if not isinstance(primary_artifact, dict):
        primary_artifact = {}
    primary_solver = str(
        primary_artifact.get(
            "solver",
            descriptor.get("primary_solver", ""),
        )
    ).casefold()
    # Older bundles recorded Hybrid PO as solver "po" plus a simulation_mode.
    legacy_mode = primary_artifact.get(
        "simulation_mode",
        descriptor.get("primary_simulation_mode"),
    )
    if legacy_mode == "hybrid_po":
        primary_solver = "hybrid_po"
    if primary_solver in _SOLVER_LABELS:
        return f"Primary {_SOLVER_LABELS[primary_solver]} simulation"
    return "Primary simulation"


def _bundle_scene_path(bundle) -> Path | None:
    """Resolve the portable scene declared directly or through environment."""

    return None if bundle.scene_path is None else Path(bundle.scene_path)


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
            "evaluated motion" if bundle.evaluated_motion_path is not None else "static target"
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
    target = descriptor.get("target") or {}
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
                (Path(bundle.root) / str(manifest_path)).read_text(encoding="utf-8")
            )
            material = manifest.get("scene", {}).get("parameters", {}).get("material")
            if material:
                return str(material)
        except (AttributeError, json.JSONDecodeError, OSError, TypeError):
            pass
    summary_path = Path(bundle.root) / "bundle_summary.json"
    if summary_path.is_file():
        try:
            material = json.loads(summary_path.read_text(encoding="utf-8")).get("material")
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
            scene_source_name=(scene_path.name if scene_path is not None else loaded.bundle_id),
        )
    orientation_deg = np.rad2deg(np.asarray(bundle.sensor_config.orientation, dtype=float))
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
    frame_index: int = 0,
):
    """Plot the bundle's channel mean on a relative-power scale."""

    palette = _plot_theme_tokens(plot_template)
    return _mean_range_profile_figure(
        [
            (
                "Range profile",
                palette["blue"],
                products.range_time_ranges_m,
                _profile_channels(
                    getattr(products, "channel_range_profile_power", None),
                    products.range_profile_power,
                ),
            )
        ],
        go,
        title=title,
        plot_template=plot_template,
        relative=True,
        frame_index=frame_index,
        height=330,
    )


def _bundle_range_profile_comparison_figure(
    primary_label,
    primary_products,
    candidates,
    go,
    *,
    plot_template: str,
    frame_index: int = 0,
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
    series = [(primary_label, primary_products), *list(candidates)]
    return _mean_range_profile_figure(
        [
            (
                str(label),
                colors[index % len(colors)],
                products.range_time_ranges_m,
                _profile_channels(
                    getattr(products, "channel_range_profile_power", None),
                    products.range_profile_power,
                ),
            )
            for index, (label, products) in enumerate(series)
        ],
        go,
        title=("Range profile comparison" if candidates else f"{primary_label} range profile"),
        plot_template=plot_template,
        relative=True,
        frame_index=frame_index,
    )


def _human_room_results_view(
    mode_results,
    pn,
    go,
    *,
    plot_template: str,
):
    """Build product-first views that compare every completed solver mode."""
    return _HumanRoomResultsView(mode_results, pn, go, plot_template).run()


class _HumanRoomResultsView:
    """Per-call state with explicit preparation, channel/product and output stages."""

    def __init__(self, mode_results, pn, go, plot_template):
        self.mode_results = mode_results
        self.pn = pn
        self.go = go
        self.plot_template = plot_template

    def _mode_label(self, result):
        return _SOLVER_LABELS.get(result.solver, result.solver)

    def range_doppler_figure(self, result, frame_index: int):
        return _range_doppler_figure(
            result,
            self.go,
            title=f"{self._mode_label(result)} range-Doppler map",
            frame_index=int(frame_index),
            motion_time_s=_radar_frame_motion_time(result, frame_index),
            plot_template=self.plot_template,
        )

    def reset_plot(self, pane, figure):
        pane.relayout_data = {}
        pane.viewport = None
        pane.object = figure

    def reset_range_profiles(self, frame_index: int):
        self.reset_plot(
            self.profile,
            _mean_range_profile_figure(
                self.profile_series,
                self.go,
                title="Range profile comparison",
                plot_template=self.plot_template,
                frame_index=frame_index,
                height=380,
            ),
        )
        for pane, (result, channels, labels) in zip(
            self.channel_profile_panes, self.channel_profile_results
        ):
            self.reset_plot(
                pane,
                _channel_range_profile_figure(
                    channels,
                    result.ranges_m,
                    self.go,
                    labels=labels,
                    title=f"{_SOLVER_LABELS[result.solver]} · range profiles by channel",
                    plot_template=self.plot_template,
                    frame_index=frame_index,
                ),
            )
        if self.hybrid_component_pane is not None:
            self.reset_plot(
                self.hybrid_component_pane,
                _hybrid_po_component_profile_figure(
                    self.hybrid_result,
                    self.go,
                    plot_template=self.plot_template,
                    frame_index=frame_index,
                ),
            )

    def reset_range_doppler(self, display_frame: int):
        frame_index = int(display_frame)
        self.previous_range_doppler_frame.disabled = display_frame <= 0
        self.next_range_doppler_frame.disabled = (
            display_frame >= self.range_doppler_frame_count - 1
        )
        self.range_doppler_frame_status.object = (
            f"**Displayed frame index:** {display_frame} / "
            f"{max(self.range_doppler_frame_count - 1, 0)}"
        )
        for result, pane in zip(self.results, self.range_doppler_panes):
            self.reset_plot(pane, self.range_doppler_figure(result, frame_index))
        self.reset_range_profiles(frame_index)

    def step_range_doppler_frame(self, offset: int):
        self.range_doppler_frame.value = int(
            np.clip(
                int(self.range_doppler_frame.value) + int(offset),
                0,
                max(self.range_doppler_frame_count - 1, 0),
            )
        )

    def reset_active_product(self, event):
        if int(event.new) == 0:
            self.reset_range_profiles(int(self.range_doppler_frame.value))
        elif int(event.new) == 1:
            for result, pane in zip(self.results, self.range_time_panes):
                self.reset_plot(
                    pane,
                    _range_time_figure(
                        result,
                        self.go,
                        title=f"{self._mode_label(result)} range-time map",
                        plot_template=self.plot_template,
                    ),
                )
        else:
            self.reset_range_doppler(int(self.range_doppler_frame.value))

    def _stage_profiles(self):
        """Profiles for this call."""
        self.results = [self.mode_results[mode] for mode in SOLVERS if mode in self.mode_results]
        if not self.results:
            raise ValueError("at least one completed simulation mode is required")

        self.palette = _plot_theme_tokens(self.plot_template)
        self.colors = (
            self.palette["blue"],
            self.palette["orange"],
            self.palette["green"],
            self.palette["purple"],
        )
        self.profile_series = []
        for self.color, self.result in zip(self.colors, self.results):
            self.mode_label = _SOLVER_LABELS.get(self.result.solver, self.result.solver)
            self.channels = getattr(self.result, "channel_range_profiles", None) or {}
            self.profile_series.append(
                (
                    self.mode_label,
                    self.color,
                    self.result.ranges_m,
                    _profile_channels(
                        self.channels.get("total"), self.result.range_profiles["total"]
                    ),
                )
            )
        self.profile_figure = _mean_range_profile_figure(
            self.profile_series,
            self.go,
            title="Range profile comparison",
            plot_template=self.plot_template,
            height=380,
        )
        self.profile = self.pn.pane.Plotly(
            self.go.Figure(self.profile_figure),
            height=self.profile_figure.layout.height,
        )
        self.hybrid_component_pane = None
        self.hybrid_component_figure = None
        self.hybrid_result = next(
            (result for result in self.results if result.solver == "hybrid_po"),
            None,
        )
        if self.hybrid_result is not None:
            self.hybrid_component_figure = _hybrid_po_component_profile_figure(
                self.hybrid_result,
                self.go,
                plot_template=self.plot_template,
            )
            if self.hybrid_component_figure is not None:
                self.hybrid_component_pane = self.pn.pane.Plotly(
                    self.go.Figure(self.hybrid_component_figure),
                    height=self.hybrid_component_figure.layout.height,
                )
        self.channel_profile_panes = []
        self.channel_profile_results = []
        for self.result in self.results:
            self.channels = (getattr(self.result, "channel_range_profiles", None) or {}).get(
                "total"
            )
            if self.channels is None:
                continue
            self.labels = _simulation_channel_labels(
                self.result.manifest.radar.parameters,
                self.channels.shape[-1],
                board_model=self.result.manifest.radar.board_model,
            )
            self.channel_profile_results.append((self.result, self.channels, self.labels))
            self.channel_profile_panes.append(
                self.pn.pane.Plotly(
                    _channel_range_profile_figure(
                        self.channels,
                        self.result.ranges_m,
                        self.go,
                        labels=self.labels,
                        title=f"{_SOLVER_LABELS[self.result.solver]} · range profiles by channel",
                        plot_template=self.plot_template,
                    ),
                    height=420,
                )
            )
        self.profile_panes = [self.profile, *self.channel_profile_panes]
        if self.hybrid_component_pane is not None:
            self.hybrid_channel_index = next(
                (
                    index
                    for index, (result, _, _) in enumerate(self.channel_profile_results)
                    if result.solver == "hybrid_po"
                ),
                len(self.channel_profile_panes),
            )
            self.profile_panes.insert(1 + self.hybrid_channel_index, self.hybrid_component_pane)

        self.profile_view = self.pn.Column(*self.profile_panes, sizing_mode="stretch_width")

    def _stage_doppler_controls(self):
        """Doppler controls for this call."""
        self.range_time_panes = [
            self.pn.pane.Plotly(
                _range_time_figure(
                    result,
                    self.go,
                    title=f"{self._mode_label(result)} range-time map",
                    plot_template=self.plot_template,
                ),
                height=380,
            )
            for result in self.results
        ]
        self.range_doppler_frame_count = min(
            int(result.range_doppler_power.shape[0]) for result in self.results
        )
        self.range_doppler_frame = self.pn.widgets.IntSlider(
            label=f"Radar frame index (0–{max(self.range_doppler_frame_count - 1, 0)})",
            start=0,
            end=max(self.range_doppler_frame_count - 1, 1),
            value=0,
            step=1,
            disabled=self.range_doppler_frame_count <= 1,
        )
        self.previous_range_doppler_frame = self.pn.widgets.Button(
            label="Previous frame",
            icon="chevron-left",
            color="light",
            disabled=True,
        )
        self.next_range_doppler_frame = self.pn.widgets.Button(
            label="Next frame",
            icon="chevron-right",
            color="light",
            disabled=self.range_doppler_frame_count <= 1,
        )
        self.range_doppler_frame_status = self.pn.pane.Markdown(
            f"**Displayed frame index:** 0 / {max(self.range_doppler_frame_count - 1, 0)}"
        )

        self.range_doppler_panes = [
            self.pn.pane.Plotly(self.range_doppler_figure(result, 0), height=380)
            for result in self.results
        ]
        self.product_tabs = self.pn.Tabs(
            ("Range profile", self.profile_view),
            ("Range-time", self.pn.Column(*self.range_time_panes)),
            (
                "Range-Doppler",
                self.pn.Column(*self.range_doppler_panes),
            ),
            dynamic=True,
            sizing_mode="stretch_width",
        )

    def _stage_metadata_and_bindings(self):
        """Metadata and bindings for this call."""
        self.range_doppler_frame.param.watch(
            lambda event: self.reset_range_doppler(int(event.new)), "value"
        )

        self.previous_range_doppler_frame.on_click(
            lambda _event: self.step_range_doppler_frame(-1)
        )
        self.next_range_doppler_frame.on_click(lambda _event: self.step_range_doppler_frame(1))

        self.product_tabs.param.watch(self.reset_active_product, "active")
        self.metadata_view = {}
        self.refinement_notes = []
        for self.result in self.results:
            self.metadata = self.result.metadata
            self.metadata_view[self._mode_label(self.result)] = {
                "solver": self.result.solver,
                "manifest_sha256": self.result.manifest.fingerprint,
                "adc_shape": list(self.result.adc.shape),
                "components": sorted(self.result.components),
                "runtime_s": self.result.runtime_s,
                "motion_time_window_s": [
                    float(self.result.manifest.scene.parameters["motion_start_s"]),
                    float(self.result.manifest.scene.parameters["motion_end_s"]),
                ],
                "radar_frames": int(self.result.adc.shape[0]),
                "path_count_max": (
                    int(np.max(self.metadata.path_counts))
                    if self.metadata.path_counts is not None
                    else None
                ),
                "visible_face_count_max": (
                    int(np.max(self.metadata.visible_face_counts))
                    if self.metadata.visible_face_counts is not None
                    else None
                ),
                "po_calibration": self.metadata.po_calibration,
                "runtime_profile_s": self.metadata.runtime_profile_s,
            }
            for self.event in getattr(self.metadata, "events", None) or []:
                if self.event.get("reason") != "adaptive_po_refinement":
                    continue
                self.metadata_view[self._mode_label(self.result)]["po_refinement"] = self.event
                if self.event["limited_passes"]:
                    self.refinement_notes.append(
                        f"**{self._mode_label(self.result)} — PO refinement limit:** "
                        f"up to {self.event['max_unresolved_parent_faces']:,} facets remained above a tolerance "
                        "when subdivision reached its limit. Raise **PO maximum refinement depth** "
                        "or **PO subfaces per parent (0 = unlimited)** in **Settings → Solver settings**."
                    )
        _fit_active_tab_height(self.product_tabs)
        self.result_sections = [
            self.pn.Column(
                self.pn.Row(
                    self.previous_range_doppler_frame,
                    self.next_range_doppler_frame,
                    self.range_doppler_frame,
                ),
                self.range_doppler_frame_status,
                visible=self.range_doppler_frame_count > 1,
            ),
            self.product_tabs,
            *[self.pn.pane.Alert(note, alert_type="warning") for note in self.refinement_notes],
        ]
        self.rt_segment_figure = _human_room_rt_segment_histogram(
            self.results,
            self.go,
            plot_template=self.plot_template,
        )
        if self.rt_segment_figure is not None:
            self.result_sections.extend(
                (
                    self.pn.pane.Markdown("### RT path segments"),
                    self.pn.pane.Plotly(self.rt_segment_figure, height=340),
                )
            )
        self.summary = [
            "| Mode | Runtime | RT paths (max) | PO visible facets (max) | Calibration |",
            "| --- | ---: | ---: | ---: | --- |",
        ]

    def _stage_result(self):
        """Result for this call."""
        self.breakdown = []
        for self.label, self.details in self.metadata_view.items():
            self.calibration = self.details["po_calibration"] or {}
            self.calibration_label = (
                str(self.calibration.get("mode", "applied"))
                if self.calibration.get("applied")
                else "Uncalibrated"
            )
            self.paths = self.details["path_count_max"]
            self.facets = self.details["visible_face_count_max"]
            self.summary.append(
                f"| {self.label} | {self.details['runtime_s']:.3g} s | "
                f"{self.paths if self.paths is not None else '—'} | "
                f"{self.facets if self.facets is not None else '—'} | {self.calibration_label} |"
            )
            self.stages = self.details["runtime_profile_s"] or {}
            self.stage_table = _runtime_stage_table(self.stages)
            if self.stage_table:
                self.breakdown.append(f"**{self.label} measured stages**\n\n{self.stage_table}")
        self.result_sections.extend(
            (
                self.pn.pane.Markdown("### Run metadata"),
                self.pn.pane.Markdown("\n".join(self.summary)),
                self.pn.pane.Markdown("\n\n".join(self.breakdown)),
                self.pn.Card(
                    self.pn.pane.JSON(
                        self.metadata_view,
                        depth=2,
                        height=380,
                        stylesheets=[_comparison_json_stylesheet(self.plot_template)],
                    ),
                    title="Advanced · Raw run metadata",
                    collapsed=True,
                    sizing_mode="stretch_width",
                ),
            )
        )
        self.view = self.pn.Column(*self.result_sections, sizing_mode="stretch_width")
        _readable_markdown(self.view, self.pn, self.plot_template)
        _style_secondary_buttons(self.view, self.pn, self.plot_template)
        return self.view

    def run(self):
        empty_result = self._stage_profiles()
        if empty_result is not None:
            return empty_result
        self._stage_doppler_controls()
        self._stage_metadata_and_bindings()
        return self._stage_result()


def _human_room_result_mode_view(result, pn, go, *, plot_template: str):
    """Backward-compatible single-mode wrapper around the product-first view."""

    return _human_room_results_view(
        {result.solver: result},
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
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _obj_bytes(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    lines = ["# HERMES transformed static target"]
    lines.extend(
        f"v {float(x):.9g} {float(y):.9g} {float(z):.9g}" for x, y, z in np.asarray(vertices)
    )
    lines.extend(f"f {int(i) + 1} {int(j) + 1} {int(k) + 1}" for i, j, k in np.asarray(faces))
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
    sensor["fmcw"] = dict(sensor["fmcw"])
    sensor["fmcw"]["carrier_reference"] = "first_adc_sample"
    if sensor["fmcw"].get("tx_schedule") == "tdm":
        sensor["fmcw"]["tdm_tx_order"] = list(
            sensor.get("tx_indices_zero_based", range(sensor["fmcw"]["num_tx"]))
        )
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
        "target_anchor_m": [float(value) for value in _target_position(result)],
    }
    summary = {
        "profile": "hermes",
        "target_type": result.target_type,
        "target_position_m": [float(value) for value in _target_position(result)],
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
    relative_permittivity = float(result.target_parameters["relative_permittivity"])
    conductivity = float(result.target_parameters["conductivity_s_per_m"])
    thickness = float(result.target_parameters["thickness_m"])
    scattering = float(result.target_parameters["rt_diffuse_reflection_coefficient"])
    descriptor["target"]["material"] = result.material_name
    descriptor["target"]["material_parameters"] = {
        "relative_permittivity": relative_permittivity,
        "conductivity": conductivity,
        "thickness": thickness,
        "scattering_coefficient": scattering,
    }
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
        "</scene>\n"
    ).encode("utf-8")
    with BundleWriter() as bundle:
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
    return BytesIO(bundle.to_bytes())


def _human_room_scene_xml(result) -> bytes:
    """Serialize the uploaded scene or the prepared room for a bundle."""

    custom_scene = getattr(result, "scene_xml", None)
    if custom_scene is None:
        custom_scene = getattr(result.preview, "scene_xml", None)
    if custom_scene is not None:
        return bytes(custom_scene)

    root = ET.fromstring(prepared_room_scene_xml_for_boxes(result.preview.room_boxes))
    human_material = ET.SubElement(
        root, "bsdf", {"type": "radio-material", "id": "human-material"}
    )
    for name, value in (
        ("relative_permittivity", "38"),
        ("conductivity", "1.5"),
        ("thickness", "0.01"),
    ):
        ET.SubElement(human_material, "float", {"name": name, "value": value})
    # Target/motion resources are injected by the simulator. Keeping a baked
    # first pose here would add a second static copy during re-simulation.
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _human_room_export(result, *, solver_results=None) -> BytesIO:
    """Build a loadable bundle containing one or more room solver results."""
    return _HumanRoomBundleExport(result, solver_results).run()


class _HumanRoomBundleExport:
    """Per-call state with explicit preparation, channel/product and output stages."""

    def __init__(self, result, solver_results):
        self.result = result
        self.solver_results = solver_results

    def _stage_descriptor(self):
        """Descriptor for this call."""
        self.primary_solver = getattr(self.result, "solver", "hybrid_po")
        self.solver_results = dict(self.solver_results or {self.primary_solver: self.result})
        self.solver_results.setdefault(self.primary_solver, self.result)
        self.components = dict(self.result.components)
        self.solver_paths = {
            solver: f"simulated_adc_{solver}.npz" for solver in self.solver_results
        }
        self.adc_artifacts = {
            "primary": {
                "path": "radar_adc.npz",
                "origin": "simulation",
                "solver": self.primary_solver,
            },
            **{
                solver: {
                    "path": path,
                    "origin": "simulation",
                    "solver": solver,
                }
                for solver, path in self.solver_paths.items()
            },
        }
        self.descriptor = {
            "schema_version": 1,
            "profile": "hermes",
            "primary_adc": "primary",
            "adcs": self.adc_artifacts,
            "primary_solver": self.primary_solver,
            "target": {"mesh": "target_mesh.npz", "obj": "target.obj"},
            "motion": {"evaluated_mesh": "human_motion.npz"},
            "scene": {
                "file": "scene.xml",
                "assets": ["target.obj", "human_motion.npz"],
            },
            "environment": "environment.json",
            "manifests": {
                "primary": "experiment_manifest.json",
                **{solver: f"experiment_manifest_{solver}.json" for solver in self.solver_results},
            },
            "diagnostics": {"solver": "solver_diagnostics.npz"},
        }
        # Hybrid PO keeps its separable terms (human PO, blocked room, coupling).
        self.hybrid_result = self.solver_results.get("hybrid_po")
        self.hybrid_components = {
            name: np.asarray(value)
            for name, value in dict(getattr(self.hybrid_result, "components", None) or {}).items()
        }
        if self.hybrid_components:
            self.descriptor["diagnostics"]["hybrid_po_components"] = "hybrid_po_components.npz"
        self.motion_payload = getattr(self.result.preview, "motion_payload", None)
        if self.motion_payload is not None:
            self.descriptor["motion"]["parameters"] = "amass_sequence.npz"
        self.radar_parameters = dict(self.result.manifest.radar.parameters)
        self.scene_parameters = dict(self.result.manifest.scene.parameters)
        self.motion_start_s = float(self.scene_parameters.get("motion_start_s", 0.0))
        self.frame_period_s = float(self.radar_parameters["frame_period_s"])
        self.time_window = {
            "start_s": self.motion_start_s,
            "end_s": float(
                self.scene_parameters.get(
                    "motion_end_s",
                    self.motion_start_s + int(self.result.adc.shape[0]) * self.frame_period_s,
                )
            ),
        }
        self.descriptor["motion"]["time_window"] = dict(self.time_window)
        self.tx_schedule = str(self.radar_parameters["tx_schedule"])
        self.tx_indices = list(self.radar_parameters["tx_indices_zero_based"])
        self.rx_indices = list(self.radar_parameters["rx_indices_zero_based"])

    def _stage_sensor_and_frames(self):
        """Sensor and frames for this call."""
        self.sensor = {
            "name": "human-room-radar",
            "board_model": self.result.manifest.radar.board_model,
            "position": [0.0, 0.0, 0.0],
            "orientation": self.result.radar_orientation_rad.tolist(),
            "orientation_deg": self.radar_parameters["orientation_deg"],
            "pattern_mode": self.radar_parameters["antenna_pattern_mode"],
            "cosine_3db_beamwidth_deg": self.radar_parameters["cosine_3db_beamwidth_deg"],
            "tx_schedule": self.tx_schedule,
            "tx_indices_zero_based": self.tx_indices,
            "rx_indices_zero_based": self.rx_indices,
            "virtual_channel_order": "tx_major",
            "fmcw": {
                "carrier_reference": "first_adc_sample",
                "carrier_frequency_hz": self.radar_parameters["carrier_frequency_hz"],
                "slope_hz_per_s": self.radar_parameters["slope_hz_per_s"],
                "chirp_duration_s": self.radar_parameters["chirp_duration_s"],
                "chirp_repetition_time_s": self.radar_parameters["chirp_repetition_time_s"],
                "sampling_frequency_hz": self.radar_parameters["sampling_frequency_hz"],
                "adc_mode": self.radar_parameters.get("adc_mode", "complex1x"),
                "if_bandwidth_hz": self.radar_parameters.get("if_bandwidth_hz"),
                "num_adc_samples": self.radar_parameters["num_adc_samples"],
                "num_chirps_per_frame": self.radar_parameters["num_chirps_per_frame"],
                "frame_period_s": self.radar_parameters["frame_period_s"],
                "num_tx": len(self.tx_indices),
                "tx_schedule": self.tx_schedule,
            },
            "metadata": {
                "generated_by": "HERMES GUI",
                "scenario": "human-in-room",
                "primary_solver": self.primary_solver,
                "manifest_sha256": self.result.manifest.fingerprint,
            },
        }
        self.source_motion_times = None
        if self.tx_schedule == "tdm":
            self.sensor["fmcw"]["tdm_tx_order"] = list(self.tx_indices)
        if self.motion_payload is not None:
            self.timing_key = "bundle_times" if "bundle_times" in self.motion_payload else "times"
            self.source_motion_times = np.asarray(
                self.motion_payload[self.timing_key],
                dtype=float,
            )
        self.frames = {
            "frames": [
                {
                    "bundle_index": frame,
                    # Radar frame start on the motion's own timeline.
                    "motion_time_s": float(
                        self.result.adc_times_s[frame, 0]
                        if self.source_motion_times is None
                        else self.source_motion_times[0]
                        + self.motion_start_s
                        + frame * self.frame_period_s
                    ),
                    "radar_frame_id": f"simulation-frame-{frame}",
                    "metadata": {"prepared_room": self.result.preview.room_preset},
                }
                for frame in range(self.result.adc.shape[0])
            ]
        }
        self.environment = {
            "scene_path": "scene.xml",
            "coordinate_system": (
                "right-handed radar-relative frame; radar phase center is "
                "(0, 0, 0), +x is boresight at zero orientation, +y is left, "
                "and +z is up"
            ),
            "scene_editable": False,
            "prepared_scene": (
                self.result.preview.room_preset
                if getattr(self.result.preview, "scene_xml", None) is None
                else None
            ),
            "scene_source": getattr(
                self.result.preview,
                "scene_source_name",
                self.result.preview.room_preset,
            ),
            "target_geometry": "target_mesh.npz",
            "motion_mesh": "human_motion.npz",
            "human_position_m": self.result.preview.human_position_m.tolist(),
            "human_yaw_deg": self.result.preview.human_yaw_deg,
            "motion_coordinate_settings": getattr(
                self.result.preview, "motion_coordinate_settings", None
            ),
            "motion_time_window": dict(self.time_window),
        }
        if self.motion_payload is not None:
            self.environment["motion_parameters"] = "amass_sequence.npz"

    def _stage_diagnostics(self):
        """Diagnostics for this call."""
        self.summary = {
            "profile": "hermes",
            "scenario": "human-in-room",
            "scene_source": getattr(
                self.result.preview,
                "scene_source_name",
                self.result.preview.room_preset,
            ),
            "motion_source": self.result.preview.source_name,
            "motion_time_window": dict(self.time_window),
            "primary_solver": self.primary_solver,
            "solvers": list(self.solver_results),
            "solver_runtime_s": {
                solver: float(solver_result.runtime_s)
                for solver, solver_result in self.solver_results.items()
            },
            "adc_shape": list(map(int, self.result.adc.shape)),
            "components": sorted(self.components),
            "runtime_s": float(self.result.runtime_s),
            "manifest_sha256": self.result.manifest.fingerprint,
            "public_release_note": (
                "The bundle contains evaluated motion geometry and, when "
                "available, source AMASS-like parameters. Redistribution rights "
                "remain the responsibility of the bundle creator."
            ),
        }
        self.diagnostic_arrays = {
            "adc_if_filter_json": np.asarray(
                json.dumps(getattr(self.result.metadata, "adc_if_filter", None))
            ),
            "ranges_m": np.asarray(self.result.ranges_m),
            **{
                f"range_profile_{name}": np.asarray(values)
                for name, values in self.result.range_profiles.items()
            },
        }
        for self.name in (
            "range_time_power",
            "range_time_ranges_m",
            "range_doppler_power",
            "range_doppler_ranges_m",
            "velocities_mps",
        ):
            self.value = getattr(self.result, self.name, None)
            if self.value is not None:
                self.diagnostic_arrays[self.name] = np.asarray(self.value)
        for self.name in (
            "path_counts",
            "visible_face_counts",
            "effective_visible_area_fraction",
            "static_path_counts",
            "blocked_static_path_counts",
            "human_env_path_counts",
            "env_human_path_counts",
            "path_depth_histogram",
        ):
            self.value = getattr(self.result.metadata, self.name, None)
            if self.value is not None:
                self.diagnostic_arrays[self.name] = np.asarray(self.value)

        self.sequence = self.result.preview.mesh_sequence

    def _stage_geometry(self):
        """Geometry for this call."""
        self.physical_chirps = int(self.radar_parameters["num_chirps_per_frame"]) * (
            len(self.tx_indices) if self.tx_schedule == "tdm" else 1
        )
        self.final_time = (self.result.adc.shape[0] - 1) * self.frame_period_s + (
            self.physical_chirps - 1
        ) * self.radar_parameters["chirp_repetition_time_s"]
        self.source_sequence = getattr(self.sequence, "_base", self.sequence)
        self.knots = np.asarray(self.source_sequence.times, dtype=float) - float(
            getattr(self.sequence, "motion_start_s", 0.0)
        )
        self.geometry_times = np.unique(
            np.concatenate(
                [
                    np.asarray(self.sequence.times, dtype=float),
                    self.knots[(self.knots >= 0) & (self.knots <= self.final_time)],
                    [0.0, self.final_time],
                ]
            )
        )
        self.geometry_vertices = np.stack(
            [self.sequence.vertices_at(float(t)) for t in self.geometry_times]
        )
        self.geometry_origin = (
            0.0
            if self.source_motion_times is None
            else self.source_motion_times[0] + self.motion_start_s
        )
        self.first_vertices = np.asarray(self.sequence.vertices_at(float(self.sequence.times[0])))

    def _stage_result(self):
        """Result for this call."""
        with BundleWriter() as self.bundle:
            self.bundle.writestr("bundle.json", _json_bytes(self.descriptor))
            self.bundle.writestr("sensor.json", _json_bytes(self.sensor))
            self.bundle.writestr("frames.json", _json_bytes(self.frames))
            self.bundle.writestr(
                "radar_adc.npz",
                _npz_bytes(adc=self.result.adc, times=self.result.adc_times_s),
            )
            for self.solver, self.solver_result in self.solver_results.items():
                self.bundle.writestr(
                    self.solver_paths[self.solver],
                    _npz_bytes(
                        adc=self.solver_result.adc,
                        times=self.solver_result.adc_times_s,
                    ),
                )
                self.bundle.writestr(
                    f"experiment_manifest_{self.solver}.json",
                    _json_bytes(self.solver_result.manifest.to_dict()),
                )
            self.bundle.writestr(
                "target_mesh.npz",
                _npz_bytes(
                    vertices=self.first_vertices,
                    faces=self.sequence.faces,
                    times=np.asarray([0.0]),
                ),
            )
            self.bundle.writestr(
                "human_motion.npz",
                _npz_bytes(
                    vertices=self.geometry_vertices,
                    faces=self.sequence.faces,
                    times=self.geometry_times + self.geometry_origin,
                ),
            )
            if self.motion_payload is not None:
                self.bundle.writestr(
                    "amass_sequence.npz",
                    _npz_bytes(**dict(self.motion_payload)),
                )
            self.bundle.writestr(
                "target.obj",
                _obj_bytes(self.first_vertices, self.sequence.faces),
            )
            self.bundle.writestr("scene.xml", _human_room_scene_xml(self.result))
            self.bundle.writestr("environment.json", _json_bytes(self.environment))
            self.bundle.writestr(
                "experiment_manifest.json",
                _json_bytes(self.result.manifest.to_dict()),
            )
            self.bundle.writestr(
                "solver_diagnostics.npz",
                _npz_bytes(**self.diagnostic_arrays),
            )
            if self.hybrid_components:
                self.bundle.writestr(
                    "hybrid_po_components.npz",
                    _npz_bytes(times=self.hybrid_result.adc_times_s, **self.hybrid_components),
                )
            self.bundle.writestr("bundle_summary.json", _json_bytes(self.summary))
            self.bundle.writestr(
                "README.txt",
                "HERMES prepared-room human simulation bundle\n"
                f"radar_adc.npz contains the {self.primary_solver} raw ADC cube. "
                "Each selected solver is also stored in a "
                "simulated_adc_<solver>.npz archive, and Hybrid PO's separable "
                "terms in hybrid_po_components.npz.\n"
                "human_motion.npz contains the evaluated mesh sequence. When "
                "present, amass_sequence.npz contains the source AMASS-like pose "
                "parameters; licensed SMPL model files are not embedded.\n"
                f"Manifest SHA-256: {self.result.manifest.fingerprint}\n",
            )
        return BytesIO(self.bundle.to_bytes())

    def run(self):
        self._stage_descriptor()
        self._stage_sensor_and_frames()
        self._stage_diagnostics()
        self._stage_geometry()
        return self._stage_result()


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
    candidate = public_root / "data" / "AMASS" / "walking_poses_cmu_105_02.npz"
    return candidate if candidate.is_file() else None


def build_app(*, initial_bundle: str | None = None, remote_access: bool = False):
    """Build the HERMES app, restricting server paths for remote sessions."""
    from .assembly import AppBuilder

    return AppBuilder(initial_bundle=initial_bundle, remote_access=remote_access).build()


__all__ = ["build_app"]
