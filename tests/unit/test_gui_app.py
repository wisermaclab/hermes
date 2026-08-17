# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Optional-dependency smoke tests for the local HERMES GUI."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import inspect
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import zipfile

import numpy as np
import param
import pytest
from bokeh.document import Document
from panel.models.reactive_html import DOMEvent


pn = pytest.importorskip("panel")
go = pytest.importorskip("plotly.graph_objects")
make_subplots = pytest.importorskip("plotly.subplots").make_subplots

from mmWaveRadar.gui import build_app  # noqa: E402
import mmWaveRadar.gui.app as gui_app_module  # noqa: E402
import mmWaveRadar.gui.cli as gui_cli_module  # noqa: E402
from mmWaveRadar.gui.app import (  # noqa: E402
    _absolute_db_limits,
    _angle_fft_range_products,
    _angle_map_figure,
    _default_human_mesh_path,
    _dynamic_scene_interaction_component,
    _human_room_export,
    _human_room_profile_figure,
    _human_room_results_view,
    _human_room_scene_figure,
    _human_room_scene_xml,
    _human_room_window_values,
    _lock_widget_disabled_states,
    _bundle_primary_label,
    _bundle_scene_boxes,
    _bundle_scene_figure,
    _cache_theme_comparison,
    _cache_theme_gui_results,
    _get_theme_comparison,
    _get_theme_gui_results,
    _motion_fast_playback_step,
    _motion_playback_interval_ms,
    _motion_playback_step,
    _physics_scene_figure,
    _physics_export,
    _plot_template,
    _range_profile_figure,
    _range_doppler_figure,
    _range_time_figure,
    _rounded_gui_float,
    _rounded_target_size,
    _safe_markdown_code,
    _safe_markdown_text,
    _selected_angle_fft_power,
    _solver_diagnostics_summary,
    _style_figure,
    _target_trace_update,
    _update_human_room_scene_frame,
    _validated_solver_settings_payload,
    _restore_widget_disabled_states,
)
from mmWaveRadar.experiments import (  # noqa: E402
    ExperimentManifest,
    RadarExperimentConfig,
    SceneExperimentConfig,
    SolverExperimentConfig,
    make_trihedral_corner_mesh,
    prepared_room_boxes,
    prepare_human_room_preview,
    run_static_target_experiment,
)
from mmWaveRadar.targets import MeshSequence  # noqa: E402
from mmWaveRadar.measurements import load_bundle  # noqa: E402


def test_gui_defaults_to_public_smpl_model_directory(monkeypatch):
    monkeypatch.delenv("MMWAVE_SMPL_MODEL_DIR", raising=False)
    public_root = Path(gui_app_module.__file__).resolve().parents[3]
    expected = public_root / "models" / "smpl_models"

    assert gui_app_module._default_smpl_model_dir() == expected
    assert gui_cli_module._default_smpl_model_dir() == expected


@pytest.mark.parametrize(
    "address",
    ["localhost", "127.0.0.1", "127.12.34.56", "::1", "[::1]"],
)
def test_gui_launcher_recognizes_explicit_loopback_addresses(address):
    assert gui_cli_module._is_loopback_address(address) is True


@pytest.mark.parametrize(
    "address",
    ["0.0.0.0", "::", "192.168.1.10", "example.test", "localhost.test"],
)
def test_gui_launcher_treats_other_bind_addresses_as_remote(address):
    assert gui_cli_module._is_loopback_address(address) is False


def test_remote_smpl_directory_ignores_browser_value():
    assert gui_app_module._effective_smpl_model_dir(
        "/browser/chosen/pickles",
        "/operator/configured/models",
        remote_access=True,
    ) == "/operator/configured/models"
    assert gui_app_module._effective_smpl_model_dir(
        "/local/chosen/models",
        "/operator/configured/models",
        remote_access=False,
    ) == "/local/chosen/models"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_type", "../../external_mano", "model_type must be one of"),
        ("gender", "../../external", "gender must be one of"),
    ],
)
def test_amass_loader_rejects_untrusted_model_selectors(
    tmp_path,
    field,
    value,
    message,
):
    values = {
        "poses": np.zeros((1, 72), dtype=np.float32),
        "trans": np.zeros((1, 3), dtype=np.float32),
        "betas": np.zeros(10, dtype=np.float32),
        "times": np.zeros(1, dtype=float),
        "model_type": np.asarray("smpl"),
        "gender": np.asarray("neutral"),
    }
    values[field] = np.asarray(value)
    archive = BytesIO()
    np.savez(archive, **values)

    with pytest.raises(ValueError, match=message):
        gui_app_module.load_amass_motion(
            archive.getvalue(),
            filename="motion.npz",
            smpl_model_dir=tmp_path,
        )


def test_gui_motion_preflight_caps_optional_strings_and_faces():
    base = {
        "poses": np.zeros((1, 72), dtype=np.float32),
        "trans": np.zeros((1, 3), dtype=np.float32),
        "betas": np.zeros(10, dtype=np.float32),
        "times": np.zeros(1, dtype=np.float32),
    }
    payload = BytesIO()
    np.savez(payload, **base, model_type=np.asarray(["smpl"]))
    with pytest.raises(ValueError, match="ndim"):
        gui_app_module._preflight_gui_motion_npz(payload.getvalue())

    payload = BytesIO()
    np.savez(payload, **base, gender=np.asarray("x" * 100))
    with pytest.raises(ValueError, match="byte limit"):
        gui_app_module._preflight_gui_motion_npz(payload.getvalue())

    payload = BytesIO()
    np.savez(payload, **base, faces=np.zeros((2, 4), dtype=np.int32))
    with pytest.raises(ValueError, match="end in shape"):
        gui_app_module._preflight_gui_motion_npz(payload.getvalue())


def test_gui_obj_preflight_caps_vertices_and_fan_expansion(monkeypatch):
    obj = b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
    monkeypatch.setattr(gui_app_module, "_MAX_GUI_OBJ_VERTICES", 3)
    with pytest.raises(ValueError, match="vertex limit"):
        gui_app_module._preflight_gui_mesh_obj(obj)

    monkeypatch.setattr(gui_app_module, "_MAX_GUI_OBJ_VERTICES", 4)
    monkeypatch.setattr(gui_app_module, "_MAX_GUI_OBJ_TRIANGLES", 1)
    with pytest.raises(ValueError, match="triangle limit"):
        gui_app_module._preflight_gui_mesh_obj(obj)


def _scene_pane(tabs):
    interaction = next(
        component
        for component in tabs.select(pn.reactive.ReactiveHTML)
        if "target_gesture" in component.param
    )
    return interaction.children[0]


def _range_profile_pane(tabs):
    return next(
        pane
        for pane in tabs.select(pn.pane.Plotly)
        if pane.object is not None
        and [trace.name for trace in pane.object.data]
        == ["Physical optics (PO)", "Ray tracing (RT)"]
    )


def _angle_map_pane(tabs):
    return next(
        pane
        for pane in tabs.select(pn.pane.Plotly)
        if pane.object is not None
        and str(pane.object.layout.title.text).startswith("2D angle FFT")
    )


def test_absolute_plot_limits_ignore_a_vanishing_solver_floor():
    po_power = np.asarray([1.0e-3, 1.0e-6])
    rt_zero = np.zeros(2)
    rt_with_diffuse_scattering = np.asarray([1.5e-3, 1.0e-8])

    limits_without_rt = _absolute_db_limits(
        (po_power, rt_zero),
        dynamic_range_db=100.0,
        headroom_db=3.0,
    )
    limits_with_rt = _absolute_db_limits(
        (po_power, rt_with_diffuse_scattering),
        dynamic_range_db=100.0,
        headroom_db=3.0,
    )

    assert limits_without_rt == (-125.0, -25.0)
    assert limits_with_rt == limits_without_rt


def test_gui_builds_with_three_public_workflows_and_settings(monkeypatch):
    application = build_app()

    assert {"plotly", "filedropper"}.issubset(
        pn.extension._loaded_extensions  # pylint: disable=protected-access
    )
    assert application.title == "HERMES · Interactive mmWave Radar Simulation"
    assert application.favicon == "/apple-touch-icon.png"
    assert len(application.main) == 1
    tabs = application.main[0]
    assert tabs._names == [  # pylint: disable=protected-access
        "Static Target",
        "Dynamic Scenes",
        "Bundle Comparison",
    ]
    assert tabs.dynamic is False
    assert application.theme_toggle is False
    assert application.theme is pn.template.DarkTheme
    assert application.busy_indicator is None
    assert len(application.header) == 1
    header_buttons = application.header[0].select(pn.widgets.Button)
    assert [button.icon for button in header_buttons] == ["settings", "sun"]
    assert all(button.label == "" for button in header_buttons)
    assert all(button.icon_size == "30px" for button in header_buttons)
    settings_storage = next(
        component
        for component in application.header[0].select(
            pn.reactive.ReactiveHTML
        )
        if "loaded_settings" in component.param
    )
    storage_render_script = settings_storage._scripts[  # pylint: disable=protected-access
        "render"
    ]
    storage_save_script = settings_storage._scripts[  # pylint: disable=protected-access
        "saved_settings"
    ]
    assert "window.localStorage.getItem" in storage_render_script
    assert "window.localStorage.setItem" in storage_save_script
    gui_storage = next(
        component
        for component in application.header[0].select(
            pn.reactive.ReactiveHTML
        )
        if "loaded_state" in component.param
        and "loaded_uploads" in component.param
    )
    assert "window.sessionStorage.getItem" in gui_storage._scripts[  # pylint: disable=protected-access
        "render"
    ]
    assert "window.indexedDB.open" in gui_storage._scripts[  # pylint: disable=protected-access
        "render"
    ]
    assert "window.sessionStorage.setItem" in gui_storage._scripts[  # pylint: disable=protected-access
        "saved_state"
    ]
    assert "store.put" in gui_storage._scripts[  # pylint: disable=protected-access
        "saved_uploads"
    ]
    dynamic_view_tabs = next(
        nested
        for nested in tabs.select(pn.Tabs)
        if nested._names == [  # pylint: disable=protected-access
            "Scene and motion",
            "Simulation results",
        ]
    )
    assert dynamic_view_tabs.dynamic is False
    measurement_view_tabs = next(
        nested
        for nested in tabs.select(pn.Tabs)
        if nested._names == [  # pylint: disable=protected-access
            "Scene",
            "ADC",
            "Range profile",
            "Range time",
            "Range Doppler",
            "Report",
        ]
    )
    assert measurement_view_tabs._names == [  # pylint: disable=protected-access
        "Scene",
        "ADC",
        "Range profile",
        "Range time",
        "Range Doppler",
        "Report",
    ]
    assert measurement_view_tabs.dynamic is False
    assert not any(
        card.title == "Bundle report" for card in tabs.select(pn.Card)
    )
    gui_storage.loaded_state = {
        "schema_version": 1,
        "top_tab": 1,
        "dynamic_tab": 1,
        "measurement_tab": 5,
        "has_motion": False,
        "load_sequence": 1,
    }
    assert tabs.active == 1
    assert dynamic_view_tabs.active == 1
    assert measurement_view_tabs.active == 5
    assert len(application.modal) == 1
    settings = application.modal[0]
    assert settings.width is None
    assert settings.min_width == 680
    assert settings.max_width == 980
    assert settings.sizing_mode == "stretch_width"
    settings_tabs = next(
        component
        for component in settings.select(pn.Tabs)
        if component._names == [  # pylint: disable=protected-access
            "Radar configuration",
            "Solver settings",
        ]
    )
    assert settings_tabs._names == [  # pylint: disable=protected-access
        "Radar configuration",
        "Solver settings",
    ]
    assert next(
        widget
        for widget in settings.select(pn.widgets.Select)
        if widget.label == "PO facet integration"
    ).value == "parent_face_quadrature"
    assert next(
        widget
        for widget in settings.select(pn.widgets.IntInput)
        if widget.label == "RT ray samples per source"
    ).value == 100_000
    assert next(
        widget
        for widget in settings.select(pn.widgets.IntInput)
        if widget.label == "RT maximum paths per source"
    ).value == 5_000
    rt_help = {
        widget.label: widget.description
        for widget in settings.select(pn.widgets.IntInput)
        if widget.label
        in {
            "RT ray samples per source",
            "RT maximum paths per source",
            "RT maximum path depth",
        }
    }
    assert set(rt_help) == {
        "RT ray samples per source",
        "RT maximum paths per source",
        "RT maximum path depth",
    }
    assert "rays launched" in rt_help["RT ray samples per source"]
    assert "paths retained" in rt_help["RT maximum paths per source"]
    assert "scene interactions" in rt_help["RT maximum path depth"]
    assert any(
        button.label == "Apply and recompute"
        for button in settings.select(pn.widgets.Button)
    )
    assert any(
        "saved in this browser" in str(pane.object)
        for pane in settings.select(pn.pane.Markdown)
    )

    scene_pane = _scene_pane(tabs)
    range_profile_pane = _range_profile_pane(tabs)
    assert not tabs.select(pn.widgets.RadioButtonGroup)
    refine = next(
        button
        for button in tabs.select(pn.widgets.Button)
        if button.label == "Recompute at higher fidelity"
    )
    assert any(
        selector.label == "Static object"
        for selector in tabs.select(pn.widgets.Select)
    )
    assert not any(
        card.title == "Reproducibility" for card in tabs.select(pn.Card)
    )
    range_figure = range_profile_pane.object
    assert [trace.name for trace in range_figure.data] == [
        "Physical optics (PO)",
        "Ray tracing (RT)",
    ]
    assert range_figure.data[1].line.dash == "solid"
    assert "absolute simulated power" in range_figure.layout.title.text
    assert (
        range_figure.layout.yaxis.title.text
        == "Power [dB, simulator units]"
    )
    assert tuple(range_figure.layout.yaxis.range) != (-80, 2)
    scene_figure = scene_pane.object
    radar_trace = next(
        trace for trace in scene_figure.data if trace.name == "Radar"
    )
    assert tuple(radar_trace.x) == (0.0,)
    assert tuple(radar_trace.y) == (0.0,)
    assert tuple(radar_trace.z) == (0.0,)
    boresight_trace = next(
        trace for trace in scene_figure.data if trace.name == "Radar boresight"
    )
    local_up_trace = next(
        trace for trace in scene_figure.data if trace.name == "Radar local up"
    )
    assert boresight_trace.x[-1] > 0.0
    assert boresight_trace.y[-1] == pytest.approx(0.0)
    assert local_up_trace.z[-1] > 0.0
    assert scene_figure.layout.scene.dragmode == "orbit"
    assert scene_figure.layout.scene.aspectmode == "manual"
    aspect_ratio = scene_figure.layout.scene.aspectratio
    assert (
        aspect_ratio.x * aspect_ratio.y * aspect_ratio.z
    ) == pytest.approx(1.0)
    assert scene_figure.layout.scene.camera.up.z == pytest.approx(1.0)
    assert scene_figure.layout.scene.xaxis.range[0] < 0.0
    assert scene_figure.layout.scene.xaxis.title.text == "x · range forward [m]"
    assert (
        scene_figure.layout.scene.yaxis.title.text
        == "y · lateral, radar-left [m]"
    )
    assert scene_figure.layout.scene.zaxis.title.text == "z · vertical, up [m]"
    assert scene_figure.layout.scene.camera.eye.y < 0.0
    assert scene_pane.config["scrollZoom"] is True
    assert scene_pane.config["modeBarButtonsToRemove"] == [
        "resetCameraDefault3d"
    ]
    assert any(
        "shift + drag" in str(pane.object).lower()
        for pane in tabs.select(pn.pane.Markdown)
    )
    assert scene_figure.layout.uirevision == "hermes-scene-camera"
    assert not any(
        nested._names == ["Physics Microscope", "Human-in-Room"]
        for nested in tabs.select(pn.Tabs)
    )
    physics_controls_card = next(
        card
        for card in tabs.select(pn.Card)
        if card.title == "Static target controls"
    )
    dynamic_controls_card = next(
        card
        for card in tabs.select(pn.Card)
        if card.title == "Dynamic scene controls"
    )
    simulation_controls = physics_controls_card[0]
    material_selector = next(
        widget
        for widget in tabs.select(pn.widgets.Select)
        if widget.label == "Material preset"
    )
    static_radar_yaw = next(
        widget
        for widget in physics_controls_card.select(pn.widgets.FloatSlider)
        if widget.label == "Radar yaw [deg]"
    )
    dynamic_radar_yaw = next(
        widget
        for widget in dynamic_controls_card.select(pn.widgets.FloatSlider)
        if widget.label == "Radar yaw [deg]"
    )
    assert material_selector in simulation_controls
    assert static_radar_yaw in simulation_controls
    assert dynamic_radar_yaw not in simulation_controls
    assert not settings.select(pn.widgets.FloatSlider)
    assert material_selector.options[
        "PEC · ideal PO, rough/diffuse RT"
    ] == "pec"
    assert any(
        "RT diffuse scattering coefficient:** `0.20`" in str(pane.object)
        for pane in simulation_controls.select(pn.pane.Markdown)
    )
    headings = [
        str(pane.object)
        for pane in simulation_controls.objects
        if isinstance(pane, pn.pane.Markdown)
    ]
    assert "## Target" in headings
    assert "## Radar orientation" in headings
    assert refine not in physics_controls_card.select(pn.widgets.Button)
    refine_toolbar = next(
        row
        for row in tabs.select(pn.Row)
        if refine in row.objects
    )
    assert refine_toolbar.objects[-1] is refine
    assert isinstance(refine_toolbar.objects[0], pn.Spacer)
    assert refine.width == 250
    assert refine.height == 40
    assert refine.sizing_mode == "fixed"
    controls_card = physics_controls_card
    assert controls_card.sizing_mode == "stretch_width"
    assert controls_card.width is None
    assert controls_card.max_width == 440
    assert next(
        widget
        for widget in settings.select(pn.widgets.Checkbox)
        if widget.label == "Enable TDM-MIMO"
    ).value is False
    tx_selector = next(
        widget
        for widget in settings.select(pn.widgets.MultiChoice)
        if widget.label == "Active Tx antennas"
    )
    rx_selector = next(
        widget
        for widget in settings.select(pn.widgets.MultiChoice)
        if widget.label == "Active Rx antennas"
    )
    assert tx_selector.value == [0, 1, 2]
    assert rx_selector.value == [0, 1, 2, 3]
    for label, value in (
        ("Target x / range-forward [m]", 2.0),
        ("Target y / lateral-left [m]", 0.0),
        ("Target z / vertical-up [m]", 0.0),
    ):
        assert next(
            widget
            for widget in tabs.select(pn.widgets.FloatSlider)
            if widget.label == label
        ).value == pytest.approx(value)
    for label in ("Radar yaw [deg]", "Radar pitch [deg]", "Radar roll [deg]"):
        orientation_controls = [
            widget
            for widget in tabs.select(pn.widgets.FloatSlider)
            if widget.label == label
        ]
        assert len(orientation_controls) == 2
        assert all(
            widget.value == pytest.approx(0.0)
            for widget in orientation_controls
        )
    antenna_pattern = next(
        widget
        for widget in settings.select(pn.widgets.Select)
        if widget.label == "Antenna pattern"
    )
    assert antenna_pattern.value == "cosine"
    assert antenna_pattern.options == {
        "Digitized TI profile": "digitized",
        "Omnidirectional": "none",
        "Cosine": "cosine",
    }
    cosine_beamwidth = next(
        widget
        for widget in settings.select(pn.widgets.FloatInput)
        if widget.label == "Cosine 3 dB beamwidth [deg]"
    )
    assert cosine_beamwidth.value == pytest.approx(60.0)
    assert cosine_beamwidth.visible is True
    assert next(
        widget
        for widget in settings.select(pn.widgets.FloatInput)
        if widget.label == "Chirp slope [MHz/µs]"
    ).value == pytest.approx(68.0)
    carrier = next(
        widget
        for widget in settings.select(pn.widgets.FloatInput)
        if widget.label == "Carrier frequency [GHz]"
    )
    assert carrier.value == 60.0
    assert carrier.format == "0.000"
    assert next(
        widget
        for widget in settings.select(pn.widgets.IntInput)
        if widget.label == "ADC samples per chirp"
    ).value == 225
    angle_bins = next(
        widget
        for widget in tabs.select(pn.widgets.IntRangeSlider)
        if widget.label == "Angle-FFT range bins (inclusive)"
    )
    assert angle_bins.value[0] == angle_bins.value[1]
    assert angle_bins.value[0] > 0
    angle_figure = _angle_map_pane(tabs).object
    assert len(angle_figure.data) == 2
    assert angle_figure.layout.xaxis.title.text == (
        "Horizontal direction cosine u"
    )
    assert angle_figure.layout.yaxis.title.text == (
        "Vertical direction cosine v"
    )
    assert np.asarray(angle_figure.data[0].z).shape == (64, 128)
    assert any(
        button.label == "Start"
        for button in tabs.select(pn.widgets.Button)
    )
    stop_button = next(
        button
        for button in tabs.select(pn.widgets.Button)
        if button.label == "Stop"
    )
    assert stop_button.disabled is True
    rt_overlay = next(
        toggle
        for toggle in tabs.select(pn.widgets.Toggle)
        if toggle.label == "Show" and toggle.icon == "route"
    )
    static_top_rt_paths = next(
        widget
        for widget in tabs.select(pn.widgets.IntInput)
        if widget.label == "Top RT paths"
    )
    static_rt_row = next(
        row
        for row in tabs.select(pn.Row)
        if rt_overlay in row.objects
    )
    assert static_rt_row.objects == [static_top_rt_paths, rt_overlay]
    assert static_top_rt_paths.width == 135
    assert rt_overlay.width == 100
    assert rt_overlay.align == "end"
    assert rt_overlay.value is False
    assert not any(
        "Solver diagnostics" in nested._names  # pylint: disable=protected-access
        for nested in tabs.select(pn.Tabs)
    )
    assert not any(
        "Radar x" in str(getattr(widget, "label", ""))
        or "Radar position" in str(getattr(widget, "label", ""))
        for widget in tabs.select(pn.widgets.Widget)
    )
    human_scene_interaction = next(
        component
        for component in tabs.select(pn.reactive.ReactiveHTML)
        if component.name == "Dynamic scene camera interaction"
    )
    human_scene = human_scene_interaction.children[0]
    human_radar = next(
        trace for trace in human_scene.object.data if trace.name == "Radar"
    )
    assert tuple(human_radar.x) == (0.0,)
    assert tuple(human_radar.y) == (0.0,)
    assert tuple(human_radar.z) == (0.0,)
    motion_frame = next(
        widget
        for widget in tabs.select(pn.widgets.IntInput)
        if widget.label.startswith("Displayed frame (max ")
    )
    top_rt_rays = next(
        widget
        for widget in tabs.select(pn.widgets.IntInput)
        if widget.label == "Top RT rays"
    )
    overlay_button = next(
        widget
        for widget in tabs.select(pn.widgets.Button)
        if widget.label == "Show" and widget.icon == "route"
    )
    compact_frame_row = next(
        row
        for row in tabs.select(pn.Row)
        if motion_frame in row.objects
    )
    assert compact_frame_row.objects == [
        motion_frame,
        top_rt_rays,
        overlay_button,
    ]
    motion_player = next(
        widget
        for widget in tabs.select(pn.widgets.Player)
        if widget.label == "Motion playback"
    )
    scene_motion_column = next(
        column
        for column in tabs.select(pn.Column)
        if human_scene_interaction in column.objects
        and motion_player in column.objects
    )
    assert scene_motion_column.objects.index(motion_player) > (
        scene_motion_column.objects.index(human_scene_interaction)
    )
    assert motion_player.start == 0
    assert motion_player.end == 0
    assert motion_player.disabled is True
    assert motion_player.sizing_mode == "stretch_width"
    assert motion_player.loop_policy == "loop"
    assert motion_player.visible_buttons == []
    transport_labels = [
        "Prev",
        "Reverse",
        "Fast reverse",
        "Play",
        "Fast forward",
        "Next",
    ]
    transport_buttons = [
        next(
            widget
            for widget in tabs.select(pn.widgets.Button)
            if widget.label == label
        )
        for label in transport_labels
    ]
    transport_row = next(
        row
        for row in tabs.select(pn.Row)
        if all(button in row.objects for button in transport_buttons)
    )
    assert [
        item.label
        for item in transport_row.objects
        if isinstance(item, pn.widgets.Button)
    ] == transport_labels
    fast_reverse = next(
        widget
        for widget in tabs.select(pn.widgets.Button)
        if widget.label == "Fast reverse"
    )
    fast_forward = next(
        widget
        for widget in tabs.select(pn.widgets.Button)
        if widget.label == "Fast forward"
    )
    assert fast_reverse.disabled is True
    assert fast_forward.disabled is True
    for button in transport_buttons:
        assert button.disabled is True
        assert button.description is None
        button.disabled = False
    motion_player.end = 3
    motion_player.disabled = False
    motion_frame.end = 3
    motion_frame.disabled = False
    motion_player.value = 2
    assert motion_frame.value == 2
    fast_forward.clicks += 1
    assert motion_player.value == 2
    assert motion_player.direction == 1
    assert motion_player.step > 1
    assert transport_buttons[3].label == "Pause"
    motion_player.direction = 0
    assert motion_player.step == 1
    assert transport_buttons[3].label == "Play"
    fast_reverse.clicks += 1
    assert motion_player.value == 2
    assert motion_player.direction == -1
    assert motion_player.step > 1
    transport_buttons[3].clicks += 1
    assert motion_player.direction == 0
    assert motion_player.step == 1
    assert transport_buttons[3].label == "Play"
    transport_buttons[0].clicks += 1
    assert motion_player.value == 1
    transport_buttons[1].clicks += 1
    assert motion_player.value == 1
    assert motion_player.direction == -1
    assert motion_player.step == 1
    assert transport_buttons[3].label == "Pause"
    transport_buttons[3].clicks += 1
    assert motion_player.direction == 0
    assert transport_buttons[3].label == "Play"
    transport_buttons[3].clicks += 1
    assert motion_player.value == 1
    assert motion_player.direction == 1
    assert transport_buttons[3].label == "Pause"
    transport_buttons[3].clicks += 1
    transport_buttons[5].clicks += 1
    assert motion_player.value == 2
    assert motion_player.direction == 0
    motion_player.direction = 1
    motion_frame.value = 1
    frame_callback = next(
        watcher.fn
        for watcher in motion_frame.param.watchers["value"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_human_room_frame"
    )
    frame_callback(SimpleNamespace(new=1))
    assert motion_player.value == 1
    assert motion_player.direction == 0
    begin_frame = next(
        widget
        for widget in tabs.select(pn.widgets.IntInput)
        if widget.label == "Begin frame (inclusive)"
    )
    end_frame = next(
        widget
        for widget in tabs.select(pn.widgets.IntInput)
        if widget.label.startswith("End frame (max ")
    )
    assert end_frame.label == "End frame (max 0)"
    assert begin_frame.value == end_frame.value == 0
    assert begin_frame.end == end_frame.end == 0
    assert begin_frame.disabled is True
    assert end_frame.disabled is True
    assert not any(
        widget.label == "Radar frames to simulate"
        for widget in tabs.select(pn.widgets.IntInput)
    )
    begin_frame.end = 3
    end_frame.end = 3
    begin_frame.disabled = False
    end_frame.disabled = False
    begin_frame.value = 2
    assert end_frame.value == 2
    end_frame.value = 1
    assert begin_frame.value == 1
    mode_checkboxes = {
        widget.label: widget
        for widget in tabs.select(pn.widgets.Checkbox)
        if widget.label in {
            "Full RT",
            "Coherent RT",
            "Human-only PO",
            "Hybrid PO",
        }
    }
    assert set(mode_checkboxes) == {
        "Full RT",
        "Coherent RT",
        "Human-only PO",
        "Hybrid PO",
    }
    assert all(
        isinstance(widget, pn.widgets.Checkbox)
        for widget in mode_checkboxes.values()
    )
    assert mode_checkboxes["Hybrid PO"].value is True
    assert not any(
        widget.value
        for label, widget in mode_checkboxes.items()
        if label != "Hybrid PO"
    )
    mode_help = {
        icon.value
        for icon in tabs.select(pn.widgets.TooltipIcon)
    }
    assert {
        "Full RT retraces every chirp.",
        "Coherent RT updates a path bank between frame retraces.",
        "Human-only PO excludes environment channels.",
        (
            "Hybrid PO includes blocking, coupling, and coherent-RT "
            "sequence calibration."
        ),
    }.issubset(mode_help)
    assert not any(
        widget.label == "Enable human ↔ environment coupling"
        for widget in tabs.select(pn.widgets.Checkbox)
    )
    assert any(
        "Simulation results" in getattr(child, "_names", [])
        for child in tabs.select(pn.Tabs)
    )
    motion_upload = next(
        widget
        for widget in tabs.select(pn.widgets.FileInput)
        if widget.label == "AMASS-like human motion (.npz)"
    )
    assert motion_upload.accept == ".npz"
    assert motion_upload.description == (
        "Upload a pickle-free AMASS-like pose archive with poses, trans, "
        "betas, and timing. The configured SMPL model evaluates its mesh. "
        "Browser uploads are limited to 64 MiB."
    )
    assert any(
        "input[type=file].bk-input" in str(stylesheet)
        and "background: transparent" in str(stylesheet)
        and "color: transparent" in str(stylesheet)
        and "::-webkit-file-upload-button" in str(stylesheet)
        and "background: #182536" in str(stylesheet)
        and "width: 132px" in str(stylesheet)
        for stylesheet in motion_upload.stylesheets
    )
    motion_upload_row = next(
        row
        for row in tabs.select(pn.Row)
        if motion_upload in row.objects
    )
    motion_filename = next(
        widget
        for widget in motion_upload_row.objects
        if isinstance(widget, pn.widgets.StaticText)
    )
    assert motion_upload.width == 132
    assert motion_upload.height == 40
    assert "Bundled default:" in motion_filename.value
    motion_upload.filename = "selected_walk.npz"
    assert motion_filename.value == "selected_walk.npz"
    with param.parameterized.discard_events(motion_upload):
        motion_upload.value = b"session motion bytes"
    persist_uploads = next(
        watcher.fn
        for watcher in motion_upload.param.watchers["value"]["value"]
        if getattr(watcher.fn, "__name__", "") == "persist_gui_uploads"
    )
    persist_uploads(
        SimpleNamespace(name="value", new=motion_upload.value),
        SimpleNamespace(name="filename", new=motion_upload.filename),
    )
    assert gui_storage.saved_uploads["motion"]["filename"] == (
        "selected_walk.npz"
    )
    assert base64.b64decode(
        gui_storage.saved_uploads["motion"]["payload"]
    ) == (
        b"session motion bytes"
    )
    assert gui_storage.saved_state["has_motion"] is True
    assert next(
        widget
        for widget in tabs.select(pn.widgets.FileInput)
        if widget.label == "Static environment scene (.xml, optional)"
    ).accept == ".xml"
    assert any(
        widget.label == "Licensed SMPL model directory"
        for widget in tabs.select(pn.widgets.TextInput)
    )
    dynamic_controls = next(
        card
        for card in tabs.select(pn.Card)
        if card.title == "Dynamic scene controls"
    )
    run_button = next(
        widget
        for widget in dynamic_controls.select(pn.widgets.Button)
        if widget.label == "Start"
    )
    scene_upload = next(
        widget
        for widget in dynamic_controls.select(pn.widgets.FileInput)
        if widget.label == "Static environment scene (.xml, optional)"
    )
    action_row = next(
        row
        for row in dynamic_controls.select(pn.Row)
        if run_button in row.objects
    )
    assert dynamic_controls.objects.index(action_row) < (
        dynamic_controls.objects.index(scene_upload)
    )
    dynamic_copy = "\n".join(
        str(pane.object)
        for pane in dynamic_controls.select(pn.pane.Markdown)
    )
    assert "### Simulation modes" in dynamic_copy
    assert "Run simulation" not in dynamic_copy
    assert "Use the prepared room or upload a static-environment XML." not in dynamic_copy
    assert (
        "The radar phase center is fixed at **(0, 0, 0)** for this run."
        not in dynamic_copy
    )
    assert "Without an uploaded XML" not in dynamic_copy
    assert "**Motion input:**" not in dynamic_copy
    assert not any(
        "Load a bundle, then choose an embedded solver result or external ADC."
        in str(pane.object)
        for pane in tabs.select(pn.pane.Markdown)
    )
    comparison_source = next(
        widget
        for widget in tabs.select(pn.widgets.Select)
        if widget.label == "Comparison source"
    )
    comparison_modes = next(
        widget
        for widget in tabs.select(pn.widgets.MultiChoice)
        if widget.label == "Bundle simulation modes"
    )
    assert comparison_source.options == {
        "Run simulation from loaded bundle": "run",
        "Load previously saved ADC or bundle": "saved",
    }
    assert comparison_source.value == "run"
    assert comparison_modes.visible is True
    assert comparison_modes.value == ["human_only_po"]
    channel_zero = next(
        widget
        for widget in tabs.select(pn.widgets.Checkbox)
        if widget.label == "Channel 0 only"
    )
    assert channel_zero.visible is True
    compare_button = next(
        button
        for button in tabs.select(pn.widgets.Button)
        if button.label == "Run simulation"
    )
    simulated_download = next(
        widget
        for widget in tabs.select(pn.widgets.FileDownload)
        if widget.description == "Download simulated ADC"
    )
    channel_download_row = next(
        row
        for row in tabs.select(pn.Row)
        if channel_zero in row.objects and simulated_download in row.objects
    )
    comparison_action_row = next(
        column
        for column in tabs.select(pn.Column)
        if channel_download_row in column.objects
        and compare_button in column.objects
    )
    assert comparison_action_row.margin == (2, 0)
    assert comparison_action_row.objects[0] is channel_download_row
    assert channel_download_row.objects[0] is channel_zero
    assert isinstance(channel_download_row.objects[1], pn.Spacer)
    assert channel_download_row.objects[2] is simulated_download
    dependency_hint = comparison_action_row.objects[2]
    assert "Load and validate a primary bundle" in str(
        dependency_hint.object
    )
    assert dependency_hint.visible is True
    assert comparison_action_row.objects[3] is compare_button
    assert compare_button.sizing_mode == "stretch_width"
    assert compare_button.width is None
    assert compare_button.height == 46
    assert compare_button.margin == 0
    assert compare_button.disabled is True
    panel_row = next(
        layout
        for layout in tabs.select(pn.FlexBox)
        if {
            card.title
            for card in layout.objects
            if isinstance(card, pn.Card)
        }
        == {"Step 1 · Primary bundle", "Step 2 · Bundle comparison"}
    )
    assert panel_row.flex_direction == "row"
    assert panel_row.flex_wrap == "wrap"
    assert panel_row.gap == "20px"
    compare_callback = next(
        watcher.fn
        for watcher in compare_button.param.watchers["clicks"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_compare"
    )
    comparison_closure = inspect.getclosurevars(compare_callback)
    comparison_status = next(
        pane
        for pane in tabs.select(pn.pane.Markdown)
        if "hermes-comparison-output" in pane.css_classes
    )
    comparison_metrics = next(
        pane
        for pane in tabs.select(pn.pane.JSON)
        if "hermes-comparison-output" in pane.css_classes
    )
    assert comparison_status.styles["color"] == "#e7eef8"
    assert comparison_metrics.styles["color"] == "#e7eef8"
    assert any(
        "color: #e7eef8 !important" in str(stylesheet)
        for stylesheet in comparison_metrics.stylesheets
    )
    assert comparison_metrics in measurement_view_tabs.objects[5].select(
        pn.pane.JSON
    )
    assert "smpl_model_dir" not in comparison_closure.unbound
    assert comparison_closure.nonlocals[
        "human_room_smpl_model_dir"
    ] is next(
        widget
        for widget in tabs.select(pn.widgets.TextInput)
        if widget.label == "Licensed SMPL model directory"
    )
    load_button = next(
        button
        for button in tabs.select(pn.widgets.Button)
        if button.label == "Load and validate bundle"
    )
    load_callback = next(
        watcher.fn
        for watcher in load_button.param.watchers["clicks"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_load_bundle"
    )
    asyncio.run(load_callback(None))
    assert compare_button.disabled is False
    assert dependency_hint.visible is False
    bundle_status = next(
        pane
        for pane in tabs.select(pn.pane.Markdown)
        if "hermes-bundle-status" in pane.css_classes
    )
    assert "Bundle loaded and validated" in str(bundle_status.object)
    assert "Frames `1`" in str(bundle_status.object)
    assert "ADC shape `(1, 64, 256, 192)`" in str(bundle_status.object)
    assert "fingerprint `" in str(bundle_status.object)
    comparison_source.value = "saved"
    assert compare_button.disabled is True
    assert "Drop a saved ADC file or bundle" in str(dependency_hint.object)
    simulated_path_for_gate = comparison_closure.nonlocals["simulated_path"]
    simulated_path_for_gate.value = "/tmp/candidate-adc.npz"
    assert compare_button.disabled is False
    simulated_path_for_gate.value = ""
    assert compare_button.disabled is True
    comparison_source.value = "run"
    assert compare_button.disabled is False
    scene_figures = [
        pane.object
        for pane in measurement_view_tabs.objects[0].select(pn.pane.Plotly)
        if pane.object is not None
    ]
    assert len(scene_figures) == 1
    assert scene_figures[0].layout.title.text.startswith(
        "Primary bundle scene"
    )
    assert any(
        "Material: wood" in str(trace.hovertemplate)
        for trace in scene_figures[0].data
    )
    adc_row = measurement_view_tabs.objects[1]
    profile_plot = measurement_view_tabs.objects[2]
    range_time_row = measurement_view_tabs.objects[3]
    range_doppler_row = measurement_view_tabs.objects[4]
    report_column = measurement_view_tabs.objects[5]
    assert isinstance(adc_row, pn.Row)
    assert isinstance(profile_plot, pn.pane.Plotly)
    assert isinstance(range_time_row, pn.Row)
    assert isinstance(range_doppler_row, pn.Row)
    assert isinstance(report_column, pn.Column)
    assert profile_plot.object is not None
    assert len(profile_plot.object.data) == 1
    candidate_results = (
        adc_row.objects[1],
        range_time_row.objects[1],
        range_doppler_row.objects[1],
    )
    for row in (adc_row, range_time_row, range_doppler_row):
        assert row.objects[0].object is not None
    for results in candidate_results:
        assert results.objects == []
        assert results.visible is False

    assert simulated_download.label == "\u200b"
    assert simulated_download.icon == "download"
    assert simulated_download.margin == 0
    assert simulated_download.sizing_mode == "fixed"
    assert simulated_download.width == 44
    assert simulated_download.height == 40
    assert not simulated_download.stylesheets
    assert simulated_download.visible is True
    assert simulated_download.disabled is True

    import validation.runner as validation_runner

    comparison_state = inspect.getclosurevars(compare_callback).nonlocals[
        "measurement_state"
    ]
    expected_simulated = np.asarray(
        comparison_state["bundle"].primary_adc
    ).copy()

    def simulate_immediately(_bundle, **kwargs):
        assert callable(kwargs["cancel_check"])
        return (
            expected_simulated[..., :1]
            if kwargs["channel_zero_only"]
            else expected_simulated
        )

    monkeypatch.setattr(
        validation_runner,
        "simulate_bundle_adc",
        simulate_immediately,
    )
    asyncio.run(compare_callback(None))
    for results in candidate_results:
        assert results.visible is True
        assert len(results.objects) == 1
        assert results.objects[0].object.layout.title.text.startswith(
            "Simulated "
        )
    assert profile_plot.object.layout.title.text == "Range profile comparison"
    assert [trace.name for trace in profile_plot.object.data] == [
        "Measured",
        "Simulated Human-only PO",
    ]
    assert simulated_download.disabled is False
    assert simulated_download.filename == "simulated_adc_human_only_po.npz"
    assert "Comparison complete" in str(comparison_status.object)
    assert {
        "range_time_correlation",
        "range_doppler_correlation",
        "complex_correlation",
        "scale_adjusted_complex_nmse",
        "peak_range_error_m",
    }.issubset(comparison_metrics.object)
    with np.load(simulated_download.callback(), allow_pickle=False) as archive:
        assert np.array_equal(archive["adc"], expected_simulated)

    comparison_modes.value = ["human_only_po", "rt_coherent_bank"]
    assert comparison_state["comparisons"] == {}
    assert comparison_state["simulated_adcs"] == {}
    assert simulated_download.disabled is True
    assert simulated_download.callback().getvalue() == b""
    channel_zero.value = True
    asyncio.run(compare_callback(None))
    for results in candidate_results:
        assert len(results.objects) == 2
    assert [trace.name for trace in profile_plot.object.data] == [
        "Measured",
        "Simulated Human-only PO",
        "Simulated Coherent RT",
    ]
    assert simulated_download.filename == "simulated_adc_results.zip"
    with zipfile.ZipFile(simulated_download.callback()) as archive:
        assert set(archive.namelist()) == {
            "simulated_adc_human_only_po.npz",
            "simulated_adc_rt_coherent_bank.npz",
        }
        with np.load(
            BytesIO(archive.read("simulated_adc_human_only_po.npz")),
            allow_pickle=False,
        ) as exported:
            assert exported["adc"].shape[-1] == 1

    cancellation_started = threading.Event()

    def simulate_until_cancelled(_bundle, **kwargs):
        cancel_check = kwargs["cancel_check"]
        cancellation_started.set()
        while not cancel_check():
            threading.Event().wait(0.005)
        raise InterruptedError("Simulation cancelled by user")

    monkeypatch.setattr(
        validation_runner,
        "simulate_bundle_adc",
        simulate_until_cancelled,
    )
    stop_confirmation = next(
        component
        for component in tabs.select(pn.reactive.ReactiveHTML)
        if "confirmation_sequence" in component.param
    )
    confirmation_sequence = stop_confirmation.confirmation_sequence
    stop_confirmation.open = True
    stop_confirmation._process_event(  # pylint: disable=protected-access
        DOMEvent(None, node="cancel", data={"type": "click"})
    )
    assert stop_confirmation.open is False
    assert stop_confirmation.confirmation_sequence == confirmation_sequence

    async def stop_running_comparison():
        task = asyncio.create_task(compare_callback(None))
        while not cancellation_started.is_set():
            await asyncio.sleep(0.005)
        assert compare_button.label == "Stop simulation"
        assert compare_button.color == "danger"
        await compare_callback(None)
        assert stop_confirmation.open is True
        stop_confirmation._process_event(  # pylint: disable=protected-access
            DOMEvent(None, node="confirm", data={"type": "click"})
        )
        await task

    asyncio.run(stop_running_comparison())
    assert stop_confirmation.open is False
    assert compare_button.label == "Run simulation"
    assert "Simulation stopped" in str(comparison_status.object)
    assert simulated_download.disabled is True
    assert not tabs.select(pn.widgets.FileSelector)
    assert not any(
        button.label == "File explorer"
        for button in tabs.select(pn.widgets.Button)
    )
    file_droppers = {
        dropper.label: dropper
        for dropper in tabs.select(pn.widgets.FileDropper)
    }
    assert set(file_droppers) == {
        "Drop primary bundle ZIP or folder",
        "Drop simulated ADC, bundle ZIP, or folder",
    }
    assert all(dropper.multiple for dropper in file_droppers.values())
    measurement_dropper = file_droppers[
        "Drop primary bundle ZIP or folder"
    ]
    simulated_dropper = file_droppers[
        "Drop simulated ADC, bundle ZIP, or folder"
    ]
    assert simulated_dropper.visible is False
    measurement_source = next(
        widget
        for widget in tabs.select(pn.widgets.Select)
        if widget.label == "Measurement benchmark"
    )
    assert measurement_dropper.visible is False
    measurement_source.value = ""
    assert measurement_dropper.visible is True
    assert compare_button.disabled is True
    comparison_source.value = "saved"
    assert comparison_modes.visible is False
    assert channel_zero.visible is False
    assert simulated_dropper.visible is True
    assert compare_button.label == "Load saved ADC and compare"
    measurement_dropper.value = {"dropped-bundle.zip": b"zip bytes"}
    assert any(
        "Bundle upload ready" in str(pane.object)
        and "`dropped-bundle.zip`" in str(pane.object)
        for pane in tabs.select(pn.pane.Markdown)
    )
    included_fixture = next(
        value for value in measurement_source.options.values() if value
    )
    measurement_source.value = included_fixture
    assert measurement_dropper.visible is False
    measurement_source.value = ""
    assert measurement_dropper.visible is True


def test_gui_template_initializes_server_document_resources():
    application = build_app()
    document = Document()

    application.server_doc(document)

    assert "template_resources" in document.template_variables


def test_bundle_comparison_state_restores_across_theme_reload():
    application = build_app()
    tabs = application.main[0]
    storage = next(
        component
        for component in application.header[0].select(
            pn.reactive.ReactiveHTML
        )
        if "loaded_state" in component.param
    )
    fixture = next(
        widget
        for widget in tabs.select(pn.widgets.Select)
        if widget.label == "Measurement benchmark"
    )
    load_button = next(
        widget
        for widget in tabs.select(pn.widgets.Button)
        if widget.label == "Load and validate bundle"
    )
    load_callback = next(
        watcher.fn
        for watcher in load_button.param.watchers["clicks"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_load_bundle"
    )
    bundle_path = inspect.getclosurevars(load_callback).nonlocals["bundle_path"]
    background = next(
        widget
        for widget in tabs.select(pn.widgets.Checkbox)
        if widget.label == "Background subtraction"
    )
    clutter = next(
        widget
        for widget in tabs.select(pn.widgets.Select)
        if widget.label == "Clutter removal"
    )
    comparison_source = next(
        widget
        for widget in tabs.select(pn.widgets.Select)
        if widget.label == "Comparison source"
    )
    comparison_modes = next(
        widget
        for widget in tabs.select(pn.widgets.MultiChoice)
        if widget.label == "Bundle simulation modes"
    )
    channel_zero = next(
        widget
        for widget in tabs.select(pn.widgets.Checkbox)
        if widget.label == "Channel 0 only"
    )
    compare_button = next(
        widget
        for widget in tabs.select(pn.widgets.Button)
        if widget.label == "Run simulation"
    )
    compare_callback = next(
        watcher.fn
        for watcher in compare_button.param.watchers["clicks"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_compare"
    )
    simulated_path = inspect.getclosurevars(compare_callback).nonlocals[
        "simulated_path"
    ]
    included_fixture = next(value for value in fixture.options.values() if value)
    cached_bundle = load_bundle(included_fixture)
    comparison_cache_key = _cache_theme_comparison(
        bundle_fingerprint=cached_bundle.fingerprint,
        comparison_kind="saved",
        candidates=[
            (
                "saved",
                "external ADC",
                "Loaded",
                "External Adc",
                cached_bundle.primary_adc,
            )
        ],
        background_subtraction=False,
        clutter_removal="mean",
        channel_indices=None,
    )
    restore_callback = next(
        watcher.fn
        for watcher in storage.param.watchers["loaded_state"]["value"]
        if getattr(watcher.fn, "__name__", "") == "restore_gui_session"
    )
    restore_state = inspect.getclosurevars(restore_callback).nonlocals[
        "gui_session_state"
    ]

    async def restore_state_and_bundle():
        storage.loaded_state = {
            "schema_version": 1,
            "top_tab": 2,
            "dynamic_tab": 0,
            "measurement_tab": 3,
            "bundle_fixture": included_fixture,
            "bundle_path": included_fixture,
            "bundle_loaded": True,
            "background_subtraction": False,
            "clutter_removal": "mean",
            "comparison_source": "saved",
            "comparison_simulation_modes": [
                "human_only_po",
                "rt_coherent_bank",
            ],
            "comparison_channel_zero": True,
            "simulated_path": "/tmp/persisted-candidate.npz",
            "comparison_completed": True,
            "comparison_kind": "saved",
            "comparison_cache_key": comparison_cache_key,
            "has_motion": False,
            "load_sequence": 2,
        }
        restore_task = restore_state["bundle_restore_task"]
        assert restore_task is not None
        await restore_task

    asyncio.run(restore_state_and_bundle())

    assert tabs.active == 2
    assert fixture.value == included_fixture
    assert bundle_path.value == included_fixture
    assert background.value is False
    assert clutter.value == "mean"
    assert comparison_source.value == "saved"
    assert comparison_modes.value == ["human_only_po", "rt_coherent_bank"]
    assert channel_zero.value is True
    assert simulated_path.value == "/tmp/persisted-candidate.npz"
    measurement_state = inspect.getclosurevars(restore_callback).nonlocals[
        "measurement_state"
    ]
    assert measurement_state["bundle"] is not None
    assert measurement_state["comparison_cache_key"] == comparison_cache_key
    assert set(measurement_state["comparisons"]) == {"saved"}
    assert storage.saved_state["bundle_loaded"] is True

    channel_zero.value = False
    assert storage.saved_state["comparison_channel_zero"] is False
    assert storage.saved_state["comparison_simulation_modes"] == [
        "human_only_po",
        "rt_coherent_bank",
    ]
    background.value = True
    assert storage.saved_state["background_subtraction"] is True


def test_stale_bundle_load_cannot_commit_after_selection_changes(monkeypatch):
    application = build_app()
    tabs = application.main[0]
    load_button = next(
        button
        for button in tabs.select(pn.widgets.Button)
        if button.label == "Load and validate bundle"
    )
    load_callback = next(
        watcher.fn
        for watcher in load_button.param.watchers["clicks"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_load_bundle"
    )
    closure = inspect.getclosurevars(load_callback).nonlocals
    bundle_path = closure["bundle_path"]
    measurement_state = closure["measurement_state"]
    original_bundle = load_bundle(bundle_path.value)
    started = threading.Event()
    release = threading.Event()

    def delayed_load(_path, **_kwargs):
        started.set()
        release.wait(timeout=5)
        return original_bundle

    monkeypatch.setattr(gui_app_module, "load_bundle", delayed_load)

    async def change_selection_during_load():
        task = asyncio.create_task(load_callback(None))
        assert await asyncio.to_thread(started.wait, 5.0)
        bundle_path.value = "/tmp/a-newer-selection.zip"
        release.set()
        await task

    asyncio.run(change_selection_during_load())

    assert measurement_state["bundle"] is None
    assert measurement_state["loading"] is False
    assert load_button.disabled is False


def test_static_dynamic_controls_and_uploads_restore_across_theme_reload():
    application = build_app()
    tabs = application.main[0]
    storage = next(
        component
        for component in application.header[0].select(
            pn.reactive.ReactiveHTML
        )
        if "loaded_uploads" in component.param
    )
    all_widgets = [
        *tabs.select(pn.widgets.Widget),
        *application.modal[0].select(pn.widgets.Widget),
    ]
    widgets = {
        widget.label: widget
        for widget in all_widgets
        if getattr(widget, "label", "")
    }
    static_range = widgets["Target x / range-forward [m]"]
    static_material = widgets["Material preset"]
    static_radar_yaw = next(
        widget
        for widget in tabs.select(pn.widgets.FloatSlider)
        if widget.label == "Radar yaw [deg]"
    )
    dynamic_radar_yaw = [
        widget
        for widget in tabs.select(pn.widgets.FloatSlider)
        if widget.label == "Radar yaw [deg]"
    ][1]
    human_x = widgets["Human x / range-forward [m]"]
    human_yaw = widgets["Human yaw [deg]"]
    smpl_dir = widgets["Licensed SMPL model directory"]
    tx_antennas = widgets["Active Tx antennas"]
    rt_samples = widgets["RT ray samples per source"]
    full_rt = widgets["Full RT"]
    hybrid_po = widgets["Hybrid PO"]
    static_upload = widgets["Human mesh (.obj or .npz)"]
    motion_upload = widgets["AMASS-like human motion (.npz)"]
    scene_upload = widgets["Static environment scene (.xml, optional)"]

    changed_values = {
        static_range: 3.1,
        static_material: "aluminum",
        static_radar_yaw: 17.0,
        dynamic_radar_yaw: -22.0,
        human_x: 2.15,
        human_yaw: 35.0,
        smpl_dir: "/private/models/smpl",
        tx_antennas: [0],
        rt_samples: 600,
        full_rt: True,
        hybrid_po: False,
    }
    for widget, value in changed_values.items():
        with param.parameterized.discard_events(widget):
            widget.value = value
    for widget, filename, payload in (
        (static_upload, "subject.obj", b"mesh bytes"),
        (motion_upload, "walk.npz", b"motion bytes"),
        (scene_upload, "room.xml", b'<scene version="3.0.0"/>'),
    ):
        with param.parameterized.discard_events(widget):
            widget.filename = filename
            widget.value = payload

    persist_uploads = next(
        watcher.fn
        for watcher in motion_upload.param.watchers["value"]["value"]
        if getattr(watcher.fn, "__name__", "") == "persist_gui_uploads"
    )
    persist_uploads()
    saved_state = dict(storage.saved_state)
    saved_uploads = dict(storage.saved_uploads)

    assert saved_state["schema_version"] == 2
    assert saved_state["static_controls"]["target_range"] == 3.1
    assert saved_state["radar_controls"]["tx_antennas"] == [0]
    assert saved_state["dynamic_controls"]["human_yaw"] == 35.0
    assert saved_state["solver_settings"]["rt_samples_per_source"] == 600
    assert saved_state["dynamic_modes"] == {
        "full_rt": True,
        "coherent_rt": False,
        "human_only_po": False,
        "hybrid_po": False,
    }
    assert set(saved_uploads) == {"static_mesh", "motion", "scene_xml"}
    assert _get_theme_gui_results(saved_state["gui_result_cache_key"])[
        "static_result"
    ] is not None

    reset_values = {
        static_range: 1.0,
        static_material: "pec",
        static_radar_yaw: 0.0,
        dynamic_radar_yaw: 0.0,
        human_x: 0.5,
        human_yaw: 0.0,
        smpl_dir: "",
        tx_antennas: list(tx_antennas.options.values()),
        rt_samples: 100_000,
        full_rt: False,
        hybrid_po: True,
    }
    for widget, value in reset_values.items():
        with param.parameterized.discard_events(widget):
            widget.value = value
    for widget in (static_upload, motion_upload, scene_upload):
        with param.parameterized.discard_events(widget):
            widget.filename = ""
            widget.value = b""

    storage.loaded_state = saved_state
    storage.loaded_uploads = saved_uploads

    assert static_range.value == 3.1
    assert static_material.value == "aluminum"
    assert static_radar_yaw.value == 17.0
    assert dynamic_radar_yaw.value == -22.0
    assert human_x.value == 2.15
    assert human_yaw.value == 35.0
    assert smpl_dir.value == "/private/models/smpl"
    assert tx_antennas.value == [0]
    assert rt_samples.value == 600
    assert full_rt.value is True
    assert hybrid_po.value is False
    assert static_upload.filename == "subject.obj"
    assert static_upload.value == b"mesh bytes"
    assert motion_upload.filename == "walk.npz"
    assert motion_upload.value == b"motion bytes"
    assert scene_upload.filename == "room.xml"
    assert scene_upload.value == b'<scene version="3.0.0"/>'


def test_completed_dynamic_results_restore_from_theme_cache(monkeypatch):
    application = build_app()
    tabs = application.main[0]
    storage = next(
        component
        for component in application.header[0].select(
            pn.reactive.ReactiveHTML
        )
        if "loaded_uploads" in component.param
    )
    fake_result = SimpleNamespace(simulation_mode="hybrid_po")
    cache_key = _cache_theme_gui_results(
        dynamic_results={"hybrid_po": fake_result},
        dynamic_status="**SIMULATION READY** · restored",
        dynamic_diagnostics=None,
        dynamic_diagnostics_status="Restored overlays.",
    )
    monkeypatch.setattr(
        gui_app_module,
        "_human_room_results_view",
        lambda *_args, **_kwargs: pn.Column(
            pn.pane.Markdown("restored dynamic products")
        ),
    )
    restore_callback = next(
        watcher.fn
        for watcher in storage.param.watchers["loaded_state"]["value"]
        if getattr(watcher.fn, "__name__", "") == "restore_gui_session"
    )
    human_room_state = inspect.getclosurevars(restore_callback).nonlocals[
        "human_room_state"
    ]
    full_rt = next(
        widget
        for widget in tabs.select(pn.widgets.Checkbox)
        if widget.label == "Full RT"
    )
    hybrid_po = next(
        widget
        for widget in tabs.select(pn.widgets.Checkbox)
        if widget.label == "Hybrid PO"
    )
    rt_samples = next(
        widget
        for widget in application.modal[0].select(pn.widgets.IntInput)
        if widget.label == "RT ray samples per source"
    )
    mode_events = []
    full_rt.param.watch(lambda event: mode_events.append(event.new), "value")
    dynamic_tabs = next(
        component
        for component in tabs.select(pn.Tabs)
        if component._names == ["Scene and motion", "Simulation results"]
    )
    navigation_events = []
    dynamic_tabs.param.watch(
        lambda event: navigation_events.append(event.new),
        "active",
    )

    storage.loaded_state = {
        "schema_version": 2,
        "top_tab": 1,
        "dynamic_tab": 1,
        "measurement_tab": 0,
        "gui_result_cache_key": cache_key,
        "solver_settings": {
            "po_integration_mode": "parent_face_quadrature",
            "po_quadrature_phase_span_scale_rad": 1.0,
            "po_quadrature_max_refinement_depth": 2,
            "po_quadrature_max_subfaces_per_parent": 16,
            "rt_samples_per_source": 600,
            "rt_max_paths_per_source": 5_000,
            "rt_max_depth": 3,
        },
        "dynamic_modes": {
            "full_rt": True,
            "coherent_rt": False,
            "human_only_po": False,
            "hybrid_po": False,
        },
        "has_uploads": False,
        "bundle_loaded": False,
    }

    assert human_room_state["result"] is fake_result
    assert human_room_state["results"] == {"hybrid_po": fake_result}
    assert full_rt.value is True
    assert hybrid_po.value is False
    assert rt_samples.value == 600
    assert mode_events == [True]
    assert navigation_events == [1]
    solver_storage = next(
        component
        for component in application.header[0].select(
            pn.reactive.ReactiveHTML
        )
        if "loaded_settings" in component.param
    )
    solver_restore = next(
        watcher.fn
        for watcher in solver_storage.param.watchers["loaded_settings"][
            "value"
        ]
        if getattr(watcher.fn, "__name__", "")
        == "on_solver_settings_loaded"
    )
    asyncio.run(
        solver_restore(
            SimpleNamespace(
                new={
                    "schema_version": 1,
                    "settings": storage.loaded_state["solver_settings"],
                }
            )
        )
    )
    assert human_room_state["result"] is fake_result
    assert any(
        "restored dynamic products" in str(pane.object)
        for pane in tabs.select(pn.pane.Markdown)
    )
    export_button = next(
        button
        for button in tabs.select(pn.widgets.FileDownload)
        if button.label == "Export loadable bundle"
        and button.filename == "hermes-human-room-bundle.zip"
    )
    assert export_button.disabled is False


def test_dynamic_run_locks_and_restores_every_input_widget(monkeypatch):
    application = build_app()
    tabs = application.main[0]
    run_button = next(
        button
        for button in tabs.select(pn.widgets.Button)
        if button.label == "Start"
    )
    run_callback = next(
        watcher.fn
        for watcher in run_button.param.watchers["clicks"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_run_human_room"
    )
    run_closure = inspect.getclosurevars(run_callback)
    run_inputs = run_closure.nonlocals["human_room_run_input_widgets"]
    assert len({id(widget) for widget in run_inputs}) == len(run_inputs)

    by_label = {
        widget.label: widget
        for widget in run_inputs
        if getattr(widget, "label", "")
    }
    required_labels = {
        "AMASS-like human motion (.npz)",
        "Hybrid PO",
        "Human x / range-forward [m]",
        "Carrier frequency [GHz]",
        "RT ray samples per source",
        "Begin frame (inclusive)",
    }
    assert required_labels.issubset(by_label)
    motion_upload = by_label["AMASS-like human motion (.npz)"]
    motion_payload = BytesIO()
    np.savez(
        motion_payload,
        poses=np.zeros((1, 72), dtype=np.float32),
        trans=np.zeros((1, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        times=np.zeros(1, dtype=np.float32),
    )
    with param.parameterized.discard_events(motion_upload):
        motion_upload.filename = "locked-run.npz"
        motion_upload.value = motion_payload.getvalue()

    started = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(
        gui_app_module,
        "load_amass_motion",
        lambda *_args, **_kwargs: object(),
    )

    def interruptible_run(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=5.0)
        raise InterruptedError("test run stopped")

    monkeypatch.setattr(
        gui_app_module,
        "run_human_room_experiment",
        interruptible_run,
    )
    original_states = tuple(
        (widget, bool(widget.disabled)) for widget in run_inputs
    )

    async def exercise_run():
        task = asyncio.create_task(run_callback(None))
        assert await asyncio.to_thread(started.wait, 5.0)
        assert all(widget.disabled for widget in run_inputs)
        release.set()
        await asyncio.wait_for(task, timeout=5.0)

    asyncio.run(exercise_run())

    assert all(
        widget.disabled is disabled for widget, disabled in original_states
    )


def test_widget_lock_restores_preexisting_disabled_states():
    enabled = pn.widgets.TextInput(label="Enabled", disabled=False)
    disabled = pn.widgets.TextInput(label="Disabled", disabled=True)

    states = _lock_widget_disabled_states((enabled, disabled, enabled))

    assert len(states) == 2
    assert enabled.disabled is True
    assert disabled.disabled is True
    _restore_widget_disabled_states(states)
    assert enabled.disabled is False
    assert disabled.disabled is True


def test_dropped_folder_materialization_preserves_tree_and_rejects_traversal(
    tmp_path,
):
    selected = gui_app_module._materialize_dropped_files(
        {
            "bundle/bundle.json": "{}",
            "bundle/data/radar_adc.npz": b"adc",
        },
        tmp_path / "valid-drop",
    )

    assert selected == tmp_path / "valid-drop" / "bundle"
    assert (selected / "bundle.json").read_text() == "{}"
    assert (selected / "data" / "radar_adc.npz").read_bytes() == b"adc"
    with pytest.raises(ValueError, match="unsafe dropped-file path"):
        gui_app_module._materialize_dropped_files(
            {"../outside.json": b"bad"},
            tmp_path / "unsafe-drop",
        )


def test_dropped_file_materialization_enforces_resource_limits(tmp_path):
    with pytest.raises(ValueError, match="contains 2 files"):
        gui_app_module._materialize_dropped_files(
            {"one.npz": b"1", "two.npz": b"2"},
            tmp_path / "too-many",
            max_files=1,
        )
    assert not (tmp_path / "too-many").exists()

    with pytest.raises(ValueError, match="per-file limit"):
        gui_app_module._materialize_dropped_files(
            {"large.npz": b"1234"},
            tmp_path / "too-large",
            max_file_bytes=3,
        )
    assert not (tmp_path / "too-large").exists()

    with pytest.raises(ValueError, match="aggregate"):
        gui_app_module._materialize_dropped_files(
            {"one.npz": b"123", "two.npz": b"456"},
            tmp_path / "too-large-total",
            max_file_bytes=3,
            max_total_bytes=5,
        )
    assert not (tmp_path / "too-large-total").exists()


def test_single_file_input_payload_enforces_resource_limit():
    assert gui_app_module._validated_upload_payload(
        b"123",
        label="motion",
        max_bytes=3,
    ) == b"123"
    with pytest.raises(ValueError, match="the limit is 3 bytes"):
        gui_app_module._validated_upload_payload(
            b"1234",
            label="motion",
            max_bytes=3,
        )


def test_gui_launcher_serves_browser_icon_probe_routes(monkeypatch):
    served = {}

    def capture_serve(*args, **kwargs):
        served["args"] = args
        served["kwargs"] = kwargs

    monkeypatch.setattr(pn, "serve", capture_serve)

    assert gui_cli_module.main(["--no-browser"]) == 0
    patterns = served["kwargs"]["extra_patterns"]
    assert len(patterns) == 1
    assert patterns[0][0] == (
        r"/(?:apple-touch-icon(?:-precomposed)?\.png|favicon\.ico)"
    )
    assert patterns[0][1].__name__ == "HermesIconHandler"


def test_gui_launcher_rejects_remote_bind_without_unsafe_override(capsys):
    with pytest.raises(SystemExit) as exc_info:
        gui_cli_module.main(
            ["--address", "0.0.0.0", "--no-browser"],
        )

    assert exc_info.value.code == 2
    assert "refusing non-loopback --address" in capsys.readouterr().err


def test_gui_launcher_marks_unsafe_remote_app_as_restricted(
    monkeypatch,
    capsys,
):
    served = {}
    built = {}

    def capture_serve(*args, **kwargs):
        served["args"] = args
        served["kwargs"] = kwargs

    def capture_build_app(**kwargs):
        built.update(kwargs)
        return object()

    monkeypatch.setattr(pn, "serve", capture_serve)
    monkeypatch.setattr(gui_app_module, "build_app", capture_build_app)

    assert gui_cli_module.main(
        [
            "--address",
            "192.0.2.10",
            "--unsafe-allow-remote",
            "--no-browser",
        ]
    ) == 0
    application_factory = served["args"][0]["/"]
    application_factory()

    assert built["remote_access"] is True
    assert served["kwargs"]["address"] == "192.0.2.10"
    assert served["kwargs"]["websocket_origin"] == ["192.0.2.10:5006"]
    warning = capsys.readouterr().err
    assert "UNSAFE REMOTE GUI MODE" in warning
    assert "does not provide TLS or user management" in warning
    assert "HERMES per-launch password:" in warning
    assert len(served["kwargs"]["cookie_secret"]) == 64
    assert served["kwargs"]["auth_provider"].get_user is not None
    expected_websocket_limit = gui_cli_module._websocket_message_limit(
        64 * 1024 * 1024
    )
    assert served["kwargs"]["websocket_max_message_size"] == (
        expected_websocket_limit
    )
    assert served["kwargs"]["http_server_kwargs"] == {
        "max_header_size": 16 * 1024,
        "max_body_size": 64 * 1024,
        "max_buffer_size": expected_websocket_limit,
    }


def test_gui_launch_auth_rejects_cross_site_before_session_creation():
    provider = gui_cli_module._build_launch_auth_provider("secret")

    class Request:
        def __init__(self, fetch_site):
            self.headers = {"Sec-Fetch-Site": fetch_site}

    class Handler:
        def __init__(self, fetch_site, cookie):
            self.request = Request(fetch_site)
            self.cookie = cookie
            self.headers = {}

        def set_header(self, name, value):
            self.headers[name] = value

        def get_secure_cookie(self, _name, max_age_days=None):
            del max_age_days
            return self.cookie

    same_origin = Handler("same-origin", b"hermes")
    assert provider.get_user(same_origin) == "hermes"
    assert same_origin.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in same_origin.headers[
        "Content-Security-Policy"
    ]

    assert provider.get_user(Handler("cross-site", b"hermes")) is None
    assert provider.get_user(Handler("", b"hermes")) is None
    assert provider.get_user(Handler("same-origin", None)) is None


def test_gui_launch_auth_accepts_authenticated_websocket_without_headers():
    from types import SimpleNamespace

    from tornado.websocket import WebSocketHandler

    provider = gui_cli_module._build_launch_auth_provider("secret")
    handler = object.__new__(WebSocketHandler)
    handler.request = SimpleNamespace(
        headers={"Sec-Fetch-Site": "same-origin"}
    )

    def reject_websocket_headers(*_args, **_kwargs):
        raise RuntimeError("Method not supported for Web Sockets")

    handler.set_header = reject_websocket_headers
    handler.get_secure_cookie = lambda *_args, **_kwargs: b"hermes"

    assert provider.get_user(handler) == "hermes"


def test_gui_launcher_requires_origins_for_unsafe_wildcard_bind(capsys):
    with pytest.raises(SystemExit) as exc_info:
        gui_cli_module.main(
            [
                "--address",
                "0.0.0.0",
                "--unsafe-allow-remote",
                "--no-browser",
            ]
        )

    assert exc_info.value.code == 2
    assert "requires at least one explicit" in capsys.readouterr().err


def test_gui_launcher_uses_explicit_wildcard_origin_allowlist(monkeypatch):
    served = {}
    monkeypatch.setattr(
        pn,
        "serve",
        lambda *args, **kwargs: served.update(args=args, kwargs=kwargs),
    )

    assert gui_cli_module.main(
        [
            "--address",
            "::",
            "--port",
            "6006",
            "--unsafe-allow-remote",
            "--allow-websocket-origin",
            "radar-lab.example",
            "--allow-websocket-origin",
            "[2001:db8::7]:7007",
            "--no-browser",
        ]
    ) == 0

    assert served["kwargs"]["websocket_origin"] == [
        "radar-lab.example:6006",
        "[2001:db8::7]:7007",
    ]


@pytest.mark.parametrize(
    "origin",
    ["*", "https://radar.example", "radar.example/path", "0.0.0.0"],
)
def test_gui_launcher_rejects_unsafe_websocket_origins(origin, capsys):
    with pytest.raises(SystemExit) as exc_info:
        gui_cli_module.main(
            [
                "--allow-websocket-origin",
                origin,
                "--no-browser",
            ]
        )

    assert exc_info.value.code == 2
    assert "websocket origin" in capsys.readouterr().err


def test_gui_launcher_formats_ipv6_loopback_websocket_origin(monkeypatch):
    served = {}
    monkeypatch.setattr(
        pn,
        "serve",
        lambda *args, **kwargs: served.update(args=args, kwargs=kwargs),
    )

    assert gui_cli_module.main(
        ["--address", "::1", "--no-browser"],
    ) == 0

    assert served["kwargs"]["websocket_origin"] == ["[::1]:5006"]


def test_remote_gui_locks_and_enforces_server_side_inputs(monkeypatch):
    monkeypatch.setattr(
        gui_app_module,
        "_default_human_mesh_path",
        lambda: pytest.fail("remote app must not inspect the private mesh"),
    )
    application = build_app(remote_access=True)
    tabs = application.main[0]
    widgets = {
        widget.label: widget
        for widget in tabs.select(pn.widgets.Widget)
        if getattr(widget, "label", "")
    }

    smpl_dir = widgets["Licensed SMPL model directory"]
    scene_upload = widgets["Static environment scene (.xml, optional)"]
    assert smpl_dir.disabled is True
    assert scene_upload.disabled is True

    load_button = widgets["Load and validate bundle"]
    load_callback = next(
        watcher.fn
        for watcher in load_button.param.watchers["clicks"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_load_bundle"
    )
    load_closure = inspect.getclosurevars(load_callback).nonlocals
    bundle_path = load_closure["bundle_path"]
    assert bundle_path.disabled is True

    with param.parameterized.discard_events(bundle_path):
        bundle_path.value = "/etc/passwd"
    asyncio.run(load_callback(None))
    status = next(
        pane
        for pane in tabs.select(pn.pane.Markdown)
        if "hermes-bundle-status" in pane.css_classes
    )
    assert "PermissionError" in str(status.object)
    assert "cannot read arbitrary server paths" in str(status.object)

    compare_button = widgets["Run simulation"]
    compare_callback = next(
        watcher.fn
        for watcher in compare_button.param.watchers["clicks"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_compare"
    )
    simulated_path = inspect.getclosurevars(compare_callback).nonlocals[
        "simulated_path"
    ]
    assert simulated_path.disabled is True

    scene_callback = next(
        watcher.fn
        for watcher in scene_upload.param.watchers["value"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_human_room_controls"
    )
    refresh_preview = inspect.getclosurevars(scene_callback).nonlocals[
        "refresh_human_room_preview"
    ]
    current_scene = inspect.getclosurevars(refresh_preview).nonlocals[
        "current_human_room_scene"
    ]
    with param.parameterized.discard_events(scene_upload):
        scene_upload.filename = "remote.xml"
        scene_upload.value = b'<scene><include filename="/etc/passwd"/></scene>'
    assert current_scene() == (None, None)


def test_motion_playback_interval_uses_timing_with_responsive_limits():
    assert _motion_playback_interval_ms(np.asarray([0.0])) == 100
    assert _motion_playback_interval_ms(np.arange(61) / 60.0) == 50
    assert _motion_playback_interval_ms(np.asarray([0.0, 0.1, 0.2])) == 100
    assert _motion_playback_interval_ms(np.asarray([0.0, 2.0])) == 1000
    assert _motion_playback_interval_ms(np.arange(121) / 120.0) == 50


def test_normal_motion_playback_decimates_to_a_dividing_20_or_25_hz_rate():
    assert _motion_playback_step(np.asarray([0.0])) == 1
    assert _motion_playback_step(np.arange(121) / 120.0) == 6
    assert _motion_playback_step(np.arange(101) / 100.0) == 4
    assert _motion_playback_step(np.arange(61) / 60.0) == 3
    assert _motion_playback_step(np.arange(51) / 50.0) == 2
    assert _motion_playback_step(np.arange(31) / 30.0) == 1
    assert _motion_playback_step(np.arange(11) / 10.0) == 1


def test_fast_motion_playback_uses_a_larger_stride_at_source_cadence():
    one_twenty_hz = np.arange(241, dtype=float) / 120.0
    sixty_hz = np.arange(121, dtype=float) / 60.0
    ten_hz = np.arange(21, dtype=float) / 10.0

    assert _motion_fast_playback_step(np.asarray([0.0])) == 1
    assert _motion_fast_playback_step(one_twenty_hz) == 24
    assert _motion_fast_playback_step(sixty_hz) == 12
    assert _motion_fast_playback_step(ten_hz) == 4


def test_human_room_window_defaults_to_one_frame_and_clamps_existing_window():
    assert _human_room_window_values(
        last_frame=99,
        previous_begin=0,
        previous_end=99,
        reset_window=True,
    ) == (0, 0)
    assert _human_room_window_values(
        last_frame=7,
        previous_begin=3,
        previous_end=99,
        reset_window=False,
    ) == (3, 7)


def test_human_room_range_maps_support_frame_selection():
    products = SimpleNamespace(
        range_time_power=np.arange(12, dtype=float).reshape(4, 3) + 1.0,
        range_time_ranges_m=np.asarray([0.0, 0.5, 1.0]),
        adc_times_s=np.asarray([[0.0, 0.001], [0.05, 0.051]]),
        range_doppler_power=(
            np.arange(30, dtype=float).reshape(2, 5, 3) + 1.0
        ),
        range_doppler_ranges_m=np.asarray([0.0, 0.5, 1.0]),
        velocities_mps=np.linspace(-1.0, 1.0, 5),
    )

    range_time = _range_time_figure(
        products,
        go,
        title="Prepared-room range-time map",
        plot_template="plotly_dark",
    )
    range_doppler = _range_doppler_figure(
        products,
        go,
        title="Prepared-room range-Doppler map",
        frame_index=1,
        source_frame_index=7,
        plot_template="plotly_dark",
    )

    assert np.asarray(range_time.data[0].z).shape == (4, 3)
    assert range_time.layout.yaxis.title.text == "Time [s]"
    assert np.asarray(range_doppler.data[0].z).shape == (5, 3)
    assert "simulated frame 2" in range_doppler.layout.title.text
    assert "source motion frame 7" in range_doppler.layout.title.text


def test_human_room_scene_renders_dynamic_solver_overlays_and_room_bounds():
    sequence = MeshSequence(
        vertices=np.asarray(
            [[[1.0, -0.2, 0.0], [1.0, 0.2, 0.0], [1.0, 0.0, 0.5]]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
        times=np.asarray([0.0]),
    )
    preview = SimpleNamespace(
        room_boxes=prepared_room_boxes(),
        mesh_sequence=sequence,
        scene_source_name="tutorial_bedroom",
        source_name="walk.npz",
    )
    diagnostics = SimpleNamespace(
        po=SimpleNamespace(visible_face_count=1),
        po_face_power=np.asarray([1.0]),
        rt=SimpleNamespace(
            segment_starts_m=np.asarray([[0.0, 0.0, 0.0]]),
            segment_ends_m=np.asarray([[1.0, 0.0, 0.25]]),
        ),
    )

    figure = _human_room_scene_figure(
        preview,
        go,
        frame_index=0,
        diagnostics=diagnostics,
        plot_template="plotly_dark",
    )

    names = [trace.name for trace in figure.data]
    assert "PO face power" in names
    assert "Top target-touching RT rays" in names
    floor_trace = next(trace for trace in figure.data if trace.name == "Floor")
    assert "Object: Floor" in floor_trace.hovertemplate
    assert "Material: Wood" in floor_trace.hovertemplate
    human_trace = next(trace for trace in figure.data if trace.name == "PO face power")
    assert "Material: Human skin" in human_trace.hovertemplate
    assert len(human_trace.facecolor) == 1
    assert figure.layout.scene.xaxis.range[0] <= 0.0
    assert figure.layout.scene.xaxis.range[1] >= 4.0
    assert figure.layout.scene.yaxis.range[0] <= -1.2
    room_y_max = max(
        float(box["translate"][1]) + float(box["scale"][1])
        for box in preview.room_boxes
    )
    assert figure.layout.scene.yaxis.range[1] >= room_y_max


def test_human_room_playback_restyles_only_the_moving_mesh():
    sequence = MeshSequence(
        vertices=np.asarray(
            [
                [[1.0, -0.2, 0.0], [1.0, 0.2, 0.0], [1.0, 0.0, 0.5]],
                [[1.2, -0.2, 0.0], [1.2, 0.2, 0.0], [1.2, 0.0, 0.5]],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
        times=np.asarray([0.0, 0.1]),
    )
    preview = SimpleNamespace(
        room_boxes=prepared_room_boxes(),
        mesh_sequence=sequence,
        scene_source_name="tutorial_bedroom",
        source_name="walk.npz",
    )
    figure = _human_room_scene_figure(
        preview,
        go,
        frame_index=0,
        plot_template="plotly_dark",
    )
    room_trace = figure.data[0]

    assert _update_human_room_scene_frame(
        figure,
        preview,
        frame_index=1,
    )

    human_trace = next(trace for trace in figure.data if trace.name == "Human mesh")
    assert np.allclose(np.asarray(human_trace.x), 1.2)
    assert figure.data[0] is room_trace
    assert figure.layout.title.text.endswith("frame 1")


def test_human_room_playback_does_not_relayout_during_camera_interaction():
    sequence = MeshSequence(
        vertices=np.asarray(
            [
                [[1.0, -0.2, 0.0], [1.0, 0.2, 0.0], [1.0, 0.0, 0.5]],
                [[1.2, -0.2, 0.0], [1.2, 0.2, 0.0], [1.2, 0.0, 0.5]],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
        times=np.asarray([0.0, 0.1]),
    )
    preview = SimpleNamespace(
        room_boxes=prepared_room_boxes(),
        mesh_sequence=sequence,
        scene_source_name="tutorial_bedroom",
        source_name="walk.npz",
    )
    figure = _human_room_scene_figure(
        preview,
        go,
        frame_index=0,
        plot_template="plotly_dark",
    )
    initial_title = figure.layout.title.text
    initial_camera = figure.layout.scene.camera.to_plotly_json()

    assert _update_human_room_scene_frame(
        figure,
        preview,
        frame_index=1,
        update_title=False,
    )

    human_trace = next(trace for trace in figure.data if trace.name == "Human mesh")
    assert np.allclose(np.asarray(human_trace.x), 1.2)
    assert figure.layout.title.text == initial_title
    assert figure.layout.scene.camera.to_plotly_json() == initial_camera


def test_dynamic_scene_camera_interaction_coalesces_playback_redraws():
    plot = pn.pane.Plotly(go.Figure())
    interaction = _dynamic_scene_interaction_component(pn, plot)
    script = interaction._scripts["render"]  # pylint: disable=protected-access
    remove_script = interaction._scripts[  # pylint: disable=protected-access
        "remove"
    ]

    assert interaction.children == [plot]
    assert interaction.name == "Dynamic scene camera interaction"
    assert 'plot.addEventListener("pointerdown"' in script
    assert 'plot.addEventListener("wheel"' in script
    assert 'window.addEventListener("pointerup"' in script
    assert "record.interacting" in script
    assert "record.pending = args" in script
    assert "record.inFlight" in script
    assert "runLatest(target, record)" in script
    assert 'eventData?.["scene.camera"]' in script
    assert '{"scene.camera": camera}' in script
    assert "originalUpdate(" in script
    assert 'plot.on("plotly_relayout", state.captureCamera)' in script
    assert 'plot.on("plotly_relayouting", state.captureCamera)' in script
    assert "requestAnimationFrame(() => requestAnimationFrame(state.flush))" in script
    assert "setTimeout(state.flush, 180)" in script
    assert 'plot.removeEventListener("pointerdown"' in remove_script
    assert 'plot.removeEventListener("wheel"' in remove_script
    assert 'removeListener?.("plotly_relayout"' in remove_script


def test_bundle_scene_evaluates_parameter_only_human_at_primary_time(monkeypatch):
    fixture = (
        Path(gui_app_module.__file__).resolve().parents[3]
        / "data"
        / "validation_bundles"
        / "rtpose_seq10_frame0020"
    )
    loaded = load_bundle(fixture)
    sequence = MeshSequence(
        vertices=np.asarray(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.0, 0.0, 0.4]],
                [[2.0, 0.0, 0.0], [2.0, 0.2, 0.0], [2.0, 0.0, 0.4]],
                [[3.0, 0.0, 0.0], [3.0, 0.2, 0.0], [3.0, 0.0, 0.4]],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
        times=np.asarray([-0.1, 0.0, 0.1]),
    )
    monkeypatch.setattr(
        gui_app_module,
        "load_amass_motion",
        lambda *_args, **_kwargs: SimpleNamespace(sequence=sequence),
    )

    figure = _bundle_scene_figure(
        loaded,
        go,
        smpl_model_dir="/licensed/smpl",
        plot_template="plotly_dark",
    )

    human = next(trace for trace in figure.data if trace.name == "Bundle human")
    assert np.allclose(np.asarray(human.x, dtype=float), 2.0)
    assert "Material: Human skin" in human.hovertemplate
    assert figure.layout.meta["bundle_mesh_source"] == "AMASS parameters"
    assert figure.layout.meta["bundle_has_target_preview"] is True
    assert figure.layout.meta["bundle_mesh_error"] is None


def test_bundle_scene_boxes_include_referenced_material(tmp_path):
    scene_path = tmp_path / "scene.xml"
    scene_path.write_text(
        """<scene version="3.0.0">
  <bsdf type="itu-radio-material" id="mat-wood">
    <string name="type" value="wood"/>
  </bsdf>
  <shape type="cube" id="desk">
    <transform name="to_world">
      <scale x="1" y="2" z="0.5"/>
      <translate x="3" y="0" z="1"/>
    </transform>
    <ref name="bsdf" id="mat-wood"/>
  </shape>
</scene>
""",
        encoding="utf-8",
    )

    boxes = _bundle_scene_boxes(scene_path)

    assert len(boxes) == 1
    assert boxes[0]["id"] == "desk"
    assert boxes[0]["material"] == "wood"
    assert boxes[0]["material_id"] == "mat-wood"


def test_human_room_results_group_modes_by_product_and_reset_plot_state():
    manifest = ExperimentManifest(
        name="human-only test",
        scene=SceneExperimentConfig(
            scenario="tutorial_bedroom",
            parameters={
                "motion_begin_frame_index": 4,
                "motion_end_frame_index": 5,
            },
        ),
        radar=RadarExperimentConfig(board_model="IWR6843AOPEVM"),
        solver=SolverExperimentConfig(mode="target_po"),
    )
    result = SimpleNamespace(
        simulation_mode="human_only_po",
        manifest=manifest,
        adc=np.ones((2, 2, 8, 1), dtype=np.complex64),
        adc_times_s=np.asarray([[0.0, 0.001], [0.05, 0.051]]),
        ranges_m=np.asarray([0.0, 0.5, 1.0]),
        range_profiles={"total": np.asarray([1.0, 2.0, 1.0])},
        range_time_power=np.ones((4, 3)),
        range_time_ranges_m=np.asarray([0.0, 0.5, 1.0]),
        range_doppler_power=np.ones((2, 2, 3)),
        range_doppler_ranges_m=np.asarray([0.0, 0.5, 1.0]),
        velocities_mps=np.asarray([-0.5, 0.5]),
        components={},
        runtime_s=0.25,
        metadata=SimpleNamespace(
            path_counts=np.ones((2, 2), dtype=np.int64),
            visible_face_counts=np.ones((2, 2), dtype=np.int64),
            po_calibration={"mode": "none", "applied": False},
            runtime_profile_s={"human_po": 0.2},
        ),
    )

    coherent = SimpleNamespace(**vars(result))
    coherent.simulation_mode = "coherent_rt"
    coherent.manifest = ExperimentManifest(
        name="coherent test",
        scene=manifest.scene,
        radar=manifest.radar,
        solver=SolverExperimentConfig(mode="rt_scattering"),
    )
    coherent.range_profiles = {
        "total": np.asarray([0.5, 1.5, 0.5])
    }
    coherent.metadata = SimpleNamespace(
        **vars(result.metadata),
        path_depth_histogram=np.asarray(
            [
                [[0, 2, 1], [0, 1, 3]],
                [[0, 4, 2], [0, 2, 1]],
            ],
            dtype=np.int64,
        ),
    )
    view = _human_room_results_view(
        {"coherent_rt": coherent, "human_only_po": result},
        pn,
        go,
        plot_template="plotly_dark",
    )
    product_tabs = next(
        child
        for child in view.select(pn.Tabs)
        if child._names == ["Range profile", "Range-time", "Range-Doppler"]
    )
    profile = product_tabs.objects[0]
    profile.relayout_data = {"yaxis.autorange": True}
    product_tabs.active = 1
    product_tabs.active = 0

    assert profile.relayout_data == {}
    assert profile.object.layout.title.text == "Range profile comparison"
    assert [trace.name for trace in profile.object.data] == [
        "Coherent RT",
        "Human-only PO",
    ]
    assert len(product_tabs.objects[1].select(pn.pane.Plotly)) == 2
    assert len(product_tabs.objects[2].select(pn.pane.Plotly)) == 2
    assert not any(
        set(getattr(child, "_names", []))
        & {"Coherent RT", "Human-only PO"}
        for child in view.select(pn.Tabs)
    )
    range_doppler_frame = next(
        widget
        for widget in view.select(pn.widgets.IntSlider)
        if widget.label.startswith("Range-Doppler frame (")
    )
    assert range_doppler_frame.label == "Range-Doppler frame (1–2)"
    assert range_doppler_frame.start == 1
    assert range_doppler_frame.end == 2
    assert range_doppler_frame.value == 1
    assert range_doppler_frame.disabled is False
    previous_button = next(
        widget
        for widget in view.select(pn.widgets.Button)
        if widget.label == "Previous frame"
    )
    next_button = next(
        widget
        for widget in view.select(pn.widgets.Button)
        if widget.label == "Next frame"
    )
    assert previous_button.disabled is True
    assert next_button.disabled is False
    next_button.clicks += 1
    assert range_doppler_frame.value == 2
    assert previous_button.disabled is False
    assert next_button.disabled is True
    previous_button.clicks += 1
    assert range_doppler_frame.value == 1
    range_doppler_frame.value = 2
    range_doppler_panes = product_tabs.objects[2].select(pn.pane.Plotly)
    assert all(
        "simulated frame 2" in pane.object.layout.title.text
        for pane in range_doppler_panes
    )
    assert all(
        "source motion frame 5" in pane.object.layout.title.text
        for pane in range_doppler_panes
    )
    rt_histogram = next(
        pane.object
        for pane in view.select(pn.pane.Plotly)
        if pane.object.layout.title.text == "RT path segment-count histogram"
    )
    assert [trace.name for trace in rt_histogram.data] == ["Coherent RT"]
    assert list(rt_histogram.data[0].x) == [1, 2, 3]
    assert list(rt_histogram.data[0].y) == [0, 9, 7]

    hybrid = SimpleNamespace(**vars(result))
    hybrid.simulation_mode = "hybrid_po"
    hybrid.manifest = ExperimentManifest(
        name="hybrid test",
        scene=manifest.scene,
        radar=manifest.radar,
        solver=SolverExperimentConfig(mode="hybrid_rt_po"),
    )
    hybrid.range_profiles = {
        "total": np.asarray([2.0, 4.0, 2.0]),
        "human_po": np.asarray([1.0, 2.0, 1.0]),
        "static_environment_blocked": np.asarray([0.5, 1.0, 0.5]),
        "human_env": np.asarray([0.25, 0.5, 0.25]),
        "env_human": np.asarray([0.125, 0.25, 0.125]),
    }
    hybrid.components = {
        name: np.ones_like(result.adc)
        for name in (
            "human_po",
            "static_environment_blocked",
            "human_env",
            "env_human",
        )
    }
    hybrid_view = _human_room_results_view(
        {"hybrid_po": hybrid},
        pn,
        go,
        plot_template="plotly_dark",
    )
    hybrid_tabs = next(
        child
        for child in hybrid_view.select(pn.Tabs)
        if child._names == ["Range profile", "Range-time", "Range-Doppler"]
    )
    hybrid_profile_figures = [
        pane.object
        for pane in hybrid_tabs.objects[0].select(pn.pane.Plotly)
    ]
    component_figure = next(
        figure
        for figure in hybrid_profile_figures
        if figure.layout.title.text
        == "Hybrid PO component range profiles"
    )
    assert [trace.name for trace in component_figure.data] == [
        "Human only",
        "Environment RT with human blockage",
        "Human → environment",
        "Environment → human",
    ]


def test_radar_change_refreshes_tx_and_rx_choices():
    application = build_app()
    settings = application.modal[0]
    board = next(
        widget
        for widget in settings.select(pn.widgets.Select)
        if widget.label == "TI radar"
    )
    tx_selector = next(
        widget
        for widget in settings.select(pn.widgets.MultiChoice)
        if widget.label == "Active Tx antennas"
    )
    rx_selector = next(
        widget
        for widget in settings.select(pn.widgets.MultiChoice)
        if widget.label == "Active Rx antennas"
    )
    callback = next(
        watcher.fn
        for watcher in board.param.watchers["value"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_board_selection_change"
    )
    document = Document()
    tx_model = tx_selector.get_root(document)
    rx_model = rx_selector.get_root(document)

    import param

    with param.parameterized.discard_events(board):
        board.value = "MMWCAS_RF_EVM"
    callback(SimpleNamespace(obj=board))

    assert list(tx_selector.options.values()) == list(range(12))
    assert list(rx_selector.options.values()) == list(range(16))
    assert tx_selector.value == [0, 1, 2, 9]
    assert rx_selector.value == [4, 5, 6, 7]
    assert len(tx_model.options) == 12
    assert len(rx_model.options) == 16
    assert tx_model.value == ["TX1", "TX2", "TX3", "TX10"]
    assert rx_model.value == ["RX5", "RX6", "RX7", "RX8"]

    with param.parameterized.discard_events(board):
        board.value = "IWR6843AOPEVM"
    callback(SimpleNamespace(obj=board))

    assert list(tx_selector.options.values()) == [0, 1, 2]
    assert list(rx_selector.options.values()) == [0, 1, 2, 3]
    assert tx_selector.value == [0, 1, 2]
    assert rx_selector.value == [0, 1, 2, 3]
    assert len(tx_model.options) == 3
    assert len(rx_model.options) == 4
    assert len(tx_model.value) == 3
    assert len(rx_model.value) == 4


def test_angle_fft_defaults_to_target_centroid_and_compares_rt_po():
    result = run_static_target_experiment(
        target_type="plate",
        range_m=2.0,
        target_y_m=0.3,
        target_z_m=0.4,
        width_m=0.3,
        height_m=0.3,
        tdm_enabled=True,
    )
    products = _angle_fft_range_products(result)
    centroid_bin = int(products["centroid_range_bin"])
    ranges_m = np.asarray(products["ranges_m"])
    assert products["fmcw"].tdm_enabled is True

    expected_range_m = np.sqrt(4.25)
    assert float(products["centroid_range_m"]) == pytest.approx(
        expected_range_m
    )
    assert centroid_bin == int(
        np.argmin(np.abs(ranges_m - expected_range_m))
    )

    figure = _angle_map_figure(
        products,
        make_subplots,
        go,
        range_bins=(centroid_bin, centroid_bin),
        plot_template="plotly_dark",
    )

    assert len(figure.data) == 2
    assert "absolute simulated power" in figure.layout.title.text
    assert "normalization" not in figure.layout.title.text
    assert f"bin {centroid_bin}" in figure.layout.title.text
    po_power, _u_axis, _v_axis = _selected_angle_fft_power(
        products,
        solver="po",
        first_bin=centroid_bin,
        last_bin=centroid_bin,
    )
    assert np.nanmax(np.asarray(figure.data[0].z, dtype=float)) == pytest.approx(
        10.0 * np.log10(np.max(po_power))
    )
    assert figure.data[0].zmin == figure.data[1].zmin
    assert figure.data[0].zmax == figure.data[1].zmax
    assert np.isnan(np.asarray(figure.data[0].z, dtype=float)[0, 0])
    assert "u %{x:.3f}" in figure.data[0].hovertemplate
    assert "Azimuth %{customdata[0]:.1f}°" in figure.data[0].hovertemplate

    range_figure = _range_profile_figure(
        result,
        go,
        plot_template="plotly_dark",
    )
    assert np.max(np.asarray(range_figure.data[0].y)) == pytest.approx(
        10.0 * np.log10(np.max(result.po_range_profile_power))
    )
    rt_peak = float(np.max(result.rt_range_profile_power))
    if rt_peak > 0.0:
        assert np.max(np.asarray(range_figure.data[1].y)) == pytest.approx(
            10.0 * np.log10(rt_peak)
        )
    else:
        assert np.max(np.asarray(range_figure.data[1].y)) < -3_000.0


def test_angle_range_bin_selection_updates_map_without_rerunning_simulation():
    application = build_app()
    tabs = application.main[0]
    angle_bins = next(
        widget
        for widget in tabs.select(pn.widgets.IntRangeSlider)
        if widget.label == "Angle-FFT range bins (inclusive)"
    )
    centroid_bin = angle_bins.value[0]
    angle_bins.value = (centroid_bin - 1, centroid_bin + 1)
    callback = next(
        watcher.fn
        for watcher in angle_bins.param.watchers["value_throttled"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_angle_range_change"
    )

    asyncio.run(callback(None))

    angle_figure = _angle_map_pane(tabs).object
    assert f"bins {centroid_bin - 1}–{centroid_bin + 1}" in (
        angle_figure.layout.title.text
    )


def test_shift_drag_bridge_previews_target_without_hijacking_camera():
    application = build_app()
    tabs = application.main[0]
    interaction = next(
        component
        for component in tabs.select(pn.reactive.ReactiveHTML)
        if "target_gesture" in component.param
    )
    script = interaction._scripts["render"]  # pylint: disable=protected-access

    assert 'event.key !== "Shift"' in script
    assert 'target_overlay.addEventListener("pointerdown"' in script
    assert 'target_overlay.addEventListener("mousedown"' not in script
    assert 'window.addEventListener("keydown"' in script
    assert 'window.addEventListener("pointermove"' in script
    assert 'window.addEventListener("mousemove"' not in script
    assert "setPointerCapture" in script
    assert "state.findPlot(scene)" in script
    assert 'target_overlay.style.pointerEvents = "auto"' in script
    assert "window.Plotly.restyle" in script
    assert "window.Plotly.relayout" not in script
    assert "scene.camera" not in script
    assert "state.previewInFlight" in script
    assert "state.previewPending" in script
    assert "state.finalPreview" in script
    assert "applySettledPreview" in script
    assert "requestPreview()" in script
    assert "(event.buttons & 1) === 0" in script
    assert '"lostpointercapture"' in script
    assert "Replace every intermediate request" in script
    assert "emitGesture(false)" not in script
    assert script.count("emitGesture()") == 1
    assert "event.stopImmediatePropagation()" in script
    update_script = interaction._scripts[  # pylint: disable=protected-access
        "target_result"
    ]
    assert "window.Plotly.restyle" in update_script
    assert "Plotly.relayout" not in update_script


def test_radar_float_input_noise_is_normalized():
    assert _rounded_gui_float(60.400000000000006) == 60.4
    assert _rounded_target_size(0.17999999999999988) == 0.18


def test_static_experiment_honors_preflight_cancellation():
    with pytest.raises(InterruptedError, match="cancelled by user"):
        run_static_target_experiment(cancel_check=lambda: True)


def test_untrusted_markdown_values_are_rendered_as_text():
    payload = '<img src=x onerror="window.pwned=true"> `escape`'

    assert _safe_markdown_text(payload).startswith("&lt;img")
    safe_code = _safe_markdown_code(payload)
    assert "<img" not in safe_code
    assert "&lt;img" in safe_code
    pane = pn.pane.Markdown(f"**Failed:** {safe_code}")
    model = pane._get_model(Document())  # pylint: disable=protected-access

    assert "<img" not in model.text
    assert "&amp;amp;lt;img" in model.text


def test_static_failure_invalidates_previous_export(monkeypatch):
    application = build_app()
    tabs = application.main[0]
    width = next(
        widget
        for widget in tabs.select(pn.widgets.FloatInput)
        if widget.label == "Plate width [m]"
    )
    callback = next(
        watcher.fn
        for watcher in width.param.watchers["value"]["value"]
        if getattr(watcher.fn, "__name__", "")
        == "on_physics_control_change"
    )
    update_physics = inspect.getclosurevars(callback).nonlocals["update_physics"]
    update_closure = inspect.getclosurevars(update_physics).nonlocals
    export_button = next(
        button
        for button in tabs.select(pn.widgets.FileDownload)
        if button.filename == "hermes-static-target-bundle.zip"
    )

    def fail_update(**_kwargs):
        raise ValueError('<img src=x onerror="boom">')

    monkeypatch.setattr(
        gui_app_module,
        "run_static_target_experiment",
        fail_update,
    )
    asyncio.run(callback(SimpleNamespace(obj=width)))

    assert update_closure["physics_state"]["result"] is None
    assert export_button.disabled is True
    assert export_button.callback().getvalue() == b""
    status = str(update_closure["physics_status"].object)
    assert "Update failed" in status
    assert "<img" not in status
    assert "&lt;img" in status


def test_static_run_can_be_stopped_at_a_safe_boundary(monkeypatch):
    application = build_app()
    tabs = application.main[0]
    width = next(
        widget
        for widget in tabs.select(pn.widgets.FloatInput)
        if widget.label == "Plate width [m]"
    )
    update_callback = next(
        watcher.fn
        for watcher in width.param.watchers["value"]["value"]
        if getattr(watcher.fn, "__name__", "")
        == "on_physics_control_change"
    )
    stop_button = next(
        button
        for button in tabs.select(pn.widgets.Button)
        if button.description.startswith("Stop the active static")
    )
    stop_callback = next(
        watcher.fn
        for watcher in stop_button.param.watchers["clicks"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_stop_physics"
    )
    update_closure = inspect.getclosurevars(
        inspect.getclosurevars(update_callback).nonlocals["update_physics"]
    ).nonlocals
    started = threading.Event()

    def run_until_cancelled(*, cancel_check, **_kwargs):
        started.set()
        while not cancel_check():
            threading.Event().wait(0.005)
        raise InterruptedError("Simulation cancelled by user")

    monkeypatch.setattr(
        gui_app_module,
        "run_static_target_experiment",
        run_until_cancelled,
    )

    async def stop_active_run():
        task = asyncio.create_task(
            update_callback(SimpleNamespace(obj=width))
        )
        while not started.is_set():
            await asyncio.sleep(0.005)
        stop_callback(None)
        await task

    asyncio.run(stop_active_run())

    assert update_closure["physics_state"]["running"] is False
    assert update_closure["physics_state"]["result"] is None
    assert stop_button.disabled is True
    assert "STOPPED" in str(update_closure["physics_status"].object)


def test_display_only_static_controls_keep_theme_result_cache():
    application = build_app()
    tabs = application.main[0]
    storage = next(
        component
        for component in application.header[0].select(pn.reactive.ReactiveHTML)
        if "loaded_state" in component.param
    )
    angle_bins = next(
        widget
        for widget in tabs.select(pn.widgets.IntRangeSlider)
        if widget.label == "Angle-FFT range bins (inclusive)"
    )
    top_paths = next(
        widget
        for widget in tabs.select(pn.widgets.IntInput)
        if widget.label == "Top RT paths"
    )

    selected = int(angle_bins.value[0])
    angle_bins.value = (selected, min(selected + 1, angle_bins.end))
    top_paths.value += 1
    cached = _get_theme_gui_results(
        storage.saved_state["gui_result_cache_key"]
    )

    assert cached is not None
    assert cached["static_result"] is not None


def test_target_size_inputs_use_two_decimal_precision():
    application = build_app()
    tabs = application.main[0]
    size_inputs = {
        widget.label: widget
        for widget in tabs.select(pn.widgets.FloatInput)
        if widget.label
        in {
            "Plate width [m]",
            "Plate height [m]",
            "Corner edge length [m]",
        }
    }
    assert set(size_inputs) == {
        "Plate width [m]",
        "Plate height [m]",
        "Corner edge length [m]",
    }
    assert all(widget.format == "0.00" for widget in size_inputs.values())

    width = size_inputs["Plate width [m]"]
    width.value = 0.17999999999999988
    callback = next(
        watcher.fn
        for watcher in width.param.watchers["value"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_physics_control_change"
    )

    asyncio.run(callback(SimpleNamespace(obj=width)))

    assert width.value == 0.18


def test_trihedral_scene_has_stable_frame_and_po_power_colorbar():
    figures = []
    results = []
    for yaw_deg, pitch_deg in ((0.0, 0.0), (37.0, -24.0)):
        vertices, faces = make_trihedral_corner_mesh(
            range_m=2.0,
            edge_m=0.2,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            max_edge_m=0.05,
        )
        result = SimpleNamespace(
            vertices=vertices,
            faces=faces,
            face_power=np.linspace(0.0, 1.0, faces.shape[0]),
            target_type="trihedral",
            target_parameters={
                "range_m": 2.0,
                "yaw_deg": yaw_deg,
                "pitch_deg": pitch_deg,
                "roll_deg": 0.0,
            },
        )
        results.append(result)
        figures.append(
            _physics_scene_figure(result, go, plot_template="plotly_dark")
        )

    mesh = figures[0].data[0]
    assert mesh.facecolor == tuple(
        gui_app_module._face_colors(results[0].face_power)
    )
    colorbar = next(
        trace
        for trace in figures[0].data
        if trace.name == "PO relative facet power"
    )
    assert colorbar.marker.colorscale is not None
    assert colorbar.marker.cmin == pytest.approx(-60.0)
    assert colorbar.marker.cmax == pytest.approx(0.0)
    assert colorbar.marker.showscale is True
    assert colorbar.marker.colorbar.title.text == "PO relative<br>power [dB]"
    assert mesh.lighting.ambient == pytest.approx(0.88)
    assert mesh.lighting.diffuse == pytest.approx(0.30)
    assert mesh.lighting.specular == pytest.approx(0.0)
    assert mesh.lightposition.x == pytest.approx(-1000.0)
    assert mesh.lightposition.y == pytest.approx(-1000.0)
    assert mesh.lightposition.z == pytest.approx(1000.0)
    for axis in ("xaxis", "yaxis", "zaxis"):
        initial_range = tuple(getattr(figures[0].layout.scene, axis).range)
        rotated_range = tuple(getattr(figures[1].layout.scene, axis).range)
        assert rotated_range == pytest.approx(initial_range)
        assert getattr(figures[1].layout.scene, axis).dtick == pytest.approx(
            getattr(figures[0].layout.scene, axis).dtick
        )

    update = _target_trace_update(results[1], sequence=3)
    assert update["sequence"] == 3
    assert set(update) == {"sequence", "x", "y", "z", "facecolor", "meta"}
    post_gesture_update = _target_trace_update(
        results[1],
        sequence=4,
        include_geometry=False,
    )
    assert set(post_gesture_update) == {"sequence", "facecolor", "meta"}


def test_human_scene_shows_only_po_power_and_lights_from_radar():
    vertices = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.2, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
        dtype=np.uint32,
    )
    result = SimpleNamespace(
        vertices=vertices,
        faces=faces,
        face_power=np.asarray([1.0, 0.0, 0.25, 0.0]),
        rt_face_hit_counts=np.asarray([0, 2, 1, 0]),
        target_type="human_mesh",
        target_parameters={
            "range_m": 2.0,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        },
        radar_position=np.zeros(3),
    )

    figure = _physics_scene_figure(
        result,
        go,
        plot_template="plotly_dark",
    )

    mesh = figure.data[0]
    assert mesh.facecolor == tuple(
        gui_app_module._face_colors(result.face_power)
    )
    assert mesh.lightposition.x == pytest.approx(0.0)
    assert mesh.lightposition.y == pytest.approx(0.0)
    assert mesh.lightposition.z == pytest.approx(0.0)
    assert not any(trace.showlegend for trace in figure.data)
    assert figure.layout.title.text == "PO facet contribution · Human Mesh"


def test_solver_diagnostics_summary_omits_duplicate_path_and_facet_tables():
    diagnostics = SimpleNamespace(
        rt=SimpleNamespace(valid_link_path_count=7),
        po=SimpleNamespace(visible_face_count=13),
    )

    summary = _solver_diagnostics_summary(diagnostics)

    assert "RT valid link-paths:** `7`" in summary
    assert "PO visible facets:** `13`" in summary
    assert "Strongest target-touching RT paths" not in summary
    assert "Strongest PO facets" not in summary
    assert "|" not in summary


def test_scene_radar_pose_vectors_show_yaw_pitch_and_roll():
    vertices, faces = make_trihedral_corner_mesh(
        range_m=2.0,
        edge_m=0.2,
        max_edge_m=0.05,
    )
    result = SimpleNamespace(
        vertices=vertices,
        faces=faces,
        face_power=np.ones(faces.shape[0]),
        target_type="trihedral",
        target_parameters={
            "range_m": 2.0,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        },
        sensor_parameters={
            "orientation": list(np.deg2rad((90.0, 0.0, 90.0))),
        },
    )

    figure = _physics_scene_figure(
        result,
        go,
        plot_template="plotly_dark",
    )

    boresight = next(
        trace for trace in figure.data if trace.name == "Radar boresight"
    )
    local_up = next(
        trace for trace in figure.data if trace.name == "Radar local up"
    )
    assert boresight.x[-1] == pytest.approx(0.0, abs=1e-10)
    assert boresight.y[-1] > 0.0
    assert boresight.z[-1] == pytest.approx(0.0)
    assert local_up.x[-1] > 0.0
    assert local_up.z[-1] == pytest.approx(0.0, abs=1e-10)


def test_shift_drag_updates_orientation_controls_and_recomputes_target():
    application = build_app()
    tabs = application.main[0]
    scene = _scene_pane(tabs)
    yaw = next(
        slider
        for slider in tabs.select(pn.widgets.FloatSlider)
        if slider.label == "Yaw / aspect [deg]"
    )
    pitch = next(
        slider
        for slider in tabs.select(pn.widgets.FloatSlider)
        if slider.label == "Pitch [deg]"
    )
    interaction = next(
        component
        for component in tabs.select(pn.reactive.ReactiveHTML)
        if "target_gesture" in component.param
    )
    yaw_value_events = []
    pitch_value_events = []
    yaw_throttled_events = []
    pitch_throttled_events = []
    yaw.param.watch(yaw_value_events.append, "value")
    pitch.param.watch(pitch_value_events.append, "value")
    yaw.param.watch(yaw_throttled_events.append, "value_throttled")
    pitch.param.watch(pitch_throttled_events.append, "value_throttled")
    before = np.asarray(scene.object.data[0].x, dtype=float)
    event = SimpleNamespace(
        new={
            "sequence": 1,
            "yaw_deg": -25.0,
            "pitch_deg": 10.0,
        }
    )
    callback = next(
        watcher.fn
        for watcher in interaction.param.watchers["target_gesture"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_target_gesture"
    )

    asyncio.run(callback(event))

    assert yaw.value == pytest.approx(-25.0)
    assert pitch.value == pytest.approx(10.0)
    assert [event.new for event in yaw_value_events] == [-25.0]
    assert [event.new for event in pitch_value_events] == [10.0]
    assert yaw_throttled_events == []
    assert pitch_throttled_events == []
    assert "x" not in interaction.target_result
    assert interaction.target_result["meta"]["yaw_deg"] == pytest.approx(-25.0)
    assert interaction.target_result["meta"]["pitch_deg"] == pytest.approx(10.0)
    assert np.allclose(np.asarray(scene.object.data[0].x, dtype=float), before)


def test_saved_solver_settings_are_validated_restored_and_normalized():
    assert _validated_solver_settings_payload(
        {
            "schema_version": 1,
            "settings": {
                "po_integration_mode": "face_centroid",
                "po_quadrature_phase_span_scale_rad": 0.75,
                "po_quadrature_max_refinement_depth": 1,
                "po_quadrature_max_subfaces_per_parent": 8,
                "rt_samples_per_source": 600,
                "rt_max_paths_per_source": 700,
                "rt_max_depth": 2,
            },
        }
    ) == {
        "po_integration_mode": "face_centroid",
        "po_quadrature_phase_span_scale_rad": 0.75,
        "po_quadrature_max_refinement_depth": 1,
        "po_quadrature_max_subfaces_per_parent": 8,
        "rt_samples_per_source": 600,
        "rt_max_paths_per_source": 700,
        "rt_max_depth": 2,
    }
    assert _validated_solver_settings_payload(
        {
            "schema_version": 1,
            "settings": {
                "po_integration_mode": "unsupported",
                "rt_samples_per_source": -1,
                "rt_max_depth": True,
            },
        }
    ) == {}
    assert _validated_solver_settings_payload(
        {"schema_version": 0, "settings": {}}
    ) == {}

    application = build_app()
    solver_settings = application.modal[0]
    settings_storage = next(
        component
        for component in application.header[0].select(
            pn.reactive.ReactiveHTML
        )
        if "loaded_settings" in component.param
    )
    restored_payload = {
        "schema_version": 1,
        "settings": {
            "po_integration_mode": "face_centroid",
            "po_quadrature_phase_span_scale_rad": 0.75,
            "po_quadrature_max_refinement_depth": 1,
            "po_quadrature_max_subfaces_per_parent": 8,
            "rt_samples_per_source": 600,
            "rt_max_paths_per_source": 700,
            "rt_max_depth": 2,
        },
    }
    callback = next(
        watcher.fn
        for watcher in settings_storage.param.watchers[
            "loaded_settings"
        ]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_solver_settings_loaded"
    )

    asyncio.run(callback(SimpleNamespace(new=restored_payload)))

    solver_widgets = [
        *solver_settings.select(pn.widgets.Select),
        *solver_settings.select(pn.widgets.FloatInput),
        *solver_settings.select(pn.widgets.IntInput),
    ]
    values_by_label = {
        widget.label: widget.value for widget in solver_widgets
    }
    assert values_by_label["PO facet integration"] == "face_centroid"
    assert values_by_label["PO subdivision phase span [rad]"] == 0.75
    assert values_by_label["PO maximum refinement depth"] == 1
    assert values_by_label["PO subfaces per parent (0 = unlimited)"] == 8
    assert values_by_label["RT ray samples per source"] == 600
    assert values_by_label["RT maximum paths per source"] == 700
    assert values_by_label["RT maximum path depth"] == 2
    assert settings_storage.saved_settings == restored_payload


def test_physics_controls_recompute_the_interactive_preview():
    application = build_app()
    tabs = application.main[0]
    aspect = next(
        slider
        for slider in tabs.select(pn.widgets.FloatSlider)
        if slider.label == "Yaw / aspect [deg]"
    )
    scene = _scene_pane(tabs)
    aspect.value = 35
    callbacks = aspect.param.watchers["value_throttled"]["value"]
    callback = next(
        watcher.fn
        for watcher in callbacks
        if getattr(watcher.fn, "__name__", "") == "on_physics_control_change"
    )

    asyncio.run(callback(None))

    assert np.ptp(np.asarray(scene.object.data[0].x, dtype=float)) > 0.1

    target_y = next(
        slider
        for slider in tabs.select(pn.widgets.FloatSlider)
        if slider.label == "Target y / lateral-left [m]"
    )
    target_y.value = 0.35
    target_position_callback = next(
        watcher.fn
        for watcher in target_y.param.watchers["value_throttled"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_physics_control_change"
    )

    asyncio.run(
        target_position_callback(SimpleNamespace(obj=target_y))
    )

    target_trace = scene.object.data[0]
    assert np.mean(np.asarray(target_trace.y, dtype=float)) == pytest.approx(
        0.35
    )
    assert tuple(target_trace.meta["orientation_pivot"]) == pytest.approx(
        (2.0, 0.35, 0.0)
    )

    radar_yaw = next(
        slider
        for slider in tabs.select(pn.widgets.FloatSlider)
        if slider.label == "Radar yaw [deg]"
    )
    radar_yaw.value = 30.0
    radar_callback = next(
        watcher.fn
        for watcher in radar_yaw.param.watchers["value_throttled"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_physics_control_change"
    )

    asyncio.run(radar_callback(SimpleNamespace(obj=radar_yaw)))

    boresight = next(
        trace for trace in scene.object.data if trace.name == "Radar boresight"
    )
    assert boresight.x[-1] > 0.0
    assert boresight.y[-1] > 0.0


def test_material_selection_updates_diffuse_scattering_description():
    application = build_app()
    tabs = application.main[0]
    material = next(
        widget
        for widget in tabs.select(pn.widgets.Select)
        if widget.label == "Material preset"
    )
    material.value = "concrete"
    callback = next(
        watcher.fn
        for watcher in material.param.watchers["value"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_material_change"
    )

    asyncio.run(callback(SimpleNamespace(obj=material)))

    assert any(
        "RT diffuse scattering coefficient:** `0.35`" in str(pane.object)
        and "Backscattering lobe mixture λ:** `0.35`" in str(pane.object)
        for pane in tabs.select(pn.pane.Markdown)
    )


def test_selecting_human_uses_internal_default_mesh():
    default_mesh = _default_human_mesh_path()
    if default_mesh is None:
        pytest.skip("private source-checkout mesh is not installed")
    assert default_mesh.name == "mesh_sequence.npz"
    assert default_mesh.parent.name == "rtpose_seq186_frame0020"

    application = build_app()
    tabs = application.main[0]
    selector = next(
        widget
        for widget in tabs.select(pn.widgets.Select)
        if widget.label == "Static object"
    )
    selector.value = "human_mesh"
    callback = next(
        watcher.fn
        for watcher in selector.param.watchers["value"]["value"]
        if getattr(watcher.fn, "__name__", "") == "on_target_type_change"
    )

    asyncio.run(callback(None))

    scene = _scene_pane(tabs)
    assert len(scene.object.data[0].x) == 6890
    assert scene.object.data[0].name == "Human Mesh"
    assert any(
        "RT-Pose seq186 frame0020" in str(pane.object)
        for pane in tabs.select(pn.pane.Markdown)
    )


def test_gui_export_is_a_loadable_static_target_bundle(tmp_path):
    result = run_static_target_experiment(
        target_type="plate",
        yaw_deg=0.0,
        target_y_m=-0.25,
        target_z_m=0.5,
        radar_yaw_deg=8.0,
        radar_pitch_deg=-3.0,
        radar_roll_deg=5.0,
        selected_tx_indices=(1,),
        selected_rx_indices=(0, 2),
        antenna_pattern_mode="cosine",
        cosine_3db_beamwidth_deg=44.0,
    )
    path = tmp_path / "static.zip"
    path.write_bytes(_physics_export(result).getvalue())
    with zipfile.ZipFile(path) as archive:
        assert {
            "bundle.json",
            "sensor.json",
            "frames.json",
            "radar_adc.npz",
            "simulated_adc_rt.npz",
            "simulated_adc_po.npz",
            "target_mesh.npz",
            "target.obj",
            "scene.xml",
            "experiment_manifest.json",
        }.issubset(archive.namelist())
        assert b'type="obj"' in archive.read("scene.xml")
        sensor = json.loads(archive.read("sensor.json"))
        assert sensor["orientation_deg"] == [8.0, -3.0, 5.0]
        assert sensor["pattern_mode"] == "cosine"
        assert sensor["cosine_3db_beamwidth_deg"] == 44.0
        assert sensor["tx_indices_zero_based"] == [1]
        assert sensor["rx_indices_zero_based"] == [0, 2]
        summary = json.loads(archive.read("bundle_summary.json"))
        environment = json.loads(archive.read("environment.json"))
        assert summary["target_position_m"] == [2.0, -0.25, 0.5]
        assert environment["target_anchor_m"] == [2.0, -0.25, 0.5]
        with np.load(
            BytesIO(archive.read("solver_diagnostics.npz")),
            allow_pickle=False,
        ) as diagnostics:
            assert diagnostics["rt_face_hit_counts"].shape == (
                result.faces.shape[0],
            )

    loaded = load_bundle(path)

    assert loaded.profile == "hermes"
    assert loaded.data_origin == "simulation"
    assert loaded.available_simulations == ("po", "rt")
    assert loaded.can_resimulate is False
    assert loaded.integrity_report["static_target"]["present"] is True
    assert loaded.load_bundled_simulation("rt").shape == loaded.primary_adc.shape
    scene = _bundle_scene_figure(
        loaded,
        go,
        plot_template="plotly_dark",
    )
    target = next(trace for trace in scene.data if trace.name == "Bundle target")
    assert "Material: pec" in target.hovertemplate


def test_prepared_room_export_preserves_rough_wall_scattering():
    result = SimpleNamespace(
        scene_xml=None,
        preview=SimpleNamespace(
            scene_xml=None,
            room_boxes=prepared_room_boxes(),
        ),
    )

    scene_xml = _human_room_scene_xml(result).decode("utf-8")

    assert scene_xml.count(
        '<float name="scattering_coefficient" value="0.2"/>'
    ) == 2
    assert (
        '<bsdf type="itu-radio-material" id="mat-metal">'
        '<string name="type" value="metal"/>'
        '<float name="scattering_coefficient" value="0.2"/>'
        "</bsdf>"
    ) in scene_xml
    assert (
        '<bsdf type="itu-radio-material" id="mat-plasterboard">'
        '<string name="type" value="plasterboard"/>'
        '<float name="scattering_coefficient" value="0.2"/>'
        "</bsdf>"
    ) in scene_xml


def test_human_room_export_is_loadable_and_keeps_radar_at_origin(tmp_path):
    vertices = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.0, 0.4],
            ],
            [
                [0.02, 0.0, 0.0],
                [0.22, 0.0, 0.0],
                [0.02, 0.2, 0.0],
                [0.02, 0.0, 0.4],
            ],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
        dtype=np.uint32,
    )
    preview = prepare_human_room_preview(
        MeshSequence(
            vertices=vertices,
            faces=faces,
            times=np.asarray([0.0, 1.0 / 30.0]),
        ),
        human_motion_filename="test_motion.npz",
        human_position_m=(1.5, 0.2, 0.1),
        human_yaw_deg=20.0,
    )
    custom_scene = b'<scene version="2.1.0"></scene>'
    preview = replace(
        preview,
        room_boxes=(),
        motion_payload={
            "poses": np.zeros((2, 72), dtype=np.float32),
            "trans": np.zeros((2, 3), dtype=np.float32),
            "betas": np.zeros(10, dtype=np.float32),
            "times": np.asarray([0.0, 1.0 / 30.0]),
            "bundle_times": np.asarray([0.0, 1.0 / 30.0]),
            "faces": faces,
            "gender": np.asarray("neutral"),
            "model_type": np.asarray("smpl"),
        },
        scene_xml=custom_scene,
        scene_source_name="lab.xml",
    )
    radar_parameters = {
        "position_m": [0.0, 0.0, 0.0],
        "orientation_deg": [10.0, -3.0, 2.0],
        "carrier_frequency_hz": 60e9,
        "slope_hz_per_s": 68e12,
        "chirp_duration_s": 58e-6,
        "chirp_repetition_time_s": 65e-6,
        "sampling_frequency_hz": 4.5e6,
        "num_adc_samples": 8,
        "num_chirps_per_frame": 2,
        "frame_period_s": 50e-3,
        "tdm_enabled": True,
        "tx_indices_zero_based": [0, 1, 2],
        "rx_indices_zero_based": [0, 1, 2, 3],
        "antenna_pattern_mode": "cosine",
        "cosine_3db_beamwidth_deg": 60.0,
    }
    manifest = ExperimentManifest(
        name="human room test",
        scene=SceneExperimentConfig(
            scenario="tutorial_bedroom",
            parameters={
                "scene_editable": False,
                "source_motion_frame_count": 2,
                "motion_begin_frame_index": 1,
                "motion_end_frame_index": 1,
            },
        ),
        radar=RadarExperimentConfig(
            board_model="IWR6843AOPEVM",
            profile="gui_human_room",
            parameters=radar_parameters,
        ),
        solver=SolverExperimentConfig(mode="hybrid_rt_po"),
    )
    adc = np.ones((1, 2, 8, 12), dtype=np.complex64)
    times = np.asarray([[0.0, 65e-6]])
    result = SimpleNamespace(
        simulation_mode="hybrid_po",
        manifest=manifest,
        preview=preview,
        radar_orientation_rad=np.deg2rad((10.0, -3.0, 2.0)),
        adc=adc,
        adc_times_s=times,
        ranges_m=np.linspace(0.0, 2.0, 8),
        range_profiles={"total": np.ones(8), "human_po": np.ones(8) * 0.25},
        range_time_power=np.ones((2, 8)),
        range_time_ranges_m=np.linspace(0.0, 2.0, 8),
        range_doppler_power=np.ones((1, 2, 8)),
        range_doppler_ranges_m=np.linspace(0.0, 2.0, 8),
        velocities_mps=np.asarray([-0.5, 0.5]),
        components={
            "human_po": adc * 0.25,
            "static_environment_blocked": adc * 0.75,
        },
        metadata=SimpleNamespace(
            path_counts=np.ones((1, 2), dtype=np.int64),
            visible_face_counts=np.ones((1, 2), dtype=np.int64),
            effective_visible_area_fraction=np.ones((1, 2)),
            static_path_counts=np.ones((1, 2), dtype=np.int64),
            blocked_static_path_counts=np.zeros((1, 2), dtype=np.int64),
            human_env_path_counts=np.zeros((1, 2), dtype=np.int64),
            env_human_path_counts=np.zeros((1, 2), dtype=np.int64),
            path_depth_histogram=np.ones((1, 2, 3), dtype=np.int64),
        ),
        runtime_s=0.5,
        scene_xml=custom_scene,
        scene_source_name="lab.xml",
    )
    full_rt_result = SimpleNamespace(**vars(result))
    full_rt_result.simulation_mode = "full_rt"
    full_rt_result.adc = adc * 2.0
    full_rt_result.manifest = replace(
        manifest,
        name="human room full RT test",
        solver=SolverExperimentConfig(mode="rt_scattering"),
    )
    path = tmp_path / "human-room.zip"
    path.write_bytes(
        _human_room_export(
            result,
            mode_results={
                "full_rt": full_rt_result,
                "hybrid_po": result,
            },
        ).getvalue()
    )

    with zipfile.ZipFile(path) as archive:
        assert {
            "bundle.json",
            "sensor.json",
            "frames.json",
            "radar_adc.npz",
            "simulated_adc_rt.npz",
            "simulated_adc_po.npz",
            "target_mesh.npz",
            "human_motion.npz",
            "amass_sequence.npz",
            "target.obj",
            "scene.xml",
            "environment.json",
            "experiment_manifest.json",
            "experiment_manifest_hybrid_po.json",
            "experiment_manifest_full_rt.json",
            "simulated_adc_full_rt.npz",
            "simulated_adc_hybrid_po.npz",
        }.issubset(archive.namelist())
        descriptor = json.loads(archive.read("bundle.json"))
        assert descriptor["profile"] == "hermes"
        assert descriptor["primary_solver"] == "po"
        assert descriptor["primary_simulation_mode"] == "hybrid_po"
        assert descriptor["motion"] == {
            "evaluated_mesh": "human_motion.npz",
            "parameters": "amass_sequence.npz",
            "frame_window": {
                "begin_frame_index": 1,
                "end_frame_index": 1,
            },
        }
        assert descriptor["adcs"]["full_rt"]["path"] == (
            "simulated_adc_full_rt.npz"
        )
        assert descriptor["adcs"]["hybrid_po"]["path"] == (
            "simulated_adc_hybrid_po.npz"
        )
        with np.load(
            BytesIO(archive.read("simulated_adc_rt.npz")),
            allow_pickle=False,
        ) as rt_comparison:
            assert np.array_equal(rt_comparison["adc"], full_rt_result.adc)
        sensor = json.loads(archive.read("sensor.json"))
        assert sensor["position"] == [0.0, 0.0, 0.0]
        frames = json.loads(archive.read("frames.json"))
        assert frames["frames"][0]["source_motion_frame_index"] == 1
        assert frames["frames"][0]["motion_time_s"] == pytest.approx(1.0 / 30.0)
        assert archive.read("scene.xml") == custom_scene
        with np.load(
            BytesIO(archive.read("amass_sequence.npz")),
            allow_pickle=False,
        ) as motion:
            assert motion["poses"].shape == (2, 72)
        environment = json.loads(archive.read("environment.json"))
        assert environment["scene_editable"] is False
        assert environment["motion_mesh"] == "human_motion.npz"
        assert environment["motion_parameters"] == "amass_sequence.npz"
        assert environment["scene_source"] == "lab.xml"
        assert environment["motion_frame_window"] == {
            "begin_frame_index": 1,
            "end_frame_index": 1,
        }
        with np.load(
            BytesIO(archive.read("solver_diagnostics.npz")),
            allow_pickle=False,
        ) as diagnostics:
            assert diagnostics["range_time_power"].shape == (2, 8)
            assert diagnostics["range_doppler_power"].shape == (1, 2, 8)

    loaded = load_bundle(path)

    assert loaded.profile == "hermes"
    assert loaded.data_origin == "simulation"
    assert loaded.available_simulations == (
        "full_rt",
        "hybrid_po",
        "po",
        "rt",
    )
    assert loaded.primary_adc.shape == adc.shape
    assert loaded.can_resimulate is True


@pytest.mark.parametrize(
    ("data_origin", "descriptor", "expected"),
    (
        ("measurement", {}, "Measured"),
        (
            "simulation",
            {"primary_simulation_mode": "full_rt", "primary_solver": "rt"},
            "Primary Full RT simulation",
        ),
        (
            "simulation",
            {"primary_simulation_mode": "hybrid_po", "primary_solver": "po"},
            "Primary Hybrid PO simulation",
        ),
        ("simulation", {"primary_solver": "rt"}, "Primary RT simulation"),
        ("simulation", {}, "Primary simulation"),
    ),
)
def test_bundle_primary_label_uses_declared_mode_and_solver(
    data_origin,
    descriptor,
    expected,
):
    loaded = SimpleNamespace(
        data_origin=data_origin,
        bundle=SimpleNamespace(descriptor=descriptor),
    )

    assert _bundle_primary_label(loaded) == expected


def test_plot_theme_tracks_the_single_panel_theme_switch():
    assert _plot_template({}) == "plotly_dark"
    assert _plot_template({"theme": [b"dark"]}) == "plotly_dark"
    assert _plot_template({"theme": [b"default"]}) == "plotly_white"


def test_theme_switch_copies_live_dynamic_modes_and_result_cache(monkeypatch):
    callbacks = []
    restore_callbacks = []
    original_jscallback = pn.widgets.Button.jscallback
    original_storage_jscallback = pn.reactive.ReactiveHTML.jscallback

    def capture_jscallback(widget, *args, **kwargs):
        if widget.icon in {"sun", "moon"}:
            callbacks.append(kwargs)
        return original_jscallback(widget, *args, **kwargs)

    monkeypatch.setattr(
        pn.widgets.Button,
        "jscallback",
        capture_jscallback,
    )

    def capture_storage_jscallback(component, *args, **kwargs):
        if "loaded_state" in component.param:
            restore_callbacks.append(kwargs)
        return original_storage_jscallback(component, *args, **kwargs)

    monkeypatch.setattr(
        pn.reactive.ReactiveHTML,
        "jscallback",
        capture_storage_jscallback,
    )

    application = build_app()

    assert len(callbacks) == 1
    assert len(restore_callbacks) == 1
    callback = callbacks[0]
    arguments = callback["args"]
    script = callback["clicks"]
    storage = arguments["session_bridge"]
    assert storage in application.header[0].select(pn.reactive.ReactiveHTML)
    assert arguments["dynamic_full_rt"].label == "Full RT"
    assert arguments["dynamic_coherent_rt"].label == "Coherent RT"
    assert arguments["dynamic_human_only_po"].label == "Human-only PO"
    assert arguments["dynamic_hybrid_po"].label == "Hybrid PO"
    assert "...bridged" in script
    assert "session_bridge.data.saved_state" in script
    assert "schema_version: 2" in script
    assert "full_rt: dynamic_full_rt.active" in script
    assert "coherent_rt: dynamic_coherent_rt.active" in script
    assert "human_only_po: dynamic_human_only_po.active" in script
    assert "hybrid_po: dynamic_hybrid_po.active" in script
    assert "bridged.gui_result_cache_key" in script
    restore_script = restore_callbacks[0]["loaded_state"]
    assert "workflow_tabs.active =" in restore_script
    assert "dynamic_tabs.active =" in restore_script
    assert "dynamic_human_only_po.active" in restore_script


def test_theme_comparison_cache_is_scoped_to_the_primary_bundle():
    adc = np.zeros((1, 2, 4, 1), dtype=np.complex64)
    key = _cache_theme_comparison(
        bundle_fingerprint="bundle-a",
        comparison_kind="run",
        candidates=[
            (
                "human_only_po",
                "new Human-only PO simulation",
                "Simulated",
                "Human-only PO",
                adc,
            )
        ],
        background_subtraction=False,
        clutter_removal="mean",
        channel_indices=(0,),
    )

    restored = _get_theme_comparison(
        key,
        bundle_fingerprint="bundle-a",
    )

    assert restored is not None
    assert restored["comparison_kind"] == "run"
    assert restored["clutter_removal"] == "mean"
    assert np.array_equal(restored["candidates"][0][4], adc)
    assert _get_theme_comparison(
        key,
        bundle_fingerprint="bundle-b",
    ) is None


def test_light_gui_uses_a_dark_accessible_accent(monkeypatch):
    monkeypatch.setattr(
        gui_app_module,
        "_plot_template",
        lambda _session_args: "plotly_white",
    )

    application = build_app()

    assert application.theme is pn.template.DefaultTheme
    assert application.accent_base_color == "#0369a1"
    assert application.header_background == "#101b2d"
    assert application.header_accent_base_color == "#38bdf8"
    assert "hermes-theme-light" in application.main[0].css_classes
    assert "hermes-theme-light" in application.modal[0].css_classes
    assert "--neutral-foreground-rest: #172033" in gui_app_module._APP_CSS
    motion_upload = next(
        widget
        for widget in application.main[0].select(pn.widgets.FileInput)
        if widget.label == "AMASS-like human motion (.npz)"
    )
    assert any(
        "background: #f8fafc" in str(stylesheet)
        and "color: #172033" in str(stylesheet)
        for stylesheet in motion_upload.stylesheets
    )


def test_light_plot_style_uses_high_contrast_text_axes_and_grids():
    figure = _style_figure(
        go.Figure(go.Scatter(x=[0.0, 1.0], y=[1.0, 2.0])),
        plot_template="plotly_white",
    )

    assert figure.layout.font.color == "#172033"
    assert figure.layout.title.font.color == "#101827"
    assert figure.layout.plot_bgcolor == "#ffffff"
    for axis in (figure.layout.xaxis, figure.layout.yaxis):
        assert axis.gridcolor == "#b6c2d0"
        assert axis.linecolor == "#64748b"
        assert axis.tickcolor == "#64748b"
        assert axis.zerolinecolor == "#64748b"
        assert axis.showline is True
        assert axis.tickfont.color == "#172033"
        assert axis.title.font.color == "#172033"


def test_light_3d_scene_style_uses_explicit_text_and_grid_contrast():
    figure = _style_figure(
        go.Figure(go.Scatter3d(x=[0.0], y=[0.0], z=[0.0])),
        plot_template="plotly_white",
    )

    assert figure.layout.scene.bgcolor == "#ffffff"
    for axis in (
        figure.layout.scene.xaxis,
        figure.layout.scene.yaxis,
        figure.layout.scene.zaxis,
    ):
        assert axis.backgroundcolor == "#ffffff"
        assert axis.gridcolor == "#b6c2d0"
        assert axis.linecolor == "#64748b"
        assert axis.zerolinecolor == "#64748b"
        assert axis.showbackground is True
        assert axis.tickfont.color == "#172033"
        assert axis.title.font.color == "#172033"


def test_dark_plot_style_preserves_existing_visual_language():
    figure = _style_figure(
        go.Figure(go.Scatter(x=[0.0, 1.0], y=[1.0, 2.0])),
        plot_template="plotly_dark",
    )

    assert figure.layout.font.color == "#e7eef8"
    assert figure.layout.plot_bgcolor == "#101b2d"
    assert figure.layout.xaxis.gridcolor == "rgba(148,163,184,0.18)"


def test_light_profile_traces_use_dark_canvas_safe_colors():
    static_result = SimpleNamespace(
        ranges_m=np.asarray([0.0, 1.0]),
        po_range_profile_power=np.asarray([1.0, 0.25]),
        rt_range_profile_power=np.asarray([0.5, 0.125]),
    )
    static_figure = _range_profile_figure(
        static_result,
        go,
        plot_template="plotly_white",
    )
    room_result = SimpleNamespace(
        simulation_mode="hybrid_po",
        ranges_m=np.asarray([0.0, 1.0]),
        range_profiles={
            "total": np.asarray([1.0, 0.5]),
            "human_po": np.asarray([0.5, 0.25]),
        },
    )
    room_figure = _human_room_profile_figure(
        room_result,
        go,
        plot_template="plotly_white",
    )

    assert [trace.line.color for trace in static_figure.data] == [
        "#0369a1",
        "#b45309",
    ]
    assert room_figure.data[0].line.color == "#0f172a"
