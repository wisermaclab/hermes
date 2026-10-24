# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Declared dependencies of each GUI workflow.

Protocols describe live controls, state and callbacks without importing the
optional GUI stack. The assembly context implements their union; workflows
are typed against only their own protocol and never accept arbitrary names.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from panel import layout, pane, widgets
    from panel.template import FastListTemplate
    from param.parameterized import Parameters

    class RequiredFloatInput(widgets.FloatInput):
        """A GUI numeric control initialized with a required float value."""

        value: float
        start: float
        end: float

    class RequiredIntInput(widgets.IntInput):
        """A GUI numeric control initialized with a required integer value."""

        value: int
        start: int
        end: int

    class RequiredIntSlider(widgets.IntSlider):
        """A frame selector initialized with integer limits and value."""

        value: int
        start: int
        end: int


class SettingsStorage(Protocol):
    param: Parameters
    saved_settings: dict[str, Any]


class SessionStorage(Protocol):
    param: Parameters
    saved_state: dict[str, Any]
    saved_uploads: dict[str, Any]

    def jscallback(self, **callbacks: Any) -> Any: ...


class SceneInteraction(Protocol):
    param: Parameters
    target_result: dict[str, Any]


class ConfirmationDialog(Protocol):
    param: Parameters
    open: bool


class ApplicationWorkflowContext(Protocol):
    """Dependencies used by application callbacks."""

    diagnostics_state: dict[str, Any]
    gui_activity_state: dict[str, bool]
    gui_session_state: dict[str, Any]
    human_room_state: dict[str, Any]
    load_motion_preview: widgets.Button
    measurement_state: dict[str, Any]
    physics_state: dict[str, Any]
    refresh_result_validity: Callable[..., Any]
    settings_button: widgets.Button
    theme_button: widgets.Button


class BundleContext(Protocol):
    """Dependencies used by bundles callbacks."""

    adc_selectors: dict[str, RequiredIntInput]
    background: widgets.Checkbox
    bundle_palette: dict[str, str]
    bundle_path: widgets.TextInput
    bundle_scene_plot: pane.Plotly
    bundle_scene_status: pane.Markdown
    bundle_step_marker: Callable[..., Any]
    bundle_step_one_label: pane.HTML
    bundle_step_one_marker: pane.HTML
    bundle_step_two_label: pane.HTML
    bundle_step_two_marker: pane.HTML
    candidate_adc_results: layout.Column
    candidate_product_results: tuple[layout.Column, ...]
    candidate_range_doppler_results: layout.Column
    candidate_range_time_results: layout.Column
    capture_result_settings: Callable[..., Any]
    clear_candidate_products: Callable[..., Any]
    clear_primary_products: Callable[..., Any]
    clutter: widgets.Select
    compare_button: widgets.Button
    comparison_channel_zero: widgets.Checkbox
    comparison_dependency_hint: pane.Markdown
    comparison_solvers: widgets.MultiChoice
    comparison_source: widgets.Select
    comparison_status: pane.Markdown
    configure_adc_selection: Callable[..., Any]
    current_comparison_signature: Callable[..., Any]
    custom_bundle_path: dict[str, str]
    default_model_dir: Path
    export_simulated_adc: Callable[..., Any]
    fixture: widgets.Select
    go: ModuleType
    gui_session_state: dict[str, Any]
    human_room_smpl_model_dir: widgets.TextInput
    hybrid_calibration: widgets.Select
    load_bundle_button: widgets.Button
    materialize_drop: Callable[..., Any]
    measurement_dropper: widgets.FileDropper
    measurement_state: dict[str, Any]
    measurement_status: pane.Markdown
    metrics_pane: pane.JSON
    persist_gui_session: Callable[..., Any]
    plot_template: str
    pn: ModuleType
    po_adaptive_visibility: widgets.Checkbox
    po_area_fraction: RequiredFloatInput
    po_normal_deg: RequiredFloatInput
    po_phase_error: RequiredFloatInput
    po_visibility_edge_margin: RequiredFloatInput
    po_visibility_samples: RequiredIntInput
    primary_adc_plot: pane.Plotly
    primary_product_plots: tuple[pane.Plotly, ...]
    primary_range_doppler_plot: pane.Plotly
    primary_range_time_plot: pane.Plotly
    range_profile_frame: RequiredIntSlider
    range_profile_frame_controls: layout.Row
    range_profile_next: widgets.Button
    range_profile_plot: pane.Plotly
    range_profile_previous: widgets.Button
    refresh_adc_selection: Callable[..., Any]
    refresh_profile_selection: Callable[..., Any]
    refresh_result_validity: Callable[..., Any]
    remember_result_settings: Callable[..., Any]
    remote_access: bool
    render_comparison_results: Callable[..., Any]
    result_records: dict[str, Any]
    rt_refresh: RequiredIntInput
    set_comparison_status: Callable[..., Any]
    set_gui_activity: Callable[..., Any]
    set_measurement_status: Callable[..., Any]
    simulated_adc_download: widgets.FileDownload
    simulated_dropper: widgets.FileDropper
    simulated_path: widgets.TextInput
    stop_simulation_confirmation: ConfirmationDialog
    trusted_bundle_paths: set[str]
    trusted_simulated_paths: set[str]
    update_bundle_step_rail: Callable[..., Any]
    update_comparison_source_visibility: Callable[..., Any]
    uploaded_bundle_paths: set[str]
    uploaded_simulated_paths: set[str]


class HumanRoomContext(Protocol):
    """Dependencies used by human room callbacks."""

    adc_if_bandwidth: RequiredFloatInput
    adc_sampling_mode: widgets.Select
    antenna_pattern: widgets.Select
    array_model: widgets.Select
    array_rx_groups: widgets.TextInput
    array_tx_groups: widgets.TextInput
    board: widgets.Select
    capture_result_settings: Callable[..., Any]
    carrier_frequency: RequiredFloatInput
    chirp_duration: RequiredFloatInput
    chirp_repetition: RequiredFloatInput
    chirp_slope: RequiredFloatInput
    configure_human_room_frame_controls: Callable[..., Any]
    configure_human_room_window_controls: Callable[..., Any]
    cosine_3db_beamwidth: RequiredFloatInput
    current_dynamic_radar_orientation: Callable[..., Any]
    current_human_position: Callable[..., Any]
    current_human_room_scene: Callable[..., Any]
    current_human_room_source: Callable[..., Any]
    current_motion_cache_key: Callable[..., Any]
    current_motion_coordinate_arguments: Callable[..., Any]
    current_room_boxes_override: Callable[..., Any]
    default_amass_motion: Path | None
    default_model_dir: Path
    diagnostics_for_time: Callable[..., Any]
    display_human_room_frame: Callable[..., Any]
    dynamic_radar_pitch: RequiredFloatInput
    dynamic_radar_roll: RequiredFloatInput
    dynamic_radar_yaw: RequiredFloatInput
    frame_period: RequiredFloatInput
    go: ModuleType
    gui_activity_state: dict[str, bool]
    gui_session_state: dict[str, Any]
    human_room_diagnostic_top_k: RequiredIntInput
    human_room_diagnostics_status: pane.Markdown
    human_room_diffuse: RequiredFloatInput
    human_room_download: widgets.FileDownload
    human_room_end_time: RequiredFloatInput
    human_room_floor_status: pane.Markdown
    human_room_frame_counter: widgets.StaticText
    human_room_play_pause: widgets.Button
    human_room_playback_state: dict[str, Any]
    human_room_player: widgets.Player
    human_room_pose_counter: Callable[..., Any]
    human_room_position_mode: widgets.Select
    human_room_position_state: dict[str, Any]
    human_room_preset: widgets.Select
    human_room_preview_time: RequiredFloatInput
    human_room_results: layout.Column
    human_room_run_input_widgets: tuple[widgets.Widget, ...]
    human_room_scene: pane.Plotly
    human_room_scene_upload: widgets.FileInput
    human_room_smpl_model_dir: widgets.TextInput
    human_room_solver_checkboxes: dict[str, widgets.Checkbox]
    human_room_start_time: RequiredFloatInput
    human_room_state: dict[str, Any]
    human_room_status: pane.Markdown
    human_room_transport_buttons: tuple[widgets.Button, ...]
    human_room_upload: widgets.FileInput
    human_room_upload_filename: widgets.StaticText
    human_room_window_frame_count: Callable[..., Any]
    human_room_window_state: dict[str, Any]
    human_room_window_summary: pane.Markdown
    human_room_x: RequiredFloatInput
    human_room_y: RequiredFloatInput
    human_room_yaw: RequiredFloatInput
    human_room_z: RequiredFloatInput
    hybrid_calibration: widgets.Select
    hybrid_coupling: widgets.Checkbox
    hybrid_max_reflectors: RequiredIntInput
    hybrid_reflector_cutoff: RequiredFloatInput
    mark_human_room_stale: Callable[..., Any]
    motion_coordinate_state: dict[str, Any]
    num_adc_samples: RequiredIntInput
    num_chirps: RequiredIntInput
    on_human_room_controls: Callable[..., Any]
    on_human_run_option_change: Callable[..., Any]
    pause_human_room_playback: Callable[..., Any]
    persist_gui_session: Callable[..., Any]
    plot_template: str
    pn: ModuleType
    po_adaptive_visibility: widgets.Checkbox
    po_area_fraction: RequiredFloatInput
    po_curvature: RequiredFloatInput
    po_integration_mode: widgets.Select
    po_normal_deg: RequiredFloatInput
    po_phase_error: RequiredFloatInput
    po_phase_span: RequiredFloatInput
    po_refinement_depth: RequiredIntInput
    po_refresh: RequiredIntInput
    po_subface_cap: RequiredIntInput
    po_visibility_edge_margin: RequiredFloatInput
    po_visibility_fade: RequiredIntInput
    po_visibility_refresh: RequiredIntInput
    po_visibility_samples: RequiredIntInput
    refresh_human_room_preview: Callable[..., Any]
    refresh_result_validity: Callable[..., Any]
    remember_result_settings: Callable[..., Any]
    remote_access: bool
    result_records: dict[str, Any]
    rt_depth: RequiredIntInput
    rt_human_bounces: widgets.Select
    rt_path_cap: RequiredIntInput
    rt_refresh: RequiredIntInput
    rt_samples: RequiredIntInput
    rt_visibility_fade: RequiredIntInput
    rt_visibility_refresh: RequiredIntInput
    run_human_room: widgets.Button
    run_human_room_diagnostics_button: widgets.Button
    rx_antennas: widgets.MultiChoice
    sampling_frequency: RequiredFloatInput
    set_gui_activity: Callable[..., Any]
    settings_dialog_state: dict[str, Any]
    settings_match_result: Callable[..., Any]
    show_human_room_playing_state: Callable[..., Any]
    start_fast_human_room_playback: Callable[..., Any]
    start_human_room_playback: Callable[..., Any]
    step_human_room_frame: Callable[..., Any]
    stop_human_room: widgets.Button
    sync_human_position_bounds: Callable[..., Any]
    sync_motion_coordinate_controls: Callable[..., Any]
    tx_antennas: widgets.MultiChoice
    tx_schedule: widgets.Select
    update_human_room_floor_status: Callable[..., Any]
    update_human_room_window_summary: Callable[..., Any]


class RadarContext(Protocol):
    """Dependencies used by radar callbacks."""

    adc_if_bandwidth: RequiredFloatInput
    adc_sampling_mode: widgets.Select
    antenna_pattern: widgets.Select
    antenna_pattern_note: pane.Markdown
    board: widgets.Select
    carrier_frequency: RequiredFloatInput
    chirp_duration: RequiredFloatInput
    chirp_repetition: RequiredFloatInput
    chirp_slope: RequiredFloatInput
    cosine_3db_beamwidth: RequiredFloatInput
    frame_period: RequiredFloatInput
    gui_session_state: dict[str, Any]
    ignored_radar_events: dict[int, int]
    material: widgets.Select
    material_details: pane.Markdown
    num_adc_samples: RequiredIntInput
    num_chirps: RequiredIntInput
    on_physics_control_change: Callable[..., Any]
    on_radar_control_change: Callable[..., Any]
    pn: ModuleType
    radar_summary: pane.Markdown
    refresh_result_validity: Callable[..., Any]
    rx_antennas: widgets.MultiChoice
    sampling_frequency: RequiredFloatInput
    set_radar_defaults: Callable[..., Any]
    settings_dialog_state: dict[str, Any]
    tx_antennas: widgets.MultiChoice
    tx_schedule: widgets.Select
    update_antenna_pattern_controls: Callable[..., Any]
    update_material_details: Callable[..., Any]
    update_radar_summary: Callable[..., Any]


class SessionContext(Protocol):
    """Dependencies used by session callbacks."""

    adc_selectors: dict[str, RequiredIntInput]
    apply_pending_dynamic_frame_state: Callable[..., Any]
    background: widgets.Checkbox
    board: widgets.Select
    bundle_path: widgets.TextInput
    cache_current_gui_results: Callable[..., Any]
    capture_result_settings: Callable[..., Any]
    clutter: widgets.Select
    comparison_channel_zero: widgets.Checkbox
    comparison_solvers: widgets.MultiChoice
    comparison_source: widgets.Select
    configure_human_room_frame_controls: Callable[..., Any]
    configure_human_room_window_controls: Callable[..., Any]
    corner_edge: RequiredFloatInput
    current_dynamic_radar_orientation: Callable[..., Any]
    current_human_room_source: Callable[..., Any]
    current_room_boxes_override: Callable[..., Any]
    current_solver_settings_payload: Callable[..., Any]
    custom_bundle_path: dict[str, str]
    decoded_upload: Callable[..., Any]
    default_amass_motion: Path | None
    diagnostic_top_k: RequiredIntInput
    diagnostics_depth: pane.Plotly
    diagnostics_state: dict[str, Any]
    diagnostics_status: pane.Markdown
    diagnostics_summary: pane.Markdown
    display_human_room_frame: Callable[..., Any]
    dynamic_result_view_state: Callable[..., Any]
    dynamic_session_widgets: dict[str, widgets.Widget]
    encoded_upload: Callable[..., Any]
    fixture: widgets.Select
    frame_period: RequiredFloatInput
    go: ModuleType
    gui_activity_state: dict[str, bool]
    gui_session_state: dict[str, Any]
    gui_session_storage: SessionStorage
    human_diffuse: widgets.FloatSlider
    human_room_diagnostics_status: pane.Markdown
    human_room_download: widgets.FileDownload
    human_room_end_time: RequiredFloatInput
    human_room_player: widgets.Player
    human_room_preview_time: RequiredFloatInput
    human_room_results: layout.Column
    human_room_scene: pane.Plotly
    human_room_scene_upload: widgets.FileInput
    human_room_solver_checkboxes: dict[str, widgets.Checkbox]
    human_room_start_time: RequiredFloatInput
    human_room_state: dict[str, Any]
    human_room_status: pane.Markdown
    human_room_upload: widgets.FileInput
    human_room_view_tabs: layout.Tabs
    human_source: pane.Markdown
    human_upload: widgets.FileInput
    load_motion_preview: widgets.Button
    material: widgets.Select
    material_details: pane.Markdown
    measurement_state: dict[str, Any]
    measurement_view_tabs: layout.Tabs
    motion_axis_options: dict[str, str]
    motion_forward_axis: widgets.Select
    on_compare: Callable[..., Any]
    on_human_room_upload_filename: Callable[..., Any]
    on_load_bundle: Callable[..., Any]
    on_run_human_room: Callable[..., Any]
    persist_gui_session: Callable[..., Any]
    persist_gui_uploads: Callable[..., Any]
    persist_solver_settings: Callable[..., Any]
    physics_download: widgets.FileDownload
    physics_scene: pane.Plotly
    physics_state: dict[str, Any]
    physics_status: pane.Markdown
    plate_height: RequiredFloatInput
    plate_width: RequiredFloatInput
    plot_template: str
    pn: ModuleType
    present_physics_result: Callable[..., Any]
    radar_session_widgets: dict[str, widgets.Widget]
    refresh_human_room_preview: Callable[..., Any]
    refresh_result_validity: Callable[..., Any]
    remember_result_settings: Callable[..., Any]
    remote_access: bool
    render_comparison_results: Callable[..., Any]
    restore_cached_gui_results: Callable[..., Any]
    restore_dynamic_result_view: Callable[..., Any]
    restore_gui_session: Callable[..., Any]
    restore_gui_uploads: Callable[..., Any]
    restore_solver_settings: Callable[..., Any]
    restore_widget_values: Callable[..., Any]
    result_banner_text: dict[str, pane.Markdown]
    result_banners: dict[str, layout.Column]
    result_records: dict[str, Any]
    result_revert_buttons: dict[str, widgets.Button]
    result_widgets: dict[str, dict[str, widgets.Widget]]
    run_human_room: widgets.Button
    run_human_room_diagnostics_button: widgets.Button
    rx_antennas: widgets.MultiChoice
    session_value: Callable[..., Any]
    set_comparison_status: Callable[..., Any]
    settings_dialog_state: dict[str, Any]
    show_rt_paths: widgets.Toggle
    simulated_adc_download: widgets.FileDownload
    simulated_path: widgets.TextInput
    solver_setting_widgets: dict[str, widgets.Widget]
    solver_settings_state: dict[str, Any]
    static_session_widgets: dict[str, widgets.Widget]
    sync_human_position_bounds: Callable[..., Any]
    sync_motion_coordinate_controls: Callable[..., Any]
    tabs: layout.Tabs
    target_type: widgets.Select
    trusted_bundle_paths: set[str]
    trusted_simulated_paths: set[str]
    tx_antennas: widgets.MultiChoice
    update_antenna_pattern_controls: Callable[..., Any]
    update_comparison_source_visibility: Callable[..., Any]
    update_human_room_window_summary: Callable[..., Any]
    update_material_details: Callable[..., Any]
    update_physics: Callable[..., Any]
    update_radar_summary: Callable[..., Any]
    upload_identities: dict[widgets.Widget, tuple[bytes, str]]
    widget_values: Callable[..., Any]


class SettingsContext(Protocol):
    """Dependencies used by settings callbacks."""

    board: widgets.Select
    current_solver_settings_payload: Callable[..., Any]
    gui_activity_state: dict[str, bool]
    gui_session_state: dict[str, Any]
    human_room_state: dict[str, Any]
    ignored_radar_events: dict[int, int]
    motion_axes_mode: widgets.Select
    motion_axis_options: dict[str, str]
    motion_coordinate_state: dict[str, Any]
    motion_coordinate_summary: pane.Markdown
    motion_coordinate_widgets: dict[str, widgets.Widget]
    motion_forward_axis: widgets.Select
    motion_handedness: widgets.Select
    motion_length_unit: widgets.Select
    motion_up_axis: widgets.Select
    on_human_room_controls: Callable[..., Any]
    persist_gui_session: Callable[..., Any]
    persist_solver_settings: Callable[..., Any]
    phase_span_by_mode: dict[str, float]
    pn: ModuleType
    po_curvature: RequiredFloatInput
    po_integration_mode: widgets.Select
    po_phase_span: RequiredFloatInput
    radar_session_widgets: dict[str, widgets.Widget]
    refresh_result_validity: Callable[..., Any]
    rerun_after_apply: widgets.Checkbox
    rerun_page: Callable[..., Any]
    restore_solver_settings: Callable[..., Any]
    rx_antennas: widgets.MultiChoice
    session_value: Callable[..., Any]
    settings_dialog_state: dict[str, Any]
    solver_setting_widgets: dict[str, widgets.Widget]
    solver_settings_state: dict[str, Any]
    solver_settings_storage: SettingsStorage
    sync_motion_coordinate_controls: Callable[..., Any]
    tabs: layout.Tabs
    template: FastListTemplate
    theme_button: widgets.Button
    tx_antennas: widgets.MultiChoice
    update_antenna_pattern_controls: Callable[..., Any]
    update_radar_summary: Callable[..., Any]


class StaticTargetContext(Protocol):
    """Dependencies used by static target callbacks."""

    adc_if_bandwidth: RequiredFloatInput
    adc_sampling_mode: widgets.Select
    angle_map: pane.Plotly
    angle_range_bins: widgets.IntRangeSlider
    angle_state: dict[str, Any]
    angle_status: pane.Markdown
    antenna_pattern: widgets.Select
    array_model: widgets.Select
    array_rx_groups: widgets.TextInput
    array_tx_groups: widgets.TextInput
    board: widgets.Select
    capture_result_settings: Callable[..., Any]
    carrier_frequency: RequiredFloatInput
    chirp_duration: RequiredFloatInput
    chirp_repetition: RequiredFloatInput
    chirp_slope: RequiredFloatInput
    compute_rt_path_overlay: Callable[..., Any]
    corner_edge: RequiredFloatInput
    cosine_3db_beamwidth: RequiredFloatInput
    current_physics_inputs: Callable[..., Any]
    default_human_mesh: Path | None
    diagnostic_top_k: RequiredIntInput
    diagnostics_depth: pane.Plotly
    diagnostics_state: dict[str, Any]
    diagnostics_status: pane.Markdown
    diagnostics_summary: pane.Markdown
    frame_period: RequiredFloatInput
    go: ModuleType
    gui_session_state: dict[str, Any]
    human_diffuse: widgets.FloatSlider
    human_source: pane.Markdown
    human_upload: widgets.FileInput
    ignored_physics_events: dict[int, int]
    invalidate_physics_result: Callable[..., Any]
    make_subplots: Callable[..., Any]
    material: widgets.Select
    material_details: pane.Markdown
    num_adc_samples: RequiredIntInput
    num_chirps: RequiredIntInput
    persist_gui_session: Callable[..., Any]
    physics_channel_profiles: layout.Column
    physics_download: widgets.FileDownload
    physics_profile: pane.Plotly
    physics_scene: pane.Plotly
    physics_state: dict[str, Any]
    physics_status: pane.Markdown
    pitch: RequiredFloatInput
    plate_height: RequiredFloatInput
    plate_width: RequiredFloatInput
    plot_template: str
    pn: ModuleType
    po_curvature: RequiredFloatInput
    po_integration_mode: widgets.Select
    po_phase_span: RequiredFloatInput
    po_refinement_depth: RequiredIntInput
    po_subface_cap: RequiredIntInput
    present_physics_result: Callable[..., Any]
    radar_pitch: RequiredFloatInput
    radar_roll: RequiredFloatInput
    radar_yaw: RequiredFloatInput
    refine_physics: widgets.Button
    refresh_result_validity: Callable[..., Any]
    remember_result_settings: Callable[..., Any]
    roll: RequiredFloatInput
    rt_depth: RequiredIntInput
    rt_path_cap: RequiredIntInput
    rt_samples: RequiredIntInput
    rx_antennas: widgets.MultiChoice
    sampling_frequency: RequiredFloatInput
    scene_interaction: SceneInteraction
    set_gui_activity: Callable[..., Any]
    settings_match_result: Callable[..., Any]
    show_rt_paths: widgets.Toggle
    stop_physics: widgets.Button
    target_range: RequiredFloatInput
    target_type: widgets.Select
    target_y: RequiredFloatInput
    target_z: RequiredFloatInput
    tx_antennas: widgets.MultiChoice
    tx_schedule: widgets.Select
    update_angle_status: Callable[..., Any]
    update_physics: Callable[..., Any]
    yaw: RequiredFloatInput


class AssemblyContext(Protocol):
    """Controls and callbacks used only while constructing the document."""

    bind_file_dropper: Callable[..., Any]
    bundle_step_heading: Callable[..., Any]
    cancel_session_work: Callable[..., Any]
    compact_numeric: Callable[..., RequiredFloatInput]
    invalidate_comparison_configuration: Callable[..., Any]
    invalidate_loaded_measurement: Callable[..., Any]
    numeric_row: Callable[..., Any]
    on_angle_range_change: Callable[..., Any]
    on_antenna_pattern_change: Callable[..., Any]
    on_apply_solver_settings: Callable[..., Any]
    on_board_change: Callable[..., Any]
    on_board_selection_change: Callable[..., Any]
    on_cancel_settings: Callable[..., Any]
    on_confirm_stop_simulation: Callable[..., Any]
    on_diagnostic_top_k_change: Callable[..., Any]
    on_dynamic_radar_preview_change: Callable[..., Any]
    on_human_position_input: Callable[..., Any]
    on_human_room_fast_forward: Callable[..., Any]
    on_human_room_fast_reverse: Callable[..., Any]
    on_human_room_next_frame: Callable[..., Any]
    on_human_room_play_pause: Callable[..., Any]
    on_human_room_player: Callable[..., Any]
    on_human_room_player_direction: Callable[..., Any]
    on_human_room_preview_time: Callable[..., Any]
    on_human_room_previous_frame: Callable[..., Any]
    on_human_room_reverse: Callable[..., Any]
    on_human_room_window_change: Callable[..., Any]
    on_load_radar_defaults: Callable[..., Any]
    on_material_change: Callable[..., Any]
    on_motion_coordinate_change: Callable[..., Any]
    on_open_settings: Callable[..., Any]
    on_po_integration_mode_change: Callable[..., Any]
    on_refine_physics: Callable[..., Any]
    on_run_human_room_diagnostics: Callable[..., Any]
    on_show_rt_paths: Callable[..., Any]
    on_solver_settings_loaded: Callable[..., Any]
    on_stop_human_room: Callable[..., Any]
    on_stop_physics: Callable[..., Any]
    on_target_gesture: Callable[..., Any]
    on_target_type_change: Callable[..., Any]
    revert_page_settings: Callable[..., Any]
    select_measurement_source: Callable[..., Any]
