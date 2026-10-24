# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Widget assembly for the GUI, ordered independently from workflow callbacks."""

from __future__ import annotations

from . import app as _app


class AppBuilder:
    """Build one document with an explicit session context and workflow owners."""

    def __init__(self, *, initial_bundle=None, remote_access=False):
        self.initial_bundle = initial_bundle
        self.remote_access = remote_access

    def _assemble_initialize(self):
        """Assemble initialize in dependency order."""
        from .workflows import (
            ApplicationContext,
            ApplicationWorkflow,
            SettingsWorkflow,
            RadarWorkflow,
            StaticTargetWorkflow,
            HumanRoomWorkflow,
            BundleWorkflow,
            SessionWorkflow,
        )

        self.context = ApplicationContext()
        self._application = ApplicationWorkflow(self.context)
        self._settings = SettingsWorkflow(self.context)
        self._radar = RadarWorkflow(self.context)
        self._static_target = StaticTargetWorkflow(self.context)
        self._human_room = HumanRoomWorkflow(self.context)
        self._bundles = BundleWorkflow(self.context)
        self._session = SessionWorkflow(self.context)
        self.context.remote_access = self.remote_access

        self.context.pn, self.context.go, self.context.make_subplots = _app._gui_modules()
        self.context.remote_access = bool(self.context.remote_access)
        self.context.trusted_bundle_paths = set()
        self.context.trusted_simulated_paths = set()
        self.context.uploaded_bundle_paths = set()
        self.context.uploaded_simulated_paths = set()
        self.context.pn.extension(
            "plotly",
            "filedropper",
            notifications=True,
            sizing_mode="stretch_width",
        )
        if _app._APP_CSS not in self.context.pn.config.raw_css:
            self.context.pn.config.raw_css.append(_app._APP_CSS)
        self.context.plot_template = _app._plot_template(self.context.pn.state.session_args)
        self.dark_theme = self.context.plot_template == "plotly_dark"
        self.theme_scope_class = "hermes-theme-dark" if self.dark_theme else "hermes-theme-light"
        self.context.theme_button = self.context.pn.widgets.Button(
            label="",
            icon="sun" if self.dark_theme else "moon",
            icon_size="30px",
            description=("Switch to light theme" if self.dark_theme else "Switch to dark theme"),
            width=52,
            height=44,
            margin=(0, 4),
            css_classes=["hermes-theme-button", "hermes-icon-theme"],
        )
        self.context.settings_button = self.context.pn.widgets.Button(
            label="",
            icon="settings",
            icon_size="30px",
            description="Radar and solver settings",
            width=52,
            height=44,
            margin=(0, 4),
            css_classes=["hermes-theme-button", "hermes-icon-settings"],
        )
        self.context.gui_activity_state = {
            "static": False,
            "dynamic": False,
            "comparison": False,
            "bundle_load": False,
        }
        self.context.settings_dialog_state = {"editing": False, "values": {}, "phase_spans": {}}

        self.context.set_gui_activity = self._application.set_gui_activity

    def _assemble_po_controls(self):
        """Assemble po controls in dependency order."""
        self.context.po_integration_mode = self.context.pn.widgets.Select(
            label="PO facet integration",
            options={
                "Adaptive analytic parent facet": "parent_face_adaptive_analytic",
                "Adaptive parent-facet quadrature": "parent_face_quadrature",
                "Far-field analytic parent facet": ("parent_face_far_field_analytic"),
                "Face centroid (fast approximation)": "face_centroid",
            },
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_integration_mode"],
        )
        self.context.po_phase_span = self.context.pn.widgets.FloatInput(
            label="PO subdivision phase span [rad]",
            description="Maximum linear phase span Δφmax for adaptive analytic facets; legacy quadrature uses its sampled phase span.",
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_quadrature_phase_span_scale_rad"],
            format="0.[000]",
            step=0.1,
            start=0.05,
            end=20.0,
        )
        self.context.po_curvature = self.context.pn.widgets.FloatInput(
            label="PO curvature tolerance [rad]",
            description="Split when the edge-midpoint path error δT = kc/3 × max|L(midpoint) − mean L(endpoints)| exceeds εφ.",
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_curvature_error_rad"],
            format="0.[000000]",
            step=0.001,
            start=0.000001,
            end=10.0,
        )

        # Quadrature samples the phase while analytic facets integrate it, so a
        # span suited to one mode is not carried into the other.
        self.context.phase_span_by_mode = {
            self.context.po_integration_mode.value: self.context.po_phase_span.value
        }
        self.context.solver_settings_state = {"restoring": False}

        self.context.on_po_integration_mode_change = self._settings.on_po_integration_mode_change

        self.context.po_integration_mode.param.watch(
            self.context.on_po_integration_mode_change, "value"
        )
        self.context.on_po_integration_mode_change()
        self.context.po_refinement_depth = self.context.pn.widgets.IntInput(
            label="PO maximum refinement depth",
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_quadrature_max_refinement_depth"],
            step=1,
            start=0,
            end=16,
        )
        self.context.po_subface_cap = self.context.pn.widgets.IntInput(
            label="PO subfaces per parent (0 = unlimited)",
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_quadrature_max_subfaces_per_parent"],
            step=4,
            start=0,
            end=4096,
        )
        self.context.po_refresh = self.context.pn.widgets.IntInput(
            label="PO re-integration every N frames (0 = every chirp)",
            description=(
                "Exact PO integration at the start of every N radar frames; in "
                "between, phase, delay, and spreading follow the motion. Facets "
                "whose neglected vertex phase change exceeds the configured "
                "threshold are re-integrated early. Hybrid PO coupling uses the same cadence. "
                "One frame is the default. On the tutorial walk, 4 frames stay "
                "within 0.4% of exact PO; check scenes where visibility changes."
            ),
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_refresh_frames"],
            step=1,
            start=0,
            end=1000,
        )
        self.context.po_phase_error = self.context.pn.widgets.FloatInput(
            label="PO refresh phase error [rad]",
            description=(
                "Re-integrate a facet when k × max|ΔL(vertex) − ΔL(centre)| "
                "since its last exact solve exceeds this value. Coupling uses "
                "image-source path lengths. Visibility changes follow the "
                "visibility refresh instead."
            ),
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_refresh_on_phase_error_rad"],
            format="0.[000000]",
            step=0.01,
            start=0.0,
            end=1000.0,
        )
        self.context.po_area_fraction = self.context.pn.widgets.FloatInput(
            label="PO refresh relative area change",
            description="Fractional area change since last exact solve; 0.02 means 2%.",
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_refresh_on_area_fraction"],
            step=0.01,
            start=0.0,
            end=1000.0,
        )
        self.context.po_normal_deg = self.context.pn.widgets.FloatInput(
            label="PO refresh normal change [deg]",
            description="Normal rotation since last exact solve, for direct and reflected PO.",
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_refresh_on_normal_deg"],
            step=0.1,
            start=0.0,
            end=180.0,
        )
        self.context.po_visibility_samples = self.context.pn.widgets.IntInput(
            label="PO visibility samples per face",
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_visibility_samples_per_face"],
            step=1,
            start=1,
            end=4096,
        )
        self.context.po_adaptive_visibility = self.context.pn.widgets.Checkbox(
            label="Adaptive PO visibility sampling",
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_adaptive_visibility_sampling"],
        )
        self.context.po_visibility_edge_margin = self.context.pn.widgets.FloatInput(
            label="PO grazing cosine margin",
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_adaptive_visibility_edge_margin"],
            step=0.01,
            start=0.0,
            end=1.0,
        )
        self.context.po_visibility_refresh = self.context.pn.widgets.IntInput(
            label="PO visibility every N frames (0 = every chirp)",
            description=(
                "How often facet self-occlusion on the body is recomputed. "
                "Hybrid PO coupling uses the same cadence."
            ),
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_visibility_refresh_frames"],
            step=1,
            start=0,
            end=1000,
        )
        self.context.po_visibility_fade = self.context.pn.widgets.IntInput(
            label="PO visibility fade [frames] (0 = step)",
            description=(
                "Ramp a facet in or out over this many frames when its "
                "visibility changes, instead of switching it at once."
            ),
            value=_app._SOLVER_SETTINGS_DEFAULTS["po_visibility_fade_frames"],
            step=1,
            start=0,
            end=100,
        )
        self.context.hybrid_reflector_cutoff = self.context.pn.widgets.FloatInput(
            label="Image-source power cutoff [dB relative to strongest]",
            description=(
                "Retain reflectors at or above this estimated power level toward the human motion volume. "
                "-20 dB means at least 1% of the strongest candidate's estimated power."
            ),
            value=_app._SOLVER_SETTINGS_DEFAULTS["hybrid_reflector_cutoff_db"],
            step=5.0,
            start=-120.0,
            end=0.0,
        )
        self.context.hybrid_max_reflectors = self.context.pn.widgets.IntInput(
            label="Image-source maximum count",
            description="Keep at most this many of the strongest reflectors passing the power cutoff (maximum 4).",
            value=_app._SOLVER_SETTINGS_DEFAULTS["hybrid_max_reflectors"],
            step=1,
            start=1,
            end=4,
        )
        self.context.hybrid_coupling = self.context.pn.widgets.Checkbox(
            label="Human–room coupling (human → room, room → human)",
            value=_app._SOLVER_SETTINGS_DEFAULTS["hybrid_coupling"],
        )
        self.context.hybrid_calibration = self.context.pn.widgets.Select(
            label="Hybrid PO calibration",
            description=(
                "Fit one gain from human RT and human-only PO ADC powers averaged "
                "over the first frame, then reuse it for human PO and coupling "
                "throughout the run. Whole-sequence fitting is also available (slow)."
            ),
            options={
                "RT first pose": "rt_first_pose",
                "RT whole sequence": "rt_sequence",
                "None": "none",
            },
            value=_app._SOLVER_SETTINGS_DEFAULTS["hybrid_calibration"],
        )

    def _assemble_rt_and_array_controls(self):
        """Assemble rt and array controls in dependency order."""
        self.context.rt_samples = self.context.pn.widgets.IntInput(
            label="RT ray samples per source",
            description=_app._RT_SOLVER_SETTING_HELP["rt_samples_per_source"],
            value=_app._SOLVER_SETTINGS_DEFAULTS["rt_samples_per_source"],
            step=500,
            start=100,
            end=2_000_000,
        )
        self.context.rt_path_cap = self.context.pn.widgets.IntInput(
            label="RT maximum paths per source",
            description=_app._RT_SOLVER_SETTING_HELP["rt_max_paths_per_source"],
            value=_app._SOLVER_SETTINGS_DEFAULTS["rt_max_paths_per_source"],
            step=500,
            start=100,
            end=2_000_000,
        )
        self.context.rt_depth = self.context.pn.widgets.IntInput(
            label="RT maximum path depth",
            description=_app._RT_SOLVER_SETTING_HELP["rt_max_depth"],
            value=_app._SOLVER_SETTINGS_DEFAULTS["rt_max_depth"],
            step=1,
            start=1,
            end=10,
        )
        self.context.rt_human_bounces = self.context.pn.widgets.Select(
            label="RT human bounces",
            description=(
                "Single keeps paths that touch the human at most once; any keeps "
                "every bounce count."
            ),
            options={"Single": "single", "Any": "any"},
            value=_app._SOLVER_SETTINGS_DEFAULTS["rt_human_bounces"],
        )
        self.context.rt_refresh = self.context.pn.widgets.IntInput(
            label="RT retrace every N frames (0 = every chirp)",
            description=(
                "Full ray trace at the start of every N radar frames; in between, "
                "the traced paths ride the moving mesh with coherent phase and "
                "delay updates."
            ),
            value=_app._SOLVER_SETTINGS_DEFAULTS["rt_refresh_frames"],
            step=1,
            start=0,
            end=1000,
        )
        self.context.rt_visibility_refresh = self.context.pn.widgets.IntInput(
            label="RT visibility every N frames (0 = every chirp)",
            description=(
                "How often known paths are re-tested for blocking by the moving "
                "body between retraces. Hybrid PO room blocking uses the same "
                "cadence."
            ),
            value=_app._SOLVER_SETTINGS_DEFAULTS["rt_visibility_refresh_frames"],
            step=1,
            start=0,
            end=1000,
        )
        self.context.rt_visibility_fade = self.context.pn.widgets.IntInput(
            label="RT visibility fade [frames] (0 = step)",
            description=(
                "Ramp a path in or out over this many frames when it becomes "
                "blocked or unblocked, instead of switching it at once."
            ),
            value=_app._SOLVER_SETTINGS_DEFAULTS["rt_visibility_fade_frames"],
            step=1,
            start=0,
            end=100,
        )
        self.context.array_model = self.context.pn.widgets.Select(
            label="MIMO array model",
            description=(
                "Phase center solves once at the Tx and Rx array centers and "
                "adds each channel's phase offset; runs warn when the aperture is "
                "too large for it. Subarrays solve once per pair of a Tx group "
                "and an Rx group; time grows with the number of pairs."
            ),
            options={"Phase center": "phase_center", "Subarrays": "subarray"},
            value=_app._SOLVER_SETTINGS_DEFAULTS["array_model"],
        )
        self.array_group_help = (
            "Zero-based antenna indices, as in Active antennas, with ';' between "
            "groups (e.g. 0-1; 2). Every active antenna must be in one group; "
            "empty means one group. Used only with Subarrays."
        )
        self.context.array_tx_groups = self.context.pn.widgets.TextInput(
            label="Tx groups",
            placeholder="e.g. 0-1; 2",
            value="",
            description=self.array_group_help,
        )
        self.context.array_rx_groups = self.context.pn.widgets.TextInput(
            label="Rx groups",
            placeholder="e.g. 0-1; 2-3",
            value="",
            description=self.array_group_help,
        )
        self.apply_solver_settings = self.context.pn.widgets.Button(
            label="Apply",
            icon="check",
            color="primary",
            width=120,
            height=40,
            sizing_mode="fixed",
        )
        self.cancel_solver_settings = self.context.pn.widgets.Button(
            label="Cancel",
            color="light",
            width=110,
            height=40,
            sizing_mode="fixed",
        )
        self.context.rerun_after_apply = self.context.pn.widgets.Checkbox(
            label="Apply & rerun this page",
            value=False,
            align="center",
        )
        self.context.solver_setting_widgets = {
            "array_model": self.context.array_model,
            "array_tx_groups": self.context.array_tx_groups,
            "array_rx_groups": self.context.array_rx_groups,
            "po_integration_mode": self.context.po_integration_mode,
            "po_quadrature_phase_span_scale_rad": self.context.po_phase_span,
            "po_curvature_error_rad": self.context.po_curvature,
            "po_quadrature_max_refinement_depth": self.context.po_refinement_depth,
            "po_quadrature_max_subfaces_per_parent": self.context.po_subface_cap,
            "po_refresh_frames": self.context.po_refresh,
            "po_refresh_on_phase_error_rad": self.context.po_phase_error,
            "po_refresh_on_area_fraction": self.context.po_area_fraction,
            "po_refresh_on_normal_deg": self.context.po_normal_deg,
            "po_visibility_samples_per_face": self.context.po_visibility_samples,
            "po_adaptive_visibility_sampling": self.context.po_adaptive_visibility,
            "po_adaptive_visibility_edge_margin": self.context.po_visibility_edge_margin,
            "po_visibility_refresh_frames": self.context.po_visibility_refresh,
            "po_visibility_fade_frames": self.context.po_visibility_fade,
            "rt_samples_per_source": self.context.rt_samples,
            "rt_max_paths_per_source": self.context.rt_path_cap,
            "rt_max_depth": self.context.rt_depth,
            "rt_human_bounces": self.context.rt_human_bounces,
            "rt_refresh_frames": self.context.rt_refresh,
            "rt_visibility_refresh_frames": self.context.rt_visibility_refresh,
            "rt_visibility_fade_frames": self.context.rt_visibility_fade,
            "hybrid_coupling": self.context.hybrid_coupling,
            "hybrid_reflector_cutoff_db": self.context.hybrid_reflector_cutoff,
            "hybrid_max_reflectors": self.context.hybrid_max_reflectors,
            "hybrid_calibration": self.context.hybrid_calibration,
        }

        self.context.current_solver_settings_payload = (
            self._settings.current_solver_settings_payload
        )

        self.context.restore_solver_settings = self._settings.restore_solver_settings

        self.context.solver_settings_storage = _app._solver_settings_storage_component(
            self.context.pn
        )

    def _assemble_settings_panels(self):
        """Assemble settings panels in dependency order."""
        self.context.motion_axis_options = {
            f"{sign}{axis.upper()}": f"{sign}{axis}" for axis in "xyz" for sign in ("+", "-")
        }
        self.context.motion_axes_mode = self.context.pn.widgets.Select(
            label="Motion coordinate mapping",
            css_classes=["hermes-settings-wide"],
            options={
                "Use file mapping / keep stored axes": "file",
                "Specify source coordinate frame": "manual",
            },
            value="file",
        )
        self.context.motion_up_axis = self.context.pn.widgets.Select(
            label="Source up axis",
            options=self.context.motion_axis_options,
            value="+z",
        )
        self.context.motion_forward_axis = self.context.pn.widgets.Select(
            label="Source forward axis",
            options=self.context.motion_axis_options,
            value="+x",
            description="Fixed reference direction, not the subject's changing heading.",
        )
        self.context.motion_handedness = self.context.pn.widgets.Select(
            label="Source handedness",
            options={"Right-handed": "right", "Left-handed": "left"},
            value="right",
        )
        self.context.motion_length_unit = self.context.pn.widgets.Select(
            label="Evaluated mesh units",
            options={"Metres": "m", "Centimetres": "cm", "Millimetres": "mm"},
            value="m",
            description="SMPL-generated meshes normally use metres.",
        )
        self.context.motion_coordinate_summary = self.context.pn.pane.Markdown(
            css_classes=["hermes-callout"]
        )
        self.context.motion_coordinate_widgets = {
            "motion_axes_mode": self.context.motion_axes_mode,
            "motion_up_axis": self.context.motion_up_axis,
            "motion_forward_axis": self.context.motion_forward_axis,
            "motion_handedness": self.context.motion_handedness,
            "motion_length_unit": self.context.motion_length_unit,
        }
        self.context.motion_coordinate_state = {"syncing": False}

        self.context.current_motion_coordinate_arguments = (
            self._settings.current_motion_coordinate_arguments
        )

        self.context.sync_motion_coordinate_controls = (
            self._settings.sync_motion_coordinate_controls
        )

        self.context.sync_motion_coordinate_controls()
        self.solver_controls_panel = self.context.pn.Column(
            _app._settings_note(
                self.context.pn,
                "These values control the numerical PO integration and RT search. "
                "Adaptive PO subdivision is temporary and is coherently integrated "
                "back onto each original facet. Cadences are counted in radar "
                "frames, and updates happen at frame starts. Hybrid PO always "
                "applies body blocking to room paths; coupling adds the "
                "image-source human-room paths. Solver values are saved in this "
                "browser and restored on the next visit.",
            ),
            _app._settings_section(
                self.context.pn,
                "MIMO array",
                "Static Target · Dynamic Scenes",
                _app._settings_grid(
                    self.context.pn,
                    self.context.array_model,
                    self.context.array_tx_groups,
                    self.context.array_rx_groups,
                    columns=3,
                ),
            ),
            _app._settings_section(
                self.context.pn,
                "Physical optics",
                "Static Target · Dynamic Scenes",
                _app._settings_grid(
                    self.context.pn,
                    self.context.po_integration_mode,
                    self.context.po_phase_span,
                    self.context.po_curvature,
                    self.context.po_refinement_depth,
                    self.context.po_subface_cap,
                    columns=3,
                ),
            ),
            _app._settings_section(
                self.context.pn,
                "PO temporal cadence",
                "Dynamic Scenes · PO · Hybrid PO",
                _app._settings_grid(
                    self.context.pn,
                    self.context.po_refresh,
                    self.context.po_phase_error,
                    self.context.po_area_fraction,
                    self.context.po_normal_deg,
                    self.context.po_visibility_samples,
                    self.context.po_adaptive_visibility,
                    self.context.po_visibility_edge_margin,
                    self.context.po_visibility_refresh,
                    self.context.po_visibility_fade,
                    columns=3,
                ),
            ),
            _app._settings_section(
                self.context.pn,
                "Ray tracing",
                "Static Target · Dynamic Scenes",
                _app._settings_grid(
                    self.context.pn,
                    self.context.rt_samples,
                    self.context.rt_path_cap,
                    self.context.rt_depth,
                    self.context.rt_human_bounces,
                    columns=3,
                ),
            ),
            _app._settings_section(
                self.context.pn,
                "RT temporal cadence",
                "Dynamic Scenes · RT · Hybrid PO",
                _app._settings_grid(
                    self.context.pn,
                    self.context.rt_refresh,
                    self.context.rt_visibility_refresh,
                    self.context.rt_visibility_fade,
                    columns=3,
                ),
            ),
            _app._settings_section(
                self.context.pn,
                "Hybrid PO",
                "Dynamic Scenes · Hybrid PO",
                _app._settings_grid(
                    self.context.pn,
                    self.context.hybrid_coupling,
                    self.context.hybrid_calibration,
                    self.context.hybrid_reflector_cutoff,
                    self.context.hybrid_max_reflectors,
                    columns=3,
                ),
            ),
            sizing_mode="stretch_width",
            margin=0,
        )
        self.motion_controls_panel = self.context.pn.Column(
            _app._settings_note(
                self.context.pn,
                "Map source coordinates to **+X forward, +Y left, +Z up**, in "
                "metres. Apply updates the loaded motion preview. Motion "
                "coordinates are kept with the current GUI session and recorded "
                "in exported runs.",
            ),
            _app._settings_section(
                self.context.pn,
                "Source coordinate frame",
                "Dynamic Scenes",
                _app._settings_grid(self.context.pn, self.context.motion_axes_mode, columns=4),
                _app._settings_grid(
                    self.context.pn,
                    self.context.motion_up_axis,
                    self.context.motion_forward_axis,
                    self.context.motion_handedness,
                    self.context.motion_length_unit,
                    columns=4,
                ),
                self.context.motion_coordinate_summary,
            ),
            sizing_mode="stretch_width",
            margin=0,
        )

        self.context.target_type = self.context.pn.widgets.Select(
            label="Target type",
            options={
                "Plate": "plate",
                "Trihedral corner reflector": "trihedral",
                "Human mesh": "human_mesh",
            },
            value="plate",
        )

    def _assemble_target_controls(self):
        """Assemble target controls in dependency order."""
        self.context.compact_numeric = self._settings.compact_numeric

        self.context.numeric_row = self._settings.numeric_row

        self.context.target_range = self.context.compact_numeric(
            "X [m]",
            start=0.5,
            end=5.0,
            step=0.1,
            value=2.0,
            format="0.00",
            description="Range-forward position.",
        )
        self.context.target_y = self.context.compact_numeric(
            "Y [m]",
            start=-2.0,
            end=2.0,
            step=0.05,
            value=0.0,
            format="0.00",
            description="Lateral-left position.",
        )
        self.context.target_z = self.context.compact_numeric(
            "Z [m]",
            start=-1.5,
            end=2.5,
            step=0.05,
            value=0.0,
            format="0.00",
            description="Vertical-up position.",
        )
        self.context.plate_width = self.context.compact_numeric(
            "Width [m]",
            value=0.30,
            format="0.00",
            step=0.01,
            start=0.02,
            end=3.0,
        )
        self.context.plate_height = self.context.compact_numeric(
            "Height [m]",
            value=0.30,
            format="0.00",
            step=0.01,
            start=0.02,
            end=3.0,
        )
        self.context.corner_edge = self.context.pn.widgets.FloatInput(
            label="Corner edge length [m]",
            value=0.20,
            format="0.00",
            step=0.01,
            start=0.02,
            end=2.0,
            visible=False,
        )
        self.context.yaw = self.context.compact_numeric(
            "Yaw [deg]", start=-75, end=75, step=1, value=0
        )
        self.context.pitch = self.context.compact_numeric(
            "Pitch [deg]", start=-75, end=75, step=1, value=0
        )
        self.context.roll = self.context.compact_numeric(
            "Roll [deg]", start=-180, end=180, step=1, value=0
        )
        self.context.radar_yaw = self.context.compact_numeric(
            "Yaw [deg]", start=-180, end=180, step=1, value=0
        )
        self.context.radar_pitch = self.context.compact_numeric(
            "Pitch [deg]", start=-90, end=90, step=1, value=0
        )
        self.context.radar_roll = self.context.compact_numeric(
            "Roll [deg]", start=-180, end=180, step=1, value=0
        )
        self.context.dynamic_radar_yaw = self.context.compact_numeric(
            "Yaw [deg]", start=-180, end=180, step=1, value=0
        )
        self.context.dynamic_radar_pitch = self.context.compact_numeric(
            "Pitch [deg]", start=-90, end=90, step=1, value=0
        )
        self.context.dynamic_radar_roll = self.context.compact_numeric(
            "Roll [deg]", start=-180, end=180, step=1, value=0
        )
        self.context.material = self.context.pn.widgets.Select(
            label="Material preset",
            options={
                "PEC · ideal PO, rough/diffuse RT": "pec",
                "Aluminum · smooth/specular": "aluminum",
                "Concrete · rough/diffuse": "concrete",
            },
            value=_app.MATERIAL_PRESETS[0],
        )
        self.context.material_details = self.context.pn.pane.Markdown()

        self.context.update_material_details = self._radar.update_material_details

        self.context.update_material_details()
        self.context.human_upload = self.context.pn.widgets.FileInput(
            label="Human mesh (.obj or .npz)",
            accept=".obj,.npz",
            stylesheets=[_app._file_input_stylesheet(self.context.plot_template, compact=False)],
            description="Browser uploads are limited to 64 MiB.",
            visible=False,
        )
        self.context.human_diffuse = self.context.pn.widgets.FloatSlider(
            label="Human diffuse reflection coefficient (RT)",
            start=0.0,
            end=1.0,
            step=0.05,
            value=0.35,
            visible=False,
        )
        # Never let a remotely connected browser implicitly select or export a
        # participant-derived mesh from the private source checkout.
        self.context.default_human_mesh = (
            None if self.context.remote_access else _app._default_human_mesh_path()
        )
        self.context.human_source = self.context.pn.pane.Markdown(
            (
                "**Default mesh:** regenerated RT-Pose seq186 frame0020 "
                "`mesh_sequence.npz` (internal). Upload a different `.obj` or "
                "`.npz` above to override it."
                if self.context.default_human_mesh is not None
                else "**No default mesh is installed.** Upload an `.obj` or `.npz`."
            ),
            visible=False,
        )

    def _assemble_radar_controls(self):
        """Assemble radar controls in dependency order."""
        self.context.board = self.context.pn.widgets.Select(
            label="TI radar",
            options=list(_app.available_ti_boards()),
            value="IWR6843AOPEVM",
        )
        self.board_spec = _app.get_ti_board_spec(self.context.board.value)
        self.context.tx_schedule = self.context.pn.widgets.Select(
            label="Tx schedule",
            options={
                "Simultaneous Tx": "simultaneous",
                "TDM-MIMO (Tx take turns)": "tdm",
            },
            value="simultaneous",
        )
        self.context.tx_antennas = self.context.pn.widgets.MultiChoice(
            label="Active Tx antennas",
            description=_app._ANTENNA_NUMBERING_HELP,
            options={f"TX{index}": index for index in range(self.board_spec.num_tx)},
            value=list(range(self.board_spec.num_tx)),
        )
        self.context.rx_antennas = self.context.pn.widgets.MultiChoice(
            label="Active Rx antennas",
            description=_app._ANTENNA_NUMBERING_HELP,
            options={f"RX{index}": index for index in range(self.board_spec.num_rx)},
            value=list(range(self.board_spec.num_rx)),
        )
        self.context.antenna_pattern = self.context.pn.widgets.Select(
            label="Antenna pattern",
            options={
                "Digitized TI profile": "digitized",
                "Omnidirectional": "none",
                "Cosine": "cosine",
            },
            value="cosine",
        )
        self.context.cosine_3db_beamwidth = self.context.pn.widgets.FloatInput(
            label="Cosine 3 dB beamwidth [deg]",
            value=60.0,
            step=1.0,
            start=1.0,
            end=179.0,
        )
        self.context.antenna_pattern_note = self.context.pn.pane.Markdown()
        self.context.carrier_frequency = self.context.pn.widgets.FloatInput(
            label="Carrier frequency [GHz]",
            value=_app._rounded_gui_float(self.board_spec.design_frequency_hz / 1e9),
            format="0.[000]",
            step=0.1,
            start=50.0,
            end=90.0,
        )
        self.context.chirp_slope = self.context.pn.widgets.FloatInput(
            label="Chirp slope [MHz/µs]",
            value=68.0,
            step=0.1,
            start=0.1,
            end=200.0,
        )
        self.context.chirp_duration = self.context.pn.widgets.FloatInput(
            label="Chirp duration [µs]",
            value=58.0,
            step=1.0,
            start=1.0,
            end=1000.0,
        )
        self.context.chirp_repetition = self.context.pn.widgets.FloatInput(
            label="Chirp repetition time [µs]",
            value=65.0,
            step=1.0,
            start=1.0,
            end=2000.0,
        )
        self.context.sampling_frequency = self.context.pn.widgets.FloatInput(
            label="ADC sampling rate [MS/s]",
            value=4.5,
            step=0.1,
            start=0.1,
            end=50.0,
        )
        self.context.adc_sampling_mode = self.context.pn.widgets.Select(
            label="ADC sampling mode",
            value="complex1x",
            options={
                "Complex 1x (0–fs)": "complex1x",
                "Complex 2x (0–fs/2)": "complex2x",
                "Real (0–fs/2)": "real",
            },
        )
        self.context.adc_if_bandwidth = self.context.pn.widgets.FloatInput(
            label="IF bandwidth cap [MHz] (0 = board default)",
            value=0.0,
            start=0.0,
            end=50.0,
            step=0.1,
        )
        self.context.num_adc_samples = self.context.pn.widgets.IntInput(
            label="ADC samples per chirp",
            value=225,
            step=1,
            start=16,
            end=4096,
        )
        self.context.num_chirps = self.context.pn.widgets.IntInput(
            label="Chirps per frame",
            value=16,
            step=1,
            start=1,
            end=4096,
        )
        self.context.frame_period = self.context.pn.widgets.FloatInput(
            label="Frame period [ms]",
            value=50.0,
            step=1.0,
            start=1.0,
            end=10_000.0,
        )
        self.load_radar_defaults = self.context.pn.widgets.Button(
            label="Load default FMCW values",
            color="light",
            icon="refresh",
        )
        self.context.radar_summary = self.context.pn.pane.Markdown()
        self.context.ignored_radar_events = {}

        self.context.update_antenna_pattern_controls = self._radar.update_antenna_pattern_controls

        self.context.update_radar_summary = self._radar.update_radar_summary

        self.context.set_radar_defaults = self._radar.set_radar_defaults

    def _assemble_static_products(self):
        """Assemble static products in dependency order."""
        self.context.update_antenna_pattern_controls()
        self.context.update_radar_summary()
        self.context.refine_physics = self.context.pn.widgets.Button(
            label="Recompute at higher fidelity",
            color="primary",
            icon="sparkles",
            width=250,
            height=40,
            sizing_mode="fixed",
            description=("Increase persistent mesh density and PO visibility sampling."),
        )
        self.context.stop_physics = self.context.pn.widgets.Button(
            label="Stop",
            color="light",
            icon="player-stop",
            width=105,
            height=40,
            sizing_mode="fixed",
            description="Stop the active static RT/PO run at a safe boundary.",
            disabled=True,
        )
        self.initial_result = _app.run_static_target_experiment(
            target_type=self.context.target_type.value,
            yaw_deg=self.context.yaw.value,
            pitch_deg=self.context.pitch.value,
            roll_deg=self.context.roll.value,
            target_y_m=self.context.target_y.value,
            target_z_m=self.context.target_z.value,
            radar_yaw_deg=self.context.radar_yaw.value,
            radar_pitch_deg=self.context.radar_pitch.value,
            radar_roll_deg=self.context.radar_roll.value,
            range_m=self.context.target_range.value,
            width_m=self.context.plate_width.value,
            height_m=self.context.plate_height.value,
            material_name=self.context.material.value,
            fidelity="preview",
            board_model=self.context.board.value,
            carrier_frequency_hz=self.context.carrier_frequency.value * 1e9,
            slope_hz_per_s=self.context.chirp_slope.value * 1e12,
            chirp_duration_s=self.context.chirp_duration.value * 1e-6,
            chirp_repetition_time_s=self.context.chirp_repetition.value * 1e-6,
            sampling_frequency_hz=self.context.sampling_frequency.value * 1e6,
            adc_mode=self.context.adc_sampling_mode.value,
            if_bandwidth_hz=self.context.adc_if_bandwidth.value * 1e6 or None,
            num_adc_samples=self.context.num_adc_samples.value,
            num_chirps_per_frame=self.context.num_chirps.value,
            frame_period_s=self.context.frame_period.value * 1e-3,
            tx_schedule=self.context.tx_schedule.value,
            selected_tx_indices=tuple(self.context.tx_antennas.value),
            selected_rx_indices=tuple(self.context.rx_antennas.value),
            antenna_pattern_mode=self.context.antenna_pattern.value,
            cosine_3db_beamwidth_deg=self.context.cosine_3db_beamwidth.value,
            po_integration_mode=self.context.po_integration_mode.value,
            po_quadrature_phase_span_scale_rad=self.context.po_phase_span.value,
            po_curvature_error_rad=self.context.po_curvature.value,
            po_quadrature_max_refinement_depth=self.context.po_refinement_depth.value,
            po_quadrature_max_subfaces_per_parent=self.context.po_subface_cap.value,
            rt_samples_per_source=self.context.rt_samples.value,
            rt_max_paths_per_source=self.context.rt_path_cap.value,
            rt_max_depth=self.context.rt_depth.value,
            array_model=self.context.array_model.value,
        )
        self.context.physics_scene = self.context.pn.pane.Plotly(
            _app._physics_scene_figure(
                self.initial_result,
                self.context.go,
                plot_template=self.context.plot_template,
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
        self.context.scene_interaction = _app._scene_interaction_component(
            self.context.pn, self.context.physics_scene
        )
        self.context.physics_profile = self.context.pn.pane.Plotly(
            _app._range_profile_figure(
                self.initial_result,
                self.context.go,
                plot_template=self.context.plot_template,
            ),
            height=310,
        )
        self.context.physics_profile.height = self.context.physics_profile.object.layout.height
        self.context.physics_profile.param.watch(
            lambda event: setattr(event.obj, "height", event.new.layout.height or 380),
            "object",
        )
        self.context.physics_channel_profiles = self.context.pn.Column(
            *[
                self.context.pn.pane.Plotly(figure, height=figure.layout.height)
                for figure in _app._static_channel_profile_figures(
                    self.initial_result, self.context.go, plot_template=self.context.plot_template
                )
            ],
            sizing_mode="stretch_width",
        )
        self.initial_angle_products = _app._angle_fft_range_products(self.initial_result)
        self.initial_centroid_bin = int(self.initial_angle_products["centroid_range_bin"])
        self.context.angle_range_bins = self.context.pn.widgets.IntRangeSlider(
            label="Angle-FFT range bins (inclusive)",
            start=0,
            end=len(self.initial_angle_products["ranges_m"]) - 1,
            step=1,
            value=(self.initial_centroid_bin, self.initial_centroid_bin),
            sizing_mode="stretch_width",
        )
        self.context.angle_map = self.context.pn.pane.Plotly(
            _app._angle_map_figure(
                self.initial_angle_products,
                self.context.make_subplots,
                self.context.go,
                range_bins=self.context.angle_range_bins.value,
                plot_template=self.context.plot_template,
            ),
            height=420,
        )
        self.context.angle_status = self.context.pn.pane.Markdown()

        self.context.update_angle_status = self._static_target.update_angle_status

        self.context.update_angle_status(
            self.initial_angle_products, self.context.angle_range_bins.value
        )
        self.context.physics_status = self.context.pn.pane.Markdown(
            f"**LIVE RT + PO** · {self.initial_result.faces.shape[0]:,} facets · "
            f"{self.initial_result.rt_path_count:,} RT paths · "
            f"PO {self.initial_result.po_runtime_s:.3f} s · "
            f"RT {self.initial_result.rt_runtime_s:.3f} s",
            css_classes=["hermes-status"],
        )
        self.context.physics_download = self.context.pn.widgets.FileDownload(
            callback=lambda: _app._physics_export(self.initial_result),
            filename="hermes-static-target-bundle.zip",
            label="Export loadable bundle",
            color="primary",
            variant="solid",
            height=44,
            sizing_mode="stretch_width",
            stylesheets=[".bk-btn {font-weight:700 !important;}"],
        )
        self.context.physics_state = {
            "result": self.initial_result,
            "refined": False,
            "revision": 0,
            "target_result_sequence": 0,
            "running": False,
            "pending_request": None,
            "cancel_event": None,
        }
        self.context.angle_state = {
            "products": self.initial_angle_products,
            "revision": 0,
        }
        self.context.diagnostic_top_k = self.context.pn.widgets.IntInput(
            label="Top RT paths",
            value=12,
            start=1,
            end=100,
            step=1,
            width=135,
        )
        self.context.show_rt_paths = self.context.pn.widgets.Toggle(
            label="Show",
            icon="route",
            color="primary",
            width=100,
            align="end",
        )
        self.context.diagnostics_status = self.context.pn.pane.Markdown(
            "RT path geometry is off. Enable it to run one diagnostic trace and "
            "compute the interaction-depth histogram; ADC synthesis is not rerun.",
            css_classes=["hermes-status"],
        )
        self.context.diagnostics_depth = self.context.pn.pane.Plotly(height=300, visible=False)
        self.context.diagnostics_summary = self.context.pn.pane.Markdown(
            f"**RT valid link-paths:** `{self.initial_result.rt_path_count:,}` · "
            f"**PO visible facets:** `{self.initial_result.visible_face_count:,}`"
        )
        self.context.diagnostics_state = {
            "revision": 0,
            "result": None,
            "top_k": None,
            "cancel_event": None,
        }

    def _assemble_static_bindings(self):
        """Assemble static bindings in dependency order."""
        self.context.invalidate_physics_result = self._static_target.invalidate_physics_result

        self.context.present_physics_result = self._static_target.present_physics_result

        self.context.current_physics_inputs = self._static_target.current_physics_inputs

        self.context.update_physics = self._static_target.update_physics

        self.context.on_refine_physics = self._static_target.on_refine_physics

        self.context.on_stop_physics = self._static_target.on_stop_physics

        self.context.compute_rt_path_overlay = self._static_target.compute_rt_path_overlay

        self.context.on_show_rt_paths = self._static_target.on_show_rt_paths

        self.context.on_diagnostic_top_k_change = self._static_target.on_diagnostic_top_k_change

        self.context.ignored_physics_events = {}

        self.context.on_physics_control_change = self._static_target.on_physics_control_change

        self.context.on_target_gesture = self._static_target.on_target_gesture

        self.context.on_angle_range_change = self._static_target.on_angle_range_change

        self.context.on_target_type_change = self._static_target.on_target_type_change

        self.context.on_board_selection_change = self._radar.on_board_selection_change

        self.context.on_board_change = self._radar.on_board_change

        self.context.on_load_radar_defaults = self._radar.on_load_radar_defaults

        self.context.on_radar_control_change = self._radar.on_radar_control_change

        self.context.on_material_change = self._radar.on_material_change

        self.context.on_antenna_pattern_change = self._radar.on_antenna_pattern_change

        self.context.refine_physics.on_click(self.context.on_refine_physics)
        self.context.stop_physics.on_click(self.context.on_stop_physics)
        self.context.show_rt_paths.param.watch(self.context.on_show_rt_paths, "value")
        self.context.diagnostic_top_k.param.watch(self.context.on_diagnostic_top_k_change, "value")
        self.load_radar_defaults.on_click(self.context.on_load_radar_defaults)
        self.context.target_type.param.watch(self.context.on_target_type_change, "value")
        self.context.scene_interaction.param.watch(
            self.context.on_target_gesture, "target_gesture"
        )
        self.context.angle_range_bins.param.watch(
            self.context.on_angle_range_change, "value_throttled"
        )
        for self.numeric in (
            self.context.target_range,
            self.context.target_y,
            self.context.target_z,
            self.context.yaw,
            self.context.pitch,
            self.context.roll,
            self.context.radar_yaw,
            self.context.radar_pitch,
            self.context.radar_roll,
            self.context.plate_width,
            self.context.plate_height,
            self.context.corner_edge,
        ):
            self.numeric.param.watch(self.context.on_physics_control_change, "value")
        self.context.human_diffuse.param.watch(
            self.context.on_physics_control_change, "value_throttled"
        )
        self.context.material.param.watch(self.context.on_material_change, "value")
        self.context.human_upload.param.watch(self.context.on_physics_control_change, "value")
        self.context.board.param.watch(self.context.on_board_selection_change, "value")
        self.context.board.param.watch(self.context.on_board_change, "value")
        self.context.tx_schedule.param.watch(self.context.on_radar_control_change, "value")
        self.context.tx_antennas.param.watch(self.context.on_radar_control_change, "value")
        self.context.rx_antennas.param.watch(self.context.on_radar_control_change, "value")
        self.context.antenna_pattern.param.watch(self.context.on_antenna_pattern_change, "value")
        self.context.cosine_3db_beamwidth.param.watch(
            self.context.on_radar_control_change, "value"
        )
        for radar_control in (
            self.context.carrier_frequency,
            self.context.chirp_slope,
            self.context.chirp_duration,
            self.context.chirp_repetition,
            self.context.sampling_frequency,
            self.context.adc_sampling_mode,
            self.context.adc_if_bandwidth,
            self.context.num_adc_samples,
            self.context.num_chirps,
            self.context.frame_period,
        ):
            radar_control.param.watch(self.context.on_radar_control_change, "value")

    def _assemble_radar_panel(self):
        """Assemble radar panel in dependency order."""
        self.radar_used_by = "Static Target · Dynamic Scenes"
        self.radar_controls = self.context.pn.Row(
            self.context.pn.Column(
                _app._settings_note(
                    self.context.pn,
                    "Hardware, antenna, and waveform settings apply to subsequent "
                    "static and dynamic runs. Each workflow keeps its own radar "
                    "orientation.",
                ),
                _app._settings_section(
                    self.context.pn,
                    "Hardware and antennas",
                    self.radar_used_by,
                    _app._settings_grid(
                        self.context.pn,
                        self.context.board,
                        self.context.tx_antennas,
                        self.context.rx_antennas,
                        self.context.antenna_pattern,
                        self.context.cosine_3db_beamwidth,
                        self.context.tx_schedule,
                        columns=3,
                    ),
                    self.context.antenna_pattern_note,
                ),
                _app._settings_section(
                    self.context.pn,
                    "Chirp",
                    self.radar_used_by,
                    _app._settings_grid(
                        self.context.pn,
                        self.context.carrier_frequency,
                        self.context.chirp_slope,
                        self.context.chirp_duration,
                        self.context.chirp_repetition,
                    ),
                ),
                _app._settings_section(
                    self.context.pn,
                    "Sampling",
                    self.radar_used_by,
                    _app._settings_grid(
                        self.context.pn,
                        self.context.sampling_frequency,
                        self.context.adc_sampling_mode,
                        self.context.adc_if_bandwidth,
                        self.context.num_adc_samples,
                    ),
                ),
                _app._settings_section(
                    self.context.pn,
                    "Frame",
                    self.radar_used_by,
                    _app._settings_grid(
                        self.context.pn, self.context.num_chirps, self.context.frame_period
                    ),
                ),
                sizing_mode="stretch_width",
                min_width=0,
                margin=0,
            ),
            # The derived values stay in view while the waveform is edited.
            self.context.pn.Column(
                self.context.pn.pane.HTML(
                    '<div class="hermes-group-title" role="heading" '
                    'aria-level="3">Derived</div>',
                    sizing_mode="stretch_width",
                    margin=(0, 10),
                ),
                self.context.radar_summary,
                self.load_radar_defaults,
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

    def _assemble_settings_and_static_layout(self):
        """Assemble settings and static layout in dependency order."""
        self.settings_modal = self.context.pn.Column(
            self.context.pn.pane.HTML(
                '<div class="hermes-settings-title" role="heading" aria-level="2">'
                "Settings</div>"
                '<p class="hermes-settings-sub">Edits take effect together when '
                "you Apply. Bundle Comparison uses the radar and solver settings "
                "stored in its primary bundle.</p>",
                sizing_mode="stretch_width",
                margin=(0, 10),
            ),
            self.context.pn.Tabs(
                ("Radar configuration", self.radar_controls),
                ("Solver settings", self.solver_controls_panel),
                ("Motion coordinates", self.motion_controls_panel),
                dynamic=False,
                sizing_mode="stretch_width",
            ),
            self.context.pn.Row(
                self.context.rerun_after_apply,
                self.context.pn.Spacer(sizing_mode="stretch_width"),
                self.cancel_solver_settings,
                self.apply_solver_settings,
                sizing_mode="stretch_width",
                margin=0,
                css_classes=["hermes-settings-footer"],
            ),
            min_width=0,
            max_width=980,
            sizing_mode="stretch_width",
            css_classes=[self.theme_scope_class, "hermes-settings"],
        )
        self.static_result_banner = self.context.pn.Column(
            visible=False, name="Static result settings"
        )
        self.physics_controls = self.context.pn.Column(
            _app._rail_nav_component(
                self.context.pn,
                (
                    ("target", "Target"),
                    ("orientation", "Orientation"),
                    ("material", "Material"),
                    ("radar", "Radar"),
                ),
            ),
            _app._rail_group(
                self.context.pn,
                "target",
                "Target",
                self.context.target_type,
                self.context.numeric_row(
                    "hermes-static-position",
                    self.context.target_range,
                    self.context.target_y,
                    self.context.target_z,
                ),
                self.context.numeric_row(
                    "hermes-static-size", self.context.plate_width, self.context.plate_height
                ),
                self.context.corner_edge,
            ),
            _app._rail_group(
                self.context.pn,
                "orientation",
                "Target orientation",
                self.context.numeric_row(
                    "hermes-static-orientation",
                    self.context.yaw,
                    self.context.pitch,
                    self.context.roll,
                ),
            ),
            _app._rail_group(
                self.context.pn,
                "material",
                "Material and surface",
                self.context.material,
                self.context.material_details,
                self.context.human_upload,
                self.context.human_diffuse,
                self.context.human_source,
            ),
            _app._rail_group(
                self.context.pn,
                "radar",
                "Radar orientation",
                self.context.numeric_row(
                    "hermes-static-radar-orientation",
                    self.context.radar_yaw,
                    self.context.radar_pitch,
                    self.context.radar_roll,
                ),
            ),
            _app._rail_group(self.context.pn, "export", "Export", self.context.physics_download),
            sizing_mode="stretch_width",
            min_width=380,
            max_width=380,
            margin=0,
            css_classes=["hermes-rail", "hermes-scene-controls", "hermes-static-controls"],
        )
        self.physics_workspace = self.context.pn.Column(
            self.context.pn.Column(
                self.context.pn.Row(
                    self.context.diagnostic_top_k,
                    self.context.show_rt_paths,
                    sizing_mode="stretch_width",
                ),
                self.context.diagnostics_status,
                self.context.scene_interaction,
                self.context.pn.pane.Markdown(
                    "**Mouse:** orbit camera · **Shift + drag:** rotate "
                    "target yaw/pitch · **Wheel:** zoom · **Roll:** use "
                    "the numeric field. Shift-drag previews geometry only; one "
                    "RT/PO simulation runs when the mouse is released.",
                    css_classes=["hermes-compact-hint"],
                ),
                sizing_mode="stretch_width",
                css_classes=["hermes-card"],
            ),
            self.context.pn.Column(
                self.context.physics_profile,
                self.context.physics_channel_profiles,
                sizing_mode="stretch_width",
                css_classes=["hermes-card"],
            ),
            self.context.pn.Column(
                self.context.angle_range_bins,
                self.context.angle_status,
                self.context.angle_map,
                sizing_mode="stretch_width",
                css_classes=["hermes-card"],
            ),
            self.context.pn.Column(
                self.context.pn.pane.HTML(
                    '<div class="hermes-card-title" role="heading" aria-level="3">'
                    "Solver summary</div>",
                    sizing_mode="stretch_width",
                    margin=(0, 10),
                ),
                self.context.diagnostics_summary,
                self.context.diagnostics_depth,
                sizing_mode="stretch_width",
                css_classes=["hermes-card"],
            ),
            sizing_mode="stretch_width",
            margin=0,
            styles={"gap": "16px"},
            css_classes=["hermes-workspace"],
        )
        self.physics_tab = self.context.pn.Row(
            self.physics_controls,
            self.context.pn.Column(
                _app._run_bar(
                    self.context.pn,
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
                    status=self.context.physics_status,
                    actions=(self.context.stop_physics, self.context.refine_physics),
                    banner=self.static_result_banner,
                ),
                self.physics_workspace,
                sizing_mode="stretch_width",
                min_width=0,
                margin=0,
                styles={"gap": "16px"},
                css_classes=["hermes-work"],
            ),
            sizing_mode="stretch_width",
            css_classes=["hermes-page"],
        )

    def _assemble_motion_source_controls(self):
        """Assemble motion source controls in dependency order."""
        self.context.human_room_preset = self.context.pn.widgets.Select(
            label="Prepared scene fallback",
            options={"Empty room": "empty_room", "Tutorial bedroom": "tutorial_bedroom"},
            value="empty_room",
        )
        self.context.human_room_scene_upload = self.context.pn.widgets.FileInput(
            label="Static environment scene (.xml, optional)",
            accept=".xml",
            stylesheets=[_app._file_input_stylesheet(self.context.plot_template, compact=False)],
            description=(
                "Disabled for remotely bound sessions because scene XML can "
                "reference server-side assets. Local uploads are limited to 2 MiB."
            ),
            disabled=self.context.remote_access,
        )
        self.context.default_amass_motion = _app._default_amass_motion_path()
        self.context.default_model_dir = _app._default_smpl_model_dir()
        self.human_room_upload_help = (
            "Upload a pickle-free motion archive with vertices, faces, and times, "
            "or AMASS-like poses, trans, betas, and timing. Supplied meshes are "
            "used directly; pose-only archives need the configured SMPL model. "
            f"Browser uploads are limited to {_app._MAX_MOTION_UPLOAD_BYTES // _app._MIB} MiB."
        )
        self.context.human_room_upload = self.context.pn.widgets.FileInput(
            label="AMASS-like human motion (.npz)",
            accept=".npz",
            stylesheets=[_app._file_input_stylesheet(self.context.plot_template)],
            description=self.human_room_upload_help,
            width=132,
            height=40,
            sizing_mode="fixed",
        )
        self.human_room_upload_label = self.context.pn.Row(
            self.context.pn.pane.Markdown(
                "**AMASS-like human motion (.npz)**",
                margin=0,
                styles={"white-space": "nowrap", "width": "max-content", "flex": "0 0 auto"},
                stylesheets=[
                    ":host {width:max-content !important; padding:0 !important;} "
                    "p {margin:0 !important;}"
                ],
            ),
            self.context.pn.widgets.TooltipIcon(value=self.human_room_upload_help, margin=0),
            sizing_mode="stretch_width",
            styles={"align-items": "center", "justify-content": "flex-start", "gap": "4px"},
            margin=(5, 16),
        )
        self.context.human_room_upload_filename = self.context.pn.widgets.StaticText(
            value=(
                f"Bundled default: {self.context.default_amass_motion.name}"
                if self.context.default_amass_motion is not None
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
        self.human_room_upload_control = self.context.pn.Column(
            self.human_room_upload_label,
            self.context.pn.Row(
                self.context.human_room_upload,
                self.context.human_room_upload_filename,
                sizing_mode="stretch_width",
                styles={"align-items": "center", "gap": "8px"},
            ),
            sizing_mode="stretch_width",
            margin=0,
        )
        self.context.human_room_smpl_model_dir = self.context.pn.widgets.TextInput(
            label="Licensed SMPL model directory",
            value=str(self.context.default_model_dir),
            placeholder="/path/to/smpl_models",
            description=(
                "In remote mode this server-side directory is fixed at launch "
                "from MMWAVE_SMPL_MODEL_DIR and cannot be changed by a browser."
            ),
            disabled=self.context.remote_access,
        )
        self.context.load_motion_preview = self.context.pn.widgets.Button(
            label="Load default motion",
            color="light",
            visible=self.context.default_amass_motion is not None,
            description="Load the bundled motion. Uploaded motion previews automatically.",
        )
        self.context.human_room_position_mode = self.context.pn.widgets.Select(
            label="Human position",
            options={"Use motion-file coordinates": "file", "Place manually": "manual"},
            value="file",
            description=(
                "Use the positions stored in the motion file after its coordinate "
                "conversion, or set the first pose's bounding-box centre manually."
            ),
        )
        self.context.human_room_position_state = {"syncing": False}
        self.human_room_position_limits = _app._human_position_bounds(
            _app.prepared_room_boxes(self.context.human_room_preset.value)
        )
        self.context.human_room_x = self.context.compact_numeric(
            "X [m]",
            start=self.human_room_position_limits[0][0],
            end=self.human_room_position_limits[0][1],
            step=0.01,
            format="0.00[0]",
            value=1.6,
            description="Range-forward position.",
        )
        self.context.human_room_y = self.context.compact_numeric(
            "Y [m]",
            start=self.human_room_position_limits[1][0],
            end=self.human_room_position_limits[1][1],
            step=0.01,
            format="0.00[0]",
            value=0.0,
            description="Lateral-left position.",
        )
        self.context.human_room_z = self.context.compact_numeric(
            "Z [m]",
            start=self.human_room_position_limits[2][0],
            end=self.human_room_position_limits[2][1],
            step=0.01,
            format="0.00[0]",
            value=0.0,
            description="First-pose bounding-box centre above the radar origin.",
        )
        self.human_room_position_controls = self.context.numeric_row(
            "hermes-dynamic-position",
            self.context.human_room_x,
            self.context.human_room_y,
            self.context.human_room_z,
        )
        self.human_room_position_controls.visible = False
        self.context.human_room_yaw = self.context.compact_numeric(
            "Yaw [deg]",
            start=-180,
            end=180,
            step=1,
            value=0,
        )
        self.context.human_room_diffuse = self.context.compact_numeric(
            "Diffuse (RT)",
            start=0.0,
            end=1.0,
            step=0.05,
            value=0.35,
            format="0.00",
            description="Human diffuse reflection coefficient for ray tracing.",
        )

    def _assemble_motion_playback_controls(self):
        """Assemble motion playback controls in dependency order."""
        self.context.human_room_preview_time = self.context.pn.widgets.FloatInput(
            label="Preview [s] (max 0.000)",
            start=0.0,
            end=0.0,
            step=0.01,
            value=0.0,
            format="0.000[000]",
            description="Seconds since the first motion pose. Interpolates between recorded poses.",
            disabled=True,
            width=190,
        )
        self.context.human_room_player = self.context.pn.widgets.Player(
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
        self.human_room_previous_frame = self.context.pn.widgets.Button(
            label="Prev",
            icon="player-skip-back",
            color="light",
            disabled=True,
            width=40,
            height=38,
        )
        self.human_room_reverse = self.context.pn.widgets.Button(
            label="Reverse",
            icon="player-play",
            color="light",
            disabled=True,
            width=40,
            height=38,
        )
        self.human_room_fast_reverse = self.context.pn.widgets.Button(
            label="Fast reverse",
            icon="player-track-prev",
            color="light",
            disabled=True,
            width=40,
            height=38,
        )
        self.context.human_room_play_pause = self.context.pn.widgets.Button(
            label="Play",
            icon="player-play",
            color="light",
            disabled=True,
            width=40,
            height=38,
        )
        self.human_room_fast_forward = self.context.pn.widgets.Button(
            label="Fast forward",
            icon="player-track-next",
            color="light",
            disabled=True,
            width=40,
            height=38,
        )
        self.human_room_next_frame = self.context.pn.widgets.Button(
            label="Next",
            icon="player-skip-forward",
            color="light",
            disabled=True,
            width=40,
            height=38,
        )
        self.context.human_room_transport_buttons = (
            self.human_room_previous_frame,
            self.human_room_fast_reverse,
            self.human_room_reverse,
            self.context.human_room_play_pause,
            self.human_room_fast_forward,
            self.human_room_next_frame,
        )
        for self.button in self.context.human_room_transport_buttons:
            self.button.description = self.button.label
            self.button.margin = 0
            self.button.stylesheets = [
                ".bk-btn {font-size:0 !important;} "
                ".bk-TablerIcon {font-size:18px !important; width:18px; height:18px;}"
            ]
        self.human_room_reverse.stylesheets += [".bk-TablerIcon {transform:rotate(180deg);}"]
        self.context.human_room_frame_counter = self.context.pn.widgets.StaticText(
            value="t = — s", width=180
        )
        self.human_room_playback_controls = self.context.pn.Row(
            self.context.pn.Spacer(sizing_mode="stretch_width"),
            *self.context.human_room_transport_buttons,
            self.context.human_room_frame_counter,
            self.context.pn.Spacer(sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            styles={"align-items": "center", "gap": "6px"},
        )
        self.context.human_room_start_time = self.context.pn.widgets.FloatInput(
            label="Start [s]",
            value=0.0,
            start=0.0,
            end=0.05,
            step=0.05,
            format="0.000",
            description=(
                "Motion time, in seconds after the first pose, at which radar "
                "frame 0 starts. The motion then plays in real time."
            ),
            disabled=True,
        )
        self.context.human_room_end_time = self.context.pn.widgets.FloatInput(
            label="End [s] (max 0.050)",
            value=0.05,
            start=0.0,
            end=0.05,
            step=0.05,
            format="0.000",
            description=(
                "Every radar frame whose chirps finish by this motion time is simulated."
            ),
            disabled=True,
        )
        self.context.human_room_window_summary = self.context.pn.pane.Markdown(
            "1 radar frame",
            margin=(0, 10),
            styles={"font-size": "0.9em"},
        )
        self.context.human_room_solver_checkboxes = {
            mode: self.context.pn.widgets.Checkbox(
                label=_app._SOLVER_LABELS[mode],
                value=mode == "hybrid_po",
                margin=0,
                styles={"width": "max-content", "flex": "0 0 auto"},
            )
            for mode in _app.SOLVERS
        }
        self.human_room_solver_options = [
            self.context.pn.Row(
                self.context.human_room_solver_checkboxes[mode],
                self.context.pn.widgets.TooltipIcon(value=_app._SOLVER_HELP[mode], margin=0),
                sizing_mode="stretch_width",
                min_width=0,
                margin=0,
                styles={"align-items": "center", "justify-content": "flex-start", "gap": "4px"},
            )
            for mode in _app.SOLVERS
        ]
        self.human_room_solver_controls = self.context.pn.Column(
            *(
                self.context.pn.Row(
                    *self.human_room_solver_options[index : index + 2],
                    sizing_mode="stretch_width",
                    margin=(5, 10),
                    styles={"gap": "12px"},
                )
                for index in range(0, len(self.human_room_solver_options), 2)
            ),
            sizing_mode="stretch_width",
            margin=0,
        )
        self.context.run_human_room = self.context.pn.widgets.Button(
            label="Start",
            icon="player-play",
            color="primary",
            width=120,
            height=40,
            sizing_mode="fixed",
            disabled=True,
        )
        self.context.stop_human_room = self.context.pn.widgets.Button(
            label="Stop",
            icon="player-stop",
            color="light",
            width=105,
            height=40,
            sizing_mode="fixed",
            disabled=True,
        )

    def _assemble_motion_results_and_state(self):
        """Assemble motion results and state in dependency order."""
        self.context.human_room_download = self.context.pn.widgets.FileDownload(
            callback=lambda: _app.BytesIO(),
            filename="hermes-human-room-bundle.zip",
            label="Export loadable bundle",
            color="primary",
            variant="solid",
            height=44,
            sizing_mode="stretch_width",
            stylesheets=[".bk-btn {font-weight:700 !important;}"],
            disabled=True,
        )
        self.context.human_room_status = self.context.pn.pane.Markdown(
            "Load human motion, verify its placement, then start the "
            "selected simulation modes.",
            css_classes=["hermes-status"],
        )
        self.context.human_room_results = self.context.pn.Column(
            self.context.pn.pane.Markdown("Start one or more simulation modes to view results."),
            sizing_mode="stretch_width",
        )
        self.context.human_room_diagnostic_top_k = self.context.pn.widgets.IntInput(
            label="Top RT rays",
            value=12,
            start=1,
            end=50,
            step=1,
            width=135,
        )
        self.context.run_human_room_diagnostics_button = self.context.pn.widgets.Button(
            label="Show",
            icon="route",
            color="light",
            disabled=True,
            width=100,
            align="end",
        )
        self.context.human_room_diagnostics_status = self.context.pn.pane.Markdown(
            "Run a simulation to enable one-frame solver overlays.",
            css_classes=["hermes-status"],
        )
        self.context.human_room_state = {
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
        self.context.human_room_playback_state = {
            "syncing": False,
            "normal_step": 1,
            "fast_step": 4,
        }
        self.context.human_room_window_state = {"syncing": False}

        self.context.show_human_room_playing_state = self._human_room.show_human_room_playing_state

        self.context.configure_human_room_frame_controls = (
            self._human_room.configure_human_room_frame_controls
        )

        self.context.human_room_pose_counter = self._human_room.human_room_pose_counter

        self.context.human_room_window_frame_count = self._human_room.human_room_window_frame_count

        self.context.update_human_room_window_summary = (
            self._human_room.update_human_room_window_summary
        )

        self.context.configure_human_room_window_controls = (
            self._human_room.configure_human_room_window_controls
        )

        self.context.current_human_room_source = self._human_room.current_human_room_source

        self.context.on_human_room_upload_filename = self._human_room.on_human_room_upload_filename

        self.context.current_human_room_scene = self._human_room.current_human_room_scene

        self.context.current_room_boxes_override = self._human_room.current_room_boxes_override

        self.context.sync_human_position_bounds = self._human_room.sync_human_position_bounds

        self.context.current_motion_cache_key = self._human_room.current_motion_cache_key

        self.context.current_human_position = self._human_room.current_human_position

        self.context.current_dynamic_radar_orientation = (
            self._human_room.current_dynamic_radar_orientation
        )

        self.context.mark_human_room_stale = self._human_room.mark_human_room_stale

        self.context.refresh_human_room_preview = self._human_room.refresh_human_room_preview

        self.initial_human_preview = None
        self.context.human_room_scene = self.context.pn.pane.Plotly(
            _app._human_room_scene_figure(
                self.initial_human_preview,
                self.context.go,
                frame_index=0,
                radar_orientation_deg=self.context.current_dynamic_radar_orientation(),
                room_boxes_override=self.context.current_room_boxes_override(),
                plot_template=self.context.plot_template,
            ),
            height=570,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["resetCameraDefault3d"],
            },
        )
        self.context.human_room_floor_status = self.context.pn.pane.Markdown(
            "Load human motion to check floor placement.",
            css_classes=["hermes-status"],
        )

        self.context.update_human_room_floor_status = (
            self._human_room.update_human_room_floor_status
        )

        self.context.human_room_scene.param.watch(
            self.context.update_human_room_floor_status, "object"
        )
        self.context.update_human_room_floor_status()

    def _assemble_motion_bindings(self):
        """Assemble motion bindings in dependency order."""
        self.human_room_scene_interaction = _app._dynamic_scene_interaction_component(
            self.context.pn,
            self.context.human_room_scene,
        )
        if self.initial_human_preview is not None:
            self.last_frame = max(
                len(self.initial_human_preview.mesh_sequence.times) - 1,
                0,
            )
            self.context.configure_human_room_frame_controls(
                last_frame=self.last_frame,
                selected_frame=0,
                disabled=self.last_frame == 0,
                interval_ms=_app._motion_playback_interval_ms(
                    self.initial_human_preview.mesh_sequence.times
                ),
                normal_step=_app._motion_playback_step(
                    self.initial_human_preview.mesh_sequence.times
                ),
                fast_step=_app._motion_fast_playback_step(
                    self.initial_human_preview.mesh_sequence.times
                ),
            )
            self.context.configure_human_room_window_controls(
                duration_s=float(self.initial_human_preview.mesh_sequence.times[-1]),
                reset_window=True,
                disabled=False,
            )

        self.context.on_human_room_controls = self._human_room.on_human_room_controls

        self.context.on_human_position_input = self._human_room.on_human_position_input

        self.context.on_motion_coordinate_change = self._human_room.on_motion_coordinate_change

        self.context.diagnostics_for_time = self._human_room.diagnostics_for_time

        self.context.display_human_room_frame = self._human_room.display_human_room_frame

        self.context.on_human_room_preview_time = self._human_room.on_human_room_preview_time

        self.context.on_human_room_player = self._human_room.on_human_room_player

        self.context.on_human_room_player_direction = (
            self._human_room.on_human_room_player_direction
        )

        self.context.start_fast_human_room_playback = (
            self._human_room.start_fast_human_room_playback
        )

        self.context.start_human_room_playback = self._human_room.start_human_room_playback

        self.context.pause_human_room_playback = self._human_room.pause_human_room_playback

        self.context.step_human_room_frame = self._human_room.step_human_room_frame

        self.context.on_human_room_previous_frame = self._human_room.on_human_room_previous_frame

        self.context.on_human_room_reverse = self._human_room.on_human_room_reverse

        self.context.on_human_room_fast_reverse = self._human_room.on_human_room_fast_reverse

        self.context.on_human_room_fast_forward = self._human_room.on_human_room_fast_forward

        self.context.on_human_room_play_pause = self._human_room.on_human_room_play_pause

        self.context.on_human_room_next_frame = self._human_room.on_human_room_next_frame

        self.context.on_dynamic_radar_preview_change = (
            self._human_room.on_dynamic_radar_preview_change
        )

        self.context.on_human_run_option_change = self._human_room.on_human_run_option_change

        self.context.on_human_room_window_change = self._human_room.on_human_room_window_change

        self.context.human_room_run_input_widgets = (
            self.context.settings_button,
            self.apply_solver_settings,
            *self.context.solver_setting_widgets.values(),
            *self.context.motion_coordinate_widgets.values(),
            self.context.board,
            self.context.tx_schedule,
            self.context.tx_antennas,
            self.context.rx_antennas,
            self.context.antenna_pattern,
            self.context.cosine_3db_beamwidth,
            self.context.carrier_frequency,
            self.context.chirp_slope,
            self.context.chirp_duration,
            self.context.chirp_repetition,
            self.context.sampling_frequency,
            self.context.adc_sampling_mode,
            self.context.adc_if_bandwidth,
            self.context.num_adc_samples,
            self.context.num_chirps,
            self.context.frame_period,
            self.context.human_room_preset,
            self.context.human_room_scene_upload,
            self.context.human_room_upload,
            self.context.human_room_smpl_model_dir,
            self.context.human_room_position_mode,
            self.context.human_room_x,
            self.context.human_room_y,
            self.context.human_room_z,
            self.context.human_room_yaw,
            self.context.human_room_diffuse,
            self.context.human_room_start_time,
            self.context.human_room_end_time,
            *self.context.human_room_solver_checkboxes.values(),
            self.context.dynamic_radar_yaw,
            self.context.dynamic_radar_pitch,
            self.context.dynamic_radar_roll,
        )

        self.context.on_run_human_room = self._human_room.on_run_human_room

        self.context.on_stop_human_room = self._human_room.on_stop_human_room

        self.context.on_run_human_room_diagnostics = self._human_room.on_run_human_room_diagnostics

        self.context.run_human_room.on_click(self.context.on_run_human_room)
        self.context.stop_human_room.on_click(self.context.on_stop_human_room)
        self.context.run_human_room_diagnostics_button.on_click(
            self.context.on_run_human_room_diagnostics
        )
        self.human_room_previous_frame.on_click(self.context.on_human_room_previous_frame)
        self.human_room_reverse.on_click(self.context.on_human_room_reverse)
        self.human_room_fast_reverse.on_click(self.context.on_human_room_fast_reverse)
        self.context.human_room_play_pause.on_click(self.context.on_human_room_play_pause)
        self.human_room_fast_forward.on_click(self.context.on_human_room_fast_forward)
        self.human_room_next_frame.on_click(self.context.on_human_room_next_frame)
        self.context.load_motion_preview.on_click(self.context.on_human_room_controls)
        self.context.human_room_upload.param.watch(self.context.on_human_room_controls, "value")
        self.context.human_room_upload.param.watch(
            self.context.on_human_room_upload_filename,
            "filename",
        )
        self.context.human_room_scene_upload.param.watch(
            self.context.on_human_room_controls, "value"
        )
        self.context.human_room_smpl_model_dir.param.watch(
            self.context.on_human_room_controls, "value"
        )
        self.context.human_room_preset.param.watch(self.context.on_human_room_controls, "value")
        self.context.human_room_position_mode.param.watch(
            lambda event: setattr(
                self.human_room_position_controls, "visible", event.new == "manual"
            ),
            "value",
        )
        self.context.human_room_position_mode.param.watch(
            self.context.on_human_room_controls, "value"
        )
        for self.widget in self.context.motion_coordinate_widgets.values():
            self.widget.param.watch(self.context.on_motion_coordinate_change, "value")
        for self.numeric in (
            self.context.human_room_x,
            self.context.human_room_y,
            self.context.human_room_z,
        ):
            self.numeric.param.watch(self.context.on_human_position_input, "value")
        for self.numeric in (self.context.human_room_yaw, self.context.human_room_diffuse):
            self.numeric.param.watch(self.context.on_human_room_controls, "value")
        self.context.human_room_preview_time.param.watch(
            self.context.on_human_room_preview_time, "value"
        )
        self.context.human_room_player.param.watch(self.context.on_human_room_player, "value")
        self.context.human_room_player.param.watch(
            self.context.on_human_room_player_direction,
            "direction",
        )
        self.context.human_room_start_time.param.watch(
            self.context.on_human_room_window_change,
            "value",
        )
        self.context.human_room_end_time.param.watch(
            self.context.on_human_room_window_change,
            "value",
        )
        for self.radar_timing_widget in (
            self.context.frame_period,
            self.context.num_chirps,
            self.context.chirp_repetition,
            self.context.tx_schedule,
            self.context.tx_antennas,
        ):
            self.radar_timing_widget.param.watch(
                self.context.update_human_room_window_summary,
                "value",
            )
        for self.mode_checkbox in self.context.human_room_solver_checkboxes.values():
            self.mode_checkbox.param.watch(self.context.on_human_run_option_change, "value")
        for self.widget, self.parameter in (
            (self.context.dynamic_radar_yaw, "value"),
            (self.context.dynamic_radar_pitch, "value"),
            (self.context.dynamic_radar_roll, "value"),
            (self.context.board, "value"),
            (self.context.tx_schedule, "value"),
            (self.context.tx_antennas, "value"),
            (self.context.rx_antennas, "value"),
            (self.context.antenna_pattern, "value"),
            (self.context.cosine_3db_beamwidth, "value"),
            (self.context.carrier_frequency, "value"),
            (self.context.chirp_slope, "value"),
            (self.context.chirp_duration, "value"),
            (self.context.chirp_repetition, "value"),
            (self.context.sampling_frequency, "value"),
            (self.context.adc_sampling_mode, "value"),
            (self.context.adc_if_bandwidth, "value"),
            (self.context.num_adc_samples, "value"),
            (self.context.num_chirps, "value"),
            (self.context.frame_period, "value"),
        ):
            self.widget.param.watch(self.context.on_dynamic_radar_preview_change, self.parameter)

    def _assemble_motion_layout(self):
        """Assemble motion layout in dependency order."""
        self.human_room_controls = self.context.pn.Column(
            _app._rail_nav_component(
                self.context.pn,
                (
                    ("motion", "Motion"),
                    ("modes", "Modes"),
                    ("human", "Human"),
                    ("radar", "Radar"),
                ),
            ),
            _app._rail_group(
                self.context.pn,
                "motion",
                "Motion and scene",
                self.human_room_upload_control,
                self.context.load_motion_preview,
                self.context.human_room_preset,
                self.context.human_room_scene_upload,
                self.context.human_room_smpl_model_dir,
            ),
            _app._rail_group(
                self.context.pn,
                "modes",
                "Simulation modes",
                self.human_room_solver_controls,
                self.context.pn.Row(
                    self.context.human_room_start_time,
                    self.context.human_room_end_time,
                    sizing_mode="stretch_width",
                ),
                self.context.human_room_window_summary,
            ),
            _app._rail_group(
                self.context.pn,
                "human",
                "Human placement",
                self.context.human_room_position_mode,
                self.human_room_position_controls,
                self.context.numeric_row(
                    "hermes-dynamic-orientation",
                    self.context.human_room_yaw,
                    self.context.human_room_diffuse,
                ),
            ),
            _app._rail_group(
                self.context.pn,
                "radar",
                "Radar orientation",
                self.context.numeric_row(
                    "hermes-dynamic-radar-orientation",
                    self.context.dynamic_radar_yaw,
                    self.context.dynamic_radar_pitch,
                    self.context.dynamic_radar_roll,
                ),
            ),
            _app._rail_group(
                self.context.pn, "export", "Export", self.context.human_room_download
            ),
            sizing_mode="stretch_width",
            min_width=380,
            max_width=380,
            margin=0,
            css_classes=["hermes-rail", "hermes-scene-controls", "hermes-dynamic-controls"],
        )
        self.context.human_room_view_tabs = self.context.pn.Tabs(
            (
                "Scene and motion",
                self.context.pn.Column(
                    self.context.pn.Row(
                        self.context.human_room_preview_time,
                        self.context.human_room_diagnostic_top_k,
                        self.context.run_human_room_diagnostics_button,
                        sizing_mode="stretch_width",
                    ),
                    self.context.human_room_diagnostics_status,
                    self.context.human_room_floor_status,
                    self.human_room_scene_interaction,
                    self.context.human_room_player,
                    self.human_room_playback_controls,
                ),
            ),
            (
                "Simulation results",
                self.context.human_room_results,
            ),
            # Keep both panes mounted.  Completed products can arrive while the
            # result tab is already selected (including after a theme restore);
            # lazy mounting can otherwise leave the browser attached to the old
            # placeholder even though the server has replaced the result column.
            dynamic=False,
            active=0,
            sizing_mode="stretch_width",
        )
        self.dynamic_result_banner = self.context.pn.Column(
            visible=False, name="Dynamic result settings"
        )
        self.human_room_tab = self.context.pn.Row(
            self.human_room_controls,
            self.context.pn.Column(
                _app._run_bar(
                    self.context.pn,
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
                    status=self.context.human_room_status,
                    actions=(self.context.run_human_room, self.context.stop_human_room),
                    banner=self.dynamic_result_banner,
                ),
                self.context.pn.Column(
                    self.context.human_room_view_tabs,
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

    def _assemble_bundle_source_controls(self):
        """Assemble bundle source controls in dependency order."""
        self.fixture_options = _app._source_fixture_options()
        for self.fixture_path in self.fixture_options.values():
            if self.fixture_path:
                self.context.trusted_bundle_paths.add(_app._server_path_key(self.fixture_path))
        if self.initial_bundle:
            # ``initial_bundle`` is supplied by the server operator on the CLI,
            # not by a browser session.
            self.context.trusted_bundle_paths.add(_app._server_path_key(self.initial_bundle))
        self.default_fixture = next(
            (
                value
                for label, value in self.fixture_options.items()
                if label == "rtpose_seq10_frame0020"
            ),
            "",
        )
        self.context.fixture = self.context.pn.widgets.Select(
            label="Measurement benchmark",
            options=self.fixture_options,
            value="" if self.initial_bundle else self.default_fixture,
        )
        self.context.bundle_path = self.context.pn.widgets.TextInput(
            label="HERMES bundle directory or ZIP",
            value=str(self.initial_bundle or self.context.fixture.value),
            placeholder="/path/to/hermes_bundle or hermes-bundle.zip",
            description=(
                "Direct server-side path entry is disabled for remote sessions; "
                "use an included fixture or browser upload."
            ),
            disabled=self.context.remote_access,
        )
        self.context.measurement_dropper = self.context.pn.widgets.FileDropper(
            label="Drop primary bundle ZIP or folder",
            multiple=True,
            previews=[],
            visible=self.context.fixture.value == "",
            max_file_size="256MB",
            max_files=_app._MAX_DROPPED_FILES,
            max_total_file_size="512MB",
            sizing_mode="stretch_width",
        )
        self.context.custom_bundle_path = {"value": str(self.initial_bundle or "")}

        self.context.select_measurement_source = self._bundles.select_measurement_source

        self.context.fixture.param.watch(self.context.select_measurement_source, "value")
        self.context.background = self.context.pn.widgets.Checkbox(
            label="Background subtraction",
            value=False,
            stylesheets=[_app._checkbox_stylesheet(self.context.plot_template)],
        )
        self.context.clutter = self.context.pn.widgets.Select(
            label="Clutter removal",
            options=["none", "mean"],
            value="none",
        )
        self.context.load_bundle_button = self.context.pn.widgets.Button(
            label="Load and validate bundle",
            color="primary",
            icon="folder-open",
        )
        self.context.measurement_status = self.context.pn.pane.Markdown(
            "",
            min_height=76,
            margin=(2, 10),
            css_classes=["hermes-status", "hermes-bundle-status"],
        )
        self.context.bundle_step_one_marker = self.context.pn.pane.HTML(
            "",
            width=30,
            height=30,
            sizing_mode="fixed",
            margin=0,
        )
        self.context.bundle_step_two_marker = self.context.pn.pane.HTML(
            "",
            width=30,
            height=30,
            sizing_mode="fixed",
            margin=0,
        )
        self.context.bundle_step_one_label = self.context.pn.pane.HTML(
            "<strong>Step 1 · Load and validate</strong>",
            height=30,
            margin=0,
        )
        self.context.bundle_step_two_label = self.context.pn.pane.HTML(
            "<strong>Step 2 · Configure and compare</strong>",
            height=30,
            margin=0,
        )

        self.context.bundle_step_heading = self._bundles.bundle_step_heading
        self.context.bundle_scene_status = self.context.pn.pane.Markdown(
            "Load the primary bundle to visualize its declared scene."
        )
        self.context.bundle_scene_plot = self.context.pn.pane.Plotly(
            _app._empty_figure(self.context.go, self.context.plot_template), height=570
        )
        self.context.primary_adc_plot = self.context.pn.pane.Plotly(
            _app._empty_figure(self.context.go, self.context.plot_template),
            height=330,
            sizing_mode="stretch_width",
        )
        self.context.adc_selectors = {
            name: self.context.pn.widgets.IntInput(
                label=f"ADC {name}", value=0, start=0, end=0, disabled=True, width=130
            )
            for name in ("frame", "chirp", "channel")
        }
        self.context.range_profile_frame = self.context.pn.widgets.IntSlider(
            label="Range-profile frame index (0–0)",
            start=0,
            end=1,
            value=0,
            disabled=True,
        )
        self.context.range_profile_frame.link(
            self.context.adc_selectors["frame"], value="value", bidirectional=True
        )
        self.context.range_profile_previous = self.context.pn.widgets.Button(
            label="Previous frame", disabled=True
        )
        self.context.range_profile_next = self.context.pn.widgets.Button(
            label="Next frame", disabled=True
        )
        self.context.range_profile_frame_controls = self.context.pn.Row(
            self.context.range_profile_previous,
            self.context.range_profile_next,
            self.context.range_profile_frame,
            visible=False,
        )
        self.context.candidate_adc_results = self.context.pn.Column(
            sizing_mode="stretch_width",
            visible=False,
        )
        self.context.range_profile_plot = self.context.pn.pane.Plotly(
            _app._empty_figure(self.context.go, self.context.plot_template),
            height=350,
            sizing_mode="stretch_width",
        )
        self.context.range_profile_plot.param.watch(
            lambda event: setattr(event.obj, "height", event.new.layout.height or 380),
            "object",
        )
        self.context.primary_range_time_plot = self.context.pn.pane.Plotly(
            _app._empty_figure(self.context.go, self.context.plot_template),
            height=380,
            sizing_mode="stretch_width",
        )
        self.context.candidate_range_time_results = self.context.pn.Column(
            sizing_mode="stretch_width",
            visible=False,
        )
        self.context.primary_range_doppler_plot = self.context.pn.pane.Plotly(
            _app._empty_figure(self.context.go, self.context.plot_template),
            height=380,
            sizing_mode="stretch_width",
        )
        self.context.candidate_range_doppler_results = self.context.pn.Column(
            sizing_mode="stretch_width",
            visible=False,
        )
        self.context.comparison_source = self.context.pn.widgets.Select(
            label="Comparison source",
            options={
                "Run simulation from loaded bundle": "run",
                "Load previously saved ADC or bundle": "saved",
            },
            value="run",
        )

    def _assemble_comparison_controls(self):
        """Assemble comparison controls in dependency order."""
        self.comparison_solver_options = {
            _app._SOLVER_LABELS[solver]: solver for solver in _app.SOLVERS
        }
        self.context.comparison_solvers = self.context.pn.widgets.MultiChoice(
            label="Bundle solvers",
            options=self.comparison_solver_options,
            value=["po"],
            placeholder="+ Add solver",
        )
        self.context.comparison_channel_zero = self.context.pn.widgets.Checkbox(
            label="Channel 0 only",
            value=False,
            stylesheets=[_app._checkbox_stylesheet(self.context.plot_template)],
        )
        self.context.simulated_path = self.context.pn.widgets.TextInput(
            label="External simulated ADC or HERMES bundle",
            placeholder="/path/to/simulated_adc.npz, bundle ZIP, or directory",
            description=(
                "Direct server-side path entry is disabled for remote sessions; "
                "upload the candidate through the browser."
            ),
            disabled=self.context.remote_access,
        )
        self.context.simulated_dropper = self.context.pn.widgets.FileDropper(
            label="Drop simulated ADC, bundle ZIP, or folder",
            multiple=True,
            previews=[],
            visible=False,
            max_file_size="256MB",
            max_files=_app._MAX_DROPPED_FILES,
            max_total_file_size="512MB",
            sizing_mode="stretch_width",
        )
        self.context.compare_button = self.context.pn.widgets.Button(
            label="Compare ADC products",
            color="primary",
            icon="chart-histogram",
            disabled=True,
            height=46,
            sizing_mode="stretch_width",
            margin=0,
        )
        self.context.simulated_adc_download = self.context.pn.widgets.FileDownload(
            callback=lambda: _app.BytesIO(),
            filename="simulated_adc.npz",
            label="\u200b",
            color="primary",
            variant="solid",
            icon="download",
            description="Download simulated ADC",
            width=44,
            height=40,
            sizing_mode="fixed",
            margin=0,
            disabled=True,
            css_classes=["hermes-icon-download"],
        )
        self.context.comparison_dependency_hint = self.context.pn.pane.Markdown(
            "🔒 **Load and validate a primary bundle to enable this step.**",
            margin=(2, 0, 4, 0),
            css_classes=["hermes-compact-hint"],
            styles={
                "color": _app._plot_theme_tokens(self.context.plot_template)["foreground"],
                "opacity": "0.78",
            },
        )
        self.context.stop_simulation_confirmation = _app._confirmation_dialog_component(
            self.context.pn
        )
        self.context.comparison_status = self.context.pn.pane.Markdown(
            "",
            visible=False,
            css_classes=["hermes-status", "hermes-comparison-output"],
            styles={"color": _app._plot_theme_tokens(self.context.plot_template)["foreground"]},
        )
        self.context.metrics_pane = self.context.pn.pane.JSON(
            {"status": "No comparison."},
            depth=3,
            css_classes=["hermes-comparison-output"],
            styles={"color": _app._plot_theme_tokens(self.context.plot_template)["foreground"]},
            stylesheets=[_app._comparison_json_stylesheet(self.context.plot_template)],
        )
        self.context.measurement_state = {
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
        self.context.bundle_palette = _app._plot_theme_tokens(self.context.plot_template)

        self.context.refresh_adc_selection = self._bundles.refresh_adc_selection

        self.context.refresh_profile_selection = self._bundles.refresh_profile_selection

        self.context.configure_adc_selection = self._bundles.configure_adc_selection

        for self.selector in self.context.adc_selectors.values():
            self.selector.param.watch(self.context.refresh_adc_selection, "value")
        self.context.range_profile_frame.param.watch(
            self.context.refresh_profile_selection, "value"
        )
        self.context.range_profile_previous.on_click(
            lambda _event: setattr(
                self.context.range_profile_frame,
                "value",
                max(
                    self.context.range_profile_frame.start,
                    self.context.range_profile_frame.value - 1,
                ),
            )
        )
        self.context.range_profile_next.on_click(
            lambda _event: setattr(
                self.context.range_profile_frame,
                "value",
                min(
                    self.context.adc_selectors["frame"].end,
                    self.context.range_profile_frame.value + 1,
                ),
            )
        )

        self.context.current_comparison_signature = self._bundles.current_comparison_signature

        self.context.bundle_step_marker = self._bundles.bundle_step_marker

        self.context.update_bundle_step_rail = self._bundles.update_bundle_step_rail

        self.context.set_measurement_status = self._bundles.set_measurement_status

        self.context.set_comparison_status = self._bundles.set_comparison_status

        self.context.set_measurement_status(
            "idle",
            "No bundle loaded yet",
            "Choose an included fixture or custom bundle, then load and validate it.",
        )
        self.context.update_bundle_step_rail(0)
        self.context.candidate_product_results = (
            self.context.candidate_adc_results,
            self.context.candidate_range_time_results,
            self.context.candidate_range_doppler_results,
        )
        self.context.primary_product_plots = (
            self.context.primary_adc_plot,
            self.context.range_profile_plot,
            self.context.primary_range_time_plot,
            self.context.primary_range_doppler_plot,
        )

    def _assemble_comparison_bindings(self):
        """Assemble comparison bindings in dependency order."""
        self.context.clear_candidate_products = self._bundles.clear_candidate_products

        self.context.clear_primary_products = self._bundles.clear_primary_products

        self.context.invalidate_loaded_measurement = self._bundles.invalidate_loaded_measurement

        self.context.bundle_path.param.watch(self.context.invalidate_loaded_measurement, "value")

        self.context.materialize_drop = self._bundles.materialize_drop

        self.context.bind_file_dropper = self._bundles.bind_file_dropper

        self.context.bind_file_dropper(
            "measurement",
            self.context.measurement_dropper,
            self.context.bundle_path,
            self.context.measurement_status,
        )
        self.context.bind_file_dropper(
            "simulated",
            self.context.simulated_dropper,
            self.context.simulated_path,
            self.context.comparison_status,
        )

        self.context.on_load_bundle = self._bundles.on_load_bundle

        self.context.export_simulated_adc = self._bundles.export_simulated_adc

        self.context.on_confirm_stop_simulation = self._bundles.on_confirm_stop_simulation

        self.context.stop_simulation_confirmation.param.watch(
            self.context.on_confirm_stop_simulation,
            "confirmation_sequence",
        )

        self.context.render_comparison_results = self._bundles.render_comparison_results

        self.context.on_compare = self._bundles.on_compare

        self.context.load_bundle_button.on_click(self.context.on_load_bundle)
        self.context.compare_button.on_click(self.context.on_compare)

        self.context.update_comparison_source_visibility = (
            self._bundles.update_comparison_source_visibility
        )

        self.context.comparison_source.param.watch(
            self.context.update_comparison_source_visibility,
            "value",
        )
        self.context.comparison_solvers.param.watch(
            self.context.update_comparison_source_visibility,
            "value",
        )
        self.context.simulated_path.param.watch(
            self.context.update_comparison_source_visibility,
            "value",
        )

        self.context.invalidate_comparison_configuration = (
            self._bundles.invalidate_comparison_configuration
        )

        for self.comparison_widget in (
            self.context.background,
            self.context.clutter,
            self.context.comparison_source,
            self.context.comparison_solvers,
            self.context.comparison_channel_zero,
            self.context.simulated_path,
        ):
            self.comparison_widget.param.watch(
                self.context.invalidate_comparison_configuration,
                "value",
            )
        self.context.update_comparison_source_visibility()

    def _assemble_comparison_layout(self):
        """Assemble comparison layout in dependency order."""
        self.measurement_controls = _app._rail_group(
            self.context.pn,
            "load",
            "Step 1 · Primary bundle",
            self.context.fixture,
            self.context.pn.Column(self.context.bundle_path, visible=False),
            self.context.measurement_dropper,
            self.context.pn.Row(self.context.background, self.context.clutter),
            self.context.load_bundle_button,
            self.context.measurement_status,
            heading=self.context.bundle_step_heading(
                self.context.bundle_step_one_marker, self.context.bundle_step_one_label
            ),
        )
        self.channel_download_row = self.context.pn.Row(
            self.context.comparison_channel_zero,
            self.context.pn.Spacer(sizing_mode="stretch_width"),
            self.context.simulated_adc_download,
            sizing_mode="stretch_width",
            margin=0,
            styles={"align-items": "center"},
        )
        self.comparison_actions = self.context.pn.Column(
            self.channel_download_row,
            self.context.pn.layout.Divider(margin=(2, 0)),
            self.context.comparison_dependency_hint,
            self.context.compare_button,
            sizing_mode="stretch_width",
            margin=(2, 10),
        )
        self.comparison_controls = _app._rail_group(
            self.context.pn,
            "compare",
            "Step 2 · Bundle comparison",
            self.context.comparison_source,
            self.context.comparison_solvers,
            self.context.pn.Column(self.context.simulated_path, visible=False),
            self.context.simulated_dropper,
            self.comparison_actions,
            self.context.pn.pane.Markdown(
                "Radar and solver settings: **primary bundle**. The Settings dialog is used by Static Target and Dynamic Scenes."
            ),
            heading=self.context.bundle_step_heading(
                self.context.bundle_step_two_marker, self.context.bundle_step_two_label
            ),
        )
        self.context.measurement_view_tabs = self.context.pn.Tabs(
            (
                "Scene",
                self.context.pn.Column(
                    self.context.bundle_scene_status,
                    self.context.bundle_scene_plot,
                ),
            ),
            (
                "ADC",
                self.context.pn.Column(
                    self.context.pn.Row(*self.context.adc_selectors.values()),
                    self.context.pn.pane.Markdown("Frame, chirp and channel indices start at 0."),
                    self.context.pn.Row(
                        self.context.primary_adc_plot, self.context.candidate_adc_results
                    ),
                ),
            ),
            (
                "Range profile",
                self.context.pn.Column(
                    self.context.range_profile_frame_controls, self.context.range_profile_plot
                ),
            ),
            (
                "Range time",
                self.context.pn.Row(
                    self.context.primary_range_time_plot, self.context.candidate_range_time_results
                ),
            ),
            (
                "Range Doppler",
                self.context.pn.Row(
                    self.context.primary_range_doppler_plot,
                    self.context.candidate_range_doppler_results,
                ),
            ),
            (
                "Report",
                self.context.pn.Column(
                    self.context.pn.pane.Markdown("### Detailed comparison report"),
                    self.context.metrics_pane,
                    sizing_mode="stretch_width",
                ),
            ),
            # Eagerly mount every Plotly pane. With lazy children, a theme reload
            # can restore the active index before its newly themed pane exists,
            # leaving stale or blank plots until every tab is visited manually.
            dynamic=False,
            sizing_mode="stretch_width",
        )
        self.comparison_result_banner = self.context.pn.Column(
            visible=False,
            name="Comparison result settings",
        )
        self.measurement_rail = self.context.pn.Column(
            _app._rail_nav_component(
                self.context.pn,
                (
                    ("load", "Step 1 · Load"),
                    ("compare", "Step 2 · Compare"),
                ),
            ),
            self.measurement_controls,
            self.comparison_controls,
            sizing_mode="stretch_width",
            min_width=380,
            max_width=380,
            margin=0,
            css_classes=["hermes-rail", "hermes-bundle-panels"],
        )
        self.measurement_tab = self.context.pn.Row(
            self.measurement_rail,
            self.context.pn.Column(
                _app._run_bar(
                    self.context.pn,
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
                    status=self.context.comparison_status,
                    banner=self.comparison_result_banner,
                ),
                self.context.pn.Column(
                    self.context.measurement_view_tabs,
                    sizing_mode="stretch_width",
                    margin=0,
                    css_classes=["hermes-workspace"],
                ),
                # Outside the isolated workspace so its backdrop covers the run bar.
                self.context.stop_simulation_confirmation,
                sizing_mode="stretch_width",
                min_width=0,
                margin=0,
                styles={"gap": "16px"},
                css_classes=["hermes-work"],
            ),
            sizing_mode="stretch_width",
            css_classes=["hermes-page"],
        )

    def _assemble_template(self):
        """Assemble template in dependency order."""
        self.context.tabs = self.context.pn.Tabs(
            ("Static Target", self.physics_tab),
            ("Dynamic Scenes", self.human_room_tab),
            ("Bundle Comparison", self.measurement_tab),
            # Keep the three workflow roots mounted across the theme reload.  A
            # dynamic outer Tabs model can restore its active index before Panel
            # has attached the newly selected child to the light-theme document,
            # leaving the inactive workflows visibly blank.  The nested result
            # tabs remain dynamic, where lazy rendering provides the useful win.
            dynamic=False,
            css_classes=[self.theme_scope_class],
        )
        self.context.gui_session_storage = _app._gui_session_storage_component(self.context.pn)
        self.context.gui_session_storage.jscallback(
            args={
                "workflow_tabs": self.context.tabs,
                "dynamic_tabs": self.context.human_room_view_tabs,
                "measurement_tabs": self.context.measurement_view_tabs,
                **{
                    f"dynamic_{solver}": self.context.human_room_solver_checkboxes[solver]
                    for solver in _app.SOLVERS
                },
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
  const solvers = restored.dynamic_solvers
  if (solvers != null && typeof solvers === "object") {
    if ("rt" in solvers) dynamic_rt.active = Boolean(solvers.rt)
    if ("po" in solvers) dynamic_po.active = Boolean(solvers.po)
    if ("hybrid_po" in solvers) {
      dynamic_hybrid_po.active = Boolean(solvers.hybrid_po)
    }
  }
}
""",
        )
        self.next_theme = "default" if self.dark_theme else "dark"
        self.context.theme_button.jscallback(
            args={
                "workflow_tabs": self.context.tabs,
                "dynamic_tabs": self.context.human_room_view_tabs,
                "measurement_tabs": self.context.measurement_view_tabs,
                "session_bridge": self.context.gui_session_storage,
                **{
                    f"dynamic_{solver}": self.context.human_room_solver_checkboxes[solver]
                    for solver in _app.SOLVERS
                },
                "bundle_fixture": self.context.fixture,
                "bundle_path_input": self.context.bundle_path,
                "background_toggle": self.context.background,
                "clutter_selector": self.context.clutter,
                "comparison_source_selector": self.context.comparison_source,
                "comparison_modes_selector": self.context.comparison_solvers,
                "channel_zero_toggle": self.context.comparison_channel_zero,
                "simulated_path_input": self.context.simulated_path,
                **{
                    f"adc_{name}_selector": widget
                    for name, widget in self.context.adc_selectors.items()
                },
            },
            clicks=f"""
    try {{
      const raw = window.sessionStorage.getItem("{_app._GUI_SESSION_STORAGE_KEY}")
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
        dynamic_solvers: {{
          rt: dynamic_rt.active,
          po: dynamic_po.active,
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
        comparison_solvers: comparison_modes_selector.value.map(
          value => ({_app.json.dumps(self.comparison_solver_options)})[value] ?? value
        ),
        comparison_channel_zero: channel_zero_toggle.active,
        simulated_path: simulated_path_input.value,
        save_sequence: Date.now(),
      }}
      window.sessionStorage.setItem(
        "{_app._GUI_SESSION_STORAGE_KEY}", JSON.stringify(saved)
      )
    }} catch (error) {{
      console.warn("Unable to save HERMES navigation before theme switch", error)
    }}
    const url = new URL(window.location.href)
    url.searchParams.set("theme", "{self.next_theme}")
    window.location.assign(url.toString())
    """,
        )
        self.modal_events = _app._modal_events_component(self.context.pn)
        self.context.template = self.context.pn.template.FastListTemplate(
            title="HERMES · Interactive mmWave Radar Simulation",
            favicon="/apple-touch-icon.png",
            accent_base_color=_app._plot_theme_tokens(self.context.plot_template)["accent"],
            header_background="#101b2d",
            header_accent_base_color="#38bdf8",
            corner_radius=8,
            shadow=True,
            theme="dark" if self.dark_theme else "default",
            theme_toggle=False,
            busy_indicator=None,
            header=[
                self.context.pn.Row(
                    self.context.solver_settings_storage,
                    self.context.gui_session_storage,
                    self.modal_events,
                    self.context.pn.Spacer(sizing_mode="stretch_width"),
                    self.context.settings_button,
                    self.context.theme_button,
                    sizing_mode="stretch_width",
                    margin=0,
                )
            ],
            modal=[self.settings_modal],
            main=[self.context.tabs],
            main_max_width="1600px",
            main_layout=None,
        )

    def _assemble_session_controls(self):
        """Assemble session controls in dependency order."""
        self.context.static_session_widgets = {
            "target_type": self.context.target_type,
            "target_range": self.context.target_range,
            "target_y": self.context.target_y,
            "target_z": self.context.target_z,
            "plate_width": self.context.plate_width,
            "plate_height": self.context.plate_height,
            "corner_edge": self.context.corner_edge,
            "target_yaw": self.context.yaw,
            "target_pitch": self.context.pitch,
            "target_roll": self.context.roll,
            "radar_yaw": self.context.radar_yaw,
            "radar_pitch": self.context.radar_pitch,
            "radar_roll": self.context.radar_roll,
            "material": self.context.material,
            "human_diffuse": self.context.human_diffuse,
            "angle_range_bins": self.context.angle_range_bins,
            "diagnostic_top_k": self.context.diagnostic_top_k,
            "show_rt_paths": self.context.show_rt_paths,
        }
        self.context.radar_session_widgets = {
            "board": self.context.board,
            "tx_schedule": self.context.tx_schedule,
            "tx_antennas": self.context.tx_antennas,
            "rx_antennas": self.context.rx_antennas,
            "antenna_pattern": self.context.antenna_pattern,
            "cosine_3db_beamwidth": self.context.cosine_3db_beamwidth,
            "carrier_frequency": self.context.carrier_frequency,
            "chirp_slope": self.context.chirp_slope,
            "chirp_duration": self.context.chirp_duration,
            "chirp_repetition": self.context.chirp_repetition,
            "sampling_frequency": self.context.sampling_frequency,
            "adc_mode": self.context.adc_sampling_mode,
            "if_bandwidth": self.context.adc_if_bandwidth,
            "num_adc_samples": self.context.num_adc_samples,
            "num_chirps": self.context.num_chirps,
            "frame_period": self.context.frame_period,
        }
        self.context.dynamic_session_widgets = {
            **self.context.motion_coordinate_widgets,
            "room_preset": self.context.human_room_preset,
            "position_mode": self.context.human_room_position_mode,
            "smpl_model_dir": self.context.human_room_smpl_model_dir,
            "human_x": self.context.human_room_x,
            "human_y": self.context.human_room_y,
            "human_z": self.context.human_room_z,
            "human_yaw": self.context.human_room_yaw,
            "human_diffuse": self.context.human_room_diffuse,
            "radar_yaw": self.context.dynamic_radar_yaw,
            "radar_pitch": self.context.dynamic_radar_pitch,
            "radar_roll": self.context.dynamic_radar_roll,
            "diagnostic_top_k": self.context.human_room_diagnostic_top_k,
        }
        if self.context.remote_access:
            # Browser session storage must never override the operator-selected
            # directory used by the pickle-based SMPL dependency.
            self.context.dynamic_session_widgets.pop("smpl_model_dir")
        self.static_solver_session_widgets = {
            name: widget
            for name, widget in self.context.static_session_widgets.items()
            if name not in {"angle_range_bins", "diagnostic_top_k", "show_rt_paths"}
        }
        self.dynamic_solver_session_widgets = {
            name: widget
            for name, widget in self.context.dynamic_session_widgets.items()
            if name != "diagnostic_top_k"
        }

        self.context.gui_session_state = {
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

        self.context.session_value = self._session.session_value

        self.context.widget_values = self._session.widget_values

        self.context.restore_widget_values = self._session.restore_widget_values

        # Keep the settings behind the last completed products even after edits.
        # These records live in the bounded server cache. Only their opaque cache
        # key is saved in browser session state.
        self.context.result_widgets = {
            "static": {
                **{
                    f"target.{key}": widget
                    for key, widget in self.static_solver_session_widgets.items()
                },
                **{
                    f"radar.{key}": widget
                    for key, widget in self.context.radar_session_widgets.items()
                },
                **{
                    f"solver.{key}": widget
                    for key, widget in self.context.solver_setting_widgets.items()
                    if not key.startswith("hybrid_")
                },
                "upload.human_mesh": self.context.human_upload,
            },
            "dynamic": {
                **{
                    f"scene.{key}": widget
                    for key, widget in self.dynamic_solver_session_widgets.items()
                },
                **{
                    f"radar.{key}": widget
                    for key, widget in self.context.radar_session_widgets.items()
                },
                **{
                    f"solver.{key}": widget
                    for key, widget in self.context.solver_setting_widgets.items()
                },
                **{
                    f"run_solver.{key}": widget
                    for key, widget in self.context.human_room_solver_checkboxes.items()
                },
                "window.start": self.context.human_room_start_time,
                "window.end": self.context.human_room_end_time,
                "upload.motion": self.context.human_room_upload,
                **(
                    {"upload.scene": self.context.human_room_scene_upload}
                    if not self.context.remote_access
                    else {}
                ),
            },
            "comparison": {
                "solver.rt_refresh_frames": self.context.rt_refresh,
                "solver.po_refresh_on_phase_error_rad": self.context.po_phase_error,
                "solver.hybrid_calibration": self.context.hybrid_calibration,
                "comparison.source": self.context.comparison_source,
                "comparison.modes": self.context.comparison_solvers,
                "comparison.channel_zero": self.context.comparison_channel_zero,
                "comparison.background": self.context.background,
                "comparison.clutter": self.context.clutter,
                "comparison.saved_path": self.context.simulated_path,
            },
        }
        self.context.upload_identities = {}

    def _assemble_result_persistence(self):
        """Assemble result persistence in dependency order."""
        self.context.capture_result_settings = self._session.capture_result_settings

        self.context.result_records = {
            "static": {
                "settings": self.context.capture_result_settings("static"),
                "result": self.initial_result,
                "refined": False,
                "status": str(self.context.physics_status.object or ""),
            },
            "dynamic": None,
            "comparison": None,
        }
        self.context.result_banners = {
            "static": self.static_result_banner,
            "dynamic": self.dynamic_result_banner,
            "comparison": self.comparison_result_banner,
        }
        self.context.result_banner_text = {}
        self.context.result_revert_buttons = {}
        for self.page, self.banner in self.context.result_banners.items():
            self.context.result_banner_text[self.page] = self.context.pn.pane.Markdown(
                "",
                css_classes=["hermes-banner-text"],
            )
            self.context.result_revert_buttons[self.page] = self.context.pn.widgets.Button(
                label="Revert",
                color="light",
                width=100,
                description="Restore the settings used for the displayed results.",
            )
            self.banner.objects = [
                self.context.result_banner_text[self.page],
                self.context.pn.Row(
                    self.context.result_revert_buttons[self.page],
                    css_classes=["hermes-banner-actions"],
                ),
            ]
            self.banner.css_classes = ["hermes-runbar-banner"]
            # Shown inside the run bar, whose left edge also turns amber.
            self.banner.styles = {
                "background": "rgba(214, 158, 46, 0.10)",
                "border-radius": "8px",
                "padding": "2px 6px 6px",
            }
            self.banner.margin = (2, 4, 4, 4)

        self.context.settings_match_result = self._session.settings_match_result

        self.context.refresh_result_validity = self._session.refresh_result_validity

        self.context.remember_result_settings = self._session.remember_result_settings

        self.context.rerun_page = self._session.rerun_page

        self.context.revert_page_settings = self._session.revert_page_settings

        for self.page in self.context.result_banners:

            async def revert(_event, page=self.page, callback=self.context.revert_page_settings):
                await callback(page)

            self.context.result_revert_buttons[self.page].on_click(revert)

        self.context.dynamic_result_view_state = self._session.dynamic_result_view_state

        self.context.cache_current_gui_results = self._session.cache_current_gui_results

        self.context.persist_gui_session = self._session.persist_gui_session

        self.context.encoded_upload = self._session.encoded_upload

        self.context.persist_gui_uploads = self._session.persist_gui_uploads

        self.context.apply_pending_dynamic_frame_state = (
            self._session.apply_pending_dynamic_frame_state
        )

        self.context.restore_dynamic_result_view = self._session.restore_dynamic_result_view

        self.context.restore_cached_gui_results = self._session.restore_cached_gui_results

        self.context.restore_gui_session = self._session.restore_gui_session

        self.context.decoded_upload = self._session.decoded_upload

        self.context.restore_gui_uploads = self._session.restore_gui_uploads

    def _assemble_session_bindings(self):
        """Assemble session bindings in dependency order."""
        self.context.gui_session_storage.param.watch(
            self.context.restore_gui_session,
            "loaded_state",
        )
        self.context.gui_session_storage.param.watch(
            self.context.restore_gui_uploads,
            "loaded_uploads",
        )
        for self.navigation_tabs in (
            self.context.tabs,
            self.context.human_room_view_tabs,
            self.context.measurement_view_tabs,
        ):
            self.navigation_tabs.param.watch(self.context.persist_gui_session, "active")
        for self.widget in (
            *self.context.static_session_widgets.values(),
            *self.context.radar_session_widgets.values(),
            *self.context.dynamic_session_widgets.values(),
            self.context.human_room_preview_time,
            self.context.human_room_start_time,
            self.context.human_room_end_time,
            *self.context.human_room_solver_checkboxes.values(),
            *self.context.adc_selectors.values(),
        ):
            self.widget.param.watch(self.context.persist_gui_session, "value")
        for self.comparison_widget, self.parameter_name in (
            (self.context.fixture, "value"),
            (self.context.bundle_path, "value"),
            (self.context.background, "value"),
            (self.context.clutter, "value"),
            (self.context.comparison_source, "value"),
            (self.context.comparison_solvers, "value"),
            (self.context.comparison_channel_zero, "value"),
            (self.context.simulated_path, "value"),
        ):
            self.comparison_widget.param.watch(
                self.context.persist_gui_session, self.parameter_name
            )
        for self.upload_widget in (
            self.context.human_upload,
            self.context.human_room_upload,
            self.context.human_room_scene_upload,
        ):
            self.upload_widget.param.watch(
                self.context.persist_gui_uploads,
                ["value", "filename"],
            )
        self.context.persist_gui_uploads()
        self.context.persist_gui_session()

        self.context.persist_solver_settings = self._settings.persist_solver_settings

        self.context.on_solver_settings_loaded = self._settings.on_solver_settings_loaded

        self.context.on_apply_solver_settings = self._settings.on_apply_solver_settings

        self.context.on_open_settings = self._settings.on_open_settings

        self.context.on_cancel_settings = self._settings.on_cancel_settings

        self.context.settings_button.on_click(self.context.on_open_settings)
        self.cancel_solver_settings.on_click(self.context.on_cancel_settings)
        self.modal_events.param.watch(self.context.on_cancel_settings, "dismissed")

        for self.key, self.widget in {
            **self.context.radar_session_widgets,
            **self.context.solver_setting_widgets,
            **self.context.motion_coordinate_widgets,
        }.items():
            self.used_by = (
                "Dynamic Scenes"
                if self.key in self.context.motion_coordinate_widgets
                else (
                    "Dynamic Scenes (Hybrid PO)"
                    if self.key.startswith("coupling_")
                    else "Static Target, Dynamic Scenes"
                )
            )
            if "description" in self.widget.param:
                self.widget.param.update(
                    description=(getattr(self.widget, "description", "") or "")
                    + f" Used by: {self.used_by}."
                )

        self.context.solver_settings_storage.param.watch(
            self.context.on_solver_settings_loaded,
            "loaded_settings",
        )
        for self.solver_widget in self.context.solver_setting_widgets.values():
            self.solver_widget.param.watch(self.context.persist_solver_settings, "value")
        self.apply_solver_settings.on_click(self.context.on_apply_solver_settings)

        for self.workflow in (self.physics_tab, self.human_room_tab):
            for self.pane in self.workflow.select(self.context.pn.pane.Markdown):
                if "hermes-status" in self.pane.css_classes:
                    self.pane.styles = {**self.pane.styles, "min-height": "0"}
                    self.pane.margin = (3, 10)
        for self.controls in (
            self.physics_controls,
            self.human_room_controls,
            self.measurement_rail,
            self.settings_modal,
        ):
            _app._compact_controls(self.controls, self.context.pn)
        _app._readable_markdown(self.context.tabs, self.context.pn, self.context.plot_template)
        _app._readable_markdown(self.settings_modal, self.context.pn, self.context.plot_template)
        _app._style_secondary_buttons(
            self.context.tabs, self.context.pn, self.context.plot_template
        )
        _app._style_secondary_buttons(
            self.settings_modal, self.context.pn, self.context.plot_template
        )
        for self.tab_group in (
            *self.context.tabs.select(self.context.pn.Tabs),
            *self.settings_modal.select(self.context.pn.Tabs),
        ):
            _app._fit_active_tab_height(self.tab_group)

        self.context.cancel_session_work = self._application.cancel_session_work

        self.document = self.context.pn.state.curdoc
        if self.document is not None and self.document.session_context is not None:
            self.document.on_session_destroyed(self.context.cancel_session_work)
        return self.context.template

    def build(self):
        self._assemble_initialize()
        self._assemble_po_controls()
        self._assemble_rt_and_array_controls()
        self._assemble_settings_panels()
        self._assemble_target_controls()
        self._assemble_radar_controls()
        self._assemble_static_products()
        self._assemble_static_bindings()
        self._assemble_radar_panel()
        self._assemble_settings_and_static_layout()
        self._assemble_motion_source_controls()
        self._assemble_motion_playback_controls()
        self._assemble_motion_results_and_state()
        self._assemble_motion_bindings()
        self._assemble_motion_layout()
        self._assemble_bundle_source_controls()
        self._assemble_comparison_controls()
        self._assemble_comparison_bindings()
        self._assemble_comparison_layout()
        self._assemble_template()
        self._assemble_session_controls()
        self._assemble_result_persistence()
        return self._assemble_session_bindings()
