# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Settings controls and callbacks for one GUI session."""

from __future__ import annotations

from .. import app as _app
from .base import Workflow
from .contexts import SettingsContext


class SettingsWorkflow(Workflow[SettingsContext]):
    def on_po_integration_mode_change(self, _event=None):
        if (
            _event is not None
            and _event.old != _event.new
            and not self.context.solver_settings_state["restoring"]
        ):
            self.context.phase_span_by_mode[_event.old] = self.context.po_phase_span.value
            self.context.po_phase_span.value = self.context.phase_span_by_mode.get(
                _event.new,
                _app.default_phase_span_scale_rad(_event.new),
            )
        self.context.po_curvature.disabled = (
            self.context.po_integration_mode.value != "parent_face_adaptive_analytic"
        )
        self.context.po_phase_span.disabled = self.context.po_integration_mode.value not in {
            "parent_face_adaptive_analytic",
            "parent_face_quadrature",
        }

    def current_solver_settings_payload(
        self,
    ) -> dict[str, object]:
        return {
            "schema_version": _app._SOLVER_SETTINGS_SCHEMA_VERSION,
            "po_defaults_version": _app._PO_DEFAULTS_VERSION,
            "po_phase_span_by_mode": {
                **self.context.phase_span_by_mode,
                self.context.po_integration_mode.value: self.context.po_phase_span.value,
            },
            "settings": {
                name: widget.value for name, widget in self.context.solver_setting_widgets.items()
            },
        }

    def restore_solver_settings(self, payload):
        restored = _app._validated_solver_settings_payload(payload)
        spans = _app._validated_po_phase_span_by_mode(payload)
        if not restored and not spans:
            return restored, False
        if payload.get("po_defaults_version") != _app._PO_DEFAULTS_VERSION:
            self.context.phase_span_by_mode.clear()
        mode = restored.get("po_integration_mode", self.context.po_integration_mode.value)
        if "po_integration_mode" in restored or mode in spans:
            restored.setdefault(
                "po_quadrature_phase_span_scale_rad",
                spans.get(
                    mode,
                    self.context.phase_span_by_mode.get(
                        mode, _app.default_phase_span_scale_rad(mode)
                    ),
                ),
            )
        changed = False
        self.context.solver_settings_state["restoring"] = True
        try:
            for name, value in restored.items():
                widget = self.context.solver_setting_widgets[name]
                if widget.value != value:
                    # Keep normal widget events so browser models update.
                    widget.value = value
                    changed = True
        finally:
            self.context.solver_settings_state["restoring"] = False
        self.context.phase_span_by_mode.update(spans)
        # The active scalar also restores profiles saved before the map existed.
        self.context.phase_span_by_mode[self.context.po_integration_mode.value] = (
            self.context.po_phase_span.value
        )
        return restored, changed

    def current_motion_coordinate_arguments(
        self,
    ):
        if self.context.motion_axes_mode.value == "file":
            return {}
        return {
            "motion_source_up_axis": self.context.motion_up_axis.value,
            "motion_source_forward_axis": self.context.motion_forward_axis.value,
            "motion_source_handedness": self.context.motion_handedness.value,
            "motion_source_length_unit": self.context.motion_length_unit.value,
        }

    def sync_motion_coordinate_controls(
        self,
    ):
        self.context.motion_coordinate_state["syncing"] = True
        try:
            self.context.motion_forward_axis.options = {
                label: value
                for label, value in self.context.motion_axis_options.items()
                if value[-1] != self.context.motion_up_axis.value[-1]
            }
            manual = self.context.motion_axes_mode.value == "manual"
            for widget in (
                self.context.motion_up_axis,
                self.context.motion_forward_axis,
                self.context.motion_handedness,
                self.context.motion_length_unit,
            ):
                widget.disabled = not manual
            if not manual:
                self.context.motion_coordinate_summary.object = (
                    "File `output_axes` / `output_signs` are used when present. "
                    "Otherwise axes and scale stay as stored; no frame is guessed."
                )
                return
            transform = _app.motion_source_to_radar_transform(
                self.context.motion_up_axis.value,
                self.context.motion_forward_axis.value,
                handedness=self.context.motion_handedness.value,
                length_unit=self.context.motion_length_unit.value,
            )
            axes = _app.np.argmax(_app.np.abs(transform), axis=1)
            signs = _app.np.sign(transform[_app.np.arange(3), axes])
            mapping = " · ".join(
                f"GUI {target.upper()} ← {'+' if sign > 0 else '-'}{'XYZ'[axis]} source"
                for target, axis, sign in zip("xyz", axes, signs)
            )
            self.context.motion_coordinate_summary.object = (
                f"**{mapping}**  \n"
                "Converted to metres, then human yaw and position are applied. "
                "This selection overrides any file axis mapping."
            )
        finally:
            self.context.motion_coordinate_state["syncing"] = False

    def compact_numeric(self, label, *, value, start, end, step, format="0.[0]", description=None):
        return self.context.pn.widgets.FloatInput(
            label=label,
            value=value,
            start=start,
            end=end,
            step=step,
            format=format,
            description=description,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(5, 0),
        )

    def numeric_row(self, css_class, *widgets):
        return self.context.pn.Row(
            *widgets,
            sizing_mode="stretch_width",
            min_width=0,
            margin=(0, 10),
            styles={"gap": "10px"},
            css_classes=[css_class],
        )

    def persist_solver_settings(self, _event=None):
        if (
            self.context.solver_settings_state["restoring"]
            or self.context.settings_dialog_state["editing"]
        ):
            return
        self.context.solver_settings_storage.saved_settings = (
            self.context.current_solver_settings_payload()
        )
        self.context.persist_gui_session(_event)

    async def on_solver_settings_loaded(self, event):
        if self.context.gui_session_state["restoring"]:
            return
        # The session snapshot owns settings on a theme reload. A later
        # localStorage bridge event must not replace that restored snapshot.
        if self.context.gui_session_state["restored_state"]:
            self.context.persist_solver_settings()
            self.context.refresh_result_validity()
            return
        self.context.restore_solver_settings(getattr(event, "new", None))

        # Rewrite partial or outdated payloads in the current complete schema.
        self.context.persist_solver_settings()
        # A storage restore is not an edit or a request to run a solver.
        self.context.refresh_result_validity()

    async def on_apply_solver_settings(self, _event):
        motion_changed = self.context.settings_dialog_state["editing"] and any(
            widget.value != self.context.settings_dialog_state["values"][widget]
            for widget in self.context.motion_coordinate_widgets.values()
        )
        self.context.settings_dialog_state["editing"] = False
        self.context.ignored_radar_events.clear()
        self.context.persist_solver_settings()
        self.context.template.close_modal()
        self.context.theme_button.disabled = any(self.context.gui_activity_state.values())
        if motion_changed and self.context.human_room_state["loaded_motion"] is not None:
            await self.context.on_human_room_controls(None)
        if self.context.rerun_after_apply.value:
            await self.context.rerun_page(
                ("static", "dynamic", "comparison")[self.context.tabs.active]
            )

    def on_open_settings(self, _event):
        self.context.settings_dialog_state["values"] = {
            widget: self.context.session_value(widget.value)
            for widget in (
                *self.context.radar_session_widgets.values(),
                *self.context.solver_setting_widgets.values(),
                *self.context.motion_coordinate_widgets.values(),
            )
        }
        self.context.settings_dialog_state["phase_spans"] = dict(self.context.phase_span_by_mode)
        self.context.settings_dialog_state["editing"] = True
        self.context.rerun_after_apply.value = False
        self.context.theme_button.disabled = True
        self.context.template.open_modal()

    async def on_cancel_settings(self, _event):
        if not self.context.settings_dialog_state["editing"]:
            return
        self.context.gui_session_state["restoring"] = True
        self.context.solver_settings_state["restoring"] = True
        try:
            saved = self.context.settings_dialog_state["values"]
            self.context.board.value = saved[self.context.board]
            spec = _app.get_ti_board_spec(self.context.board.value)
            self.context.tx_antennas.options = {f"TX{i}": i for i in range(spec.num_tx)}
            self.context.rx_antennas.options = {f"RX{i}": i for i in range(spec.num_rx)}
            for widget, value in saved.items():
                widget.value = value
            self.context.phase_span_by_mode.clear()
            self.context.phase_span_by_mode.update(
                self.context.settings_dialog_state["phase_spans"]
            )
            self.context.update_antenna_pattern_controls()
            self.context.sync_motion_coordinate_controls()
            self.context.update_radar_summary()
            await _app.asyncio.sleep(0)
        finally:
            self.context.solver_settings_state["restoring"] = False
            self.context.gui_session_state["restoring"] = False
            self.context.settings_dialog_state["editing"] = False
            self.context.ignored_radar_events.clear()
        self.context.template.close_modal()
        self.context.theme_button.disabled = any(self.context.gui_activity_state.values())
        self.context.persist_gui_session()
