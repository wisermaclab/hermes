# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Radar controls and callbacks for one GUI session."""

from __future__ import annotations

from .. import app as _app
from .base import Workflow
from .contexts import RadarContext


class RadarWorkflow(Workflow[RadarContext]):
    def update_material_details(
        self,
    ):
        parameters = _app.material_preset_parameters(self.context.material.value)
        coefficient = float(parameters["diffuse_coefficient"])
        if coefficient == 0.0:
            self.context.material_details.object = (
                "**RT diffuse scattering coefficient:** `0.00`  \n"
                "Diffuse RT paths are disabled for this smooth-surface preset. "
                "PO still uses its electromagnetic material model."
            )
        else:
            backscattering_lambda = float(parameters["backscattering_lambda"])
            self.context.material_details.object = (
                f"**RT diffuse scattering coefficient:** `{coefficient:.2f}`  \n"
                "**Backscattering lobe mixture λ:** "
                f"`{backscattering_lambda:.2f}`  \n"
                "These RT rough-surface terms are independent of the PO "
                "electromagnetic material model."
            )

    def update_antenna_pattern_controls(
        self,
    ):
        pattern_mode = self.context.antenna_pattern.value
        self.context.cosine_3db_beamwidth.visible = pattern_mode == "cosine"
        if pattern_mode == "cosine":
            self.context.antenna_pattern_note.object = (
                "The specified width is the full **power-pattern 3 dB "
                "beamwidth**, centered on radar local +x."
            )
        elif pattern_mode == "digitized":
            self.context.antenna_pattern_note.object = (
                "Uses the packaged per-channel TI pattern profile."
                if _app.ti_digitized_pattern_asset_available()
                else (
                    "**Digitized profile unavailable:** install "
                    "`ti_digitized_patterns.npz` in the radar data package "
                    "before running this mode."
                )
            )
        else:
            self.context.antenna_pattern_note.object = (
                "Uses the board's scalar gain uniformly in every direction."
            )

    def update_radar_summary(
        self,
    ):
        selected_tx = [int(value) for value in self.context.tx_antennas.value]
        selected_rx = [int(value) for value in self.context.rx_antennas.value]
        active_tx_indices = selected_tx
        active_tx = len(active_tx_indices)
        active_rx = len(selected_rx)
        bandwidth_hz = (
            float(self.context.chirp_slope.value)
            * 1e12
            * int(self.context.num_adc_samples.value)
            / (float(self.context.sampling_frequency.value) * 1e6)
        )
        range_resolution_m = 299_792_458.0 / (2.0 * bandwidth_hz)
        pattern_summary = {
            "digitized": "digitized TI profile",
            "none": "omnidirectional",
            "cosine": (
                f"cosine, {float(self.context.cosine_3db_beamwidth.value):.1f}° " "3 dB width"
            ),
        }[self.context.antenna_pattern.value]
        board_if = _app.get_ti_board_spec(self.context.board.value).adc.get("if_bandwidth_hz")
        requested_if = float(self.context.adc_if_bandwidth.value) * 1e6 or board_if
        effective_if = _app.if_cutoff_hz(
            _app.SimpleNamespace(
                adc_mode=self.context.adc_sampling_mode.value,
                sampling_frequency=float(self.context.sampling_frequency.value) * 1e6,
                if_bandwidth_hz=min(requested_if, board_if) if board_if else requested_if,
            )
        )
        if_range_m = (
            299_792_458.0
            * effective_if
            / (2.0 * abs(float(self.context.chirp_slope.value) * 1e12))
        )
        self.context.radar_summary.object = (
            f"**{active_tx} Tx × {active_rx} Rx · "
            f"{active_tx * active_rx} channels**  \n"
            f"Active `TX {list(active_tx_indices)}` · "
            f"`RX {list(selected_rx)}`  \n"
            f"Sampled bandwidth `{bandwidth_hz / 1e9:.3f} GHz` · range "
            f"resolution `{range_resolution_m * 100:.2f} cm`  \n"
            f"Usable IF `{effective_if/1e6:.3f} MHz` · maximum range `{if_range_m:.2f} m`  \n"
            f"Antenna `{pattern_summary}`"
        )

    def set_radar_defaults(
        self,
    ):
        selected_spec = _app.get_ti_board_spec(self.context.board.value)
        antenna_defaults = _app._RADAR_ACTIVE_ANTENNA_DEFAULTS.get(
            selected_spec.key,
            {
                "tx": tuple(range(selected_spec.num_tx)),
                "rx": tuple(range(selected_spec.num_rx)),
            },
        )
        default_tx = list(antenna_defaults["tx"])
        default_rx = list(antenna_defaults["rx"])
        defaults = {
            self.context.carrier_frequency: _app._rounded_gui_float(
                selected_spec.design_frequency_hz / 1e9
            ),
            self.context.chirp_slope: 68.0,
            self.context.chirp_duration: 58.0,
            self.context.chirp_repetition: 65.0,
            self.context.sampling_frequency: 4.5,
            self.context.adc_sampling_mode: "complex1x",
            self.context.adc_if_bandwidth: 0.0,
            self.context.num_adc_samples: 225,
            self.context.num_chirps: 16,
            self.context.frame_period: 50.0,
            self.context.tx_schedule: "simultaneous",
            self.context.tx_antennas: default_tx,
            self.context.rx_antennas: default_rx,
        }
        for selector, prefix, count, desired in (
            (self.context.tx_antennas, "TX", selected_spec.num_tx, default_tx),
            (self.context.rx_antennas, "RX", selected_spec.num_rx, default_rx),
        ):
            if list(selector.value) != desired:
                self.context.ignored_radar_events[id(selector)] = (
                    self.context.ignored_radar_events.get(id(selector), 0) + 1
                )
            selector.options = {f"{prefix}{index}": index for index in range(count)}
        for widget, value in defaults.items():
            if widget.value != value:
                if widget not in (self.context.tx_antennas, self.context.rx_antennas):
                    self.context.ignored_radar_events[id(widget)] = (
                        self.context.ignored_radar_events.get(id(widget), 0) + 1
                    )
                widget.value = value
        self.context.update_radar_summary()

    def on_board_selection_change(self, _event):
        if self.context.gui_session_state["restoring"]:
            return
        self.context.set_radar_defaults()

    async def on_board_change(self, _event):
        if (
            self.context.gui_session_state["restoring"]
            or self.context.settings_dialog_state["editing"]
        ):
            return
        self.context.refresh_result_validity()

    async def on_load_radar_defaults(self, _event):
        self.context.set_radar_defaults()

    async def on_radar_control_change(self, event):
        if self.context.gui_session_state["restoring"]:
            return
        changed_widget = getattr(event, "obj", None)
        ignored_count = self.context.ignored_radar_events.get(id(changed_widget), 0)
        if ignored_count:
            if ignored_count == 1:
                self.context.ignored_radar_events.pop(id(changed_widget), None)
            else:
                self.context.ignored_radar_events[id(changed_widget)] = ignored_count - 1
            self.context.update_radar_summary()
            return
        if changed_widget in (self.context.tx_antennas, self.context.rx_antennas):
            if not changed_widget.value:
                import param

                with param.parameterized.discard_events(changed_widget):
                    changed_widget.value = [0]
                self.context.pn.state.notifications.warning(
                    "At least one antenna must remain active.",
                    duration=4000,
                )
        if isinstance(changed_widget, self.context.pn.widgets.FloatInput):
            normalized = _app._rounded_gui_float(changed_widget.value)
            if normalized != changed_widget.value:
                import param

                with param.parameterized.discard_events(changed_widget):
                    changed_widget.value = normalized
        self.context.update_radar_summary()
        if not self.context.settings_dialog_state["editing"]:
            self.context.refresh_result_validity()

    async def on_material_change(self, event):
        self.context.update_material_details()
        await self.context.on_physics_control_change(event)

    async def on_antenna_pattern_change(self, event):
        self.context.update_antenna_pattern_controls()
        await self.context.on_radar_control_change(event)
