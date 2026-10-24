# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Session controls and callbacks for one GUI session."""

from __future__ import annotations

from typing import Any, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from panel.widgets import FileInput

from .. import app as _app
from .base import Workflow
from .contexts import SessionContext


class SessionWorkflow(Workflow[SessionContext]):
    def session_value(self, value):
        if isinstance(value, _app.np.generic):
            return value.item()
        if isinstance(value, (tuple, list)):
            return [self.context.session_value(item) for item in value]
        return value

    def widget_values(self, widgets):
        return {name: self.context.session_value(widget.value) for name, widget in widgets.items()}

    def restore_widget_values(self, widgets, values):
        if not isinstance(values, dict):
            return
        for name, widget in widgets.items():
            if name not in values:
                continue
            value = values[name]
            if isinstance(widget, self.context.pn.widgets.IntRangeSlider):
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    continue
                value = tuple(int(item) for item in value)
            try:
                # Normal events synchronize the new browser models. Solver
                # handlers ignore restoration, including queued async events.
                widget.value = value
            except (TypeError, ValueError):
                continue

    def capture_result_settings(self, page):
        values: dict[str, Any] = {}
        uploads = {}
        for key, widget in self.context.result_widgets[page].items():
            if isinstance(widget, self.context.pn.widgets.FileInput):
                payload = widget.value or b""
                previous = self.context.upload_identities.get(widget)
                if previous is None or previous[0] is not payload:
                    previous = (payload, _app.hashlib.sha256(payload).hexdigest())
                    self.context.upload_identities[widget] = previous
                values[key] = [str(widget.filename or ""), previous[1]]
                uploads[key] = (widget.filename, widget.value)
            else:
                value = (
                    self.context.settings_dialog_state["values"].get(widget, widget.value)
                    if self.context.settings_dialog_state["editing"]
                    else widget.value
                )
                values[key] = self.context.session_value(value)
        if page == "comparison":
            loaded = self.context.measurement_state.get("bundle")
            values["bundle"] = getattr(loaded, "fingerprint", None)
        return {
            "fingerprint": _app._result_settings_fingerprint(values),
            "values": values,
            "uploads": uploads,
        }

    def settings_match_result(self, page):
        record = self.context.result_records[page]
        return bool(
            record
            and record["settings"]["fingerprint"]
            == self.context.capture_result_settings(page)["fingerprint"]
        )

    def refresh_result_validity(
        self,
    ):
        self.context.load_motion_preview.visible = (
            self.context.default_amass_motion is not None
            and not self.context.human_room_upload.value
            and self.context.human_room_state["loaded_motion"] is None
        )
        self.context.run_human_room.disabled = (
            self.context.human_room_state["loaded_motion"] is None
            or self.context.human_room_state["running"]
            or self.context.gui_activity_state["dynamic"]
        )
        for index, (page, title) in enumerate(
            (
                ("static", "Static Target"),
                ("dynamic", "Dynamic Scenes"),
                ("comparison", "Bundle Comparison"),
            )
        ):
            record = self.context.result_records[page]
            was_stale = self.context.result_banners[page].visible
            current = self.context.capture_result_settings(page)
            stale = bool(record and record["settings"]["fingerprint"] != current["fingerprint"])
            if stale:
                previous_values = record["settings"]["values"]
                changed = [
                    key
                    for key, value in current["values"].items()
                    if previous_values.get(key) != value
                ]
                labels = [
                    (
                        str(self.context.result_widgets[page][key].label or key)
                        if key in self.context.result_widgets[page]
                        else "Primary bundle"
                    )
                    for key in changed
                ]
                self.context.result_banner_text[page].object = (
                    "**Results need a rerun.** Changed: "
                    + ", ".join(_app._safe_markdown_code(label) for label in labels)
                    + ". The plots show the last completed run."
                )
                if (
                    page == "static"
                    and not self.context.gui_activity_state["static"]
                    and (
                        not was_stale
                        or str(self.context.physics_status.object).startswith(
                            ("**LIVE", "**HIGHER FIDELITY")
                        )
                    )
                ):
                    self.context.physics_status.object = (
                        "**LAST RUN (out of date)** · The displayed plots use previous settings."
                    )
                elif (
                    page == "dynamic"
                    and not self.context.gui_activity_state["dynamic"]
                    and (
                        not was_stale
                        or str(self.context.human_room_status.object).startswith(
                            ("**SIMULATION READY", "**READY TO RUN")
                        )
                    )
                ):
                    self.context.human_room_status.object = (
                        "**LAST RUN (out of date)** · The displayed plots use previous settings."
                    )
            self.context.result_banners[page].visible = stale
            if (
                was_stale
                and not stale
                and record
                and not any(self.context.gui_activity_state.values())
            ):
                if page == "static":
                    self.context.physics_status.object = record.get(
                        "status", self.context.physics_status.object
                    )
                elif page == "dynamic":
                    self.context.human_room_status.object = record.get(
                        "status", self.context.human_room_status.object
                    )
            tab_title = title + (" •" if stale else "")
            if self.context.tabs._names[index] != tab_title:  # pylint: disable=protected-access
                self.context.tabs._names[index] = tab_title  # pylint: disable=protected-access
                # Panel reuses unchanged TabPanels without updating titles.
                # Keep the mounted workflow models and update only the title.
                for (
                    model,
                    _parent,
                ) in self.context.tabs._models.values():  # pylint: disable=protected-access

                    def update_title(model=model, index=index, title=tab_title):
                        from bokeh.models import CustomJS

                        panel = model.tabs[index]
                        if not panel.js_property_callbacks.get("change:title"):
                            panel.js_on_change(
                                "title",
                                CustomJS(
                                    args={"workflow_tabs": model},
                                    code="workflow_tabs.properties.tabs.change.emit()",
                                ),
                            )
                        panel.title = title

                    document = model.document
                    if document is not None and document.session_context is not None:
                        document.add_next_tick_callback(update_title)
                    else:
                        update_title()
            if page == "static" and record:
                self.context.physics_state["result"] = None if stale else record["result"]
                self.context.physics_state["refined"] = record["refined"]
                self.context.gui_session_state["static_cacheable"] = not stale
                self.context.physics_download.disabled = (
                    stale or self.context.physics_state["running"]
                )
                diagnostic_busy = self.context.diagnostics_state.get("cancel_event") is not None
                self.context.show_rt_paths.disabled = (
                    stale or self.context.physics_state["running"] or diagnostic_busy
                )
                self.context.diagnostic_top_k.disabled = (
                    stale or self.context.physics_state["running"] or diagnostic_busy
                )
            elif page == "dynamic" and record:
                self.context.human_room_state["results"] = {} if stale else record["results"]
                self.context.human_room_state["result"] = None if stale else record["result"]
                self.context.gui_session_state["dynamic_cacheable"] = not stale
                self.context.human_room_download.disabled = (
                    stale or self.context.human_room_state["running"]
                )
                self.context.run_human_room_diagnostics_button.disabled = (
                    stale
                    or self.context.human_room_state["running"]
                    or self.context.human_room_state.get("diagnostic_cancel_event") is not None
                )
            elif page == "comparison" and record:
                self.context.simulated_adc_download.disabled = (
                    stale or not self.context.measurement_state["simulated_adcs"]
                )
            running = any(self.context.gui_activity_state.values())
            self.context.result_revert_buttons[page].disabled = running

    def remember_result_settings(self, page, settings=None):
        settings = settings or self.context.capture_result_settings(page)
        if page == "static":
            self.context.result_records[page] = {
                "settings": settings,
                "result": self.context.physics_state["result"],
                "refined": self.context.physics_state["refined"],
                "status": str(self.context.physics_status.object or ""),
            }
        elif page == "dynamic":
            self.context.result_records[page] = {
                "settings": settings,
                "result": self.context.human_room_state["result"],
                "results": dict(self.context.human_room_state["results"]),
                "status": str(self.context.human_room_status.object or ""),
            }
        else:
            self.context.result_records[page] = {"settings": settings}

    async def rerun_page(self, page):
        if page == "static":
            record = self.context.result_records[page]
            fidelity = "high_fidelity" if record and record["refined"] else "preview"
            await self.context.update_physics(fidelity=fidelity, force=True)
        elif page == "dynamic":
            await self.context.on_run_human_room(None)
        else:
            await self.context.on_compare(None)

    async def revert_page_settings(self, page):
        record = self.context.result_records[page]
        if record is None:
            return
        self.context.gui_session_state["restoring"] = True
        try:
            settings = record["settings"]
            # Board options must exist before restoring its antenna indices.
            if page in {"static", "dynamic"}:
                self.context.board.value = settings["values"]["radar.board"]
                spec = _app.get_ti_board_spec(self.context.board.value)
                self.context.tx_antennas.options = {f"TX{i}": i for i in range(spec.num_tx)}
                self.context.rx_antennas.options = {f"RX{i}": i for i in range(spec.num_rx)}
            for key, widget in self.context.result_widgets[page].items():
                if key in settings["uploads"]:
                    upload = cast("FileInput", widget)
                    upload.filename, upload.value = settings["uploads"][key]
                else:
                    widget.value = settings["values"][key]
            self.context.update_radar_summary()
        finally:
            self.context.gui_session_state["restoring"] = False
        if page == "dynamic":
            await self.context.refresh_human_room_preview(mark_stale=False)
            self.context.human_room_status.object = record.get(
                "status", self.context.human_room_status.object
            )
        self.context.persist_solver_settings()
        self.context.persist_gui_uploads()
        self.context.persist_gui_session()

    def dynamic_result_view_state(
        self,
    ):
        result_tabs = next(
            (
                component
                for component in self.context.human_room_results.select(self.context.pn.Tabs)
                if component._names  # pylint: disable=protected-access
                == ["Range profile", "Range-time", "Range-Doppler"]
            ),
            None,
        )
        range_doppler_slider = next(
            (
                widget
                for widget in self.context.human_room_results.select(
                    self.context.pn.widgets.IntSlider
                )
                if str(widget.label).startswith("Radar frame")
            ),
            None,
        )
        return {
            "product_tab": int(result_tabs.active) if result_tabs else 0,
            "range_doppler_frame_index": (
                int(range_doppler_slider.value) if range_doppler_slider is not None else 0
            ),
        }

    def cache_current_gui_results(
        self,
    ) -> str:
        static_record = self.context.result_records["static"]
        dynamic_record = self.context.result_records["dynamic"]
        static_result = static_record["result"] if static_record else None
        dynamic_results = dict(dynamic_record["results"]) if dynamic_record else {}
        signature = (
            id(static_result),
            bool(self.context.physics_state.get("refined")),
            id(self.context.diagnostics_state.get("result")),
            bool(self.context.show_rt_paths.value),
            int(self.context.diagnostic_top_k.value),
            id(self.context.human_room_state.get("preview")),
            id(self.context.human_room_state.get("loaded_motion")),
            tuple((mode, id(result)) for mode, result in dynamic_results.items()),
            id(self.context.human_room_state.get("diagnostics")),
            static_record["settings"]["fingerprint"] if static_record else None,
            dynamic_record["settings"]["fingerprint"] if dynamic_record else None,
        )
        if signature == self.context.gui_session_state["result_signature"]:
            return str(self.context.gui_session_state["result_cache_key"])
        cache_key = _app._cache_theme_gui_results(
            static_result=static_result,
            static_settings=static_record["settings"] if static_record else None,
            static_refined=bool(self.context.physics_state.get("refined")),
            static_status=static_record.get("status", "") if static_record else "",
            static_diagnostics=self.context.diagnostics_state.get("result"),
            static_diagnostics_status=str(self.context.diagnostics_status.object or ""),
            static_show_rt_paths=bool(self.context.show_rt_paths.value),
            static_diagnostic_top_k=int(self.context.diagnostic_top_k.value),
            dynamic_preview=self.context.human_room_state.get("preview"),
            dynamic_loaded_motion=self.context.human_room_state.get("loaded_motion"),
            dynamic_motion_cache_key=self.context.human_room_state.get("motion_cache_key"),
            dynamic_results=dynamic_results,
            dynamic_settings=dynamic_record["settings"] if dynamic_record else None,
            dynamic_diagnostics=self.context.human_room_state.get("diagnostics"),
            dynamic_status=dynamic_record.get("status", "") if dynamic_record else "",
            dynamic_diagnostics_status=str(
                self.context.human_room_diagnostics_status.object or ""
            ),
        )
        self.context.gui_session_state["result_signature"] = signature
        self.context.gui_session_state["result_cache_key"] = cache_key
        return cache_key

    def persist_gui_session(self, _event=None):
        if (
            self.context.gui_session_state["restoring"]
            or self.context.solver_settings_state["restoring"]
            or self.context.settings_dialog_state["editing"]
        ):
            return
        self.context.refresh_result_validity()
        self.context.gui_session_state["save_sequence"] += 1
        uploads_present = bool(
            self.context.human_upload.value
            or self.context.human_room_upload.value
            or (self.context.human_room_scene_upload.value and not self.context.remote_access)
        )
        self.context.gui_session_storage.saved_state = {
            "schema_version": 2,
            "po_defaults_version": _app._PO_DEFAULTS_VERSION,
            "image_source_selection_version": _app._IMAGE_SOURCE_SELECTION_VERSION,
            "po_phase_span_by_mode": self.context.current_solver_settings_payload()[
                "po_phase_span_by_mode"
            ],
            "top_tab": int(self.context.tabs.active),
            "dynamic_tab": int(self.context.human_room_view_tabs.active),
            "measurement_tab": int(self.context.measurement_view_tabs.active),
            "adc_selection": {
                name: int(widget.value) for name, widget in self.context.adc_selectors.items()
            },
            "static_controls": self.context.widget_values(self.context.static_session_widgets),
            "radar_controls": self.context.widget_values(self.context.radar_session_widgets),
            "dynamic_controls": self.context.widget_values(self.context.dynamic_session_widgets),
            "solver_settings": self.context.widget_values(self.context.solver_setting_widgets),
            "dynamic_solvers": {
                solver: bool(checkbox.value)
                for solver, checkbox in self.context.human_room_solver_checkboxes.items()
            },
            "dynamic_preview_time_s": float(self.context.human_room_preview_time.value),
            "dynamic_start_s": float(self.context.human_room_start_time.value),
            "dynamic_end_s": float(self.context.human_room_end_time.value),
            "dynamic_result_view": self.context.dynamic_result_view_state(),
            "has_uploads": uploads_present,
            "has_motion": bool(self.context.human_room_upload.value),
            "bundle_fixture": str(self.context.fixture.value or ""),
            "bundle_path": str(self.context.bundle_path.value or ""),
            "bundle_loaded": self.context.measurement_state["bundle"] is not None,
            "background_subtraction": bool(self.context.background.value),
            "clutter_removal": str(self.context.clutter.value),
            "comparison_source": str(self.context.comparison_source.value),
            "comparison_solvers": list(self.context.comparison_solvers.value),
            "comparison_channel_zero": bool(self.context.comparison_channel_zero.value),
            "simulated_path": str(self.context.simulated_path.value or ""),
            "comparison_completed": bool(self.context.measurement_state["comparisons"]),
            "comparison_kind": self.context.measurement_state["comparison_kind"],
            "comparison_cache_key": self.context.measurement_state["comparison_cache_key"],
            "gui_result_cache_key": self.context.cache_current_gui_results(),
            "save_sequence": self.context.gui_session_state["save_sequence"],
        }

    def encoded_upload(self, widget, fallback_filename: str, max_bytes: int):
        payload = widget.value
        if not payload:
            return None
        try:
            payload = _app._validated_upload_payload(
                payload,
                label=str(widget.label or "file"),
                max_bytes=max_bytes,
            )
        except (TypeError, ValueError) as exc:
            self.context.pn.state.notifications.error(str(exc), duration=8000)
            return None
        return {
            "filename": str(widget.filename or fallback_filename),
            "payload": _app.base64.b64encode(payload).decode("ascii"),
        }

    def persist_gui_uploads(self, *_events):
        if (
            self.context.gui_session_state["restoring_uploads"]
            or self.context.gui_session_state["restoring"]
        ):
            return
        changed_widgets = {getattr(event, "obj", None) for event in _events}
        if self.context.human_upload in changed_widgets:
            self.context.gui_session_state["static_cacheable"] = False
        if changed_widgets.intersection(
            {self.context.human_room_upload, self.context.human_room_scene_upload}
        ):
            self.context.gui_session_state["dynamic_cacheable"] = False
        uploads = {
            "static_mesh": self.context.encoded_upload(
                self.context.human_upload,
                "human_mesh.npz",
                _app._MAX_STATIC_MESH_UPLOAD_BYTES,
            ),
            "motion": self.context.encoded_upload(
                self.context.human_room_upload,
                "amass_motion.npz",
                _app._MAX_MOTION_UPLOAD_BYTES,
            ),
            "scene_xml": (
                None
                if self.context.remote_access
                else self.context.encoded_upload(
                    self.context.human_room_scene_upload,
                    "scene.xml",
                    _app._MAX_SCENE_XML_UPLOAD_BYTES,
                )
            ),
        }
        self.context.gui_session_storage.saved_uploads = {
            name: upload for name, upload in uploads.items() if upload is not None
        }
        self.context.persist_gui_session()

    def apply_pending_dynamic_frame_state(self, restored):
        preview = self.context.human_room_state.get("preview")
        if preview is None:
            return
        import param

        last_frame = max(len(preview.mesh_sequence.times) - 1, 0)
        selected_frame = int(_app.np.clip(int(restored.get("dynamic_frame", 0)), 0, last_frame))
        duration_s = float(preview.mesh_sequence.times[-1])
        selected_time_s = float(preview.mesh_sequence.times[selected_frame])
        try:
            restored_time_s = float(restored["dynamic_preview_time_s"])
            if _app.np.isfinite(restored_time_s):
                selected_time_s = float(_app.np.clip(restored_time_s, 0.0, duration_s))
                selected_frame = _app._nearest_pose_index(
                    preview.mesh_sequence.times, selected_time_s
                )
        except (KeyError, TypeError, ValueError):
            pass
        frame_period_s = float(self.context.frame_period.value) * 1e-3
        try:
            restored_window = (
                float(restored["dynamic_start_s"]),
                float(restored["dynamic_end_s"]),
            )
            if not _app.np.all(_app.np.isfinite(restored_window)):
                raise ValueError("non-finite window")
            start_s, end_s = _app._human_room_window_values(
                duration_s=duration_s,
                frame_period_s=frame_period_s,
                previous_start=restored_window[0],
                previous_end=restored_window[1],
                reset_window=False,
            )
        except (KeyError, TypeError, ValueError):
            start_s, end_s = _app._human_room_window_values(
                duration_s=duration_s,
                frame_period_s=frame_period_s,
                previous_start=0.0,
                previous_end=0.0,
                reset_window=True,
            )
        self.context.configure_human_room_frame_controls(
            last_frame=last_frame,
            selected_frame=selected_frame,
            disabled=last_frame == 0,
            interval_ms=_app._motion_playback_interval_ms(preview.mesh_sequence.times),
            normal_step=_app._motion_playback_step(preview.mesh_sequence.times),
            fast_step=_app._motion_fast_playback_step(preview.mesh_sequence.times),
            selected_time_s=selected_time_s,
        )
        self.context.configure_human_room_window_controls(
            duration_s=duration_s,
            reset_window=False,
            disabled=False,
        )
        with param.parameterized.discard_events(self.context.human_room_start_time):
            self.context.human_room_start_time.value = start_s
        with param.parameterized.discard_events(self.context.human_room_end_time):
            self.context.human_room_end_time.value = end_s
        self.context.update_human_room_window_summary()
        self.context.display_human_room_frame(selected_frame, motion_time_s=selected_time_s)

    def restore_dynamic_result_view(self, restored):
        view_state = restored.get("dynamic_result_view", {})
        if not isinstance(view_state, dict):
            return
        import param

        result_tabs = next(
            (
                component
                for component in self.context.human_room_results.select(self.context.pn.Tabs)
                if component._names  # pylint: disable=protected-access
                == ["Range profile", "Range-time", "Range-Doppler"]
            ),
            None,
        )
        if result_tabs is not None:
            active = int(_app.np.clip(int(view_state.get("product_tab", 0)), 0, 2))
            with param.parameterized.discard_events(result_tabs):
                result_tabs.active = active
        range_doppler_slider = next(
            (
                widget
                for widget in self.context.human_room_results.select(
                    self.context.pn.widgets.IntSlider
                )
                if str(widget.label).startswith("Radar frame")
            ),
            None,
        )
        if range_doppler_slider is not None:
            frame = int(
                _app.np.clip(
                    int(
                        view_state.get(
                            "range_doppler_frame_index",
                            int(view_state.get("range_doppler_frame", 1)) - 1,
                        )
                    ),
                    int(range_doppler_slider.start),
                    0 if range_doppler_slider.disabled else int(range_doppler_slider.end),
                )
            )
            range_doppler_slider.value = frame

    def restore_cached_gui_results(self, restored):
        cache_key = str(restored.get("gui_result_cache_key", "") or "")
        cached = _app._get_theme_gui_results(cache_key)
        if cached is None:
            return
        static_result = cached.get("static_result")
        if static_result is not None:
            self.context.gui_session_state["static_cacheable"] = True
            self.context.present_physics_result(
                static_result,
                refined=bool(cached.get("static_refined")),
                preserve_scene_frame=False,
                geometry_already_previewed=False,
                result_settings=cached.get("static_settings"),
            )
            self.context.physics_status.object = str(
                cached.get("static_status") or self.context.physics_status.object
            )
            self.context.result_records["static"]["status"] = self.context.physics_status.object
            static_diagnostics = cached.get("static_diagnostics")
            if static_diagnostics is not None:
                self.context.diagnostics_state["result"] = static_diagnostics
                self.context.diagnostics_state["top_k"] = int(
                    cached.get("static_diagnostic_top_k", self.context.diagnostic_top_k.value)
                )
                self.context.diagnostics_summary.object = _app._solver_diagnostics_summary(
                    static_diagnostics
                )
                self.context.diagnostics_depth.object = _app._path_depth_figure(
                    static_diagnostics,
                    self.context.go,
                    plot_template=self.context.plot_template,
                )
                self.context.diagnostics_depth.visible = True
                if bool(cached.get("static_show_rt_paths")):
                    self.context.physics_scene.object = _app._solver_diagnostics_scene_figure(
                        static_result,
                        static_diagnostics,
                        self.context.go,
                        plot_template=self.context.plot_template,
                    )
                self.context.diagnostics_status.object = str(
                    cached.get("static_diagnostics_status")
                    or self.context.diagnostics_status.object
                )

        preview = cached.get("dynamic_preview")
        if preview is not None:
            self.context.human_room_state["preview"] = preview
            self.context.human_room_state["loaded_motion"] = cached.get("dynamic_loaded_motion")
            self.context.human_room_state["motion_cache_key"] = cached.get(
                "dynamic_motion_cache_key"
            )
            self.context.apply_pending_dynamic_frame_state(restored)
        dynamic_results = dict(cached.get("dynamic_results") or {})
        selection_changed = (
            restored.get("image_source_selection_version") != _app._IMAGE_SOURCE_SELECTION_VERSION
        )
        if selection_changed:
            dynamic_results.pop("hybrid_po", None)
        if dynamic_results:
            self.context.gui_session_state["dynamic_cacheable"] = True
            primary_solver = (
                "hybrid_po" if "hybrid_po" in dynamic_results else next(iter(dynamic_results))
            )
            self.context.human_room_state["result"] = dynamic_results[primary_solver]
            self.context.human_room_state["results"] = dynamic_results
            self.context.remember_result_settings("dynamic", cached.get("dynamic_settings"))
            self.context.human_room_state["diagnostics"] = (
                None if selection_changed else cached.get("dynamic_diagnostics")
            )
            self.context.human_room_results.objects = [
                _app._human_room_results_view(
                    dynamic_results,
                    self.context.pn,
                    self.context.go,
                    plot_template=self.context.plot_template,
                )
            ]
            self.context.human_room_status.object = str(
                cached.get("dynamic_status") or "**SIMULATION READY**"
            )
            self.context.result_records["dynamic"][
                "status"
            ] = self.context.human_room_status.object
            self.context.human_room_download.callback = lambda: (
                _app._human_room_export(
                    self.context.human_room_state["result"],
                    solver_results=self.context.human_room_state["results"],
                )
                if self.context.human_room_state["result"] is not None
                else _app.BytesIO()
            )
            self.context.human_room_download.disabled = False
            self.context.run_human_room_diagnostics_button.disabled = False
            self.context.human_room_diagnostics_status.object = str(
                cached.get("dynamic_diagnostics_status")
                or "Choose a preview time, then show solver overlays."
            )
            if self.context.human_room_state["diagnostics"] is not None:
                self.context.display_human_room_frame(
                    int(self.context.human_room_player.value),
                    motion_time_s=float(self.context.human_room_preview_time.value),
                )
            self.context.restore_dynamic_result_view(restored)
        self.context.gui_session_state["result_cache_key"] = cache_key

    def restore_gui_session(self, event, *, page_ready=False):
        # After a theme reload the browser reports its saved state before
        # its document_ready event. Until then Panel leaves held layout
        # updates unsent while their new models are already in the document;
        # a browser message arriving in between makes Bokeh treat those
        # models as known, so restored result views would reach the browser
        # without their definitions. Restore once the page is connected.
        # (Panel reports ``loaded`` only after its onload callbacks finish,
        # so the deferred call says it is ready instead of asking again.)
        if not page_ready and not self.context.pn.state.loaded:
            self.context.pn.state.onload(
                lambda: self.context.restore_gui_session(event, page_ready=True)
            )
            return
        restored = getattr(event, "new", None)
        if not isinstance(restored, dict) or restored.get("schema_version") not in {
            1,
            2,
        }:
            return
        self.context.gui_session_state["restored_state"] = dict(restored)
        self.context.gui_session_state["restoring"] = True
        try:
            for navigation, value, upper in (
                (self.context.tabs, restored.get("top_tab", 0), 2),
                # Always land on the scene; cached results remain in their tab.
                (self.context.human_room_view_tabs, 0, 1),
                (self.context.measurement_view_tabs, restored.get("measurement_tab", 0), 5),
            ):
                # These changes must emit Param events so Panel updates the
                # browser-side Bokeh models.  Suppressing the events changes
                # only the Python objects, leaving the newly themed document
                # on its default tabs.  ``restoring`` already prevents the
                # persistence watcher from writing an intermediate state.
                navigation.active = int(_app.np.clip(int(value), 0, upper))

            radar_values = restored.get("radar_controls", {})
            if isinstance(radar_values, dict) and "board" in radar_values:
                restored_board = str(radar_values["board"])
                if restored_board in self.context.board.options:
                    self.context.board.value = restored_board
                    selected_spec = _app.get_ti_board_spec(self.context.board.value)
                    self.context.tx_antennas.options = {
                        f"TX{index}": index for index in range(selected_spec.num_tx)
                    }
                    self.context.rx_antennas.options = {
                        f"RX{index}": index for index in range(selected_spec.num_rx)
                    }
            self.context.restore_widget_values(self.context.radar_session_widgets, radar_values)
            self.context.restore_widget_values(
                self.context.static_session_widgets,
                restored.get("static_controls", {}),
            )
            self.context.motion_forward_axis.options = self.context.motion_axis_options
            dynamic_values = restored.get("dynamic_controls", {})
            if isinstance(dynamic_values, dict) and "position_mode" not in dynamic_values:
                # Older sessions only offered the furnished room and manual
                # placement. Adopt the new GUI defaults on their first load.
                dynamic_values = {
                    **dynamic_values,
                    "room_preset": "empty_room",
                    "position_mode": "file",
                }
            self.context.restore_widget_values(
                self.context.dynamic_session_widgets,
                dynamic_values,
            )
            # Uploaded scenes arrive in a separate restoration event. Do not
            # clamp their saved placement against the initial room meanwhile.
            if not (restored.get("has_uploads") or restored.get("has_motion")):
                self.context.sync_human_position_bounds()
            self.context.sync_motion_coordinate_controls()
            restored_solver_settings, _ = self.context.restore_solver_settings(
                {
                    "schema_version": _app._SOLVER_SETTINGS_SCHEMA_VERSION,
                    "po_defaults_version": restored.get("po_defaults_version"),
                    "po_phase_span_by_mode": restored.get("po_phase_span_by_mode"),
                    "settings": restored.get("solver_settings", {}),
                }
            )
            previous_solver_settings = restored.get("solver_settings", {})
            po_defaults_changed = isinstance(previous_solver_settings, dict) and any(
                name.startswith("po_") and previous_solver_settings.get(name) != value
                for name, value in restored_solver_settings.items()
            )
            restored_dynamic_solvers = restored.get("dynamic_solvers", {})
            if isinstance(restored_dynamic_solvers, dict):
                for solver, checkbox in self.context.human_room_solver_checkboxes.items():
                    if solver in restored_dynamic_solvers:
                        # Let Panel propagate the value to the rendered
                        # checkbox.  Solver-change callbacks may clear the
                        # initial empty result pane here; cached products are
                        # restored immediately below after all solvers settle.
                        checkbox.value = bool(restored_dynamic_solvers[solver])

            is_plate = self.context.target_type.value == "plate"
            is_corner = self.context.target_type.value == "trihedral"
            is_human = self.context.target_type.value == "human_mesh"
            self.context.plate_width.visible = is_plate
            self.context.plate_height.visible = is_plate
            self.context.corner_edge.visible = is_corner
            self.context.material.visible = not is_human
            self.context.material_details.visible = not is_human
            self.context.human_upload.visible = is_human
            self.context.human_diffuse.visible = is_human
            self.context.human_source.visible = is_human
            self.context.update_material_details()
            self.context.update_antenna_pattern_controls()
            self.context.update_radar_summary()
            if not po_defaults_changed:
                self.context.restore_cached_gui_results(restored)

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
                fixture_values = set(self.context.fixture.options.values())
                self.context.fixture.value = (
                    restored_fixture if restored_fixture in fixture_values else ""
                )
                restored_bundle_path = str(restored.get("bundle_path", ""))
                if self.context.fixture.value:
                    restored_bundle_path = str(self.context.fixture.value)
                elif self.context.remote_access and restored_bundle_path:
                    try:
                        restored_bundle_path = _app._authorized_server_path(
                            restored_bundle_path,
                            remote_access=True,
                            trusted_paths=self.context.trusted_bundle_paths,
                        )
                    except (OSError, PermissionError, RuntimeError, ValueError):
                        restored_bundle_path = ""
                if self.context.fixture.value == "":
                    self.context.custom_bundle_path["value"] = restored_bundle_path
                self.context.bundle_path.value = restored_bundle_path
                self.context.background.value = bool(restored.get("background_subtraction", False))
                restored_clutter = str(restored.get("clutter_removal", "none"))
                self.context.clutter.value = (
                    restored_clutter
                    if restored_clutter in self.context.clutter.options
                    else "none"
                )
                restored_solvers = restored.get("comparison_solvers", ["po"])
                if isinstance(restored_solvers, list):
                    self.context.comparison_solvers.value = [
                        str(solver) for solver in restored_solvers if str(solver) in _app.SOLVERS
                    ]
                self.context.comparison_channel_zero.value = bool(
                    restored.get("comparison_channel_zero", False)
                )
                restored_simulated_path = str(restored.get("simulated_path", ""))
                if self.context.remote_access and restored_simulated_path:
                    try:
                        restored_simulated_path = _app._authorized_server_path(
                            restored_simulated_path,
                            remote_access=True,
                            trusted_paths=self.context.trusted_simulated_paths,
                        )
                    except (OSError, PermissionError, RuntimeError, ValueError):
                        restored_simulated_path = ""
                self.context.simulated_path.value = restored_simulated_path
                restored_source = str(restored.get("comparison_source", "run"))
                if restored_source in self.context.comparison_source.options.values():
                    self.context.comparison_source.value = restored_source
                self.context.update_comparison_source_visibility()
        except (TypeError, ValueError):
            return
        finally:

            def finish_restoration():
                self.context.gui_session_state["restoring"] = False
                self.context.refresh_result_validity()
                self.context.persist_gui_session()

            document = self.context.pn.state.curdoc
            if document is not None and document.session_context is not None:
                document.add_next_tick_callback(finish_restoration)
            else:
                try:
                    loop = _app.asyncio.get_running_loop()
                except RuntimeError:
                    finish_restoration()
                else:
                    # Queued widget watchers must see the restoration guard.
                    loop.call_soon(finish_restoration)

        if restored.get("has_uploads") or restored.get("has_motion"):
            pass
        elif self.context.human_room_state.get("preview") is None:
            self.context.human_room_scene.object = _app._human_room_scene_figure(
                None,
                self.context.go,
                frame_index=0,
                radar_orientation_deg=self.context.current_dynamic_radar_orientation(),
                room_boxes_override=self.context.current_room_boxes_override(),
                plot_template=self.context.plot_template,
            )
            source, _filename = self.context.current_human_room_source()
            if source is not None:
                try:
                    loop = _app.asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None:
                    self.context.gui_session_state["dynamic_restore_task"] = loop.create_task(
                        self.context.refresh_human_room_preview(mark_stale=False)
                    )

        if not restored.get("bundle_loaded") or not self.context.bundle_path.value:
            self.context.persist_gui_session()
            return

        async def restore_loaded_bundle():
            await self.context.on_load_bundle(None)
            if self.context.measurement_state["bundle"] is None:
                return
            for name, widget in self.context.adc_selectors.items():
                value = restored.get("adc_selection", {}).get(name, 0)
                widget.value = int(_app.np.clip(int(value), 0, widget.end))
            restored_source = str(restored.get("comparison_source", "run"))
            if restored_source in self.context.comparison_source.options.values():
                self.context.comparison_source.value = restored_source
            self.context.update_comparison_source_visibility()
            comparison_cache_key = str(restored.get("comparison_cache_key", "") or "")
            cached_comparison = _app._get_theme_comparison(
                comparison_cache_key,
                bundle_fingerprint=self.context.measurement_state["bundle"].fingerprint,
            )
            if bool(restored.get("comparison_completed")) and cached_comparison:
                self.context.set_comparison_status(
                    "info",
                    "Restoring comparison",
                    "Applying the selected theme.",
                )
                cached_results = []
                for candidate in cached_comparison["candidates"]:
                    key, label, heading, detail, adc = candidate
                    comparison = await _app.asyncio.to_thread(
                        self.context.measurement_state["bundle"].compare_adc,
                        adc,
                        background_subtraction=cached_comparison["background_subtraction"],
                        clutter_removal=cached_comparison["clutter_removal"],
                        channel_indices=cached_comparison["channel_indices"],
                    )
                    cached_results.append((key, label, heading, detail, adc, comparison))
                self.context.measurement_state["comparison_cache_key"] = comparison_cache_key
                self.context.render_comparison_results(
                    cached_results,
                    comparison_kind=str(cached_comparison["comparison_kind"]),
                    configuration_signature=cached_comparison.get("configuration_signature"),
                    result_settings=cached_comparison.get("result_settings"),
                )
                self.context.persist_gui_session()
                return
            should_restore_saved_comparison = (
                bool(restored.get("comparison_completed"))
                and restored.get("comparison_kind") == "saved"
                and restored_source != "run"
                and (restored_source != "saved" or bool(self.context.simulated_path.value))
            )
            if should_restore_saved_comparison:
                await self.context.on_compare(None)
            self.context.persist_gui_session()

        try:
            loop = _app.asyncio.get_running_loop()
        except RuntimeError:
            return
        self.context.gui_session_state["bundle_restore_task"] = loop.create_task(
            restore_loaded_bundle()
        )

    def decoded_upload(self, restored, extensions, max_bytes):
        if not isinstance(restored, dict):
            return None
        filename = _app.Path(str(restored.get("filename", ""))).name
        encoded = restored.get("payload")
        if not filename.casefold().endswith(tuple(extensions)) or not isinstance(encoded, str):
            return None
        maximum_encoded_length = 4 * ((int(max_bytes) + 2) // 3)
        if len(encoded) > maximum_encoded_length:
            return None
        try:
            payload = _app.base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            return None
        if len(payload) > int(max_bytes):
            return None
        return (filename, payload) if payload else None

    def restore_gui_uploads(self, event, *, page_ready=False):
        # Same ordering as restore_gui_session: wait for the connected page.
        if not page_ready and not self.context.pn.state.loaded:
            self.context.pn.state.onload(
                lambda: self.context.restore_gui_uploads(event, page_ready=True)
            )
            return
        restored = getattr(event, "new", None)
        if not isinstance(restored, dict):
            return
        import param

        upload_specs = [
            (
                "static_mesh",
                self.context.human_upload,
                (".obj", ".npz"),
                _app._MAX_STATIC_MESH_UPLOAD_BYTES,
            ),
            (
                "motion",
                self.context.human_room_upload,
                (".npz",),
                _app._MAX_MOTION_UPLOAD_BYTES,
            ),
        ]
        if not self.context.remote_access:
            upload_specs.append(
                (
                    "scene_xml",
                    self.context.human_room_scene_upload,
                    (".xml",),
                    _app._MAX_SCENE_XML_UPLOAD_BYTES,
                )
            )
        self.context.gui_session_state["restoring_uploads"] = True
        try:
            for name, widget, extensions, max_bytes in upload_specs:
                decoded = self.context.decoded_upload(
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
            self.context.on_human_room_upload_filename(
                type("UploadEvent", (), {"new": self.context.human_room_upload.filename})()
            )
        finally:
            self.context.gui_session_state["restoring_uploads"] = False

        self.context.sync_human_position_bounds()
        self.context.refresh_result_validity()

        if self.context.human_room_state.get("preview") is not None:
            self.context.persist_gui_session()
            return

        async def restore_dynamic_uploads():
            await self.context.refresh_human_room_preview(mark_stale=False)
            self.context.apply_pending_dynamic_frame_state(
                self.context.gui_session_state["restored_state"]
            )
            self.context.persist_gui_session()

        try:
            loop = _app.asyncio.get_running_loop()
        except RuntimeError:
            return
        self.context.gui_session_state["dynamic_restore_task"] = loop.create_task(
            restore_dynamic_uploads()
        )
