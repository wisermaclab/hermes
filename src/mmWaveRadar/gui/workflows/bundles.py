# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Bundle controls and callbacks for one GUI session."""

from __future__ import annotations

from .. import app as _app
from .base import Workflow
from .contexts import BundleContext


class BundleWorkflow(Workflow[BundleContext]):
    def select_measurement_source(self, event):
        custom = event.new == ""
        self.context.measurement_dropper.visible = custom
        self.context.bundle_path.value = (
            self.context.custom_bundle_path["value"] if custom else str(event.new)
        )

    def bundle_step_heading(self, marker, label):
        return self.context.pn.Row(
            marker,
            label,
            sizing_mode="stretch_width",
            margin=(0, 10),
            styles={"align-items": "center", "gap": "8px"},
        )

    def refresh_adc_selection(self, _event=None):
        products = self.context.measurement_state["adc_primary_products"]
        if products is None or self.context.measurement_state["configuring_adc"]:
            return
        loaded = self.context.measurement_state["bundle"]
        selection = {
            f"{name}_index": int(widget.value)
            for name, widget in self.context.adc_selectors.items()
        }
        suffix = " / ".join(
            f"{name} {widget.value}" for name, widget in self.context.adc_selectors.items()
        )
        self.context.primary_adc_plot.object = _app._adc_trace_figure(
            products,
            self.context.go,
            title=f"{_app._bundle_primary_label(loaded)} ADC · {suffix}",
            plot_template=self.context.plot_template,
            **selection,
        )
        self.context.candidate_adc_results.objects = [
            self.context.pn.pane.Plotly(
                _app._adc_trace_figure(
                    candidate,
                    self.context.go,
                    title=f"{label} ADC · {suffix}",
                    plot_template=self.context.plot_template,
                    **selection,
                ),
                height=330,
                sizing_mode="stretch_width",
            )
            for label, candidate in self.context.measurement_state["adc_candidates"]
        ]

    def refresh_profile_selection(self, _event=None):
        products = self.context.measurement_state["adc_primary_products"]
        if products is None or self.context.measurement_state["configuring_adc"]:
            return
        frame = int(self.context.range_profile_frame.value)
        self.context.range_profile_previous.disabled = frame <= 0
        self.context.range_profile_next.disabled = frame >= products.adc.shape[0] - 1
        self.context.range_profile_plot.object = _app._bundle_range_profile_comparison_figure(
            _app._bundle_primary_label(self.context.measurement_state["bundle"]),
            products,
            self.context.measurement_state["adc_candidates"],
            self.context.go,
            plot_template=self.context.plot_template,
            frame_index=frame,
        )

    def configure_adc_selection(self, products):
        self.context.measurement_state["adc_primary_products"] = products
        self.context.measurement_state["configuring_adc"] = True
        try:
            shape = products.adc.shape
            last_frame = max(int(shape[0]) - 1, 0)
            # Bokeh requires unequal slider bounds, including when hidden.
            self.context.range_profile_frame.end = max(last_frame, 1)
            self.context.range_profile_frame.label = f"Range-profile frame index (0–{last_frame})"
            self.context.range_profile_frame.disabled = shape[0] <= 1
            self.context.range_profile_frame_controls.visible = shape[0] > 1
            for widget, limit in zip(
                self.context.adc_selectors.values(), (shape[0], shape[1], shape[3])
            ):
                widget.end = max(int(limit) - 1, 0)
                widget.value = min(int(widget.value), widget.end)
                widget.disabled = False
        finally:
            self.context.measurement_state["configuring_adc"] = False
        self.context.refresh_adc_selection()
        self.context.refresh_profile_selection()

    def current_comparison_signature(
        self,
    ) -> tuple[object, ...]:
        loaded = self.context.measurement_state.get("bundle")
        return (
            getattr(loaded, "fingerprint", None),
            str(self.context.comparison_source.value),
            tuple(str(mode) for mode in self.context.comparison_solvers.value),
            bool(self.context.comparison_channel_zero.value),
            bool(self.context.background.value),
            str(self.context.clutter.value),
            str(self.context.simulated_path.value or ""),
        )

    def bundle_step_marker(self, number: int, state: str) -> str:
        if state == "complete":
            content = "&#10003;"
            background = "#3ecf8e"
            border = "#3ecf8e"
            foreground = "#06141f"
            shadow = "none"
        elif state == "active":
            content = str(number)
            background = "transparent"
            border = self.context.bundle_palette["accent"]
            foreground = self.context.bundle_palette["accent"]
            shadow = f"0 0 0 4px {self.context.bundle_palette['accent']}24"
        else:
            content = str(number)
            background = "transparent"
            border = self.context.bundle_palette["axis"]
            foreground = self.context.bundle_palette["axis"]
            shadow = "none"
        return (
            '<div style="align-items:center; background:'
            f"{background}; border:2px solid {border}; border-radius:50%; "
            f"box-shadow:{shadow}; color:{foreground}; display:flex; "
            "font-family:ui-monospace,monospace; font-size:12px; "
            "font-weight:700; height:26px; justify-content:center; "
            f'width:26px;">{content}</div>'
        )

    def update_bundle_step_rail(self, stage: int) -> None:
        stage = int(_app.np.clip(stage, 0, 2))
        first_state = "complete" if stage >= 1 else "active"
        second_state = "complete" if stage >= 2 else "active" if stage == 1 else "idle"
        self.context.bundle_step_one_marker.object = self.context.bundle_step_marker(
            1, first_state
        )
        self.context.bundle_step_two_marker.object = self.context.bundle_step_marker(
            2, second_state
        )
        self.context.bundle_step_one_label.styles = {
            "align-items": "center",
            "color": (
                self.context.bundle_palette["foreground"]
                if stage >= 0
                else self.context.bundle_palette["axis"]
            ),
            "display": "flex",
            "height": "30px",
        }
        self.context.bundle_step_two_label.styles = {
            "align-items": "center",
            "color": (
                self.context.bundle_palette["foreground"]
                if stage >= 1
                else self.context.bundle_palette["axis"]
            ),
            "display": "flex",
            "height": "30px",
        }

    def set_measurement_status(self, kind: str, title: str, detail: str) -> None:
        colors = {
            "idle": (self.context.bundle_palette["axis"], "rgba(100, 116, 139, 0.08)"),
            "info": (self.context.bundle_palette["accent"], "rgba(79, 163, 216, 0.10)"),
            "success": ("#3ecf8e", "rgba(62, 207, 142, 0.11)"),
            "error": ("#ef4444", "rgba(239, 68, 68, 0.10)"),
        }
        border, background_color = colors[kind]
        self.context.measurement_status.object = f"**{title}**  \n{detail}"
        self.context.measurement_status.styles = {
            "background": background_color,
            "border-left": f"3px solid {border}",
            "border-radius": "8px",
            "color": self.context.bundle_palette["foreground"],
            "padding": "12px 14px",
        }

    def set_comparison_status(
        self,
        kind: str,
        title: str,
        detail: str = "",
    ) -> None:
        colors = {
            "info": (self.context.bundle_palette["accent"], "rgba(79, 163, 216, 0.10)"),
            "success": ("#3ecf8e", "rgba(62, 207, 142, 0.11)"),
            "error": ("#ef4444", "rgba(239, 68, 68, 0.10)"),
        }
        border, background_color = colors[kind]
        self.context.comparison_status.object = f"**{title}**" + (
            f"  \n{detail}" if detail else ""
        )
        self.context.comparison_status.styles = {
            "background": background_color,
            "border-left": f"3px solid {border}",
            "border-radius": "8px",
            "color": self.context.bundle_palette["foreground"],
            "padding": "12px 14px",
        }
        self.context.comparison_status.visible = True

    def clear_candidate_products(
        self,
    ):
        self.context.result_records["comparison"] = None
        self.context.measurement_state["comparisons"] = {}
        self.context.measurement_state["comparison_kind"] = None
        self.context.measurement_state["comparison_cache_key"] = None
        self.context.measurement_state["comparison_signature"] = None
        self.context.measurement_state["simulated_adcs"] = {}
        self.context.measurement_state["adc_candidates"] = []
        self.context.metrics_pane.object = {"status": "No comparison."}
        for results in self.context.candidate_product_results:
            results.objects = []
            results.visible = False
        self.context.simulated_adc_download.callback = lambda: _app.BytesIO()
        self.context.simulated_adc_download.filename = "simulated_adc.npz"
        self.context.simulated_adc_download.disabled = True
        self.context.update_bundle_step_rail(
            1 if self.context.measurement_state["bundle"] is not None else 0
        )

    def clear_primary_products(
        self,
    ):
        self.context.measurement_state["adc_primary_products"] = None
        self.context.range_profile_frame.disabled = True
        self.context.range_profile_frame_controls.visible = False
        for selector in self.context.adc_selectors.values():
            selector.disabled = True
        for pane in self.context.primary_product_plots:
            pane.object = _app._empty_figure(self.context.go, self.context.plot_template)
        self.context.bundle_scene_plot.object = _app._empty_figure(
            self.context.go, self.context.plot_template
        )
        self.context.bundle_scene_status.object = (
            "Load the primary bundle to visualize its declared scene."
        )
        self.context.clear_candidate_products()

    def invalidate_loaded_measurement(self, _event):
        self.context.measurement_state["load_revision"] += 1
        self.context.measurement_state["comparison_revision"] += 1
        cancel_event = self.context.measurement_state.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        self.context.measurement_state["bundle"] = None
        self.context.clear_primary_products()
        self.context.comparison_status.object = ""
        self.context.comparison_status.visible = False
        self.context.set_measurement_status(
            "idle",
            "Bundle selection changed",
            "Load and validate the selected bundle to enable comparison.",
        )
        self.context.update_bundle_step_rail(0)
        self.context.update_comparison_source_visibility()

    def materialize_drop(self, kind, uploaded, path_input, status_pane):
        try:
            workspace = self.context.measurement_state["drop_workspace"]
            if workspace is None:
                workspace = _app.TemporaryDirectory(prefix="hermes-gui-drops-")
                self.context.measurement_state["drop_workspace"] = workspace
            self.context.measurement_state["drop_sequence"] += 1
            destination = _app.Path(workspace.name) / (
                f"{kind}-{self.context.measurement_state['drop_sequence']:04d}"
            )
            selected = _app._materialize_dropped_files(uploaded, destination)
            trusted_paths = (
                self.context.trusted_bundle_paths
                if kind == "measurement"
                else self.context.trusted_simulated_paths
            )
            selected_key = _app._server_path_key(selected)
            trusted_paths.add(selected_key)
            uploaded_paths = (
                self.context.uploaded_bundle_paths
                if kind == "measurement"
                else self.context.uploaded_simulated_paths
            )
            uploaded_paths.add(selected_key)
            path_input.value = str(selected)
            if kind == "measurement":
                self.context.custom_bundle_path["value"] = str(selected)
                self.context.set_measurement_status(
                    "info",
                    "Bundle upload ready",
                    f"{_app._safe_markdown_code(selected.name)} · "
                    f"{len(uploaded):,} file(s). "
                    "Load and validate it to continue.",
                )
            else:
                self.context.set_comparison_status(
                    "info",
                    "Saved comparison ready",
                    f"{_app._safe_markdown_code(selected.name)} · " f"{len(uploaded):,} file(s).",
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if kind == "measurement":
                self.context.set_measurement_status(
                    "error",
                    "Bundle upload failed",
                    _app._safe_markdown_code(f"{type(exc).__name__}: {exc}"),
                )
            else:
                self.context.set_comparison_status(
                    "error",
                    "Saved comparison upload failed",
                    _app._safe_markdown_code(f"{type(exc).__name__}: {exc}"),
                )
            self.context.pn.state.notifications.error(str(exc), duration=8000)

    def bind_file_dropper(self, kind, dropper, path_input, status_pane):
        def receive_drop(event):
            if not event.new:
                return
            self.context.measurement_state["drop_tokens"][kind] += 1
            token = self.context.measurement_state["drop_tokens"][kind]
            uploaded = dict(event.new)

            def settle_folder_upload():
                if token != self.context.measurement_state["drop_tokens"][kind]:
                    return
                self.context.materialize_drop(kind, uploaded, path_input, status_pane)

            document = self.context.pn.state.curdoc
            if document is not None and document.session_context is not None:
                document.add_timeout_callback(settle_folder_upload, 250)
            else:
                self.context.materialize_drop(kind, uploaded, path_input, status_pane)

        dropper.param.watch(receive_drop, "value")

    async def on_load_bundle(self, _event):
        if self.context.measurement_state["loading"]:
            return
        self.context.measurement_state["loading"] = True
        self.context.measurement_state["load_revision"] += 1
        load_revision = self.context.measurement_state["load_revision"]
        requested_path = str(self.context.bundle_path.value)
        self.context.measurement_state["bundle"] = None
        self.context.measurement_state["untrusted_bundle"] = False
        self.context.measurement_state["remote_resimulation_allowed"] = True
        self.context.clear_primary_products()
        self.context.set_gui_activity("bundle_load", True)
        load_input_states = _app._lock_widget_disabled_states(
            (
                self.context.fixture,
                self.context.measurement_dropper,
                self.context.bundle_path,
                self.context.background,
                self.context.clutter,
            )
        )
        self.context.load_bundle_button.disabled = True
        self.context.compare_button.disabled = True
        self.context.set_measurement_status(
            "info",
            "Validating bundle",
            "Checking the bundle contract and computing scene and radar products...",
        )
        try:
            requested_path = _app._authorized_server_path(
                requested_path,
                remote_access=self.context.remote_access,
                trusted_paths=self.context.trusted_bundle_paths,
            )
            requested_key = _app._server_path_key(requested_path)
            untrusted_bundle = requested_key in self.context.uploaded_bundle_paths
            loaded = await _app.asyncio.to_thread(
                _app.load_bundle,
                requested_path,
                resource_limits=(_app.GUI_BUNDLE_RESOURCE_LIMITS if untrusted_bundle else None),
            )
            if load_revision != self.context.measurement_state[
                "load_revision"
            ] or requested_path != str(self.context.bundle_path.value):
                return
            products = await _app.asyncio.to_thread(
                loaded.primary_products,
                background_subtraction=self.context.background.value,
                clutter_removal=self.context.clutter.value,
            )
            if load_revision != self.context.measurement_state[
                "load_revision"
            ] or requested_path != str(self.context.bundle_path.value):
                return
            scene_figure = await _app.asyncio.to_thread(
                _app._bundle_scene_figure,
                loaded,
                self.context.go,
                smpl_model_dir=(
                    _app._effective_smpl_model_dir(
                        self.context.human_room_smpl_model_dir.value,
                        self.context.default_model_dir,
                        remote_access=self.context.remote_access,
                    )
                    or None
                ),
                plot_template=self.context.plot_template,
            )
            if load_revision != self.context.measurement_state[
                "load_revision"
            ] or requested_path != str(self.context.bundle_path.value):
                return
            remote_resimulation_allowed = not (self.context.remote_access and untrusted_bundle)
            self.context.measurement_state["bundle"] = loaded
            self.context.measurement_state["untrusted_bundle"] = untrusted_bundle
            self.context.measurement_state["remote_resimulation_allowed"] = (
                remote_resimulation_allowed
            )
            self.context.clear_candidate_products()
            source_options = {}
            if loaded.can_resimulate and remote_resimulation_allowed:
                source_options["Run simulation from loaded bundle"] = "run"
            source_options["Load previously saved ADC or bundle"] = "saved"
            for solver in loaded.available_simulations:
                source_options[
                    "Previously saved "
                    f"{_app._SOLVER_LABELS.get(solver, solver.upper())} ADC in "
                    "loaded bundle"
                ] = solver
            self.context.measurement_state["configuring"] = True
            try:
                self.context.comparison_source.options = source_options
                if loaded.can_resimulate and remote_resimulation_allowed:
                    self.context.comparison_source.value = "run"
                elif "rt" in loaded.available_simulations:
                    self.context.comparison_source.value = "rt"
                elif loaded.available_simulations:
                    self.context.comparison_source.value = loaded.available_simulations[0]
                else:
                    self.context.comparison_source.value = "saved"
            finally:
                self.context.measurement_state["configuring"] = False
            primary_label = _app._bundle_primary_label(loaded)
            self.context.bundle_scene_plot.object = scene_figure
            scene_meta = dict(self.context.bundle_scene_plot.object.layout.meta or {})
            if loaded.bundle.motion_path is not None and not scene_meta.get(
                "bundle_has_target_preview"
            ):
                self.context.bundle_scene_status.object = (
                    "**Human preview unavailable:** "
                    f"{_app._safe_markdown_code(scene_meta.get('bundle_mesh_error', 'unknown error'))}"
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
                self.context.bundle_scene_status.object = (
                    "The scene is shown from the primary bundle only; "
                    "candidate ADC comparisons do not replace its geometry." + preview_note
                )
            self.context.primary_adc_plot.object = _app._adc_trace_figure(
                products,
                self.context.go,
                title=f"{primary_label} ADC · frame 0 / chirp 0 / channel 0",
                plot_template=self.context.plot_template,
            )
            self.context.configure_adc_selection(products)
            self.context.primary_range_time_plot.object = _app._range_time_figure(
                products,
                self.context.go,
                title=f"{primary_label} range time",
                plot_template=self.context.plot_template,
            )
            self.context.primary_range_doppler_plot.object = _app._range_doppler_figure(
                products,
                self.context.go,
                title=f"{primary_label} range Doppler",
                plot_template=self.context.plot_template,
            )
            adc_shape = tuple(int(value) for value in loaded.primary_adc.shape)
            frame_count = adc_shape[0] if adc_shape else 0
            self.context.set_measurement_status(
                "success",
                "Bundle loaded and validated",
                f"{_app._safe_markdown_code(loaded.profile)} / "
                f"{_app._safe_markdown_code(loaded.data_origin)} · ready for "
                "simulation or saved-ADC comparison.  \n"
                f"Frames `{frame_count:,}` · ADC shape `{adc_shape}` · "
                f"fingerprint {_app._safe_markdown_code(loaded.fingerprint[:12])}",
            )
            self.context.update_bundle_step_rail(1)
            self.context.update_comparison_source_visibility()
            self.context.persist_gui_session()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if load_revision == self.context.measurement_state["load_revision"]:
                self.context.measurement_state["bundle"] = None
                self.context.clear_primary_products()
                self.context.set_measurement_status(
                    "error",
                    "Bundle validation failed",
                    _app._safe_markdown_code(f"{type(exc).__name__}: {exc}"),
                )
                self.context.update_bundle_step_rail(0)
                self.context.update_comparison_source_visibility()
                self.context.pn.state.notifications.error(str(exc), duration=8000)
        finally:
            _app._restore_widget_disabled_states(load_input_states)
            self.context.measurement_state["loading"] = False
            self.context.load_bundle_button.disabled = False
            self.context.set_gui_activity("bundle_load", False)
            self.context.update_comparison_source_visibility()

    def export_simulated_adc(
        self,
    ):
        simulated_adcs = self.context.measurement_state.get("simulated_adcs", {})
        loaded = self.context.measurement_state.get("bundle")
        if (
            not simulated_adcs
            or loaded is None
            or self.context.measurement_state.get("comparison_signature")
            != self.context.current_comparison_signature()
        ):
            return _app.BytesIO()

        def adc_payload(simulated):
            arrays = {"adc": _app.np.asarray(simulated)}
            times = loaded.bundle.times
            if (
                times is not None
                and _app.np.asarray(times).shape == _app.np.asarray(simulated).shape[:-2]
            ):
                arrays["times"] = _app.np.asarray(times)
            return _app._npz_bytes(**arrays)

        if len(simulated_adcs) == 1:
            return _app.BytesIO(adc_payload(next(iter(simulated_adcs.values()))))
        archive = _app.BytesIO()
        with _app.zipfile.ZipFile(
            archive,
            "w",
            compression=_app.zipfile.ZIP_DEFLATED,
        ) as exported:
            for mode, simulated in simulated_adcs.items():
                exported.writestr(
                    f"simulated_adc_{mode}.npz",
                    adc_payload(simulated),
                )
        archive.seek(0)
        return archive

    def on_confirm_stop_simulation(self, _event):
        self.context.stop_simulation_confirmation.open = False
        cancel_event = self.context.measurement_state.get("cancel_event")
        if cancel_event is None:
            return
        cancel_event.set()
        self.context.compare_button.disabled = True
        self.context.compare_button.label = "Stopping simulation..."
        self.context.set_comparison_status(
            "info",
            "Stopping simulation",
            "Waiting for the next safe solver boundary.",
        )

    def render_comparison_results(
        self,
        results,
        *,
        comparison_kind: str,
        configuration_signature: tuple[object, ...] | None = None,
        result_settings=None,
    ):
        loaded = self.context.measurement_state["bundle"]
        if loaded is None or not results:
            return
        comparisons = {
            key: comparison for key, _label, _heading, _detail, _adc, comparison in results
        }
        self.context.measurement_state["comparisons"] = comparisons
        self.context.measurement_state["comparison_kind"] = comparison_kind
        self.context.measurement_state["comparison_signature"] = (
            self.context.current_comparison_signature()
            if configuration_signature is None
            else configuration_signature
        )
        metrics = {
            detail: comparison.metrics.to_dict()
            for _key, _label, _heading, detail, _adc, comparison in results
        }
        self.context.metrics_pane.object = (
            next(iter(metrics.values())) if len(metrics) == 1 else metrics
        )
        primary_products = results[0][-1].primary
        primary_label = _app._bundle_primary_label(loaded)
        self.context.primary_adc_plot.object = _app._adc_trace_figure(
            primary_products,
            self.context.go,
            title=f"{primary_label} ADC · frame 0 / chirp 0 / channel 0",
            plot_template=self.context.plot_template,
        )
        self.context.measurement_state["adc_candidates"] = [
            (f"{heading} {detail}", comparison.candidate)
            for _key, _label, heading, detail, _adc, comparison in results
        ]
        self.context.configure_adc_selection(primary_products)
        self.context.primary_range_time_plot.object = _app._range_time_figure(
            primary_products,
            self.context.go,
            title=f"{primary_label} range time",
            plot_template=self.context.plot_template,
        )
        self.context.primary_range_doppler_plot.object = _app._range_doppler_figure(
            primary_products,
            self.context.go,
            title=f"{primary_label} range Doppler",
            plot_template=self.context.plot_template,
        )
        self.context.candidate_range_time_results.objects = [
            self.context.pn.pane.Plotly(
                _app._range_time_figure(
                    comparison.candidate,
                    self.context.go,
                    title=f"{heading} range time · {detail}",
                    plot_template=self.context.plot_template,
                ),
                height=380,
                sizing_mode="stretch_width",
            )
            for _key, _label, heading, detail, _adc, comparison in results
        ]
        self.context.candidate_range_doppler_results.objects = [
            self.context.pn.pane.Plotly(
                _app._range_doppler_figure(
                    comparison.candidate,
                    self.context.go,
                    title=f"{heading} range Doppler · {detail}",
                    plot_template=self.context.plot_template,
                ),
                height=380,
                sizing_mode="stretch_width",
            )
            for _key, _label, heading, detail, _adc, comparison in results
        ]
        for candidate_results in self.context.candidate_product_results:
            candidate_results.visible = True
        if comparison_kind == "run":
            self.context.measurement_state["simulated_adcs"] = {
                key: adc for key, _label, _heading, _detail, adc, _comparison in results
            }
            self.context.simulated_adc_download.callback = self.context.export_simulated_adc
            result_modes = [result[0] for result in results]
            self.context.simulated_adc_download.filename = (
                f"simulated_adc_{result_modes[0]}.npz"
                if len(result_modes) == 1
                else "simulated_adc_results.zip"
            )
            self.context.simulated_adc_download.disabled = False
        candidate_labels = ", ".join(result[1] for result in results)
        self.context.set_comparison_status(
            "success",
            "Comparison complete",
            f"Primary bundle vs {_app._safe_markdown_text(candidate_labels)}.",
        )
        self.context.update_bundle_step_rail(2)
        self.context.remember_result_settings("comparison", result_settings)

    async def on_compare(self, _event):
        if self.context.measurement_state["running"]:
            self.context.stop_simulation_confirmation.open = True
            return
        if self.context.measurement_state["action_running"]:
            return
        loaded = self.context.measurement_state["bundle"]
        if loaded is None:
            return

        comparison_source_value = str(self.context.comparison_source.value)
        run_simulation = comparison_source_value == "run"
        if run_simulation and not self.context.measurement_state.get(
            "remote_resimulation_allowed",
            True,
        ):
            self.context.set_comparison_status(
                "error",
                "Remote resimulation disabled",
                "Bundles uploaded to a remotely bound GUI cannot execute "
                "scene references. Use a server-configured trusted bundle.",
            )
            return
        selected_solvers = list(self.context.comparison_solvers.value)
        if run_simulation and not selected_solvers:
            self.context.set_comparison_status(
                "error",
                "Solver required",
                "Select at least one solver to continue.",
            )
            return
        self.context.measurement_state["comparison_revision"] += 1
        comparison_revision = self.context.measurement_state["comparison_revision"]
        configuration_signature = self.context.current_comparison_signature()
        run_settings = self.context.capture_result_settings("comparison")
        background_value = bool(self.context.background.value)
        clutter_value = str(self.context.clutter.value)
        channel_zero_value = bool(self.context.comparison_channel_zero.value)
        simulated_path_value = str(self.context.simulated_path.value or "")
        self.context.measurement_state["action_running"] = True
        self.context.set_gui_activity("comparison", True)
        run_input_states = _app._lock_widget_disabled_states(
            (
                self.context.fixture,
                self.context.measurement_dropper,
                self.context.bundle_path,
                self.context.background,
                self.context.clutter,
                self.context.load_bundle_button,
                self.context.comparison_source,
                self.context.comparison_solvers,
                self.context.comparison_channel_zero,
                self.context.simulated_dropper,
                self.context.simulated_path,
            )
        )
        cancel_event = None
        self.context.clear_candidate_products()
        if run_simulation:
            cancel_event = _app.threading.Event()
            self.context.measurement_state["cancel_event"] = cancel_event
            self.context.measurement_state["running"] = True
            self.context.compare_button.disabled = False
            self.context.compare_button.label = "Stop simulation"
            self.context.compare_button.icon = "player-stop"
            self.context.compare_button.color = "danger"
        else:
            self.context.compare_button.disabled = True

        try:
            results = []
            if run_simulation:
                assert cancel_event is not None
                from validation.simulation import simulate_bundle_adc

                for index, solver in enumerate(selected_solvers, start=1):
                    if cancel_event.is_set():
                        raise InterruptedError("Simulation cancelled by user")
                    solver_label = _app._SOLVER_LABELS[solver]
                    self.context.set_comparison_status(
                        "info",
                        f"Running {solver_label}",
                        f"Solver {index} of {len(selected_solvers)}.",
                    )
                    worker_task = _app.asyncio.create_task(
                        _app.asyncio.to_thread(
                            simulate_bundle_adc,
                            loaded.bundle,
                            solver=solver,
                            rt_refresh_frames=int(self.context.rt_refresh.value),
                            po_refresh_on_phase_error_rad=float(self.context.po_phase_error.value),
                            solver_settings={
                                "po_refresh_on_area_fraction": float(
                                    self.context.po_area_fraction.value
                                ),
                                "po_refresh_on_normal_deg": float(
                                    self.context.po_normal_deg.value
                                ),
                                "po_visibility_samples_per_face": int(
                                    self.context.po_visibility_samples.value
                                ),
                                "po_adaptive_visibility_sampling": bool(
                                    self.context.po_adaptive_visibility.value
                                ),
                                "po_adaptive_visibility_edge_margin": float(
                                    self.context.po_visibility_edge_margin.value
                                ),
                            },
                            hybrid_calibration=str(self.context.hybrid_calibration.value),
                            smpl_model_dir=(
                                _app._effective_smpl_model_dir(
                                    self.context.human_room_smpl_model_dir.value,
                                    self.context.default_model_dir,
                                    remote_access=self.context.remote_access,
                                )
                                or None
                            ),
                            channel_zero_only=channel_zero_value,
                            cancel_check=cancel_event.is_set,
                        )
                    )
                    while not worker_task.done():
                        await _app.asyncio.sleep(0.15)
                    simulated = await worker_task
                    if cancel_event.is_set():
                        raise InterruptedError("Simulation cancelled by user")
                    comparison = await _app.asyncio.to_thread(
                        loaded.compare_adc,
                        simulated,
                        background_subtraction=background_value,
                        clutter_removal=clutter_value,
                        channel_indices=((0,) if channel_zero_value else None),
                    )
                    results.append(
                        (
                            solver,
                            f"new {solver_label} simulation",
                            "Simulated",
                            solver_label,
                            _app.np.asarray(simulated),
                            comparison,
                        )
                    )
            elif comparison_source_value == "saved":
                self.context.set_comparison_status(
                    "info",
                    "Comparing saved ADC",
                    "Loading and processing the selected candidate.",
                )
                simulated_path_value = _app._authorized_server_path(
                    simulated_path_value,
                    remote_access=self.context.remote_access,
                    trusted_paths=self.context.trusted_simulated_paths,
                )
                candidate_path = _app.Path(simulated_path_value).expanduser()
                untrusted_candidate = (
                    _app._server_path_key(candidate_path) in self.context.uploaded_simulated_paths
                )
                if candidate_path.is_dir() or (candidate_path.suffix.casefold() == ".zip"):
                    candidate_bundle = await _app.asyncio.to_thread(
                        _app.load_bundle,
                        candidate_path,
                        resource_limits=(
                            _app.GUI_BUNDLE_RESOURCE_LIMITS if untrusted_candidate else None
                        ),
                    )
                    simulated = candidate_bundle.primary_adc
                    candidate_label = f"external {candidate_bundle.data_origin} bundle"
                else:
                    simulated = await _app.asyncio.to_thread(
                        _app.load_simulated_adc,
                        candidate_path,
                        npz_limits=(_app.GUI_NPZ_LIMITS if untrusted_candidate else None),
                    )
                    candidate_label = "external ADC"
                comparison = await _app.asyncio.to_thread(
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
                        _app.np.asarray(simulated),
                        comparison,
                    )
                )
            else:
                self.context.set_comparison_status(
                    "info",
                    "Comparing bundled ADC",
                    "Loading and processing the selected solver result.",
                )
                simulated = await _app.asyncio.to_thread(
                    loaded.load_bundled_simulation,
                    comparison_source_value,
                )
                candidate_label = "bundled " + _app._SOLVER_LABELS.get(
                    comparison_source_value, comparison_source_value.upper()
                )
                comparison = await _app.asyncio.to_thread(
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
                        _app.np.asarray(simulated),
                        comparison,
                    )
                )

            comparison_kind = "run" if run_simulation else "saved"
            if (
                comparison_revision != self.context.measurement_state["comparison_revision"]
                or self.context.measurement_state["bundle"] is not loaded
                or configuration_signature != self.context.current_comparison_signature()
            ):
                return
            self.context.render_comparison_results(
                results,
                comparison_kind=comparison_kind,
                configuration_signature=configuration_signature,
                result_settings=run_settings,
            )
            channel_indices = (0,) if run_simulation and channel_zero_value else None
            self.context.measurement_state["comparison_cache_key"] = _app._cache_theme_comparison(
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
            self.context.set_comparison_status(
                "info",
                "Simulation stopped",
                "Stopped at the next safe solver boundary.",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.context.set_comparison_status(
                "error",
                "Comparison failed",
                _app._safe_markdown_code(f"{type(exc).__name__}: {exc}"),
            )
            self.context.pn.state.notifications.error(str(exc), duration=8000)
        finally:
            self.context.stop_simulation_confirmation.open = False
            _app._restore_widget_disabled_states(run_input_states)
            self.context.measurement_state["action_running"] = False
            self.context.set_gui_activity("comparison", False)
            if run_simulation:
                self.context.measurement_state["running"] = False
                self.context.measurement_state["cancel_event"] = None
            self.context.update_comparison_source_visibility()
            self.context.persist_gui_session()

    def update_comparison_source_visibility(self, event=None):
        del event
        value = self.context.comparison_source.value
        run_simulation = value == "run"
        load_saved = value == "saved"
        self.context.comparison_solvers.visible = run_simulation
        self.context.comparison_channel_zero.visible = run_simulation
        self.context.simulated_dropper.visible = load_saved
        self.context.simulated_adc_download.visible = run_simulation
        self.context.simulated_adc_download.disabled = not (
            run_simulation
            and self.context.measurement_state["comparison_kind"] == "run"
            and bool(self.context.measurement_state["simulated_adcs"])
        )
        if self.context.measurement_state["running"]:
            self.context.comparison_dependency_hint.visible = False
            self.context.compare_button.disabled = False
            self.context.compare_button.label = "Stop simulation"
            self.context.compare_button.icon = "player-stop"
            self.context.compare_button.color = "danger"
            return
        self.context.compare_button.color = "primary"
        if run_simulation:
            self.context.compare_button.label = "Run simulation"
            self.context.compare_button.icon = "player-play"
        elif load_saved:
            self.context.compare_button.label = "Load saved ADC and compare"
            self.context.compare_button.icon = "folder-open"
        else:
            self.context.compare_button.label = "Compare saved bundled ADC"
            self.context.compare_button.icon = "chart-histogram"

        bundle_ready = self.context.measurement_state["bundle"] is not None
        action_ready = bundle_ready
        hint = ""
        if not bundle_ready:
            hint = "🔒 **Load and validate a primary bundle to enable this step.**"
        elif run_simulation and not self.context.comparison_solvers.value:
            action_ready = False
            hint = "▣ **Select at least one simulation mode to continue.**"
        elif load_saved:
            saved_path = str(self.context.simulated_path.value).strip()
            saved_path_allowed = bool(saved_path)
            if saved_path_allowed and self.context.remote_access:
                try:
                    saved_path_allowed = (
                        _app._server_path_key(saved_path) in self.context.trusted_simulated_paths
                    )
                except (OSError, RuntimeError, ValueError):
                    saved_path_allowed = False
            if not saved_path_allowed:
                action_ready = False
                hint = "📎 **Drop a saved ADC file or bundle to enable comparison.**"
        self.context.compare_button.disabled = not action_ready
        self.context.comparison_dependency_hint.object = hint
        self.context.comparison_dependency_hint.visible = bool(hint)

    def invalidate_comparison_configuration(self, event) -> None:
        self.context.update_comparison_source_visibility()
        if (
            self.context.measurement_state["configuring"]
            or self.context.gui_session_state["restoring"]
        ):
            return
        self.context.measurement_state["comparison_revision"] += 1
        cancel_event = self.context.measurement_state.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        if self.context.result_records["comparison"] is None:
            self.context.clear_candidate_products()
        self.context.refresh_result_validity()
        if self.context.measurement_state["bundle"] is not None:
            self.context.set_comparison_status(
                "info",
                "Comparison settings changed",
                "Run the comparison again to generate matching products.",
            )
