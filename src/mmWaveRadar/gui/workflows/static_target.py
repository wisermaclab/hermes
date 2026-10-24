# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""StaticTarget controls and callbacks for one GUI session."""

from __future__ import annotations

from .. import app as _app
from .base import Workflow
from .contexts import StaticTargetContext


class StaticTargetWorkflow(Workflow[StaticTargetContext]):
    def update_angle_status(self, products, range_bins):
        ranges_m = _app.np.asarray(products["ranges_m"], dtype=float)
        first_bin, last_bin = sorted(int(value) for value in range_bins)
        selected = (
            f"bin `{first_bin}` at `{ranges_m[first_bin]:.3f} m`"
            if first_bin == last_bin
            else (
                f"bins `{first_bin}–{last_bin}` at "
                f"`{ranges_m[first_bin]:.3f}–{ranges_m[last_bin]:.3f} m`"
            )
        )
        self.context.angle_status.object = (
            f"**Selected:** {selected} · target centroid "
            f"`{float(products['centroid_range_m']):.3f} m` "
            f"(nearest bin `{int(products['centroid_range_bin'])}`)"
        )

    def invalidate_physics_result(
        self,
    ) -> None:
        """Fail closed whenever solver inputs no longer match the result."""

        self.context.physics_state["result"] = None
        self.context.physics_state["refined"] = False
        self.context.gui_session_state["static_cacheable"] = False
        self.context.physics_download.callback = lambda: (
            _app._physics_export(self.context.physics_state["result"])
            if self.context.physics_state["result"] is not None
            else _app.BytesIO()
        )
        self.context.physics_download.disabled = True
        self.context.diagnostics_state["revision"] += 1
        self.context.diagnostics_state["result"] = None
        self.context.diagnostics_state["top_k"] = None
        diagnostic_cancel = self.context.diagnostics_state.get("cancel_event")
        if diagnostic_cancel is not None:
            diagnostic_cancel.set()
        self.context.diagnostics_state["cancel_event"] = None
        import param

        with param.parameterized.discard_events(self.context.show_rt_paths):
            self.context.show_rt_paths.value = False
        self.context.show_rt_paths.disabled = True
        self.context.diagnostic_top_k.disabled = True
        self.context.diagnostics_depth.visible = False
        self.context.diagnostics_summary.object = "**No current solver result.**"
        self.context.diagnostics_status.object = (
            "RT path geometry is unavailable until the current static simulation finishes."
        )

    def present_physics_result(
        self,
        result,
        *,
        refined: bool,
        preserve_scene_frame: bool,
        geometry_already_previewed: bool,
        result_settings=None,
    ):
        had_diagnostics_overlay = self.context.diagnostics_state["result"] is not None
        self.context.physics_state["result"] = result
        self.context.physics_state["refined"] = bool(refined)
        self.context.gui_session_state["static_cacheable"] = True
        if preserve_scene_frame and not had_diagnostics_overlay:
            self.context.physics_state["target_result_sequence"] += 1
            self.context.scene_interaction.target_result = _app._target_trace_update(
                result,
                sequence=self.context.physics_state["target_result_sequence"],
                include_geometry=not geometry_already_previewed,
            )
        else:
            self.context.physics_scene.object = _app._physics_scene_figure(
                result,
                self.context.go,
                plot_template=self.context.plot_template,
            )
        self.context.physics_profile.object = _app._range_profile_figure(
            result,
            self.context.go,
            plot_template=self.context.plot_template,
        )
        self.context.physics_channel_profiles.objects = [
            self.context.pn.pane.Plotly(figure, height=figure.layout.height)
            for figure in _app._static_channel_profile_figures(
                result, self.context.go, plot_template=self.context.plot_template
            )
        ]
        self.context.angle_state["revision"] += 1
        angle_products = _app._angle_fft_range_products(result)
        self.context.angle_state["products"] = angle_products
        centroid_bin = int(angle_products["centroid_range_bin"])
        import param

        with param.parameterized.discard_events(self.context.angle_range_bins):
            self.context.angle_range_bins.start = 0
            self.context.angle_range_bins.end = len(angle_products["ranges_m"]) - 1
            self.context.angle_range_bins.value = (centroid_bin, centroid_bin)
        self.context.angle_map.object = _app._angle_map_figure(
            angle_products,
            self.context.make_subplots,
            self.context.go,
            range_bins=self.context.angle_range_bins.value,
            plot_template=self.context.plot_template,
        )
        self.context.update_angle_status(angle_products, self.context.angle_range_bins.value)
        self.context.physics_download.callback = lambda: (
            _app._physics_export(self.context.physics_state["result"])
            if self.context.physics_state["result"] is not None
            else _app.BytesIO()
        )
        self.context.physics_download.disabled = False
        self.context.show_rt_paths.disabled = False
        self.context.diagnostic_top_k.disabled = False
        tier = "HIGHER FIDELITY" if refined else "LIVE RT + PO"
        self.context.physics_status.object = (
            f"**{tier}** · {result.faces.shape[0]:,} facets · "
            f"{result.visible_face_count:,} PO-visible · "
            f"{result.rt_path_count:,} RT paths · "
            f"PO {result.po_runtime_s:.3f} s · RT {result.rt_runtime_s:.3f} s · "
            "uncalibrated amplitudes · manifest "
            f"`{result.manifest.fingerprint[:12]}`"
        )
        self.context.diagnostics_state["revision"] += 1
        self.context.diagnostics_state["result"] = None
        self.context.diagnostics_state["top_k"] = None
        import param

        with param.parameterized.discard_events(self.context.show_rt_paths):
            self.context.show_rt_paths.value = False
        self.context.diagnostics_depth.visible = False
        self.context.diagnostics_summary.object = (
            f"**RT valid link-paths:** `{result.rt_path_count:,}` · "
            f"**PO visible facets:** `{result.visible_face_count:,}`"
        )
        self.context.diagnostics_status.object = (
            "RT path geometry is off. Enable it to inspect the current simulated state."
        )
        self.context.remember_result_settings("static", result_settings)
        self.context.persist_gui_session()

    def current_physics_inputs(
        self,
    ) -> dict:
        selected_human_mesh = self.context.human_upload.value
        selected_human_filename = self.context.human_upload.filename
        if selected_human_mesh:
            selected_human_mesh = _app._validated_upload_payload(
                selected_human_mesh,
                label="human mesh",
                max_bytes=_app._MAX_STATIC_MESH_UPLOAD_BYTES,
            )
            if _app.Path(selected_human_filename or "").suffix.casefold() == ".npz":
                _app._preflight_gui_mesh_npz(selected_human_mesh)
            elif _app.Path(selected_human_filename or "").suffix.casefold() == ".obj":
                _app._preflight_gui_mesh_obj(selected_human_mesh)
        if (
            self.context.target_type.value == "human_mesh"
            and not selected_human_mesh
            and self.context.default_human_mesh is not None
        ):
            selected_human_mesh = self.context.default_human_mesh
            selected_human_filename = self.context.default_human_mesh.name
        return {
            "target_type": self.context.target_type.value,
            "range_m": self.context.target_range.value,
            "target_y_m": self.context.target_y.value,
            "target_z_m": self.context.target_z.value,
            "width_m": self.context.plate_width.value,
            "height_m": self.context.plate_height.value,
            "corner_edge_m": self.context.corner_edge.value,
            "yaw_deg": self.context.yaw.value,
            "pitch_deg": self.context.pitch.value,
            "roll_deg": self.context.roll.value,
            "radar_yaw_deg": self.context.radar_yaw.value,
            "radar_pitch_deg": self.context.radar_pitch.value,
            "radar_roll_deg": self.context.radar_roll.value,
            "material_name": self.context.material.value,
            "human_mesh": selected_human_mesh,
            "human_mesh_filename": selected_human_filename,
            "rt_human_diffuse_coefficient": self.context.human_diffuse.value,
            "board_model": self.context.board.value,
            "carrier_frequency_hz": self.context.carrier_frequency.value * 1e9,
            "slope_hz_per_s": self.context.chirp_slope.value * 1e12,
            "chirp_duration_s": self.context.chirp_duration.value * 1e-6,
            "chirp_repetition_time_s": self.context.chirp_repetition.value * 1e-6,
            "sampling_frequency_hz": self.context.sampling_frequency.value * 1e6,
            "adc_mode": self.context.adc_sampling_mode.value,
            "if_bandwidth_hz": self.context.adc_if_bandwidth.value * 1e6 or None,
            "num_adc_samples": self.context.num_adc_samples.value,
            "num_chirps_per_frame": self.context.num_chirps.value,
            "frame_period_s": self.context.frame_period.value * 1e-3,
            "tx_schedule": self.context.tx_schedule.value,
            "selected_tx_indices": tuple(self.context.tx_antennas.value),
            "selected_rx_indices": tuple(self.context.rx_antennas.value),
            "antenna_pattern_mode": self.context.antenna_pattern.value,
            "cosine_3db_beamwidth_deg": self.context.cosine_3db_beamwidth.value,
            "po_integration_mode": self.context.po_integration_mode.value,
            "po_quadrature_phase_span_scale_rad": self.context.po_phase_span.value,
            "po_curvature_error_rad": self.context.po_curvature.value,
            "po_quadrature_max_refinement_depth": self.context.po_refinement_depth.value,
            "po_quadrature_max_subfaces_per_parent": self.context.po_subface_cap.value,
            "rt_samples_per_source": self.context.rt_samples.value,
            "rt_max_paths_per_source": self.context.rt_path_cap.value,
            "rt_max_depth": self.context.rt_depth.value,
            "array_model": self.context.array_model.value,
            "array_tx_groups": self.context.array_tx_groups.value,
            "array_rx_groups": self.context.array_rx_groups.value,
        }

    async def update_physics(
        self,
        *,
        fidelity: str,
        preserve_scene_frame: bool = False,
        geometry_already_previewed: bool = False,
        force: bool = False,
    ):
        if self.context.gui_session_state["restoring"]:
            return
        if not force and fidelity == "preview" and self.context.settings_match_result("static"):
            self.context.physics_state["revision"] += 1
            self.context.physics_state["pending_request"] = None
            cancel_event = self.context.physics_state.get("cancel_event")
            if cancel_event is not None:
                cancel_event.set()
            self.context.refresh_result_validity()
            return
        self.context.physics_state["revision"] += 1
        revision = self.context.physics_state["revision"]
        request = {
            "revision": revision,
            "fidelity": fidelity,
            "refined": fidelity == "high_fidelity",
            "preserve_scene_frame": bool(preserve_scene_frame),
            "geometry_already_previewed": bool(geometry_already_previewed),
            "inputs": self.context.current_physics_inputs(),
            "settings": self.context.capture_result_settings("static"),
        }
        self.context.physics_state["pending_request"] = request
        self.context.invalidate_physics_result()
        if (
            self.context.target_type.value == "human_mesh"
            and not self.context.human_upload.value
            and self.context.default_human_mesh is None
        ):
            self.context.physics_state["pending_request"] = None
            cancel_event = self.context.physics_state.get("cancel_event")
            if cancel_event is not None:
                cancel_event.set()
            self.context.physics_status.object = (
                "**Human mesh selected.** Upload a pickle-free `.obj` or `.npz` "
                "mesh to run RT and PO."
            )
            return
        if self.context.physics_state["running"]:
            cancel_event = self.context.physics_state.get("cancel_event")
            if cancel_event is not None:
                cancel_event.set()
            self.context.physics_status.object = (
                "**UPDATING** · The active run is stopping; the latest control state is queued."
            )
            return

        self.context.physics_state["running"] = True
        self.context.set_gui_activity("static", True)
        self.context.refine_physics.disabled = True
        self.context.stop_physics.disabled = False
        self.context.stop_physics.color = "danger"
        self.context.physics_scene.loading = True
        self.context.physics_profile.loading = True
        self.context.angle_map.loading = True
        try:
            while self.context.physics_state["pending_request"] is not None:
                active_request = self.context.physics_state["pending_request"]
                self.context.physics_state["pending_request"] = None
                cancel_event = _app.threading.Event()
                self.context.physics_state["cancel_event"] = cancel_event
                self.context.physics_status.object = (
                    "Running higher-density RT and PO..."
                    if active_request["refined"]
                    else "Updating RT and PO from the current controls..."
                )
                try:
                    result = await _app.asyncio.to_thread(
                        _app._call_with_array_groups,
                        _app.run_static_target_experiment,
                        fidelity=active_request["fidelity"],
                        cancel_check=cancel_event.is_set,
                        **active_request["inputs"],
                    )
                except InterruptedError:
                    if (
                        self.context.physics_state["pending_request"] is None
                        and active_request["revision"] == self.context.physics_state["revision"]
                    ):
                        self.context.physics_status.object = (
                            "**STOPPED** · The static simulation was "
                            "terminated at a safe solver boundary."
                        )
                    continue
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    if active_request["revision"] == self.context.physics_state["revision"]:
                        self.context.physics_status.object = (
                            "**Update failed:** "
                            f"{_app._safe_markdown_code(f'{type(exc).__name__}: {exc}')}"
                        )
                        self.context.pn.state.notifications.error(str(exc), duration=8000)
                    continue
                if (
                    active_request["revision"] == self.context.physics_state["revision"]
                    and self.context.physics_state["pending_request"] is None
                    and active_request["settings"]["fingerprint"]
                    == self.context.capture_result_settings("static")["fingerprint"]
                ):
                    self.context.present_physics_result(
                        result,
                        refined=active_request["refined"],
                        preserve_scene_frame=active_request["preserve_scene_frame"],
                        geometry_already_previewed=active_request["geometry_already_previewed"],
                        result_settings=active_request["settings"],
                    )
        finally:
            self.context.physics_state["running"] = False
            self.context.physics_state["cancel_event"] = None
            self.context.refine_physics.disabled = False
            self.context.stop_physics.disabled = True
            self.context.stop_physics.color = "light"
            self.context.set_gui_activity("static", False)
            self.context.physics_download.disabled = self.context.physics_state["result"] is None
            self.context.physics_scene.loading = False
            self.context.physics_profile.loading = False
            self.context.angle_map.loading = False
            self.context.refresh_result_validity()

    async def on_refine_physics(self, _event):
        await self.context.update_physics(fidelity="high_fidelity")

    def on_stop_physics(self, _event):
        self.context.physics_state["pending_request"] = None
        cancel_event = self.context.physics_state.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        diagnostic_cancel = self.context.diagnostics_state.get("cancel_event")
        if diagnostic_cancel is not None:
            diagnostic_cancel.set()
        self.context.stop_physics.disabled = True
        if cancel_event is not None:
            self.context.physics_status.object = (
                "**STOPPING** · Waiting for the current RT/PO kernel to reach a safe boundary."
            )
        elif diagnostic_cancel is not None:
            self.context.diagnostics_status.object = (
                "**STOPPING** · Waiting for RT path extraction to reach a safe boundary."
            )

    async def compute_rt_path_overlay(
        self,
    ):
        if self.context.physics_state["result"] is None:
            self.context.diagnostics_status.object = (
                "Run the static simulation before requesting RT paths."
            )
            return
        self.context.diagnostics_state["revision"] += 1
        revision = self.context.diagnostics_state["revision"]
        source_result = self.context.physics_state["result"]
        requested_top_k = int(self.context.diagnostic_top_k.value)
        cancel_event = _app.threading.Event()
        self.context.diagnostics_state["cancel_event"] = cancel_event
        self.context.set_gui_activity("static", True)
        self.context.show_rt_paths.disabled = True
        self.context.diagnostic_top_k.disabled = True
        self.context.stop_physics.disabled = False
        self.context.stop_physics.color = "danger"
        self.context.physics_scene.loading = True
        self.context.diagnostics_depth.loading = True
        self.context.diagnostics_status.object = (
            "Tracing the current state and extracting compact RT/PO diagnostics..."
        )
        try:
            diagnostic_result = await _app.asyncio.to_thread(
                _app.run_static_target_diagnostics,
                source_result,
                top_k=requested_top_k,
                rt_samples_per_source=int(self.context.rt_samples.value),
                rt_max_paths_per_source=int(self.context.rt_path_cap.value),
                rt_max_depth=int(self.context.rt_depth.value),
                cancel_check=cancel_event.is_set,
            )
            if (
                revision == self.context.diagnostics_state["revision"]
                and self.context.physics_state["result"] is source_result
                and requested_top_k == int(self.context.diagnostic_top_k.value)
            ):
                self.context.diagnostics_state["result"] = diagnostic_result
                self.context.diagnostics_state["top_k"] = requested_top_k
                self.context.physics_scene.object = _app._solver_diagnostics_scene_figure(
                    source_result,
                    diagnostic_result,
                    self.context.go,
                    plot_template=self.context.plot_template,
                )
                self.context.diagnostics_depth.object = _app._path_depth_figure(
                    diagnostic_result,
                    self.context.go,
                    plot_template=self.context.plot_template,
                )
                self.context.diagnostics_summary.object = _app._solver_diagnostics_summary(
                    diagnostic_result
                )
                self.context.diagnostics_depth.visible = True
                self.context.diagnostics_status.object = (
                    f"**READY** · "
                    f"{diagnostic_result.rt.selected_path_indices.size:,} "
                    "target-touching RT paths overlaid in **RT/PO results**"
                )
                self.context.persist_gui_session()
        except InterruptedError:
            if revision == self.context.diagnostics_state["revision"]:
                self.context.diagnostics_status.object = (
                    "**STOPPED** · RT path extraction was cancelled."
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if revision == self.context.diagnostics_state["revision"]:
                self.context.diagnostics_status.object = (
                    "**Diagnostic trace failed:** "
                    f"{_app._safe_markdown_code(f'{type(exc).__name__}: {exc}')}"
                )
                self.context.pn.state.notifications.error(str(exc), duration=8000)
                import param

                with param.parameterized.discard_events(self.context.show_rt_paths):
                    self.context.show_rt_paths.value = False
        finally:
            if self.context.diagnostics_state.get("cancel_event") is cancel_event:
                self.context.diagnostics_state["cancel_event"] = None
            if revision == self.context.diagnostics_state["revision"]:
                has_result = self.context.physics_state["result"] is not None
                self.context.show_rt_paths.disabled = not has_result
                self.context.diagnostic_top_k.disabled = not has_result
                self.context.stop_physics.disabled = not self.context.physics_state["running"]
                self.context.stop_physics.color = (
                    "danger" if self.context.physics_state["running"] else "light"
                )
                self.context.physics_scene.loading = False
                self.context.diagnostics_depth.loading = False
            self.context.set_gui_activity("static", bool(self.context.physics_state["running"]))

    async def on_show_rt_paths(self, event):
        if self.context.physics_state["result"] is None:
            return
        if not bool(event.new):
            self.context.physics_scene.object = _app._physics_scene_figure(
                self.context.physics_state["result"],
                self.context.go,
                plot_template=self.context.plot_template,
            )
            self.context.diagnostics_status.object = (
                "RT path geometry is off. The latest path counts and "
                "interaction-depth histogram remain below."
                if self.context.diagnostics_state["result"] is not None
                else ("RT path geometry is off. Enable it to inspect the current simulated state.")
            )
            self.context.persist_gui_session()
            return
        if self.context.diagnostics_state["result"] is not None and self.context.diagnostics_state[
            "top_k"
        ] == int(self.context.diagnostic_top_k.value):
            self.context.physics_scene.object = _app._solver_diagnostics_scene_figure(
                self.context.physics_state["result"],
                self.context.diagnostics_state["result"],
                self.context.go,
                plot_template=self.context.plot_template,
            )
            self.context.diagnostics_status.object = (
                f"**READY** · "
                f"{self.context.diagnostics_state['result'].rt.selected_path_indices.size:,} "
                "target-touching RT paths overlaid"
            )
            self.context.persist_gui_session()
            return
        await self.context.compute_rt_path_overlay()

    async def on_diagnostic_top_k_change(self, _event):
        if self.context.show_rt_paths.value:
            await self.context.compute_rt_path_overlay()

    async def on_physics_control_change(self, event):
        changed_widget = getattr(event, "obj", None)
        ignored_count = self.context.ignored_physics_events.get(id(changed_widget), 0)
        if ignored_count:
            if ignored_count == 1:
                self.context.ignored_physics_events.pop(id(changed_widget), None)
            else:
                self.context.ignored_physics_events[id(changed_widget)] = ignored_count - 1
            return
        if changed_widget in (
            self.context.plate_width,
            self.context.plate_height,
            self.context.corner_edge,
        ):
            normalized = _app._rounded_target_size(changed_widget.value)
            if normalized != changed_widget.value:
                import param

                with param.parameterized.discard_events(changed_widget):
                    changed_widget.value = normalized
        await self.context.update_physics(
            fidelity="preview",
            preserve_scene_frame=changed_widget
            in (self.context.yaw, self.context.pitch, self.context.roll),
        )

    async def on_target_gesture(self, event):
        gesture = getattr(event, "new", None)
        if not isinstance(gesture, dict):
            return
        try:
            new_yaw = float(gesture["yaw_deg"])
            new_pitch = float(gesture["pitch_deg"])
        except (KeyError, TypeError, ValueError):
            return
        if not _app.np.isfinite(new_yaw) or not _app.np.isfinite(new_pitch):
            return
        new_yaw = float(_app.np.clip(new_yaw, self.context.yaw.start, self.context.yaw.end))
        new_pitch = float(
            _app.np.clip(new_pitch, self.context.pitch.start, self.context.pitch.end)
        )
        yaw_delta = abs(new_yaw - float(self.context.yaw.value))
        pitch_delta = abs(new_pitch - float(self.context.pitch.value))
        if yaw_delta < 0.05 and pitch_delta < 0.05:
            return
        # Synchronize visible numeric fields while the gesture owns one solve.
        # Count the value events so delayed async watchers also skip this solve.
        for widget, value in (
            (self.context.yaw, round(new_yaw, 1)),
            (self.context.pitch, round(new_pitch, 1)),
        ):
            if widget.value != value:
                self.context.ignored_physics_events[id(widget)] = (
                    self.context.ignored_physics_events.get(id(widget), 0) + 1
                )
                widget.value = value
        await self.context.update_physics(
            fidelity="preview",
            preserve_scene_frame=True,
            geometry_already_previewed=True,
        )

    async def on_angle_range_change(self, _event):
        self.context.angle_state["revision"] += 1
        revision = self.context.angle_state["revision"]
        selected_bins = tuple(int(value) for value in self.context.angle_range_bins.value)
        products = self.context.angle_state["products"]
        self.context.angle_map.loading = True
        try:
            figure = await _app.asyncio.to_thread(
                _app._angle_map_figure,
                products,
                self.context.make_subplots,
                self.context.go,
                range_bins=selected_bins,
                plot_template=self.context.plot_template,
            )
            if revision == self.context.angle_state["revision"]:
                self.context.angle_map.object = figure
                self.context.update_angle_status(products, selected_bins)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if revision == self.context.angle_state["revision"]:
                self.context.pn.state.notifications.error(str(exc), duration=8000)
        finally:
            if revision == self.context.angle_state["revision"]:
                self.context.angle_map.loading = False

    async def on_target_type_change(self, _event):
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
        await self.context.update_physics(fidelity="preview")
