# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""HumanRoom controls and callbacks for one GUI session."""

from __future__ import annotations

from .. import app as _app
from .base import Workflow
from .contexts import HumanRoomContext


class HumanRoomWorkflow(Workflow[HumanRoomContext]):
    def show_human_room_playing_state(self, playing: bool):
        self.context.human_room_play_pause.label = "Pause" if playing else "Play"
        self.context.human_room_play_pause.description = self.context.human_room_play_pause.label
        self.context.human_room_play_pause.icon = "player-pause" if playing else "player-play"

    def configure_human_room_frame_controls(
        self,
        *,
        last_frame: int,
        selected_frame: int,
        disabled: bool,
        interval_ms: int,
        normal_step: int,
        fast_step: int,
        selected_time_s: float | None = None,
    ):
        self.context.human_room_playback_state["syncing"] = True
        try:
            preview = self.context.human_room_state["preview"]
            times = _app.np.asarray([0.0] if preview is None else preview.mesh_sequence.times)
            duration_s = float(times[-1])
            if selected_time_s is None:
                selected_time_s = float(times[selected_frame])
            selected_time_s = float(_app.np.clip(selected_time_s, 0.0, duration_s))
            self.context.human_room_preview_time.end = duration_s
            self.context.human_room_preview_time.label = f"Preview [s] (max {duration_s:.3f})"
            self.context.human_room_preview_time.value = selected_time_s
            self.context.human_room_frame_counter.value = self.context.human_room_pose_counter(
                selected_time_s
            )
            self.context.human_room_preview_time.disabled = bool(disabled)
            self.context.human_room_player.direction = 0
            self.context.human_room_player.end = int(last_frame)
            self.context.human_room_player.value = int(selected_frame)
            self.context.human_room_player.step = max(int(normal_step), 1)
            self.context.human_room_player.interval = int(interval_ms)
            self.context.human_room_player.disabled = bool(disabled)
            for button in self.context.human_room_transport_buttons:
                button.disabled = bool(disabled)
            self.context.human_room_playback_state["normal_step"] = max(
                int(normal_step),
                1,
            )
            self.context.human_room_playback_state["fast_step"] = max(int(fast_step), 1)
            self.context.show_human_room_playing_state(False)
        finally:
            self.context.human_room_playback_state["syncing"] = False

    def human_room_pose_counter(self, time_s: float) -> str:
        preview = self.context.human_room_state["preview"]
        if preview is None:
            return "t = — s"
        times = preview.mesh_sequence.times
        return f"t = {float(time_s):.3f} / {float(times[-1]):.3f} s"

    def human_room_window_frame_count(
        self,
    ) -> int:
        tx_count = max(len(self.context.tx_antennas.value), 1)
        slow_time_s = (
            float(self.context.chirp_repetition.value)
            * 1e-6
            * (tx_count if self.context.tx_schedule.value == "tdm" else 1)
        )
        return _app.radar_frame_count(
            float(self.context.human_room_start_time.value),
            float(self.context.human_room_end_time.value),
            frame_period_s=float(self.context.frame_period.value) * 1e-3,
            frame_duration_s=int(self.context.num_chirps.value) * slow_time_s,
        )

    def update_human_room_window_summary(self, *_events):
        count = self.context.human_room_window_frame_count()
        if count <= 0:
            self.context.human_room_window_summary.object = (
                "**No radar frame fits.** Make the window at least one frame's chirps long."
            )
            return
        self.context.human_room_window_summary.object = (
            f"**{count}** radar frame{'s' if count != 1 else ''} · "
            f"{float(self.context.frame_period.value):g} ms apart · real-time motion"
        )

    def configure_human_room_window_controls(
        self,
        *,
        duration_s: float,
        reset_window: bool,
        disabled: bool,
    ):
        self.context.human_room_window_state["syncing"] = True
        try:
            frame_period_s = float(self.context.frame_period.value) * 1e-3
            limit = _app._human_room_window_limit(duration_s, frame_period_s)
            previous_start = float(self.context.human_room_start_time.value)
            previous_end = float(self.context.human_room_end_time.value)
            self.context.human_room_start_time.end = limit
            self.context.human_room_end_time.end = limit
            self.context.human_room_end_time.label = f"End [s] (max {limit:.3f})"
            start_s, end_s = _app._human_room_window_values(
                duration_s=duration_s,
                frame_period_s=frame_period_s,
                previous_start=previous_start,
                previous_end=previous_end,
                reset_window=reset_window,
            )
            self.context.human_room_start_time.value = start_s
            self.context.human_room_end_time.value = end_s
            self.context.human_room_start_time.disabled = bool(disabled)
            self.context.human_room_end_time.disabled = bool(disabled)
        finally:
            self.context.human_room_window_state["syncing"] = False
        self.context.update_human_room_window_summary()

    def current_human_room_source(
        self,
    ):
        if self.context.human_room_upload.value:
            payload = _app._validated_upload_payload(
                self.context.human_room_upload.value,
                label="human motion",
                max_bytes=_app._MAX_MOTION_UPLOAD_BYTES,
            )
            _app._preflight_gui_motion_npz(payload)
            return payload, self.context.human_room_upload.filename
        model_dir = _app._effective_smpl_model_dir(
            self.context.human_room_smpl_model_dir.value,
            self.context.default_model_dir,
            remote_access=self.context.remote_access,
        )
        if (
            self.context.default_amass_motion is not None
            and _app.Path(model_dir).expanduser().is_dir()
        ):
            return self.context.default_amass_motion, self.context.default_amass_motion.name
        return None, None

    def on_human_room_upload_filename(self, event):
        self.context.human_room_upload_filename.value = (
            str(event.new)
            if event.new
            else (
                f"Bundled default: {self.context.default_amass_motion.name}"
                if self.context.default_amass_motion is not None
                else "No file selected"
            )
        )

    def current_human_room_scene(
        self,
    ):
        if self.context.human_room_scene_upload.value and not self.context.remote_access:
            return (
                _app._validated_upload_payload(
                    self.context.human_room_scene_upload.value,
                    label="scene XML",
                    max_bytes=_app._MAX_SCENE_XML_UPLOAD_BYTES,
                ),
                self.context.human_room_scene_upload.filename,
            )
        return None, None

    def current_room_boxes_override(
        self,
    ):
        scene_source, _scene_filename = self.context.current_human_room_scene()
        return (
            _app._scene_xml_boxes(scene_source)
            if scene_source is not None
            else _app.prepared_room_boxes(self.context.human_room_preset.value)
        )

    def sync_human_position_bounds(
        self,
    ):
        limits = _app._human_position_bounds(self.context.current_room_boxes_override())
        self.context.human_room_position_state["syncing"] = True
        try:
            for widget, (lower, upper) in zip(
                (self.context.human_room_x, self.context.human_room_y, self.context.human_room_z),
                limits,
            ):
                value = float(widget.value)
                if lower is not None:
                    value = max(value, lower)
                if upper is not None:
                    value = min(value, upper)
                widget.param.update(start=lower, end=upper, value=value)
        finally:
            self.context.human_room_position_state["syncing"] = False

    def current_motion_cache_key(self, source, filename):
        if isinstance(source, (bytes, bytearray)):
            source_identity = _app.hashlib.sha256(bytes(source)).hexdigest()
        else:
            source_path = _app.Path(source)
            source_identity = (
                str(source_path.resolve()),
                source_path.stat().st_mtime_ns,
                source_path.stat().st_size,
            )
        return (
            filename,
            source_identity,
            str(
                _app.Path(
                    _app._effective_smpl_model_dir(
                        self.context.human_room_smpl_model_dir.value,
                        self.context.default_model_dir,
                        remote_access=self.context.remote_access,
                    )
                )
                .expanduser()
                .resolve()
            ),
        )

    def current_human_position(
        self,
    ):
        if self.context.human_room_position_mode.value == "file":
            return None
        return (
            float(self.context.human_room_x.value),
            float(self.context.human_room_y.value),
            float(self.context.human_room_z.value),
        )

    def current_dynamic_radar_orientation(
        self,
    ):
        return (
            float(self.context.dynamic_radar_yaw.value),
            float(self.context.dynamic_radar_pitch.value),
            float(self.context.dynamic_radar_roll.value),
        )

    def mark_human_room_stale(self, message: str):
        if (
            self.context.gui_session_state["restoring"]
            or self.context.gui_session_state["restoring_uploads"]
        ):
            return
        if self.context.settings_match_result("dynamic"):
            self.context.refresh_result_validity()
            return
        self.context.human_room_state["result"] = None
        self.context.human_room_state["results"] = {}
        self.context.human_room_state["diagnostics"] = None
        self.context.human_room_state["diagnostic_revision"] += 1
        diagnostic_cancel = self.context.human_room_state.get("diagnostic_cancel_event")
        if diagnostic_cancel is not None:
            diagnostic_cancel.set()
        self.context.human_room_state["diagnostic_cancel_event"] = None
        self.context.gui_session_state["dynamic_cacheable"] = False
        self.context.human_room_download.disabled = True
        self.context.run_human_room_diagnostics_button.disabled = True
        self.context.human_room_diagnostics_status.object = (
            "Run a simulation to enable one-frame solver overlays."
        )
        if self.context.result_records["dynamic"] is None:
            self.context.human_room_results.objects = [self.context.pn.pane.Markdown(message)]
        self.context.persist_gui_session()

    async def refresh_human_room_preview(self, *, mark_stale: bool = True):
        if mark_stale and self.context.gui_session_state["restoring"]:
            return
        self.context.sync_human_position_bounds()
        self.context.human_room_state["preview_revision"] += 1
        preview_revision = self.context.human_room_state["preview_revision"]
        source, filename = self.context.current_human_room_source()
        scene_source, scene_filename = self.context.current_human_room_scene()
        self.context.human_room_preset.disabled = scene_source is not None
        if source is None:
            if scene_source is not None:
                await _app.asyncio.to_thread(
                    _app.load_scene_xml,
                    scene_source,
                    filename=scene_filename,
                )
                if preview_revision != self.context.human_room_state["preview_revision"]:
                    return
            self.context.human_room_state["preview"] = None
            self.context.human_room_state["loaded_motion"] = None
            self.context.human_room_state["motion_cache_key"] = None
            self.context.run_human_room.disabled = True
            self.context.configure_human_room_frame_controls(
                last_frame=0,
                selected_frame=0,
                disabled=True,
                interval_ms=100,
                normal_step=1,
                fast_step=1,
            )
            self.context.configure_human_room_window_controls(
                duration_s=0.0,
                reset_window=True,
                disabled=True,
            )
            self.context.human_room_scene.object = _app._human_room_scene_figure(
                None,
                self.context.go,
                frame_index=0,
                radar_orientation_deg=self.context.current_dynamic_radar_orientation(),
                room_boxes_override=self.context.current_room_boxes_override(),
                plot_template=self.context.plot_template,
            )
            if mark_stale:
                self.context.mark_human_room_stale(
                    "Motion input is unavailable; load a valid sequence and "
                    "run the simulation again."
                )
            self.context.human_room_status.object = (
                "**Human motion required.** Upload a pickle-free `.npz`. "
                "Pose-only archives also need a licensed SMPL model directory."
            )
            return
        cache_key = self.context.current_motion_cache_key(source, filename)
        motion_changed = cache_key != self.context.human_room_state["motion_cache_key"]
        if motion_changed:
            self.context.human_room_status.object = (
                "Loading human motion from " f"{_app._safe_markdown_code(filename)}..."
            )
            loaded_motion = await _app.asyncio.to_thread(
                _app.load_amass_motion,
                source,
                filename=filename,
                smpl_model_dir=_app._effective_smpl_model_dir(
                    self.context.human_room_smpl_model_dir.value,
                    self.context.default_model_dir,
                    remote_access=self.context.remote_access,
                ),
            )
            if preview_revision != self.context.human_room_state["preview_revision"]:
                return
            self.context.human_room_state["loaded_motion"] = loaded_motion
            self.context.human_room_state["motion_cache_key"] = cache_key
        preview = await _app.asyncio.to_thread(
            _app.prepare_human_room_preview,
            self.context.human_room_state["loaded_motion"],
            room_preset=self.context.human_room_preset.value,
            scene_xml=scene_source,
            scene_xml_filename=scene_filename,
            human_position_m=self.context.current_human_position(),
            human_yaw_deg=float(self.context.human_room_yaw.value),
            **self.context.current_motion_coordinate_arguments(),
        )
        if preview_revision != self.context.human_room_state["preview_revision"]:
            return
        self.context.human_room_state["preview"] = preview
        self.context.run_human_room.disabled = False
        frame_count = len(preview.mesh_sequence.times)
        last_frame = max(frame_count - 1, 0)
        selected_time_s = float(
            _app.np.clip(
                self.context.human_room_preview_time.value,
                0.0,
                preview.mesh_sequence.times[-1],
            )
        )
        selected_frame = _app._nearest_pose_index(preview.mesh_sequence.times, selected_time_s)
        self.context.configure_human_room_frame_controls(
            last_frame=last_frame,
            selected_frame=selected_frame,
            disabled=frame_count <= 1,
            interval_ms=_app._motion_playback_interval_ms(preview.mesh_sequence.times),
            normal_step=_app._motion_playback_step(preview.mesh_sequence.times),
            fast_step=_app._motion_fast_playback_step(preview.mesh_sequence.times),
            selected_time_s=selected_time_s,
        )
        self.context.configure_human_room_window_controls(
            duration_s=float(preview.mesh_sequence.times[-1]),
            reset_window=motion_changed,
            disabled=False,
        )
        self.context.human_room_scene.object = _app._human_room_scene_figure(
            preview,
            self.context.go,
            frame_index=selected_frame,
            motion_time_s=selected_time_s,
            radar_orientation_deg=self.context.current_dynamic_radar_orientation(),
            room_boxes_override=self.context.current_room_boxes_override(),
            plot_template=self.context.plot_template,
        )
        if mark_stale:
            self.context.mark_human_room_stale(
                "Scene, motion, or radar configuration changed; run the simulation again."
            )
        if self.context.human_room_state["result"] is None:
            self.context.human_room_status.object = (
                f"**READY TO RUN** · "
                f"{_app._safe_markdown_code(preview.source_name)} · "
                f"{frame_count:,} motion frame(s) · "
                f"{preview.mesh_sequence.vertex_count:,} vertices · "
                f"{preview.mesh_sequence.face_count:,} facets · scene "
                f"{_app._safe_markdown_code(preview.scene_source_name)}"
            )
        self.context.refresh_result_validity()

    def update_human_room_floor_status(self, _event=None):
        self.context.human_room_floor_status.object = next(
            (
                trace.meta["floor_status"]
                for trace in self.context.human_room_scene.object.data
                if trace.name == "Below-floor vertices"
            ),
            "Load human motion to check floor placement.",
        )

    async def on_human_room_controls(self, _event):
        expected_revision = self.context.human_room_state["preview_revision"] + 1
        try:
            await self.context.refresh_human_room_preview()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if expected_revision != self.context.human_room_state["preview_revision"]:
                return
            self.context.human_room_state["preview"] = None
            self.context.human_room_state["loaded_motion"] = None
            self.context.human_room_state["motion_cache_key"] = None
            self.context.mark_human_room_stale(
                "Dynamic inputs could not be loaded; correct them and run the simulation again."
            )
            self.context.human_room_status.object = (
                f"**Dynamic input load failed:** "
                f"{_app._safe_markdown_code(f'{type(exc).__name__}: {exc}')}"
            )
            self.context.pn.state.notifications.error(str(exc), duration=8000)

    def on_human_position_input(self, event):
        if self.context.human_room_position_state["syncing"]:
            return
        self.context.pn.state.execute(_app.partial(self.context.on_human_room_controls, event))

    async def on_motion_coordinate_change(self, event):
        if self.context.motion_coordinate_state["syncing"]:
            return
        self.context.sync_motion_coordinate_controls()
        if (
            self.context.gui_session_state["restoring"]
            or self.context.settings_dialog_state["editing"]
        ):
            return
        await self.context.on_human_room_controls(event)

    def diagnostics_for_time(self, time_s: float):
        diagnostics = self.context.human_room_state.get("diagnostics")
        if diagnostics is not None and _app.np.isclose(
            diagnostics.motion_time_s,
            time_s,
            rtol=0.0,
            atol=1e-9,
        ):
            return diagnostics
        return None

    def display_human_room_frame(self, frame_index: int, *, motion_time_s: float | None = None):
        preview = self.context.human_room_state["preview"]
        if motion_time_s is None:
            motion_time_s = (
                0.0 if preview is None else float(preview.mesh_sequence.times[frame_index])
            )
        self.context.human_room_frame_counter.value = self.context.human_room_pose_counter(
            motion_time_s
        )
        frame_diagnostics = self.context.diagnostics_for_time(motion_time_s)
        updated_in_place = frame_diagnostics is None and (
            _app._update_human_room_scene_frame(
                self.context.human_room_scene.object,
                preview,
                frame_index=int(frame_index),
                update_title=not bool(self.context.human_room_player.direction),
                motion_time_s=motion_time_s,
            )
        )
        if not updated_in_place:
            self.context.human_room_scene.object = _app._human_room_scene_figure(
                preview,
                self.context.go,
                frame_index=int(frame_index),
                motion_time_s=motion_time_s,
                radar_orientation_deg=self.context.current_dynamic_radar_orientation(),
                room_boxes_override=self.context.current_room_boxes_override(),
                diagnostics=frame_diagnostics,
                plot_template=self.context.plot_template,
            )
        self.context.update_human_room_floor_status()
        diagnostics = self.context.human_room_state.get("diagnostics")
        if diagnostics is not None and frame_diagnostics is None:
            self.context.human_room_diagnostics_status.object = (
                f"Solver overlays are retained for radar frame "
                f"{int(diagnostics.frame_index)} at t = "
                f"{float(diagnostics.motion_time_s):.3f} s; return to that "
                "time or recompute them for the displayed pose."
            )

    def on_human_room_preview_time(self, event):
        if self.context.human_room_playback_state["syncing"]:
            return
        preview = self.context.human_room_state["preview"]
        if preview is None:
            return
        time_s = float(
            _app.np.clip(
                _app.np.nan_to_num(
                    float(event.new or 0.0),
                    nan=0.0,
                    posinf=self.context.human_room_preview_time.end,
                    neginf=0.0,
                ),
                self.context.human_room_preview_time.start,
                self.context.human_room_preview_time.end,
            )
        )
        frame_index = _app._nearest_pose_index(preview.mesh_sequence.times, time_s)
        self.context.human_room_playback_state["syncing"] = True
        try:
            self.context.human_room_preview_time.value = time_s
            self.context.human_room_player.direction = 0
            self.context.human_room_player.step = int(
                self.context.human_room_playback_state["normal_step"]
            )
            self.context.human_room_player.value = frame_index
            self.context.show_human_room_playing_state(False)
        finally:
            self.context.human_room_playback_state["syncing"] = False
        self.context.display_human_room_frame(frame_index, motion_time_s=time_s)

    def on_human_room_player(self, event):
        if self.context.human_room_playback_state["syncing"]:
            return
        preview = self.context.human_room_state["preview"]
        if preview is None:
            return
        frame_index = int(event.new)
        self.context.human_room_playback_state["syncing"] = True
        try:
            self.context.human_room_preview_time.value = float(
                preview.mesh_sequence.times[frame_index]
            )
        finally:
            self.context.human_room_playback_state["syncing"] = False
        self.context.display_human_room_frame(frame_index)

    def on_human_room_player_direction(self, _event):
        if self.context.human_room_playback_state["syncing"]:
            return
        self.context.human_room_playback_state["syncing"] = True
        try:
            self.context.human_room_player.step = int(
                self.context.human_room_playback_state["normal_step"]
            )
            self.context.show_human_room_playing_state(
                bool(self.context.human_room_player.direction)
            )
        finally:
            self.context.human_room_playback_state["syncing"] = False

    def start_fast_human_room_playback(self, direction: int):
        if self.context.human_room_player.disabled:
            return
        self.context.human_room_playback_state["syncing"] = True
        try:
            self.context.human_room_player.step = int(
                self.context.human_room_playback_state["fast_step"]
            )
            self.context.human_room_player.direction = int(_app.np.sign(direction))
            self.context.show_human_room_playing_state(True)
        finally:
            self.context.human_room_playback_state["syncing"] = False

    def start_human_room_playback(self, direction: int):
        if self.context.human_room_player.disabled:
            return
        self.context.human_room_playback_state["syncing"] = True
        try:
            self.context.human_room_player.step = int(
                self.context.human_room_playback_state["normal_step"]
            )
            self.context.human_room_player.direction = int(_app.np.sign(direction))
            self.context.show_human_room_playing_state(True)
        finally:
            self.context.human_room_playback_state["syncing"] = False

    def pause_human_room_playback(
        self,
    ):
        self.context.human_room_playback_state["syncing"] = True
        try:
            self.context.human_room_player.direction = 0
            self.context.human_room_player.step = int(
                self.context.human_room_playback_state["normal_step"]
            )
            self.context.show_human_room_playing_state(False)
        finally:
            self.context.human_room_playback_state["syncing"] = False
        preview = self.context.human_room_state.get("preview")
        figure = self.context.human_room_scene.object
        if preview is not None and figure is not None:
            figure.plotly_relayout(
                {
                    "title.text": _app._human_room_scene_title(
                        preview,
                        int(self.context.human_room_player.value),
                        motion_time_s=float(self.context.human_room_preview_time.value),
                    )
                }
            )

    def step_human_room_frame(self, offset: int):
        if self.context.human_room_player.disabled:
            return
        self.context.pause_human_room_playback()
        next_frame = int(
            _app.np.clip(
                int(self.context.human_room_player.value) + int(offset),
                int(self.context.human_room_player.start),
                int(self.context.human_room_player.end),
            )
        )
        self.context.human_room_player.value = next_frame

    def on_human_room_previous_frame(self, _event):
        self.context.step_human_room_frame(-1)

    def on_human_room_reverse(self, _event):
        self.context.start_human_room_playback(-1)

    def on_human_room_fast_reverse(self, _event):
        self.context.start_fast_human_room_playback(-1)

    def on_human_room_fast_forward(self, _event):
        self.context.start_fast_human_room_playback(1)

    def on_human_room_play_pause(self, _event):
        if self.context.human_room_player.direction:
            self.context.pause_human_room_playback()
        else:
            self.context.start_human_room_playback(1)

    def on_human_room_next_frame(self, _event):
        self.context.step_human_room_frame(1)

    def on_dynamic_radar_preview_change(self, _event):
        if (
            self.context.gui_session_state["restoring"]
            or self.context.settings_dialog_state["editing"]
        ):
            return
        preview = self.context.human_room_state["preview"]
        self.context.human_room_scene.object = _app._human_room_scene_figure(
            preview,
            self.context.go,
            frame_index=int(self.context.human_room_player.value),
            motion_time_s=float(self.context.human_room_preview_time.value),
            radar_orientation_deg=self.context.current_dynamic_radar_orientation(),
            room_boxes_override=self.context.current_room_boxes_override(),
            plot_template=self.context.plot_template,
        )
        self.context.mark_human_room_stale(
            "Radar configuration changed; start the simulation again."
        )

    def on_human_run_option_change(self, _event):
        self.context.mark_human_room_stale(
            "Simulation options changed; start the simulation again."
        )

    def on_human_room_window_change(self, event):
        if self.context.human_room_window_state["syncing"]:
            return
        self.context.human_room_window_state["syncing"] = True
        try:
            start_s = float(self.context.human_room_start_time.value)
            end_s = float(self.context.human_room_end_time.value)
            frame_period_s = float(self.context.frame_period.value) * 1e-3
            if start_s > end_s:
                if event.obj is self.context.human_room_start_time:
                    self.context.human_room_end_time.value = min(
                        start_s + frame_period_s,
                        float(self.context.human_room_end_time.end),
                    )
                else:
                    self.context.human_room_start_time.value = max(
                        end_s - frame_period_s,
                        0.0,
                    )
        finally:
            self.context.human_room_window_state["syncing"] = False
        self.context.update_human_room_window_summary()
        self.context.on_human_run_option_change(event)

    def _run_arguments(
        self, filename, scene_source, scene_filename, cancel_event, publish_progress
    ):
        """Collect the motion, scene, radar and solver controls for a run."""
        return dict(
            human_motion=self.context.human_room_state["loaded_motion"],
            human_motion_filename=filename,
            smpl_model_dir=_app._effective_smpl_model_dir(
                self.context.human_room_smpl_model_dir.value,
                self.context.default_model_dir,
                remote_access=self.context.remote_access,
            ),
            room_preset=self.context.human_room_preset.value,
            scene_xml=scene_source,
            scene_xml_filename=scene_filename,
            human_position_m=self.context.current_human_position(),
            human_yaw_deg=float(self.context.human_room_yaw.value),
            **self.context.current_motion_coordinate_arguments(),
            radar_orientation_deg=self.context.current_dynamic_radar_orientation(),
            board_model=self.context.board.value,
            carrier_frequency_hz=float(self.context.carrier_frequency.value) * 1e9,
            slope_hz_per_s=float(self.context.chirp_slope.value) * 1e12,
            chirp_duration_s=float(self.context.chirp_duration.value) * 1e-6,
            chirp_repetition_time_s=float(self.context.chirp_repetition.value) * 1e-6,
            sampling_frequency_hz=float(self.context.sampling_frequency.value) * 1e6,
            adc_mode=self.context.adc_sampling_mode.value,
            if_bandwidth_hz=self.context.adc_if_bandwidth.value * 1e6 or None,
            num_adc_samples=int(self.context.num_adc_samples.value),
            num_chirps_per_frame=int(self.context.num_chirps.value),
            frame_period_s=float(self.context.frame_period.value) * 1e-3,
            tx_schedule=str(self.context.tx_schedule.value),
            selected_tx_indices=tuple(self.context.tx_antennas.value),
            selected_rx_indices=tuple(self.context.rx_antennas.value),
            antenna_pattern_mode=self.context.antenna_pattern.value,
            cosine_3db_beamwidth_deg=float(self.context.cosine_3db_beamwidth.value),
            rt_human_diffuse_coefficient=float(self.context.human_room_diffuse.value),
            motion_start_s=float(self.context.human_room_start_time.value),
            motion_end_s=float(self.context.human_room_end_time.value),
            array_model=str(self.context.array_model.value),
            **_app._array_group_arguments(
                str(self.context.array_model.value),
                self.context.array_tx_groups.value,
                self.context.array_rx_groups.value,
            ),
            hybrid_coupling=bool(self.context.hybrid_coupling.value),
            hybrid_reflector_cutoff_db=float(self.context.hybrid_reflector_cutoff.value),
            hybrid_max_reflectors=int(self.context.hybrid_max_reflectors.value),
            hybrid_calibration=str(self.context.hybrid_calibration.value),
            rt_samples_per_source=int(self.context.rt_samples.value),
            rt_max_paths_per_source=int(self.context.rt_path_cap.value),
            rt_max_depth=int(self.context.rt_depth.value),
            rt_human_bounces=str(self.context.rt_human_bounces.value),
            rt_refresh_frames=int(self.context.rt_refresh.value),
            rt_visibility_refresh_frames=int(self.context.rt_visibility_refresh.value),
            rt_visibility_fade_frames=int(self.context.rt_visibility_fade.value),
            po_refresh_frames=int(self.context.po_refresh.value),
            po_refresh_on_phase_error_rad=float(self.context.po_phase_error.value),
            po_refresh_on_area_fraction=float(self.context.po_area_fraction.value),
            po_refresh_on_normal_deg=float(self.context.po_normal_deg.value),
            po_visibility_refresh_frames=int(self.context.po_visibility_refresh.value),
            po_visibility_fade_frames=int(self.context.po_visibility_fade.value),
            po_visibility_samples_per_face=int(self.context.po_visibility_samples.value),
            po_adaptive_visibility_sampling=bool(self.context.po_adaptive_visibility.value),
            po_adaptive_visibility_edge_margin=float(self.context.po_visibility_edge_margin.value),
            po_integration_mode=self.context.po_integration_mode.value,
            po_quadrature_phase_span_scale_rad=float(self.context.po_phase_span.value),
            po_curvature_error_rad=float(self.context.po_curvature.value),
            po_quadrature_max_refinement_depth=int(self.context.po_refinement_depth.value),
            po_quadrature_max_subfaces_per_parent=int(self.context.po_subface_cap.value),
            compute_backend="auto",
            cancel_check=cancel_event.is_set,
            progress_callback=publish_progress,
        )

    async def on_run_human_room(self, _event):
        if self.context.human_room_state["running"] or self.context.gui_activity_state["dynamic"]:
            return
        source, filename = self.context.current_human_room_source()
        if source is None:
            self.context.human_room_status.object = (
                "**Human motion required.** Upload an `.npz` first."
            )
            return
        selected_solvers = tuple(
            solver
            for solver in _app.SOLVERS
            if self.context.human_room_solver_checkboxes[solver].value
        )
        if not selected_solvers:
            self.context.human_room_status.object = (
                "**Solver required.** Select at least one solver."
            )
            return
        if self.context.human_room_window_frame_count() <= 0:
            self.context.human_room_status.object = (
                "**Motion window too short.** Make the window from Start to "
                "End at least one radar frame's chirps long."
            )
            return
        scene_source, scene_filename = self.context.current_human_room_scene()
        run_settings = self.context.capture_result_settings("dynamic")
        self.context.human_room_state["revision"] += 1
        revision = self.context.human_room_state["revision"]
        self.context.mark_human_room_stale(
            "Simulation is in progress; completed products will appear here."
        )
        cancel_event = _app.threading.Event()
        self.context.human_room_state["cancel_event"] = cancel_event
        self.context.human_room_state["running"] = True
        self.context.human_room_state["diagnostics"] = None
        self.context.run_human_room_diagnostics_button.disabled = True
        self.context.human_room_diagnostics_status.object = (
            "Solver overlays will be available after the simulation finishes."
        )
        self.context.run_human_room.disabled = True
        self.context.stop_human_room.disabled = False
        self.context.stop_human_room.color = "danger"
        self.context.set_gui_activity("dynamic", True)
        self.context.human_room_scene.loading = True
        self.context.human_room_results.loading = True
        completed_results = {}
        progress_events = _app.queue.SimpleQueue()
        selected_frame_count = self.context.human_room_window_frame_count()

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
                except _app.queue.Empty:
                    break
            if latest is None:
                return
            status = _app._human_room_running_status(latest)
            if self.context.human_room_status.object != status:
                self.context.human_room_status.object = status

        self.context.human_room_status.object = (
            "Running "
            + ", ".join(_app._SOLVER_LABELS[solver] for solver in selected_solvers)
            + "..."
        )
        run_input_states = _app._lock_widget_disabled_states(
            self.context.human_room_run_input_widgets
        )
        try:
            cache_key = self.context.current_motion_cache_key(source, filename)
            if cache_key != self.context.human_room_state["motion_cache_key"]:
                self.context.human_room_state["loaded_motion"] = await _app.asyncio.to_thread(
                    _app.load_amass_motion,
                    source,
                    filename=filename,
                    smpl_model_dir=_app._effective_smpl_model_dir(
                        self.context.human_room_smpl_model_dir.value,
                        self.context.default_model_dir,
                        remote_access=self.context.remote_access,
                    ),
                )
                self.context.human_room_state["motion_cache_key"] = cache_key
            run_arguments = self._run_arguments(
                filename,
                scene_source,
                scene_filename,
                cancel_event,
                publish_progress,
            )

            def run_selected_solvers():
                for solver in selected_solvers:
                    if cancel_event.is_set():
                        raise InterruptedError("Simulation cancelled by user")
                    publish_progress(
                        {
                            "solver": solver,
                            "frame_index": 0,
                            "num_frames": selected_frame_count,
                        }
                    )
                    completed_results[solver] = _app.run_human_room_experiment(
                        solver=solver,
                        **run_arguments,
                    )
                return completed_results

            worker_task = _app.asyncio.create_task(_app.asyncio.to_thread(run_selected_solvers))
            while not worker_task.done():
                refresh_progress_status()
                await _app.asyncio.sleep(0.15)
            refresh_progress_status()
            results = await worker_task
            if revision == self.context.human_room_state["revision"]:
                primary_solver = "hybrid_po" if "hybrid_po" in results else selected_solvers[0]
                result = results[primary_solver]
                self.context.human_room_state["result"] = result
                self.context.human_room_state["results"] = results
                self.context.gui_session_state["dynamic_cacheable"] = True
                self.context.human_room_results.objects = [
                    _app._human_room_results_view(
                        results,
                        self.context.pn,
                        self.context.go,
                        plot_template=self.context.plot_template,
                    )
                ]
                runtime_summary = " · ".join(
                    f"{_app._SOLVER_LABELS[solver]} " f"{results[solver].runtime_s:.2f} s"
                    for solver in selected_solvers
                )
                self.context.human_room_status.object = (
                    f"**SIMULATION READY** · {len(results)} solver(s) · "
                    f"ADC `{tuple(result.adc.shape)}` · {runtime_summary}"
                )
                self.context.remember_result_settings("dynamic", run_settings)
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
                self.context.human_room_diagnostics_status.object = (
                    "Choose a preview time within the simulated "
                    "window, then show solver overlays."
                )
                self.context.persist_gui_session()
        except InterruptedError:
            if revision == self.context.human_room_state["revision"]:
                if completed_results:
                    primary_solver = (
                        "hybrid_po"
                        if "hybrid_po" in completed_results
                        else next(iter(completed_results))
                    )
                    self.context.human_room_state["result"] = completed_results[primary_solver]
                    self.context.human_room_state["results"] = dict(completed_results)
                    self.context.gui_session_state["dynamic_cacheable"] = True
                    self.context.human_room_results.objects = [
                        _app._human_room_results_view(
                            completed_results,
                            self.context.pn,
                            self.context.go,
                            plot_template=self.context.plot_template,
                        )
                    ]
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
                    self.context.human_room_diagnostics_status.object = (
                        "Choose a preview time within a retained "
                        "solver's simulated window, then show solver overlays."
                    )
                    retained = (
                        f" {len(completed_results)} completed mode result(s) " "were retained."
                    )
                else:
                    retained = ""
                    if self.context.result_records["dynamic"] is None:
                        self.context.human_room_results.objects = [
                            self.context.pn.pane.Markdown(
                                "The simulation was stopped before any mode "
                                "completed. Start again to generate results."
                            )
                        ]
                self.context.human_room_status.object = (
                    "**STOPPED** · The current simulation was terminated at "
                    f"the next safe solver boundary.{retained}"
                )
                if completed_results:
                    self.context.remember_result_settings("dynamic", run_settings)
                self.context.persist_gui_session()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if revision == self.context.human_room_state["revision"]:
                self.context.human_room_status.object = (
                    "**Simulation failed:** "
                    f"{_app._safe_markdown_code(f'{type(exc).__name__}: {exc}')}"
                )
                self.context.pn.state.notifications.error(str(exc), duration=8000)
                if self.context.result_records["dynamic"] is None:
                    self.context.human_room_results.objects = [
                        self.context.pn.pane.Markdown(
                            "The simulation failed. Correct the settings and start again."
                        )
                    ]
        finally:
            _app._restore_widget_disabled_states(run_input_states)
            if revision == self.context.human_room_state["revision"]:
                self.context.run_human_room.disabled = (
                    self.context.human_room_state["loaded_motion"] is None
                )
                self.context.stop_human_room.disabled = True
                self.context.stop_human_room.color = "light"
                self.context.human_room_state["running"] = False
                self.context.set_gui_activity("dynamic", False)
                self.context.human_room_state["cancel_event"] = None
                self.context.human_room_scene.loading = False
                self.context.human_room_results.loading = False
                self.context.refresh_result_validity()

    def on_stop_human_room(self, _event):
        cancel_event = self.context.human_room_state.get("cancel_event")
        diagnostic_cancel = self.context.human_room_state.get("diagnostic_cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        if diagnostic_cancel is not None:
            diagnostic_cancel.set()
        if cancel_event is None and diagnostic_cancel is None:
            return
        self.context.stop_human_room.disabled = True
        self.context.human_room_status.object = (
            "**STOPPING** · Waiting for the current RT/PO kernel to reach a safe boundary."
        )

    async def on_run_human_room_diagnostics(self, _event):
        if self.context.human_room_state["running"] or self.context.gui_activity_state["dynamic"]:
            return
        results = self.context.human_room_state.get("results", {})
        if not results:
            self.context.human_room_diagnostics_status.object = (
                "Run a simulation before requesting solver overlays."
            )
            return
        source_frame = int(self.context.human_room_player.value)
        displayed_time_s = float(self.context.human_room_preview_time.value)
        representative = (
            results.get("hybrid_po") or results.get("rt") or next(iter(results.values()))
        )
        preview = self.context.human_room_state.get("preview")
        if preview is None:
            self.context.human_room_diagnostics_status.object = (
                "Load the motion again to choose a pose for solver overlays."
            )
            return
        radar_frame = _app._radar_frame_for_motion_time(
            representative,
            displayed_time_s,
        )
        if radar_frame is None:
            scene_parameters = representative.manifest.scene.parameters
            self.context.human_room_diagnostics_status.object = (
                f"Displayed time {displayed_time_s:.3f} s is outside the "
                "simulated window "
                f"{float(scene_parameters['motion_start_s']):.3f}–"
                f"{float(scene_parameters['motion_end_s']):.3f} s."
            )
            return
        radar_frame_time_s = _app._radar_frame_motion_time(
            representative,
            radar_frame,
        )
        include_rt = "rt" in results
        include_po = bool({"po", "hybrid_po"}.intersection(results))
        self.context.human_room_state["diagnostic_revision"] += 1
        diagnostic_revision = self.context.human_room_state["diagnostic_revision"]
        requested_top_k = int(self.context.human_room_diagnostic_top_k.value)
        result_identity = tuple((mode, id(result)) for mode, result in results.items())
        cancel_event = _app.threading.Event()
        self.context.human_room_state["diagnostic_cancel_event"] = cancel_event
        self.context.run_human_room_diagnostics_button.disabled = True
        self.context.human_room_diagnostic_top_k.disabled = True
        self.context.human_room_preview_time.disabled = True
        self.context.run_human_room.disabled = True
        self.context.stop_human_room.disabled = False
        self.context.set_gui_activity("dynamic", True)
        self.context.human_room_scene.loading = True
        included = " and ".join(
            label
            for enabled, label in (
                (include_rt, "RT rays"),
                (include_po, "PO face power"),
            )
            if enabled
        )
        self.context.human_room_diagnostics_status.object = (
            f"Computing {included} for radar frame {radar_frame} "
            f"(t = {radar_frame_time_s:.3f} s)..."
        )
        try:
            diagnostics = await _app.asyncio.to_thread(
                _app.run_human_room_diagnostics,
                representative,
                frame_index=radar_frame,
                top_k=requested_top_k,
                include_rt=include_rt,
                include_po=include_po,
                cancel_check=cancel_event.is_set,
            )
            current_identity = tuple(
                (mode, id(result))
                for mode, result in self.context.human_room_state.get("results", {}).items()
            )
            if (
                diagnostic_revision == self.context.human_room_state["diagnostic_revision"]
                and result_identity == current_identity
                and displayed_time_s == float(self.context.human_room_preview_time.value)
                and requested_top_k == int(self.context.human_room_diagnostic_top_k.value)
            ):
                self.context.human_room_state["diagnostics"] = diagnostics
                overlay_time_s = float(diagnostics.motion_time_s)
                if overlay_time_s != displayed_time_s:
                    # Show the interpolated pose the overlays were computed for.
                    self.context.human_room_preview_time.value = overlay_time_s
                else:
                    self.context.display_human_room_frame(
                        source_frame, motion_time_s=overlay_time_s
                    )
                summary = []
                if diagnostics.rt is not None:
                    summary.append(f"{diagnostics.rt.selected_path_indices.size} top RT " "ray(s)")
                if diagnostics.po is not None:
                    summary.append(f"{diagnostics.po.visible_face_count:,} visible PO " "face(s)")
                self.context.human_room_diagnostics_status.object = (
                    f"**OVERLAYS READY** · radar frame {radar_frame} · "
                    f"t = {radar_frame_time_s:.3f} s · " + " · ".join(summary)
                )
                self.context.persist_gui_session()
        except InterruptedError:
            if diagnostic_revision == self.context.human_room_state["diagnostic_revision"]:
                self.context.human_room_diagnostics_status.object = (
                    "**STOPPED** · Solver overlay computation was cancelled."
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if diagnostic_revision == self.context.human_room_state["diagnostic_revision"]:
                self.context.human_room_diagnostics_status.object = (
                    f"**Overlay computation failed:** "
                    f"{_app._safe_markdown_code(f'{type(exc).__name__}: {exc}')}"
                )
                self.context.pn.state.notifications.error(str(exc), duration=8000)
        finally:
            self.context.human_room_state["diagnostic_cancel_event"] = None
            self.context.run_human_room_diagnostics_button.disabled = not bool(
                self.context.human_room_state.get("results")
            )
            self.context.human_room_diagnostic_top_k.disabled = False
            current_preview = self.context.human_room_state.get("preview")
            self.context.human_room_preview_time.disabled = (
                current_preview is None or len(current_preview.mesh_sequence.times) <= 1
            )
            self.context.run_human_room.disabled = False
            self.context.stop_human_room.disabled = True
            self.context.set_gui_activity("dynamic", False)
            self.context.human_room_scene.loading = False
