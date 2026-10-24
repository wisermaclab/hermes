# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Application controls and callbacks for one GUI session."""

from __future__ import annotations

from .. import app as _app
from .base import Workflow
from .contexts import ApplicationWorkflowContext


class ApplicationWorkflow(Workflow[ApplicationWorkflowContext]):
    def set_gui_activity(self, name: str, active: bool) -> None:
        self.context.gui_activity_state[name] = bool(active)
        self.context.theme_button.disabled = any(self.context.gui_activity_state.values())
        self.context.settings_button.disabled = any(self.context.gui_activity_state.values())
        self.context.load_motion_preview.disabled = any(self.context.gui_activity_state.values())
        self.context.refresh_result_validity()

    def cancel_session_work(self, _session_context) -> None:
        self.context.physics_state["revision"] += 1
        self.context.physics_state["pending_request"] = None
        self.context.human_room_state["revision"] += 1
        self.context.human_room_state["diagnostic_revision"] += 1
        self.context.measurement_state["load_revision"] += 1
        self.context.measurement_state["comparison_revision"] += 1
        for cancel_event in (
            self.context.physics_state.get("cancel_event"),
            self.context.diagnostics_state.get("cancel_event"),
            self.context.human_room_state.get("cancel_event"),
            self.context.human_room_state.get("diagnostic_cancel_event"),
            self.context.measurement_state.get("cancel_event"),
        ):
            if cancel_event is not None:
                cancel_event.set()
        for task_name in ("bundle_restore_task", "dynamic_restore_task"):
            task = self.context.gui_session_state.get(task_name)
            if task is not None and not task.done():
                task.cancel()
        drop_workspace = self.context.measurement_state.get("drop_workspace")
        if drop_workspace is not None:
            drop_workspace.cleanup()
            self.context.measurement_state["drop_workspace"] = None
