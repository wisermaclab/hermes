# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Workflow behavior can be tested without constructing the GUI."""

from types import SimpleNamespace

import pytest

from mmWaveRadar.gui.workflows import ApplicationContext, SettingsWorkflow, SessionWorkflow


def test_context_rejects_undeclared_dependencies_during_assembly():
    with pytest.raises(AttributeError, match="motion_axis_mode"):
        ApplicationContext(motion_axis_mode=SimpleNamespace(value="file"))
    context = ApplicationContext()
    with pytest.raises(AttributeError, match="motion_axis_mode"):
        context.motion_axis_mode = SimpleNamespace(value="file")


def test_motion_coordinate_arguments_follow_file_or_manual_selection():
    context = ApplicationContext(
        motion_axes_mode=SimpleNamespace(value="file"),
        motion_up_axis=SimpleNamespace(value="+y"),
        motion_forward_axis=SimpleNamespace(value="-z"),
        motion_handedness=SimpleNamespace(value="left"),
        motion_length_unit=SimpleNamespace(value="mm"),
    )
    workflow = SettingsWorkflow(context)
    assert workflow.current_motion_coordinate_arguments() == {}
    context.motion_axes_mode.value = "manual"
    assert workflow.current_motion_coordinate_arguments() == {
        "motion_source_up_axis": "+y",
        "motion_source_forward_axis": "-z",
        "motion_source_handedness": "left",
        "motion_source_length_unit": "mm",
    }


def test_separate_workflow_instances_keep_session_state_isolated():
    first = ApplicationContext(motion_axes_mode=SimpleNamespace(value="file"))
    second = ApplicationContext(motion_axes_mode=SimpleNamespace(value="file"))
    SettingsWorkflow(first).context.motion_axes_mode.value = "manual"
    assert second.motion_axes_mode.value == "file"


def test_upload_encoding_round_trips_without_widgets_or_a_document():
    workflow = SessionWorkflow(ApplicationContext())
    payload = workflow.encoded_upload(
        SimpleNamespace(value=b"raw data", filename="motion.npz", label="Motion"),
        "fallback.npz",
        1024,
    )
    assert workflow.decoded_upload(payload, (".npz",), 1024) == ("motion.npz", b"raw data")
