# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mmWaveRadar.targets import MeshSequence
from mmWaveRadar.tutorial_support import (
    BEDROOM_SCENE_BOXES,
    bedroom_scene_xml,
    plot_bedroom_scene_projection,
    plot_mesh_projection,
    rigid_transform_mesh_sequence,
    time_window_mesh_sequence,
)


def test_prepared_bedroom_is_tight_but_clears_the_default_walk_and_furniture():
    boxes = {box["id"]: box for box in BEDROOM_SCENE_BOXES}
    floor = boxes["floor"]
    room_min = np.asarray(floor["translate"][:2]) - np.asarray(
        floor["scale"][:2]
    )
    room_max = np.asarray(floor["translate"][:2]) + np.asarray(
        floor["scale"][:2]
    )

    assert room_min == pytest.approx([-0.45, -3.6])
    assert room_max == pytest.approx([5.45, 5.1])
    assert np.prod(room_max - room_min) < 52.0
    motion_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "AMASS"
        / "walking_poses_cmu_105_02.npz"
    )
    with np.load(motion_path, allow_pickle=False) as archive:
        translation = np.asarray(archive["trans"], dtype=float)
    centerline = np.asarray([1.6, 0.0]) + (
        translation[:, :2] - translation[0, :2]
    )
    # A 0.5 m horizontal body envelope is conservative for the bundled SMPL
    # walk while keeping this test independent of separately licensed models.
    walk_min = np.min(centerline, axis=0) - 0.5
    walk_max = np.max(centerline, axis=0) + 0.5

    assert np.min(walk_min - room_min) >= 0.05
    assert np.min(room_max - walk_max) >= 0.05

    for furniture_id in ("bed", "nightstand", "wardrobe"):
        furniture = boxes[furniture_id]
        furniture_min = np.asarray(furniture["translate"][:2]) - np.asarray(
            furniture["scale"][:2]
        )
        furniture_max = np.asarray(furniture["translate"][:2]) + np.asarray(
            furniture["scale"][:2]
        )
        assert np.min(furniture_min - room_min) >= 0.05 - 1e-9
        assert np.min(room_max - furniture_max) >= 0.05 - 1e-9
        separation = max(
            furniture_min[0] - walk_max[0],
            walk_min[0] - furniture_max[0],
            furniture_min[1] - walk_max[1],
            walk_min[1] - furniture_max[1],
        )
        assert separation >= 0.35


def test_bedroom_xml_geometry_matches_preview_boxes():
    scene = ET.fromstring(bedroom_scene_xml())
    xml_shapes = {shape.attrib["id"]: shape for shape in scene.findall("./shape")}

    assert set(xml_shapes) == {box["id"] for box in BEDROOM_SCENE_BOXES}
    for box in BEDROOM_SCENE_BOXES:
        transform = xml_shapes[box["id"]].find("./transform[@name='to_world']")
        assert transform is not None
        scale = transform.find("./scale")
        translate = transform.find("./translate")
        assert scale is not None
        assert translate is not None
        assert tuple(float(scale.attrib[axis]) for axis in "xyz") == pytest.approx(
            box["scale"]
        )
        assert tuple(
            float(translate.attrib[axis]) for axis in "xyz"
        ) == pytest.approx(box["translate"])


def test_bedroom_right_wall_uses_the_metal_reflector():
    scene = ET.fromstring(bedroom_scene_xml())
    wall_material = scene.find("./bsdf[@id='mat-wall']")
    material = scene.find("./bsdf[@id='mat-reflective-wall']")
    right_wall = scene.find("./shape[@id='right_wall']")

    assert wall_material is not None
    assert float(
        wall_material.find("./float[@name='scattering_coefficient']").attrib[
            "value"
        ]
    ) == pytest.approx(0.2)
    assert material is not None
    assert material.find("./string[@name='type']").attrib["value"] == "metal"
    assert float(
        material.find("./float[@name='scattering_coefficient']").attrib["value"]
    ) == pytest.approx(0.2)
    assert right_wall is not None
    assert right_wall.find("./ref[@name='bsdf']").attrib["id"] == (
        "mat-reflective-wall"
    )


def test_mesh_preview_uses_fixed_legend_locations(monkeypatch):
    import matplotlib.pyplot as plt

    sequence = MeshSequence(
        vertices=[[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]],
        faces=[[0, 1, 2]],
        times=[0.0],
    )
    _, ax = plt.subplots()
    legend_locations = []
    original_legend = ax.legend

    def record_legend(*args, **kwargs):
        legend_locations.append(kwargs.get("loc"))
        return original_legend(*args, **kwargs)

    monkeypatch.setattr(ax, "legend", record_legend)
    plot_mesh_projection(
        ax,
        sequence,
        0.0,
        radar_position=[-1.0, 0.0, 0.0],
        title="test mesh",
    )
    plot_bedroom_scene_projection(ax)
    plt.close(ax.figure)

    assert legend_locations == ["upper left", "upper left"]


def test_rigid_transform_mesh_sequence_rotates_and_translates_vertices():
    sequence = MeshSequence(
        vertices=[[[1.0, 2.0, 3.0]], [[2.0, 2.0, 3.0]]],
        faces=[[0, 0, 0]],
        times=[0.0, 1.0],
    )
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])

    transformed = rigid_transform_mesh_sequence(
        sequence,
        rotation=rotation,
        translation=[4.0, 5.0, 6.0],
    )

    assert np.allclose(transformed.vertices_at(0.0), [[5.0, 2.0, 8.0]])
    assert transformed.max_vertex_displacement(
        1.0,
        transformed.vertices_at(0.0),
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "rotation, message",
    [
        (np.ones((2, 2)), "3x3"),
        (np.diag([1.0, 1.0, 2.0]), "orthonormal"),
        (np.diag([1.0, 1.0, -1.0]), "right-handed"),
    ],
)
def test_rigid_transform_mesh_sequence_rejects_invalid_rotation(
    rotation,
    message,
):
    sequence = MeshSequence(
        vertices=[[[0.0, 0.0, 0.0]]],
        faces=[[0, 0, 0]],
        times=[0.0],
    )

    with pytest.raises(ValueError, match=message):
        rigid_transform_mesh_sequence(sequence, rotation=rotation)


def test_rigid_transform_mesh_sequence_defaults_to_translation_only():
    sequence = MeshSequence(
        vertices=[[[1.0, 2.0, 3.0]]],
        faces=[[0, 0, 0]],
        times=[0.0],
    )

    transformed = rigid_transform_mesh_sequence(
        sequence,
        translation=[4.0, 5.0, 6.0],
    )

    assert np.allclose(transformed.vertices_at(0.0), [[5.0, 7.0, 9.0]])


def test_time_window_mesh_sequence_rebases_times_and_vertices():
    vertices = [
        [[time_s, 0.0, 0.0], [time_s, 1.0, 0.0]]
        for time_s in (0.0, 1.0, 2.0, 3.0)
    ]
    sequence = MeshSequence(
        vertices=vertices,
        faces=[[0, 1, 1]],
        times=[0.0, 1.0, 2.0, 3.0],
    )

    window = time_window_mesh_sequence(
        sequence,
        start_time_s=1.0,
        duration_s=1.5,
    )

    assert window.start_time_s == 1.0
    assert window.end_time_s == 2.5
    assert window.times.tolist() == [0.0, 1.0, 1.5]
    assert window.vertex_count == 2
    assert window.face_count == 1
    assert window.vertices_at(0.0)[0, 0] == pytest.approx(1.0)
    assert window.vertices_at(0.5)[0, 0] == pytest.approx(1.5)
    assert window.vertices_at(5.0)[0, 0] == pytest.approx(2.5)


def test_time_window_mesh_sequence_validates_window():
    sequence = MeshSequence(
        vertices=[[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        faces=[[0, 0, 0]],
        times=[0.0, 1.0],
    )

    with pytest.raises(ValueError, match="start_time_s"):
        time_window_mesh_sequence(sequence, start_time_s=1.0)
    with pytest.raises(ValueError, match="duration_s"):
        time_window_mesh_sequence(
            sequence,
            start_time_s=0.0,
            duration_s=0.0,
        )
