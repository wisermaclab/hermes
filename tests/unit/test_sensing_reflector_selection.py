# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Image-source ranking, physical losses, cutoff and runtime cap."""

import json
from types import SimpleNamespace

from sensing_test_helpers import *  # noqa: F401,F403
from mmWaveRadar.radar.ti import SimpleCosinePattern
from mmWaveRadar.radar.sensor import _rotation_matrix_numpy
from mmWaveRadar.simulation.env_human_coupling import extract_environment_reflector_bank
import mmWaveRadar.simulation.reflector_selection as selection


def _scene(*surfaces):
    objects = {}
    for i, vertices in enumerate(surfaces):
        mesh = mi.Mesh(f"surface-{i}", 4, 2)
        params = mi.traverse(mesh)
        params["vertex_positions"] = mi.Float(np.asarray(vertices, np.float32).ravel())
        params["faces"] = mi.UInt32([0, 1, 2, 0, 2, 3])
        params.update()
        objects[f"surface-{i}"] = SimpleNamespace(
            mi_mesh=mesh,
            radio_material=PhysicalOpticsMaterial(1.0, 0.0, model="pec"),
        )
    return SimpleNamespace(objects=objects)


def _floor(z=-1.0):
    return np.array([[-10.0, -10.0, z], [10.0, -10.0, z], [10.0, 10.0, z], [-10.0, 10.0, z]])


def _target():
    return MeshTarget("human", _single_face_mesh_sequence(x0=2.0))


def test_geometry_discovery_has_no_rt_seed_or_sixteen_plane_limit(monkeypatch):
    scene = _scene(*[_floor(-1.0 - i) for i in range(20)])
    monkeypatch.setattr(
        selection,
        "_image_power_scores",
        lambda bank, *args: np.repeat(np.arange(1.0, 21.0)[:, None], 2, axis=1),
    )
    selected = selection.select_image_source_reflectors(scene, _phase_one_radar(), _target())
    assert selected.selection_metadata["candidate_count"] == 20
    assert selected.object_names == ("surface-19", "surface-18", "surface-17", "surface-16")
    assert len(selected.vertices) == 4


@pytest.mark.parametrize(
    "cutoff,cap,expected",
    [
        (-20.0, 4, (1, 4, 5, 3)),
        (-20.0, 2, (1, 4)),
        (-10.0, 4, (1, 4, 5, 3)),
        (-3.0, 4, (1,)),
        (0.0, 4, (1,)),
    ],
)
def test_power_cutoff_then_count_cap_and_metadata(monkeypatch, cutoff, cap, expected):
    scene = _scene(*[_floor(-1.0 - i) for i in range(6)])
    powers = np.array([1e-5, 1.0, 0.01, 0.1, 0.5, 0.2])
    monkeypatch.setattr(
        selection, "_image_power_scores", lambda *args: np.column_stack([powers, 0.5 * powers])
    )
    bank = selection.select_image_source_reflectors(
        scene,
        _phase_one_radar(),
        _target(),
        power_threshold_db=cutoff,
        max_count=cap,
    )
    assert bank.object_names == tuple(f"surface-{i}" for i in expected)
    np.testing.assert_allclose(bank.estimated_powers, powers[list(expected)])
    np.testing.assert_allclose(bank.relative_power_db, 10.0 * np.log10(powers[list(expected)]))
    metadata = bank.selection_metadata
    assert metadata["max_count"] == cap
    assert metadata["retained_count"] == len(expected)
    assert metadata["candidates"][0]["relative_power_db"] == 0.0
    if cutoff == -20.0:
        boundary = next(row for row in metadata["candidates"] if row["object"] == "surface-2")
        assert boundary["above_power_cutoff"]  # Exactly 1% power is -20 dB.
        assert not boundary["retained"]  # Excluded by the count cap.
    json.dumps(metadata, allow_nan=False)


def test_zero_power_and_disabled_coupling_do_not_retain_planes(monkeypatch):
    scene = _scene(_floor())
    monkeypatch.setattr(selection, "_image_power_scores", lambda *args: np.zeros((1, 2)))
    bank = selection.select_image_source_reflectors(scene, _phase_one_radar(), _target())
    assert len(bank.vertices) == 0
    assert bank.selection_metadata["candidates"][0]["relative_power_db"] is None
    json.dumps(bank.selection_metadata, allow_nan=False)
    # Disabled coupling does not evaluate a potentially expensive motion provider.
    disabled = selection.select_image_source_reflectors(scene, None, None, enabled=False)
    assert len(disabled.vertices) == 0


@pytest.mark.parametrize("cutoff", [None, True, np.nan, np.inf, 0.1])
def test_keep_all_or_invalid_threshold_is_rejected(cutoff):
    with pytest.raises(ValueError, match="hybrid_reflector_cutoff_db"):
        HybridPOConfig(hybrid_reflector_cutoff_db=cutoff)
    with pytest.raises(ValueError, match="power_threshold"):
        selection.select_image_source_reflectors(None, None, None, power_threshold_db=cutoff)


@pytest.mark.parametrize("cap", [0, -1, 5, True, 2.5])
def test_count_must_be_between_one_and_four(cap):
    with pytest.raises(ValueError, match="hybrid_max_reflectors"):
        HybridPOConfig(hybrid_max_reflectors=cap)
    with pytest.raises(ValueError, match="max_count"):
        selection.select_image_source_reflectors(None, None, None, max_count=cap)


def test_floor_power_accounts_for_distance_reflection_loss_and_both_orders():
    scene = _scene(_floor())
    bank = extract_environment_reflector_bank(scene)
    radar = _phase_one_radar()
    probes = np.array([[2.0, 0.0, 0.0]])
    scores = selection._image_power_scores(bank, radar, probes, None, 1.0)
    # A 2 m direct leg and sqrt(8) m image leg, with unit PEC field reflection.
    np.testing.assert_allclose(scores, [[1.0 / 32.0, 1.0 / 32.0]], rtol=1e-12)
    farther = selection._image_power_scores(bank, radar, probes * 2.0, None, 1.0)
    np.testing.assert_allclose(farther / scores, 0.1, rtol=1e-12)
    scaled = selection._image_power_scores(bank, radar, probes, None, 0.5)
    np.testing.assert_allclose(scaled / scores, 0.25, rtol=1e-12)
    dielectric = replace(bank, materials=(PhysicalOpticsMaterial(4.0, 0.1),))
    lossy = selection._image_power_scores(dielectric, radar, probes, None, 1.0)
    assert np.all((lossy > 0.0) & (lossy < scores))
    # The floor qualifies even with no static monostatic RT path bank at all.
    assert len(selection.select_image_source_reflectors(scene, radar, _target()).vertices) == 1


@pytest.mark.parametrize("x,z", [(0.5, 0.0), (1.5, -0.5), (0.5, -0.5)])
def test_each_of_the_three_physical_legs_is_checked_for_obstruction(x, z):
    scene = _scene(_floor())
    bank = extract_environment_reflector_bank(scene)
    blocker = _scene(
        np.array([[x, -0.1, z - 0.1], [x, 0.1, z - 0.1], [x, 0.1, z + 0.1], [x, -0.1, z + 0.1]])
    )
    static = mi.load_dict(
        {
            "type": "scene",
            "floor": scene.objects["surface-0"].mi_mesh,
            "blocker": blocker.objects["surface-0"].mi_mesh,
        }
    )
    scores = selection._image_power_scores(
        bank, _phase_one_radar(), np.array([[2.0, 0.0, 0.0]]), static, 1.0
    )
    np.testing.assert_array_equal(scores, 0.0)


def test_finite_reflector_cannot_act_as_an_infinite_image_plane():
    scene = _scene(_floor() + [20.0, 0.0, 0.0])
    bank = extract_environment_reflector_bank(scene)
    scores = selection._image_power_scores(
        bank, _phase_one_radar(), np.array([[2.0, 0.0, 0.0]]), None, 1.0
    )
    np.testing.assert_array_equal(scores, 0.0)


def test_endpoint_patterns_and_polarization_rotate_with_the_radar_and_scene():
    bank = extract_environment_reflector_bank(_scene(_floor()))
    radar = _phase_one_radar()
    radar = replace(
        radar,
        hardware=replace(
            RadarHardware.from_positions(
                [[0.0, -0.05, 0.0], [0.0, 0.05, 0.0]], [[0.0, 0.0, -0.05], [0.0, 0.0, 0.05]]
            ),
            antenna_pattern=SimpleCosinePattern(4),
        ),
        fmcw=replace(radar.fmcw, num_tx=2),
    )
    probes = np.array([[2.0, 0.0, 0.0], [3.0, 0.3, 0.2]])
    expected = selection._image_power_scores(bank, radar, probes, None, 1.0)
    isotropic = selection._image_power_scores(
        bank,
        replace(radar, hardware=replace(radar.hardware, antenna_pattern=None)),
        probes,
        None,
        1.0,
    )
    assert np.all(expected < isotropic)
    orientation, offset = (0.7, -0.4, 0.2), np.array([3.0, -2.0, 1.0])
    rotation = _rotation_matrix_numpy(orientation)
    rotated = replace(
        bank,
        vertices=bank.vertices @ rotation.T + offset,
        centroids=bank.centroids @ rotation.T + offset,
        normals=bank.normals @ rotation.T,
        support_triangles=tuple(tri @ rotation.T + offset for tri in bank.support_triangles),
    )
    actual = selection._image_power_scores(
        rotated,
        replace(radar, position=tuple(offset), orientation=orientation),
        probes @ rotation.T + offset,
        None,
        1.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_motion_volume_includes_intermediate_poses_and_window_endpoints():
    vertices = np.stack([_front_facing_triangle(x) for x in (1.0, 4.0, 2.0)])
    motion = MeshSequence(vertices, np.array([[0, 1, 2]]), np.array([0.0, 0.5, 1.0]))
    lower, upper = selection.motion_volume_bounds(motion, (0.25, 0.75))
    np.testing.assert_allclose(lower, [2.5, -0.5, 0.0])
    np.testing.assert_allclose(upper, [4.0, 0.5, 1.0])
