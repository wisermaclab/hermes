# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""One raw-name policy across bundle writers, archives and NPZ members."""

import zipfile

import pytest

from bundle_prepare.archives import extract_bundle_zip
from bundle_prepare.paths import portable_relative_path, safe_bundle_file
from bundle_prepare.safe_npz import _portable_member_name
from bundle_prepare.writer import BundleWriter


@pytest.mark.parametrize(
    "name",
    [
        "../adc.npz",
        "/adc.npz",
        "C:/adc.npz",
        "room\\adc.npz",
        "room/../adc.npz",
        "room/./adc.npz",
        "room//adc.npz",
        "adc.npz/",
        "adc\0.npz",
    ],
)
def test_untrusted_names_are_rejected_consistently_before_normalization(tmp_path, name):
    with pytest.raises(ValueError):
        portable_relative_path(name)
    with pytest.raises(ValueError):
        _portable_member_name(name)
    with BundleWriter() as writer:
        with pytest.raises(ValueError):
            writer.writestr(name, b"test")
        # This test inspects staging without publishing a descriptor.
        writer.report = {}


@pytest.mark.parametrize("name", ["room/./adc.npz", "room//adc.npz", "room/../adc.npz"])
def test_zip_extraction_uses_the_raw_name_policy(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(name, b"test")
    with pytest.raises(ValueError, match="unsafe bundle ZIP member"):
        extract_bundle_zip(archive, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


@pytest.mark.parametrize(
    "names",
    [
        ("radar_adc.npz", "RADAR_ADC.npz"),
        ("room/adc.npz", "Room/adc.npz"),
        ("room/", "ROOM/"),
    ],
)
def test_zip_members_differing_only_in_case_are_duplicates(tmp_path, names):
    archive = tmp_path / "case.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for index, name in enumerate(names):
            output.writestr(name, b"" if name.endswith("/") else f"member {index}")
    with pytest.raises(ValueError, match="duplicate bundle ZIP member"):
        extract_bundle_zip(archive, tmp_path / "extracted")


def test_zip_directory_policy_accepts_one_trailing_separator():
    assert portable_relative_path("room/", directory=True).as_posix() == "room"
    with pytest.raises(ValueError):
        portable_relative_path("room//", directory=True)


def test_containment_rejects_symlinked_parent_even_when_target_is_inside_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "mesh.obj").write_text("v 0 0 0\n")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        safe_bundle_file(tmp_path, "alias/mesh.obj", label="mesh")
    assert safe_bundle_file(tmp_path, "real/mesh.obj", label="mesh") == real / "mesh.obj"


def test_scene_name_policy_remains_explicit():
    assert portable_relative_path("aux.npy").name == "aux.npy"
    with pytest.raises(ValueError):
        portable_relative_path("aux.obj", windows_names=True)


def test_zip_cannot_write_through_a_preexisting_symlink(tmp_path):
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("assets/mesh.obj", "mesh")
    destination = tmp_path / "bundle"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "assets").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        extract_bundle_zip(archive, destination)
    assert not (outside / "mesh.obj").exists()


def test_writer_checks_filesystem_paths_as_well_as_member_names(tmp_path):
    with BundleWriter() as writer:
        (writer.root / "assets").symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            writer.writestr("assets/mesh.obj", "mesh")
        writer.report = {}
    assert not (tmp_path / "mesh.obj").exists()


def test_scene_asset_graph_rejects_intermediate_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "mesh.obj").write_text("v 0 0 0\n")
    (source / "alias").symlink_to(source, target_is_directory=True)
    scene = source / "scene.xml"
    scene.write_text(
        '<scene version="3.0.0"><shape type="obj">'
        '<string name="filename" value="alias/mesh.obj"/></shape></scene>'
    )
    with BundleWriter() as writer:
        with pytest.raises(ValueError, match="symlink"):
            writer.add_scene(scene)
        writer.report = {}
