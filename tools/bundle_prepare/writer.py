# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Public, GUI-independent construction and publication of portable bundles."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shlex
import shutil
from tempfile import TemporaryDirectory, mkdtemp
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from .paths import portable_relative_path, safe_bundle_destination, safe_bundle_file
from .contract import (
    BundleSensorConfig,
    BundleValidationError,
    read_json,
    required_bundle_features,
)


def _check_prepared_directory(root: Path) -> dict:
    """New geometry-bearing bundles must support re-simulation at publication."""
    from .integrity import check_and_load_bundle

    bundle, report = check_and_load_bundle(root)
    if bundle.resimulation_issues:
        raise BundleValidationError(
            "Cannot prepare bundle for re-simulation: " + "; ".join(bundle.resimulation_issues),
            field="resimulation",
            code="resimulation.unavailable",
        )
    return report


def canonicalize_directory(root: Path) -> dict:
    """Normalize producer aliases without changing ADC samples or motion."""
    descriptor = read_json(root / "bundle.json")
    original = read_json(root / "sensor.json")
    sensor = BundleSensorConfig.from_mapping(original)
    if sensor.fmcw.tx_schedule == "tdm":
        if sensor.tdm_tx_order is None and not any(
            (sensor.metadata or {}).get(k) is not None for k in ("tx_to_enable", "tdm_tx_order")
        ):
            raise ValueError("New TDM bundles require an explicit fmcw.tdm_tx_order")
        stored = sensor.to_sensor().hardware
        metadata = stored.board_metadata
        channel_map = tuple(
            zip(
                np.asarray(metadata["physical_tx_indices"])[stored.virtual_tx_indices()].tolist(),
                np.asarray(metadata["physical_rx_indices"])[stored.virtual_rx_indices()].tolist(),
            )
        )
        sensor = replace(
            sensor, tdm_tx_order=tuple(sensor.slot_order().tolist()), channel_map=channel_map
        )
    canonical = {**original, **sensor.to_json()}
    canonical["fmcw"] = {**original.get("fmcw", {}), **canonical["fmcw"]}
    canonical["fmcw"].pop("tdm_enabled", None)
    canonical.pop("tdm_enabled", None)
    features = set(descriptor.get("required_features", [])) | required_bundle_features(
        descriptor, sensor
    )
    if features:
        descriptor["schema_version"] = 2
        descriptor["required_features"] = sorted(features)
    environment_path = descriptor.get("environment")
    if environment_path:
        environment = read_json(safe_bundle_file(root, environment_path, label="environment"))
        scene_values = [(descriptor.get("scene") or {}).get("file")]
        scene_values += [environment.get(k) for k in ("scene_path", "scene_file")]
        paths = {safe_bundle_file(root, v, label="scene") for v in scene_values if v is not None}
        if len(paths) > 1:
            raise ValueError("Conflicting scene declarations")
        if paths:
            descriptor.setdefault("scene", {})["file"] = (
                paths.pop().relative_to(root.resolve()).as_posix()
            )
    (root / "sensor.json").write_text(json.dumps(canonical, indent=2, allow_nan=False) + "\n")
    (root / "bundle.json").write_text(json.dumps(descriptor, indent=2, allow_nan=False) + "\n")
    return descriptor


def publish_staged_directory(
    staged: Path, output: Path, *, force: bool = False, ownership_check=None
) -> None:
    """Publish a checked directory, retaining the previous output for rollback."""
    output = Path(output)
    if output.is_symlink():
        raise ValueError("Bundle output must not be a symlink")
    backup = None
    reserved = False
    container = Path(mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent))
    try:
        try:
            output.mkdir()
            reserved = True
        except FileExistsError:
            if not force:
                raise FileExistsError(f"{output} exists; use --force to replace it")
            if not output.is_dir() or not (output / "bundle.json").is_file():
                raise ValueError("Refusing to replace a directory that is not a bundle")
            if ownership_check is not None and not ownership_check(output):
                raise ValueError("Refusing to replace a bundle not owned by this adapter")
            backup = container / "previous"
            os.replace(output, backup)
            if ownership_check is not None and not ownership_check(backup):
                raise ValueError("Bundle ownership changed during publication")
            output.mkdir()
            reserved = True
        os.replace(staged, output)
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        if reserved and output.is_dir():
            try:
                output.rmdir()
            except OSError:
                pass
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    finally:
        # Keep a recoverable previous bundle if a concurrent writer prevented
        # rollback. Never clean up the only remaining copy of the old output.
        if not any(container.iterdir()):
            container.rmdir()


class BundleWriter:
    """Stage files, normalize metadata, check, then publish a directory or ZIP.

    With no output path, ``to_bytes()`` returns an in-memory ZIP suitable for
    GUI downloads. The file-writing methods intentionally resemble ZipFile.
    Exceptions leave the destination untouched. Writers must be used as
    context managers so staging is always cleaned up.
    """

    def __init__(self, output: str | Path | None = None, *, force: bool = False):
        self.output = None if output is None else Path(output).expanduser().absolute()
        self.force = force
        self._archive = None
        self.report = None

    @staticmethod
    def publish_directory(
        staged: str | Path, output: str | Path, *, force=False, ownership_check=None
    ) -> dict:
        """Finish an adapter-owned staging tree without copying its large ADC."""
        staged, output = Path(staged), Path(output)
        canonicalize_directory(staged)
        report = _check_prepared_directory(staged)
        from mmWaveRadar.provenance import write_bundle_provenance

        write_bundle_provenance(staged)
        publish_staged_directory(staged, output, force=force, ownership_check=ownership_check)
        return {**report, "path": str(output)}

    def __enter__(self):
        parent = None if self.output is None else self.output.parent
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        self._temporary = TemporaryDirectory(prefix=".hermes-bundle-", dir=parent)
        self.root = Path(self._temporary.name) / "bundle"
        self.root.mkdir()
        return self

    def _destination(self, name: str) -> Path:
        result = safe_bundle_destination(self.root, name, label="Unsafe bundle path")
        result.parent.mkdir(parents=True, exist_ok=True)
        return result

    def writestr(self, name: str, payload: str | bytes) -> None:
        if self.report is not None:
            raise ValueError("Bundle writer has already finished")
        path = self._destination(name)
        if path.exists():
            raise ValueError(f"Duplicate bundle path: {name}")
        path.write_bytes(payload.encode("utf-8") if isinstance(payload, str) else payload)

    def write_json(self, name: str, value) -> None:
        self.writestr(name, json.dumps(value, indent=2, allow_nan=False) + "\n")

    def write_npz(self, name: str, **arrays) -> None:
        stream = BytesIO()
        for key, value in arrays.items():
            if np.asarray(value).dtype.hasobject:
                raise ValueError(f"NPZ {key} cannot contain Python objects")
        np.savez_compressed(stream, **arrays)
        self.writestr(name, stream.getvalue())

    def copy_file(self, source: str | Path, name: str) -> None:
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Input must be a regular file: {source}")
        target = self._destination(name)
        if target.exists():
            raise ValueError(f"Duplicate bundle path: {name}")
        shutil.copyfile(source, target)

    def add_scene(self, source: str | Path) -> dict:
        """Copy a local XML/OBJ/MTL asset graph and rewrite its references.

        Referenced files must remain within the source scene directory. XML
        includes and remote URLs are unsupported. Content-addressed names
        avoid collisions and avoid embedding absolute source paths.
        """
        source = Path(source).absolute()
        source_root = source.parent.resolve()
        source = source_root / source.name
        assets = {}
        visiting = set()

        def visit(path):
            name = path.relative_to(source_root).as_posix()
            resolved = safe_bundle_file(source_root, name, label="Scene asset")
            if resolved in assets:
                return assets[resolved]
            if resolved in visiting:
                raise ValueError("Cyclic scene asset references are unsupported")
            if resolved.stat().st_size > 256 * 1024**2:
                raise ValueError("A scene asset exceeds 256 MiB")
            payload = resolved.read_bytes()
            visiting.add(resolved)

            def reference(value):
                if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value):
                    raise ValueError("Remote scene assets are unsupported")
                relative = portable_relative_path(value, label="Scene asset reference")
                child = visit(resolved.parent / relative)
                return Path(child).name

            if resolved.suffix.lower() == ".xml":
                if len(payload) > 2 * 1024**2:
                    raise ValueError("Scene XML exceeds 2 MiB")
                tree = ET.fromstring(payload)
                for node in tree.iter():
                    tag = node.tag.rsplit("}", 1)[-1]
                    if tag == "include":
                        raise ValueError("XML includes are unsupported in portable bundles")
                    if tag == "string" and node.get("name") == "filename":
                        node.set("value", reference(node.attrib["value"]))
                payload = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
            elif resolved.suffix.lower() in {".obj", ".mtl"}:
                lines = []
                for line in payload.decode("utf-8").splitlines():
                    # OBJ object/group/material names are free text. Quotes in
                    # those names are not shell syntax and must be preserved.
                    keyword = line.lstrip().split(maxsplit=1)
                    command = keyword[0].lower() if keyword else ""
                    dependency = (
                        command == "mtllib"
                        or command.startswith("map_")
                        or command in {"bump", "disp", "decal", "refl"}
                    )
                    tokens = shlex.split(line, comments=True) if dependency else []
                    if tokens and command == "mtllib":
                        line = "mtllib " + " ".join(reference(v) for v in tokens[1:])
                    elif tokens and (
                        tokens[0].lower().startswith("map_")
                        or tokens[0].lower() in {"bump", "disp", "decal", "refl"}
                    ):
                        line = " ".join(tokens[:-1] + [reference(tokens[-1])])
                    lines.append(line)
                payload = ("\n".join(lines) + "\n").encode()
            name = f"scene/{sha256(payload).hexdigest()[:16]}{resolved.suffix.lower()}"
            visiting.remove(resolved)
            assets[resolved] = name
            existing = self.root / name
            if not existing.exists():
                self.writestr(name, payload)
            elif existing.read_bytes() != payload:
                raise ValueError("Scene asset name collision")
            return name

        scene = visit(source)
        return {"file": scene, "assets": sorted(set(assets.values()) - {scene})}

    def finish(self) -> dict:
        if self.report is not None:
            return self.report
        canonicalize_directory(self.root)
        report = _check_prepared_directory(self.root)
        from mmWaveRadar.provenance import write_bundle_provenance

        write_bundle_provenance(self.root)
        if self.output is None or self.output.suffix.lower() == ".zip":
            stream = BytesIO()
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(self.root.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(self.root).as_posix())
            self._archive = stream.getvalue()
            if self.output is not None:
                if self.output.is_symlink():
                    raise ValueError("Output must not be a symlink")
                temporary = Path(self._temporary.name) / "output.zip"
                temporary.write_bytes(self._archive)
                if self.force:
                    os.replace(temporary, self.output)
                else:
                    os.link(temporary, self.output)
        else:
            publish_staged_directory(self.root, self.output, force=self.force)
        self.report = report
        return self.report

    def to_bytes(self) -> bytes:
        self.finish()
        if self._archive is None:
            raise ValueError("Directory writers do not retain an in-memory ZIP")
        return self._archive

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.finish()
        finally:
            self._temporary.cleanup()
