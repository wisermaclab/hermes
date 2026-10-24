# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University
"""Exercise the narrowly approved fixture exceptions and binary rejections."""

import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from approved_fixtures import SOURCE_FIXTURES
from audit_distribution import audit_sdist, audit_wheel

ROOT = Path(__file__).resolve().parents[2]


class DistributionPolicyTests(unittest.TestCase):
    def make_archive(self, directory, *, wheel, damage=None):
        suffix = ".whl" if wheel else ".tar.gz"
        path = Path(directory) / ("hermes-test" + suffix)
        payloads = {"LICENSE": b"license", "THIRD_PARTY_NOTICES.md": b"notices"}
        for source in SOURCE_FIXTURES:
            name = source.removeprefix("tools/") if wheel else "hermes-test/" + source
            data = (ROOT / source).read_bytes()
            if damage == "content":
                data = bytes([data[0] ^ 1]) + data[1:]
            elif damage == "size":
                data += b"extra"
            payloads[name] = data
        if damage == "unapproved":
            payloads["bundle_prepare/data/private_capture.npz"] = b"private capture"
        if wheel:
            with zipfile.ZipFile(path, "w") as archive:
                for name, data in payloads.items():
                    archive.writestr(name, data)
        else:
            with tarfile.open(path, "w:gz") as archive:
                for name, data in payloads.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
        return path

    def test_only_original_fixture_bytes_are_allowed(self):
        for wheel in (False, True):
            for damage in (None, "content", "size", "unapproved"):
                with (
                    self.subTest(wheel=wheel, damage=damage),
                    tempfile.TemporaryDirectory() as folder,
                ):
                    path = self.make_archive(folder, wheel=wheel, damage=damage)
                    errors = []
                    (audit_wheel if wheel else audit_sdist)(path, errors)
                    self.assertEqual(bool(errors), damage is not None, errors)


if __name__ == "__main__":
    unittest.main()
