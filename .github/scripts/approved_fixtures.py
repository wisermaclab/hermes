# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University
"""Exact public-package binary exceptions, shared by both release checks."""

from pathlib import PurePosixPath

# Original generated ADC/mesh only; see their source NOTICE.txt. Measured ADC,
# body models, and neighbouring data remain prohibited in distributions.
SOURCE_FIXTURES = {
    "tools/bundle_prepare/data/static_target/radar_adc.npz": (
        6050,
        "065a745e3133061dff029baf1bc8e7f7e644358a55ef280159ec25a5c7fd9847",
    ),
    "tools/bundle_prepare/data/static_target/target_mesh.npz": (
        496,
        "8de48b4d8ea9c7b8d0fb66ef1faaf90bf4f8ff82a9ef4d5e1ef4a630596587ee",
    ),
}


def package_fixture(name: str, *, wheel: bool):
    """Resolve only the exact wheel/source-distribution package locations."""
    parts = PurePosixPath(name).parts
    source_name = "tools/" + name if wheel else "/".join(parts[1:])
    return SOURCE_FIXTURES.get(source_name)
