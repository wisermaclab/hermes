# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Reusable diagnostic tools for mmWave radar sensing studies."""

__all__ = [
    "RadialVelocityDiagnosticConfig",
    "RadialVelocityDiagnosticResult",
    "format_radial_velocity_report",
    "run_radial_velocity_diagnostic",
    "save_radial_velocity_diagnostic_plot",
    "save_radial_velocity_samples_npz",
]


def __getattr__(name):
    if name in __all__:
        from . import radial_velocity_geometry  # pylint: disable=import-outside-toplevel

        return getattr(radial_velocity_geometry, name)
    raise AttributeError(name)
