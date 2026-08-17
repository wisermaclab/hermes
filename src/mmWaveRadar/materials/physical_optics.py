# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Physical-optics material helpers."""

from ..simulation.physical_optics import (
    MaterialModel,
    PhysicalOpticsMaterial,
    coherent_surface_attenuation,
    complex_relative_permittivity,
    fresnel_coefficients,
    fresnel_interface_coefficients,
    fresnel_slab_coefficients,
)

__all__ = [
    "MaterialModel",
    "PhysicalOpticsMaterial",
    "coherent_surface_attenuation",
    "complex_relative_permittivity",
    "fresnel_coefficients",
    "fresnel_interface_coefficients",
    "fresnel_slab_coefficients",
]
