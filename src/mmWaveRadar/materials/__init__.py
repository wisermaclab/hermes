# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Material definitions and helpers for mmWave radar simulations."""

from .human import human_skin_material

_PHYSICAL_OPTICS_EXPORTS = {
    "MaterialModel",
    "PhysicalOpticsMaterial",
    "coherent_surface_attenuation",
    "complex_relative_permittivity",
    "fresnel_coefficients",
    "fresnel_interface_coefficients",
    "fresnel_slab_coefficients",
}


def __getattr__(name):
    if name in _PHYSICAL_OPTICS_EXPORTS:
        from . import physical_optics

        value = getattr(physical_optics, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MaterialModel",
    "PhysicalOpticsMaterial",
    "coherent_surface_attenuation",
    "complex_relative_permittivity",
    "fresnel_coefficients",
    "fresnel_interface_coefficients",
    "fresnel_slab_coefficients",
    "human_skin_material",
]
