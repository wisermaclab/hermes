# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Human/body material helpers for mmWave radar simulations."""

from __future__ import annotations


def human_skin_material(name: str):
    """
    Builds an explicit lossy human-skin/tissue-like mesh material.

    This is only a coarse tutorial proxy, not a validated skin/body model.
    The nonzero backscattering term avoids treating rough biological surfaces
    as perfectly specular slabs and favors monostatic radar returns.
    """
    from sionna.rt import RadioMaterial  # pylint: disable=import-outside-toplevel

    return RadioMaterial(name=name,
                         relative_permittivity=38.0,
                         conductivity=1.5,
                         thickness=0.01,
                         scattering_coefficient=0.35,
                         scattering_pattern="backscattering",
                         alpha_r=8,
                         alpha_i=20,
                         lambda_=0.35,
                         color=(0.85, 0.55, 0.45))
