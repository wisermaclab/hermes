# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Compatibility module alias for :mod:`mmWaveRadar.targets.smpl`."""

from __future__ import annotations

import sys

from .targets import smpl as _smpl

sys.modules[__name__] = _smpl
