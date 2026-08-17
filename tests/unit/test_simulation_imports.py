# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Import-cost regressions for the public simulation package."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_motion_diagnostics_import_does_not_initialize_rt_stack():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(repo_root / "src"), str(repo_root), env.get("PYTHONPATH")),
        )
    )
    code = """
import sys
import mmWaveRadar.simulation.motion_diagnostics
heavy = (
    'mitsuba',
    'sionna',
    'mmWaveRadar.simulation.physical_optics',
    'mmWaveRadar.simulation.simulator',
)
assert not any(name in sys.modules for name in heavy)
from mmWaveRadar.simulation import RadialVelocitySamples
assert RadialVelocitySamples.__module__.endswith('motion_diagnostics')
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
