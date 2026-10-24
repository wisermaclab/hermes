# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Shared conversion for validated frame refresh settings (0 = every chirp)."""


def refresh_period_chirps(frames: int, chirps_per_frame: int) -> int:
    return 1 if frames == 0 else int(frames) * int(chirps_per_frame)
