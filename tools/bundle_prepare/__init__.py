# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Dataset-neutral bundle contract and preparation integrity checks."""

from .contract import Bundle, BundleSensorConfig, FrameRecord


def check_bundle_integrity(*args, **kwargs):
    """Lazily import the checker so ``python -m ...integrity`` stays warning-free."""

    from .integrity import check_bundle_integrity as check

    return check(*args, **kwargs)

__all__ = [
    "Bundle",
    "BundleSensorConfig",
    "FrameRecord",
    "check_bundle_integrity",
]
