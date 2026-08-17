# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Optional local web interface for HERMES experiments and measurements."""


def build_app(*args, **kwargs):
    """Build the Panel application without importing GUI dependencies eagerly."""

    from .app import build_app as _build_app

    return _build_app(*args, **kwargs)


__all__ = ["build_app"]
