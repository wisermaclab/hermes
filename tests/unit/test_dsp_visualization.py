# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Tests for generic DSP plotting helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mmWaveRadar.dsp import (
    plot_axis_image,
    plot_point_cloud_projection,
    plot_range_profile,
)


def test_plot_axis_image_returns_artist_and_sets_axes():
    fig, ax = plt.subplots()

    image = np.arange(6, dtype=float).reshape(2, 3)
    im = plot_axis_image(
        image,
        np.array([-1.0, 0.0, 1.0]),
        np.array([10.0, 20.0]),
        ax=ax,
        xlabel="azimuth",
        ylabel="elevation",
        title="angle map",
        colorbar=True,
        colorbar_label="relative power",
    )

    assert im.axes is ax
    assert tuple(im.get_extent()) == (-1.0, 1.0, 10.0, 20.0)
    assert ax.get_xlabel() == "azimuth"
    assert ax.get_ylabel() == "elevation"
    assert ax.get_title() == "angle map"
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "relative power"
    plt.close(fig)


def test_plot_axis_image_rejects_mismatched_axes():
    with pytest.raises(ValueError, match="x_axis"):
        plot_axis_image(np.ones((2, 3)), np.arange(2), np.arange(2))

    with pytest.raises(ValueError, match="y_axis"):
        plot_axis_image(np.ones((2, 3)), np.arange(3), np.arange(4))


def test_plot_axis_image_supports_shared_colorbar():
    fig, axes = plt.subplots(1, 2)
    x_axis = np.arange(3)
    y_axis = np.arange(2)

    im0 = plot_axis_image(np.ones((2, 3)), x_axis, y_axis, ax=axes[0])
    plot_axis_image(2.0 * np.ones((2, 3)), x_axis, y_axis, ax=axes[1])
    fig.colorbar(im0, ax=axes, label="shared")

    assert len(fig.axes) == 3
    assert fig.axes[-1].get_ylabel() == "shared"
    plt.close(fig)


def test_plot_range_profile_applies_limits_and_target_marker():
    fig, ax = plt.subplots()

    line = plot_range_profile(
        np.linspace(0.0, 4.0, 5),
        np.arange(5),
        ax=ax,
        range_limits_m=(1.0, 3.0),
        target_range_m=2.0,
        label="profile",
        ylabel="magnitude",
        title="range profile",
        linewidth=2.0,
    )

    np.testing.assert_allclose(line.get_xdata(), [1.0, 2.0, 3.0])
    assert ax.get_xlim() == (1.0, 3.0)
    assert ax.get_xlabel() == "range [m]"
    assert ax.get_ylabel() == "magnitude"
    assert ax.get_title() == "range profile"
    assert len(ax.lines) == 2
    plt.close(fig)


def test_plot_point_cloud_projection_handles_empty_and_nonempty_points():
    fig, axes = plt.subplots(1, 2)

    empty = plot_point_cloud_projection(
        np.zeros((0, 3)),
        ax=axes[0],
        empty_label="empty",
    )
    assert empty is None
    assert axes[0].texts[0].get_text() == "empty"

    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    scatter = plot_point_cloud_projection(
        points,
        x_axis="x",
        y_axis="z",
        power=np.array([0.25, 1.0]),
        ax=axes[1],
        colorbar=True,
        colorbar_label="power",
    )

    assert scatter is not None
    np.testing.assert_allclose(scatter.get_offsets(), [[1.0, 3.0], [4.0, 6.0]])
    assert axes[1].get_xlabel() == "x [m]"
    assert axes[1].get_ylabel() == "z [m]"
    assert fig.axes[-1].get_ylabel() == "power"
    plt.close(fig)
