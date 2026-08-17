# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""DSP helpers for mmWave radar sensing cubes."""

from .angle import angle_map_2d_fft, angle_map_fft, direction_cosine_valid_mask
from .array_processing import (
    array_covariance,
    music_pseudospectrum,
    mvdr_pseudospectrum,
    steering_matrix,
)
from .cfar import ca_cfar_1d
from .clutter import remove_slow_time_mean
from .doppler import (
    doppler_time_spectrum,
    framewise_range_doppler_map,
    range_doppler_map,
)
from .fft import phase2distance, range_fft, range_time_map
from .imaging import (
    analytical_array_psf,
    backprojection,
    psf_cross_correlation,
    psf_cross_correlation_image,
    wiener_deconvolution,
    wiener_psf_image,
)
from .point_cloud import point_cloud_3d_from_adc
from .power import power_to_db, power_to_relative_db
from .profiles import range_profile_from_cube, select_range_bin
from .visualization import (
    plot_angle_map,
    plot_axis_image,
    plot_point_cloud_projection,
    plot_pseudospectrum,
    plot_range_profile,
    plot_range_time_map,
)
from .window import window_1d

__all__ = [
    "angle_map_2d_fft",
    "angle_map_fft",
    "analytical_array_psf",
    "array_covariance",
    "backprojection",
    "ca_cfar_1d",
    "doppler_time_spectrum",
    "framewise_range_doppler_map",
    "direction_cosine_valid_mask",
    "music_pseudospectrum",
    "mvdr_pseudospectrum",
    "phase2distance",
    "point_cloud_3d_from_adc",
    "power_to_db",
    "power_to_relative_db",
    "psf_cross_correlation",
    "psf_cross_correlation_image",
    "plot_angle_map",
    "plot_axis_image",
    "plot_point_cloud_projection",
    "plot_pseudospectrum",
    "plot_range_profile",
    "plot_range_time_map",
    "range_doppler_map",
    "range_fft",
    "range_profile_from_cube",
    "range_time_map",
    "remove_slow_time_mean",
    "select_range_bin",
    "steering_matrix",
    "window_1d",
    "wiener_deconvolution",
    "wiener_psf_image",
]
