# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Warning-only capture diagnostics; these never certify physical fidelity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def diagnose_bundle(path, output, *, reference_position=None, loaded=None) -> dict:
    from mmWaveRadar.measurements import load_bundle
    from mmWaveRadar.dsp import range_fft, psf_cross_correlation_image
    from mmWaveRadar.simulation.point_targets import calibrate_range_cube_channel_phase
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    owns_loaded = loaded is None
    if loaded is None:
        loaded = load_bundle(path)
    bundle = loaded.bundle
    sensor = bundle.sensor()
    fmcw = bundle.sensor_config.fmcw
    # Bound diagnostics to frame zero; no solver or body model is invoked.
    adc = np.asarray(bundle.real_adc[0])
    warnings = []

    def warn(code, message):
        warnings.append({"code": code, "message": message})

    raw_power = float(np.mean(np.abs(adc) ** 2))
    dc_power = float(np.mean(np.abs(np.mean(adc, axis=1)) ** 2))
    if raw_power == 0:
        warn("adc.zero", "The selected ADC frame contains no signal.")
    elif dc_power / raw_power > 0.2:
        warn("adc.dc", "Large DC component; inspect offsets and near-range clutter.")
    if np.iscomplexobj(adc) and fmcw.adc_mode != "real":
        i_power, q_power = float(np.mean(adc.real**2)), float(np.mean(adc.imag**2))
        if max(i_power, q_power) > 4 * max(min(i_power, q_power), np.finfo(float).tiny):
            warn(
                "adc.iq_imbalance",
                "I/Q powers differ by more than 6 dB; this may also reflect the scene.",
            )
        spectrum = np.mean(np.abs(np.fft.fft(adc, axis=1)) ** 2, axis=(0, 2))
        half = adc.shape[1] // 2
        positive = float(np.sum(spectrum[1:half]))
        negative = float(np.sum(spectrum[half + 1 :]))
        if fmcw.adc_mode == "complex1x" and fmcw.slope > 0 and negative > 2 * positive:
            warn(
                "adc.possible_iq_mirror",
                "Most energy is in negative FFT frequencies; IQ conjugation or legitimate upper-band returns can cause this.",
            )

    cube, ranges = range_fft(adc, fmcw=fmcw)
    profile = np.mean(np.abs(cube) ** 2, axis=(0, 2))
    peak_bin = int(np.argmax(profile))
    peak_range = float(ranges[peak_bin])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(ranges, 10 * np.log10(np.maximum(profile, np.finfo(float).tiny)))
    ax.set(xlabel="Range (m)", ylabel="Power (dB, input units squared)")
    fig.tight_layout()
    fig.savefig(output / "range_profile.png", dpi=150)
    plt.close(fig)

    hardware = sensor.hardware
    positions = hardware.virtual_channel_positions_lambda
    if positions is None:
        positions = (
            hardware.tx_positions[hardware.virtual_tx_indices(), 1:]
            + hardware.rx_positions[hardware.virtual_rx_indices(), 1:]
        ) / fmcw.wavelength
    calibrated = calibrate_range_cube_channel_phase(cube, hardware)
    # dirty_image is the direct matched-filter response; no deconvolution.
    image = psf_cross_correlation_image(calibrated[:, peak_bin, :], positions, grid_size=(65, 129))
    power = np.asarray(image["dirty_image"])
    mask = image.get("valid_mask")
    if mask is not None:
        power = np.where(mask, power, 0)
    index = np.unravel_index(int(np.argmax(power)), power.shape)
    u = float(image["u"][index[-1]])
    v = float(image["v"][index[0]]) if "v" in image else None
    fig, ax = plt.subplots(figsize=(6, 4))
    if power.ndim == 2:
        ax.imshow(
            10 * np.log10(np.maximum(power, np.finfo(float).tiny)),
            origin="lower",
            aspect="auto",
            extent=[image["u"][0], image["u"][-1], image["v"][0], image["v"][-1]],
        )
        ax.set(xlabel="Horizontal direction cosine", ylabel="Vertical direction cosine")
    else:
        ax.plot(image["u"], power)
        ax.set(xlabel="Horizontal direction cosine", ylabel="Power")
    ax.set_title(f"Angle response at {peak_range:.3f} m")
    fig.tight_layout()
    fig.savefig(output / "range_angle.png", dpi=150)
    plt.close(fig)

    reference = None
    if reference_position is not None:
        target = np.asarray(reference_position, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("Reference position must contain three finite world coordinates")
        local = sensor.world_to_local_points(target[None])[0]
        expected_range = float(np.linalg.norm(local))
        if expected_range == 0:
            raise ValueError("Reference target cannot coincide with the radar")
        expected_u, expected_v = float(local[1] / expected_range), float(local[2] / expected_range)
        resolution = (
            299792458.0 * fmcw.sampling_frequency / (2 * abs(fmcw.slope) * fmcw.num_adc_samples)
        )
        if abs(peak_range - expected_range) > 2 * resolution:
            warn(
                "reference.range_mismatch",
                "Dominant range peak differs from the known target by more than two range-resolution cells.",
            )
        if expected_range * 2 / 299792458.0 > fmcw.effective_if_cutoff_hz / abs(fmcw.slope):
            warn("reference.outside_if", "Known target lies outside the declared usable IF range.")
        if abs(expected_u) > 0.05 and u * expected_u < 0:
            warn(
                "reference.azimuth_sign",
                "Dominant angle response has the opposite horizontal sign from the known target.",
            )
        if v is not None and abs(expected_v) > 0.05 and v * expected_v < 0:
            warn(
                "reference.elevation_sign",
                "Dominant angle response has the opposite vertical sign from the known target.",
            )
        reference = {
            "range_m": expected_range,
            "u": expected_u,
            "v": expected_v,
            "range_error_m": abs(peak_range - expected_range),
        }
    report = {
        "frame": 0,
        "peak_range_m": peak_range,
        "peak_u": u,
        "peak_v": v,
        "reference": reference,
        "warnings": warnings,
        "note": "Warnings are hypotheses, not structural errors or a fidelity certificate. Use multiple known target positions to constrain channel permutations.",
    }
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if owns_loaded and loaded._temporary_directory is not None:
        loaded._temporary_directory.cleanup()
    return report
