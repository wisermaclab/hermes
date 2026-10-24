# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Import the supported TI FMCW CLI waveform and TX schedule subset."""

import numpy as np
from .fmcw import FMCWConfig


def parse_ti_cli_config(
    filename: str, default_carrier_frequency: float
) -> tuple[FMCWConfig, tuple[int, ...] | None]:
    """Parses TI ``profileCfg``, ``chirpCfg``, and ``frameCfg`` commands."""

    profiles: dict[int, dict[str, float | int]] = {}
    chirp_configs: list[dict[str, object]] = []
    frame_config: tuple[int, int, int, float] | None = None
    sampling_mode = "complex1x"

    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            toks = line.split()
            if toks[0] == "adcCfg":
                # TI CLI: 0 = real, 1 = complex 1x, 2 = complex 2x.
                try:
                    sampling_mode = {0: "real", 1: "complex1x", 2: "complex2x"}[int(toks[2])]
                except (IndexError, ValueError, KeyError) as exc:
                    raise ValueError(f"Could not parse adcCfg: {line}") from exc
            elif toks[0] == "profileCfg":
                try:
                    profile_id = int(float(toks[1]))
                    profiles[profile_id] = {
                        "start_freq_ghz": float(toks[2]),
                        "idle_us": float(toks[3]),
                        "adc_start_us": float(toks[4]),
                        "ramp_us": float(toks[5]),
                        "slope_mhz_per_us": float(toks[8]),
                        "num_adc": int(float(toks[10])),
                        "fs_ksps": float(toks[11]),
                    }
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Could not parse profileCfg: {line}") from exc
            elif toks[0] == "chirpCfg":
                try:
                    chirp_configs.append(
                        {
                            "start": int(float(toks[1])),
                            "end": int(float(toks[2])),
                            "profile_id": int(float(toks[3])),
                            "variations": tuple(float(toks[index]) for index in range(4, 8)),
                            "tx_mask": int(float(toks[8])),
                        }
                    )
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Could not parse chirpCfg: {line}") from exc
            elif toks[0] == "frameCfg":
                try:
                    frame_config = (
                        int(float(toks[1])),
                        int(float(toks[2])),
                        int(float(toks[3])),
                        float(toks[5]),
                    )
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Could not parse frameCfg: {line}") from exc

    if not profiles or frame_config is None:
        raise ValueError("TI config must contain profileCfg and frameCfg")

    chirp_start, chirp_end, num_loops, frame_period_ms = frame_config
    if chirp_start < 0 or chirp_end < chirp_start:
        raise ValueError("frameCfg chirp range must be nonnegative and ordered")
    if num_loops <= 0:
        raise ValueError("frameCfg numLoops must be positive")

    num_chirps_per_loop = chirp_end - chirp_start + 1
    schedule: list[dict[str, object]] = []
    if chirp_configs:
        for chirp_index in range(chirp_start, chirp_end + 1):
            matches = [
                config
                for config in chirp_configs
                if int(config["start"]) <= chirp_index <= int(config["end"])
            ]
            if not matches:
                raise ValueError(f"No chirpCfg covers in-frame chirp index {chirp_index}")
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple chirpCfg entries cover in-frame chirp index" f" {chirp_index}"
                )
            schedule.append(matches[0])
    elif len(profiles) != 1:
        raise ValueError("Multiple profileCfg entries require an in-frame chirpCfg mapping")

    if schedule:
        for config in schedule:
            profile_id = int(config["profile_id"])
            if profile_id not in profiles:
                raise ValueError(f"In-frame chirpCfg references missing profile {profile_id}")
            if any(value != 0.0 for value in config["variations"]):
                raise ValueError("In-frame chirpCfg parameter variations are not supported")
        profile_ids = tuple(dict.fromkeys(int(config["profile_id"]) for config in schedule))
    else:
        profile_ids = (next(iter(profiles)),)

    profile = profiles[profile_ids[0]]
    waveform_fields = (
        "start_freq_ghz",
        "idle_us",
        "adc_start_us",
        "ramp_us",
        "slope_mhz_per_us",
        "num_adc",
        "fs_ksps",
    )
    for profile_id in profile_ids[1:]:
        candidate = profiles[profile_id]
        if any(candidate[field] != profile[field] for field in waveform_fields):
            raise ValueError("In-frame chirpCfg entries reference incompatible profiles")

    start_freq_ghz = float(profile["start_freq_ghz"])
    carrier_frequency = (
        default_carrier_frequency if not np.isfinite(start_freq_ghz) else start_freq_ghz * 1e9
    )
    # ADC synthesis and wideband PO measure fast time from sample zero.
    carrier_frequency += (
        float(profile["adc_start_us"]) * 1e-6 * float(profile["slope_mhz_per_us"]) * 1e12
    )
    enabled_tx_indices = None
    tdm_enabled = False
    if schedule:
        tx_masks = tuple(int(config["tx_mask"]) for config in schedule)
        mask = 0
        for tx_mask in tx_masks:
            mask |= tx_mask
        if mask <= 0:
            raise ValueError("In-frame chirpCfg entries must enable a transmitter")
        enabled_tx_indices = tuple(
            index for index in range(mask.bit_length()) if mask & (1 << index)
        )
        single_tx_slots = all(tx_mask > 0 and not tx_mask & (tx_mask - 1) for tx_mask in tx_masks)
        constant_mask = len(set(tx_masks)) == 1
        complete_cycle = False
        if len(enabled_tx_indices) > 1 and single_tx_slots:
            scheduled_tx_indices = tuple(tx_mask.bit_length() - 1 for tx_mask in tx_masks)
            cycle_length = len(enabled_tx_indices)
            first_cycle = scheduled_tx_indices[:cycle_length]
            complete_cycle = (
                len(set(first_cycle)) == cycle_length
                and len(scheduled_tx_indices) % cycle_length == 0
                and all(
                    scheduled_tx_indices[start : start + cycle_length] == first_cycle
                    for start in range(0, len(scheduled_tx_indices), cycle_length)
                )
            )
            if complete_cycle:
                enabled_tx_indices = first_cycle
                tdm_enabled = True
        if not constant_mask and not complete_cycle:
            raise ValueError(
                "Changing in-frame Tx masks are unsupported; use one constant "
                "simultaneous mask or a complete repeated single-Tx TDM cycle"
            )

    num_slow_time_samples = (
        num_loops * (num_chirps_per_loop // len(enabled_tx_indices))
        if tdm_enabled
        else num_loops * num_chirps_per_loop
    )

    fmcw = FMCWConfig(
        carrier_frequency=carrier_frequency,
        slope=float(profile["slope_mhz_per_us"]) * 1e12,
        chirp_duration=float(profile["ramp_us"]) * 1e-6,
        chirp_repetition_time=(float(profile["idle_us"]) + float(profile["ramp_us"])) * 1e-6,
        sampling_frequency=float(profile["fs_ksps"]) * 1e3,
        num_adc_samples=int(profile["num_adc"]),
        num_chirps_per_frame=num_slow_time_samples,
        frame_period=frame_period_ms * 1e-3,
        num_tx=(
            len(enabled_tx_indices) if enabled_tx_indices is not None else num_chirps_per_loop
        ),
        tx_schedule="tdm" if tdm_enabled else "simultaneous",
        adc_mode=sampling_mode,
    )
    return fmcw, enabled_tx_indices
