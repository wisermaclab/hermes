#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 AND BSD-3-Clause
# Copyright (c) 2026 Wireless System Research Group, McMaster University
#
# The cascade ADC calibration and decoding logic includes portions adapted from
# Texas Instruments' mmWave cascade processing software:
#
# Copyright (C) 2018 Texas Instruments Incorporated - http://www.ti.com/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Texas Instruments Incorporated nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read one measured RT-Pose AWR2243 cascade frame as an ADC cube.

The calibration stage is a Python adaptation of TI's ``calibrationCascade``
time-domain range/phase correction distributed with the upstream RT-Pose
processing code. The retained BSD terms are above.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.io import loadmat


# RT-Pose follows TI's MIMO-processing permutation after calibration. These are
# physical AWR2243 receiver labels, not new receiver identities. Bundle
# preparation converts this source order back to RX1..RX16 before export.
RTPOSE_MIMO_RX_ORDER = (
    13,
    14,
    15,
    16,
    1,
    2,
    3,
    4,
    9,
    10,
    11,
    12,
    5,
    6,
    7,
    8,
)


@dataclass(frozen=True)
class RadarParams:
    """Fixed acquisition parameters used by the published RT-Pose captures."""

    num_adc_samples: int = 256
    adc_sample_rate: float = 5.0e6
    start_freq: float = 77.0e9
    chirp_slope: float = 6.4985e13
    chirp_idle_time: float = 5.0e-6
    adc_start_time: float = 5.0e-6
    chirp_ramp_end_time: float = 6.0e-5
    num_devices: int = 4
    num_rx_per_device: int = 4
    num_chirps_in_loop: int = 12
    num_loops: int = 64
    tx_to_enable: tuple[int, ...] = (12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
    rx_for_mimo_process: tuple[int, ...] = RTPOSE_MIMO_RX_ORDER
    slope_calib: float = 7.8986e13
    fs_calib: float = 8.0e6
    calibration_interp: int = 5
    phase_calib_only: bool = True

    @property
    def num_rx(self) -> int:
        return self.num_devices * self.num_rx_per_device

    @property
    def num_tx(self) -> int:
        return len(self.tx_to_enable)

    @property
    def num_chirps_per_frame(self) -> int:
        return self.num_chirps_in_loop * self.num_loops

    @property
    def chirp_ramp_time(self) -> float:
        return self.num_adc_samples / self.adc_sample_rate

    @property
    def range_resolution(self) -> float:
        bandwidth = self.chirp_slope * self.chirp_ramp_time
        return 3.0e8 / (2.0 * bandwidth)

    @property
    def carrier_frequency(self) -> float:
        midpoint = self.adc_start_time + self.chirp_ramp_time / 2.0
        return self.start_freq + midpoint * self.chirp_slope

    @property
    def velocity_resolution(self) -> float:
        wavelength = 3.0e8 / self.carrier_frequency
        chirp_interval = self.chirp_ramp_end_time + self.chirp_idle_time
        return wavelength / (2.0 * self.num_loops * chirp_interval * self.num_tx)


def capture_file_index(data_folder: Path) -> str:
    """Return the sole cascade capture index below an RT-Pose sequence."""

    indices = {
        match.group(1)
        for path in data_folder.glob("*_data.bin")
        if (match := re.search(r"_(\d+)_data\.bin$", path.name)) is not None
    }
    if not indices:
        raise FileNotFoundError(f"No AWR2243 *_data.bin files found in {data_folder}")
    if len(indices) != 1:
        choices = ", ".join(sorted(indices))
        raise ValueError(
            f"Expected one AWR2243 capture set in {data_folder}; found {choices}"
        )
    return next(iter(indices))


def cascade_files(data_folder: Path, file_index: str) -> dict[str, Path]:
    files = {
        device: data_folder / f"{device}_{file_index}_data.bin"
        for device in ("master", "slave1", "slave2", "slave3")
    }
    files["master_idx"] = data_folder / f"master_{file_index}_idx.bin"
    missing = [path for path in files.values() if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"RT-Pose cascade capture is incomplete: {names}")
    return files


def valid_frame_count(index_file: Path) -> int:
    header = np.fromfile(index_file, dtype="<u4", count=6)
    if header.size != 6:
        raise ValueError(f"RT-Pose index file is too small: {index_file}")
    return int(header[3])


def read_device_frame(
    data_file: Path,
    frame_index: int,
    params: RadarParams,
) -> np.ndarray:
    """Read one 1-based frame as ``[samples, loops, device_rx, tx]``."""

    words_per_frame = (
        params.num_adc_samples
        * params.num_chirps_in_loop
        * params.num_loops
        * params.num_rx_per_device
        * 2
    )
    byte_offset = (frame_index - 1) * words_per_frame * np.dtype("<i2").itemsize
    raw = np.fromfile(
        data_file,
        dtype="<i2",
        count=words_per_frame,
        offset=byte_offset,
    )
    if raw.size != words_per_frame:
        raise ValueError(
            f"Could not read radar frame {frame_index} from {data_file.name}: "
            f"expected {words_per_frame} int16 values, got {raw.size}"
        )
    iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    cube = iq.reshape(
        (
            params.num_rx_per_device,
            params.num_adc_samples,
            params.num_chirps_in_loop,
            params.num_loops,
        ),
        order="F",
    )
    return cube.transpose(1, 3, 0, 2)


def read_adc_cube(
    files: dict[str, Path],
    frame_index: int,
    params: RadarParams,
) -> np.ndarray:
    adc = np.empty(
        (params.num_adc_samples, params.num_loops, params.num_rx, params.num_tx),
        dtype=np.complex64,
    )
    for device_index, device in enumerate(("master", "slave1", "slave2", "slave3")):
        start = device_index * params.num_rx_per_device
        stop = start + params.num_rx_per_device
        adc[:, :, start:stop, :] = read_device_frame(
            files[device],
            frame_index,
            params,
        )
    return adc


def load_calibration(calibration_file: Path) -> tuple[np.ndarray, np.ndarray]:
    if not calibration_file.is_file():
        raise FileNotFoundError(
            f"RT-Pose radar calibration file was not found: {calibration_file}"
        )
    payload = loadmat(calibration_file)
    if "calibResult" not in payload:
        raise ValueError(f"Calibration file has no calibResult: {calibration_file}")
    result = payload["calibResult"][0, 0]
    return result["RangeMat"].astype(np.float64), result["PeakValMat"]


def apply_calibration(
    adc: np.ndarray,
    calibration_file: Path,
    params: RadarParams,
) -> np.ndarray:
    """Apply the RT-Pose time-domain cascade range and phase calibration."""

    range_matrix, peak_matrix = load_calibration(calibration_file)
    calibrated = np.empty_like(adc, dtype=np.complex64)
    sample_ids = np.arange(params.num_adc_samples, dtype=np.float64)[:, None]
    reference_tx = params.tx_to_enable[0] - 1
    for tx_slot, tx_id in enumerate(params.tx_to_enable):
        tx = tx_id - 1
        frequency = (
            (range_matrix[tx, :] - range_matrix[reference_tx, 0])
            * params.fs_calib
            / params.adc_sample_rate
            * params.chirp_slope
            / params.slope_calib
        )
        frequency *= 2.0 * np.pi / (
            params.num_adc_samples * params.calibration_interp
        )
        frequency_correction = np.exp(-1j * sample_ids * frequency)[:, None, :]
        phase = peak_matrix[reference_tx, 0] / peak_matrix[tx, :]
        if params.phase_calib_only:
            magnitude = np.abs(phase)
            if np.any(magnitude == 0.0):
                raise ValueError("RT-Pose calibration contains a zero phase magnitude")
            phase = phase / magnitude
        calibrated[:, :, :, tx_slot] = (
            adc[:, :, :, tx_slot]
            * frequency_correction
            * phase[None, None, :]
        )
    return calibrated


def build_measured_adc_cube(
    dataset_dir: Path,
    sequence: str,
    radar_frame_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build one calibrated ``[samples, loops, rx, tx]`` measured ADC cube."""

    data_folder = dataset_dir / f"Data/sequences/{sequence}/radar/bin"
    file_index = capture_file_index(data_folder)
    files = cascade_files(data_folder, file_index)
    frame_index = int(radar_frame_id)
    frame_count = valid_frame_count(files["master_idx"])
    if frame_index < 1 or frame_index > frame_count:
        raise ValueError(
            f"Radar frame {frame_index} is outside the measured capture range "
            f"[1, {frame_count}]"
        )
    params = RadarParams()
    adc = read_adc_cube(files, frame_index, params)
    calibration_file = (
        dataset_dir
        / "RT-POSE/data_processing/mmWave-Matlab/main/cascade/input/"
        "calibrateResults_high.mat"
    )
    adc = apply_calibration(adc, calibration_file, params)
    reorder = np.asarray(params.rx_for_mimo_process, dtype=np.int64) - 1
    adc = np.asarray(adc[:, :, reorder, :], dtype=np.complex64)
    metadata = {
        "sequence": sequence,
        "radar_frame_id": radar_frame_id,
        "stage": "adc",
        "layout": ["adc_sample", "chirp_loop", "rx_mimo_order", "tx_slot"],
        "rx_order": list(params.rx_for_mimo_process),
        "calibrated": True,
        "capture_file_index": file_index,
        "range_resolution_m": params.range_resolution,
        "velocity_resolution_mps": params.velocity_resolution,
        "params": asdict(params),
    }
    return adc, metadata
