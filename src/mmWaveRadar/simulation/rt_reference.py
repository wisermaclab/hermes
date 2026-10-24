# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Component diagnostics collected from the simulator's RT execution."""

from dataclasses import dataclass, field, replace

import numpy as np

from .types import RadarCube


@dataclass(frozen=True)
class RTBaselineResult:
    """RT cube and coherent component references for PO calibration."""

    cube: RadarCube
    human_reference_adc: np.ndarray
    human_reference_path_counts: np.ndarray
    human_touch_path_counts: np.ndarray
    one_human_touch_path_counts: np.ndarray
    single_human_only_path_counts: np.ndarray
    human_env_coupled_path_counts: np.ndarray
    human_multi_touch_path_counts: np.ndarray
    path_depth_histogram: np.ndarray
    total_path_power: np.ndarray
    human_touch_path_power: np.ndarray
    one_human_touch_path_power: np.ndarray
    single_human_only_path_power: np.ndarray
    human_env_coupled_path_power: np.ndarray
    human_multi_touch_path_power: np.ndarray
    retrace_count: int
    reference_adc_power: dict[str, np.ndarray] = field(default_factory=dict)


class RTReferenceCollector:
    """Observe the same filtered, visibility-weighted paths that produce RT ADC."""

    def __init__(self, simulator, radar, num_frames, reference_component):
        self.simulator = simulator
        self.radar = radar
        self.reference_component = reference_component
        shape = (num_frames, radar.fmcw.num_chirps_per_frame)
        self.counts = {
            name: np.zeros(shape, dtype=np.int64)
            for name in (
                "human_touch",
                "one_human_touch",
                "single_human_only",
                "human_env_coupled",
                "human_multi_touch",
            )
        }
        self.powers = {name: np.zeros(shape) for name in ("total", *self.counts)}
        self.histogram = np.zeros((*shape, simulator.rt_max_depth + 1), dtype=np.int64)
        _, dtype = simulator.numpy_dtypes_for_precision(simulator.adc_compute_precision)
        self.human_adc = np.zeros(
            (shape[1], radar.fmcw.num_adc_samples, radar.hardware.num_virtual_channels),
            dtype=dtype,
        )
        self.human_counts = np.zeros(shape[1], dtype=np.int64)
        self.reference_power = (
            {}
            if reference_component is None
            else {reference_component: np.zeros((*shape, radar.hardware.num_virtual_channels))}
        )

    def observe(self, frame, chirp, coefficients, delays, valid, target_counts, total_counts):
        masks = {
            "human_touch": valid & (target_counts > 0),
            "one_human_touch": valid & (target_counts == 1),
            "single_human_only": valid & (target_counts == 1) & (total_counts == 1),
            "human_env_coupled": valid & (target_counts > 0) & (total_counts > target_counts),
            "human_multi_touch": valid & (target_counts > 1),
        }
        power = np.nan_to_num(
            np.sum(abs(np.asarray(coefficients, complex)) ** 2, axis=0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for name, mask in masks.items():
            self.counts[name][frame, chirp] = np.count_nonzero(mask)
            self.powers[name][frame, chirp] = power[mask].sum()
        self.powers["total"][frame, chirp] = power[valid].sum()
        bins = self.histogram.shape[-1]
        self.histogram[frame, chirp] = np.bincount(
            np.clip(total_counts[valid], 0, bins - 1), minlength=bins
        )[:bins]

        def synthesize(component):
            return self.simulator.synthesize_adc(
                coefficients, delays, masks[component], self.radar.fmcw
            )

        reference_adc = None
        if self.reference_component is not None:
            reference_adc = synthesize(self.reference_component)
            self.reference_power[self.reference_component][frame, chirp] = np.mean(
                abs(reference_adc.astype(complex)) ** 2, axis=0
            )
        if frame == 0:
            self.human_counts[chirp] = self.counts["single_human_only"][frame, chirp]
            self.human_adc[chirp] = (
                reference_adc
                if self.reference_component == "single_human_only"
                else synthesize("single_human_only")
            )

    def finish(self, cube):
        diagnostics = {
            **{f"{name}_path_counts": values for name, values in self.counts.items()},
            **{f"{name}_path_power": values for name, values in self.powers.items()},
            "path_depth_histogram": self.histogram,
        }

        # Preserve the established profiling names without retaining another RT loop.
        def profile_name(key):
            return key.replace("rt_path_bank_trace", "rt_baseline_trace").replace(
                "rt_path_bank_", "rt_baseline_path_bank_"
            )

        metadata = replace(
            cube.metadata,
            **diagnostics,
            adc_if_filter=dict(self.simulator.adc_if_filter),
            runtime_profile_s={
                profile_name(k): v for k, v in (cube.metadata.runtime_profile_s or {}).items()
            },
            runtime_profile_counts={
                profile_name(k): v for k, v in (cube.metadata.runtime_profile_counts or {}).items()
            },
        )
        cube = replace(cube, metadata=metadata)
        return RTBaselineResult(
            cube=cube,
            human_reference_adc=self.human_adc,
            human_reference_path_counts=self.human_counts,
            retrace_count=len(cube.metadata.events),
            reference_adc_power=self.reference_power,
            **diagnostics,
        )
