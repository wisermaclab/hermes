# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Validated public solver controls for measurement and tradeoff benchmarks."""

from dataclasses import fields
import inspect
import math

from mmWaveRadar.naming import validate_array_model, validate_array_groups, validate_frames
from mmWaveRadar.simulation import HybridPOConfig, MmWaveRadarSimulator, POConfig

PO_KEYS = frozenset(
    f.name
    for f in fields(POConfig)
    if not (f.name.startswith("precomputed_") or f.name == "progress_callback")
)
HYBRID_KEYS = frozenset(f.name for f in fields(HybridPOConfig) if f.name != "po_config")
SIM_DEFAULTS = {
    name: p.default
    for name, p in inspect.signature(MmWaveRadarSimulator).parameters.items()
    if name.startswith("rt_")
    and name != "rt_coherent_transition_config"
    or name
    in {
        "array_model",
        "array_tx_groups",
        "array_rx_groups",
        "compute_backend",
        "compute_precision",
        "adc_compute_backend",
        "adc_compute_precision",
    }
}


def resolve_settings(overrides):
    """Expand every public numeric setting; reject typos before expensive runs."""

    if not isinstance(overrides, dict):
        raise ValueError("solver_settings must be an object")
    unknown = set(overrides) - PO_KEYS - HYBRID_KEYS - SIM_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown solver settings: {', '.join(sorted(unknown))}")
    po = POConfig(**{k: v for k, v in overrides.items() if k in PO_KEYS})
    hybrid = HybridPOConfig(
        **{k: v for k, v in overrides.items() if k in HYBRID_KEYS}, po_config=po
    )
    resolved = {**SIM_DEFAULTS, **overrides}
    resolved.update({k: getattr(po, k) for k in PO_KEYS if k not in SIM_DEFAULTS})
    resolved.update({k: getattr(hybrid, k) for k in HYBRID_KEYS})
    resolved["po_calibration"] = po.po_calibration.mode
    resolved["hybrid_calibration"] = hybrid.hybrid_calibration.mode
    resolved["array_model"] = validate_array_model(resolved["array_model"])
    for key in ("array_tx_groups", "array_rx_groups"):
        resolved[key] = validate_array_groups(resolved[key], name=key)
        if resolved[key] is not None and resolved["array_model"] != "subarray":
            raise ValueError(f"{key} requires array_model='subarray'")
    for key in ("rt_refresh_frames", "rt_visibility_refresh_frames", "rt_visibility_fade_frames"):
        resolved[key] = validate_frames(resolved[key], name=key)
    for key in ("rt_samples_per_source", "rt_max_paths_per_source", "rt_max_depth", "rt_seed"):
        value = resolved[key]
        minimum = 1 if key in {"rt_samples_per_source", "rt_max_paths_per_source"} else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in ("compute_backend", "adc_compute_backend"):
        if resolved[key] not in (
            {"auto", "numpy", "torch", None}
            if key.startswith("adc_")
            else {"auto", "numpy", "torch"}
        ):
            raise ValueError(f"Invalid {key}")
    for key in ("compute_precision", "adc_compute_precision"):
        if resolved[key] not in (
            {"float32", "float64", None} if key.startswith("adc_") else {"float32", "float64"}
        ):
            raise ValueError(f"Invalid {key}")
    if resolved["rt_human_bounces"] not in {"single", "any"}:
        raise ValueError("rt_human_bounces must be single or any")
    for key in ("rt_diffuse_reflection", "rt_refraction", "rt_human_specular_reflection"):
        if not isinstance(resolved[key], bool):
            raise ValueError(f"{key} must be boolean")
    drift = resolved["rt_refresh_on_displacement"]
    if drift is not None and (
        isinstance(drift, bool) or not math.isfinite(float(drift)) or float(drift) <= 0
    ):
        raise ValueError("rt_refresh_on_displacement must be positive metres or null")
    return resolved


def simulator_arguments(settings):
    """Build the same PO/Hybrid configuration objects used by public callers."""

    po = POConfig(**{k: v for k, v in settings.items() if k in PO_KEYS})
    hybrid = HybridPOConfig(
        **{k: v for k, v in settings.items() if k in HYBRID_KEYS}, po_config=po
    )
    return {**{k: settings[k] for k in SIM_DEFAULTS}, "po_config": po, "hybrid_config": hybrid}
