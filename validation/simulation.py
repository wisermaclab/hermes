# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Bundle-aware simulation, motion and physical TDM acquisition."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from typing import Callable
import xml.etree.ElementTree as ET
import numpy as np
from mmWaveRadar.targets.mesh import MeshSequenceView, PlacedMeshSequence
from mmWaveRadar.naming import HybridCalibration, reject_removed_arguments, validate_solver
from mmWaveRadar.simulation.types import DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD
from bundle_prepare.contract import Bundle as BenchmarkClip
from .settings import simulator_arguments
from . import channels as _channels
from . import configuration as _configuration


def simulate_bundle_adc(
    clip: BenchmarkClip,
    *,
    solver: str,
    smpl_model_dir: str | None = None,
    channel_zero_only: bool = False,
    rt_refresh_frames: int = 1,
    po_refresh_on_phase_error_rad: float = DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD,
    rt_diffuse_reflection: bool = True,
    rt_refraction: bool = False,
    rt_human_specular_reflection: bool = False,
    hybrid_calibration: HybridCalibration = "rt_first_pose",
    cancel_check: Callable[[], bool] | None = None,
    solver_settings: dict | None = None,
    **removed,
) -> np.ndarray:
    """Simulate one ADC cube directly from a validated measurement bundle."""

    reject_removed_arguments(removed, where="simulate_bundle_adc")
    solver = validate_solver(solver)
    selected_tx_indices = None
    selected_rx_indices = None
    if channel_zero_only:
        sensor = clip.sensor()
        channel_tx, channel_rx = _channels.channel_tx_rx_indices_1based(
            clip,
            sensor,
            full_channels=int(np.asarray(clip.real_adc).shape[3]),
        )
        selected_tx_indices = (int(channel_tx[0]) - 1,)
        selected_rx_indices = (int(channel_rx[0]) - 1,)
    config = _configuration.BenchmarkRunConfig(
        write_plots=False,
        simulate=True,
        solvers=(solver,),
        solver_settings={} if solver_settings is None else dict(solver_settings),
        rt_refresh_frames=rt_refresh_frames,
        po_refresh_on_phase_error_rad=po_refresh_on_phase_error_rad,
        rt_diffuse_reflection=rt_diffuse_reflection,
        rt_refraction=rt_refraction,
        rt_human_specular_reflection=rt_human_specular_reflection,
        hybrid_calibration=hybrid_calibration,
        smpl_model_dir=smpl_model_dir,
        selected_tx_indices=selected_tx_indices,
        selected_rx_indices=selected_rx_indices,
        progress=False,
        target_material=dict(
            ((getattr(clip, "descriptor", None) or {}).get("target") or {}).get(
                "material_parameters", {}
            )
        ),
    )
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError("cancel_check must be callable or None")
    if cancel_check is None:
        return simulate_adc(clip, config, solver=solver)
    return simulate_adc(clip, config, solver=solver, cancel_check=cancel_check)


def simulate_adc(
    clip: BenchmarkClip,
    config: _configuration.BenchmarkRunConfig,
    *,
    solver: str,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    if cancel_check is not None and bool(cancel_check()):
        raise InterruptedError("Simulation cancelled by user")
    if clip_uses_tdm_virtual_adc(clip):
        if config.ignore_tdm_timing:
            return simulate_simultaneous_tdm_adc(
                clip,
                config,
                solver=solver,
                cancel_check=cancel_check,
            )
        return simulate_tdm_virtual_adc(
            clip,
            config,
            solver=solver,
            cancel_check=cancel_check,
        )

    from sionna.rt import load_scene

    from mmWaveRadar.materials import human_skin_material
    from mmWaveRadar.targets import MeshTarget

    scene_path = resolve_scene_path(clip, config)
    scene = load_bundle_scene(clip, config)
    motion = simulation_mesh_sequence(clip, config)
    target = MeshTarget(
        name="benchmark_human",
        mesh_sequence=BundleFrameMeshSequence(
            motion,
            frame_times_s=clip.motion_times_s,
            frame_period_s=float(clip.sensor_config.fmcw.frame_period),
        ),
        material=target_material(config),
    )
    simulator = validation_simulator(config, solver, cancel_check)
    cube = run_observed_simulator(
        simulator, scene, clip.sensor(), target, clip.num_frames, config, solver
    )
    selection = _channels.subarray_selection(
        clip,
        config,
        full_channels=int(np.asarray(clip.real_adc).shape[3]),
    )
    return _channels.select_adc_channels(np.asarray(cube.adc), selection)


def resolve_scene_path(
    clip: BenchmarkClip, config: _configuration.BenchmarkRunConfig
) -> str | None:
    """Return the user override or the scene resolved and checked by loading."""

    value = config.scene_path
    if value is None and clip.scene_path is not None:
        value = clip.scene_path
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((Path(clip.root) / path).resolve())


def clip_uses_tdm_virtual_adc(clip: BenchmarkClip) -> bool:
    fmcw = getattr(clip.sensor_config, "fmcw", None)
    if fmcw is not None and hasattr(fmcw, "tx_schedule"):
        return fmcw.tx_schedule == "tdm"
    metadata = clip.sensor_config.metadata or {}
    if bool(metadata.get("tdm_virtual_adc", False)):
        return True
    source_axes = metadata.get("source_adc_axes")
    if source_axes == ["adc_sample", "chirp_loop", "rx", "tx"]:
        return True
    return False


def load_bundle_scene(clip: BenchmarkClip, config: _configuration.BenchmarkRunConfig):
    """Load the environment without a legacy copy of the injected target.

    Old GUI envelopes embedded target.obj in the scene as well as declaring it
    separately. Filter that exact resource reference, keeping the source bundle
    intact and resolving remaining assets against the original scene directory.
    """
    from sionna.rt import load_scene
    from bundle_prepare.paths import safe_bundle_file

    value = resolve_scene_path(clip, config)
    if value is None:
        return load_scene()
    scene_path = Path(value).resolve()
    if (
        clip.scene_path is None
        or scene_path != Path(clip.scene_path).resolve()
        or clip.target_obj_path is None
    ):
        return load_scene(str(scene_path))
    tree = ET.parse(scene_path)
    target = Path(clip.target_obj_path).resolve()
    removed = False
    for parent in tree.iter():
        for shape in list(parent):
            if shape.tag != "shape":
                continue
            filename = shape.find("./string[@name='filename']")
            if filename is None:
                continue
            path = (scene_path.parent / filename.get("value", "")).resolve()
            if not path.exists():
                path = (Path(clip.root) / filename.get("value", "")).resolve()
            if path == target:
                parent.remove(shape)
                removed = True
    if not removed:
        return load_scene(str(scene_path))
    for node in tree.iter("string"):
        if node.get("name") != "filename":
            continue
        path = (scene_path.parent / node.get("value", "")).resolve()
        if not path.exists():
            path = (Path(clip.root) / node.get("value", "")).resolve()
        relative = path.relative_to(Path(clip.root).resolve()).as_posix()
        node.set("value", str(safe_bundle_file(Path(clip.root), relative, label="scene asset")))
    with tempfile.TemporaryDirectory(prefix="hermes-environment-") as directory:
        filtered = Path(directory) / "scene.xml"
        tree.write(filtered, encoding="utf-8", xml_declaration=True)
        return load_scene(str(filtered))


def tdm_tx_order(clip: BenchmarkClip) -> np.ndarray:
    return clip.sensor_config.slot_order()


def simulate_tdm_virtual_adc(
    clip: BenchmarkClip,
    config: _configuration.BenchmarkRunConfig,
    *,
    solver: str,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Simulates real TDM-MIMO timing and reassembles virtual-channel ADC.

    The standardized RT-Pose ADC bundle stores one frame as
    ``[loop, sample, tx_major_virtual_channel]`` after separating the physical
    chirp stream into TX slots. For loop ``l`` and scheduled TX slot ``s``, the
    physical chirp time is
    ``clip.motion_times_s[frame] + (l*num_tx + s)*T_chirp``. This helper
    evaluates the moving mesh at that physical time, keeps only the active TX
    group's requested RX outputs, and writes them back into the virtual-channel
    cube. The output shape remains compatible with real ADC maps.

    This is the default TDM policy. It invokes the selected solver once
    per selected physical TX slot so each run receives that slot's timestamps.
    """

    from sionna.rt import load_scene

    from mmWaveRadar.materials import human_skin_material
    from mmWaveRadar.radar import FMCWConfig, RadarSensor
    from mmWaveRadar.targets import MeshTarget

    fmcw = clip.sensor_config.fmcw
    sensor = clip.sensor_config.to_sensor(full_array=True)
    tx_order = tdm_tx_order(clip)
    selection = _channels.subarray_selection(
        clip,
        config,
        full_channels=int(np.asarray(clip.real_adc).shape[3]),
    )
    num_frames = clip.num_frames
    num_loops = int(fmcw.num_chirps_per_frame)
    num_tx = int(fmcw.num_tx)
    num_rx = int(sensor.hardware.num_rx)
    num_adc = int(fmcw.num_adc_samples)
    slot_rx_outputs = tdm_slot_rx_outputs(
        selection,
        tx_order=tx_order,
        hardware=sensor.hardware,
    )
    out = np.zeros(
        (num_frames, num_loops, num_adc, selection.channel_count),
        dtype=np.complex128,
    )
    mesh_sequence = simulation_mesh_sequence(clip, config)

    for slot, tx_index in enumerate(tx_order):
        if cancel_check is not None and bool(cancel_check()):
            raise InterruptedError("Simulation cancelled by user")
        selected_outputs = slot_rx_outputs.get(int(slot))
        if not selected_outputs:
            continue
        tx_index_int = int(tx_index)
        slot_times = np.empty((num_frames, num_loops), dtype=float)
        for frame in range(num_frames):
            for loop in range(num_loops):
                slot_times[frame, loop] = float(clip.motion_times_s[frame]) + (
                    loop * num_tx + slot
                ) * float(fmcw.chirp_repetition_time)
        slot_sequence = TimedMeshSequence(mesh_sequence, slot_times.reshape(-1))
        slot_fmcw = FMCWConfig(
            carrier_frequency=fmcw.carrier_frequency,
            slope=fmcw.slope,
            chirp_duration=fmcw.chirp_duration,
            # Mesh lookup for this internal per-slot run uses synthetic flat
            # indices; the wrapper maps them back to physical TDM times.
            # ADC synthesis itself does not depend on chirp repetition time.
            chirp_repetition_time=1.0,
            sampling_frequency=fmcw.sampling_frequency,
            adc_mode=fmcw.adc_mode,
            if_bandwidth_hz=fmcw.if_bandwidth_hz,
            num_adc_samples=fmcw.num_adc_samples,
            num_chirps_per_frame=fmcw.num_chirps_per_frame,
            frame_period=float(num_loops),
            num_tx=1,
        )
        slot_sensor = RadarSensor(
            hardware=sensor.hardware.subset_tx([tx_index_int]),
            fmcw=slot_fmcw,
            name=f"{sensor.name}-tdm-tx{tx_index_int + 1}",
            position=sensor.position,
            orientation=sensor.orientation,
            tx_power_dbm=sensor.tx_power_dbm,
        )
        scene_path = resolve_scene_path(clip, config)
        scene = load_bundle_scene(clip, config)
        target = MeshTarget(
            name=f"benchmark_human_tdm_tx{tx_index_int + 1}",
            mesh_sequence=slot_sequence,
            material=target_material(config),
        )
        slot_config = config
        if config.effective_solver_settings["array_model"] == "subarray":
            # The existing TDM adapter solves one physical transmitter at a time.
            slot_config = replace(
                config,
                solver_settings={
                    **config.solver_settings,
                    "array_tx_groups": [[0]],
                },
            )
        simulator = validation_simulator(slot_config, solver, cancel_check)
        cube = run_observed_simulator(
            simulator, scene, slot_sensor, target, num_frames, config, solver
        )
        slot_adc = np.asarray(cube.adc)
        if slot_adc.shape != (num_frames, num_loops, num_adc, num_rx):
            raise ValueError("TDM slot simulation returned unexpected shape " f"{slot_adc.shape}")
        for out_index, rx_index in selected_outputs:
            out[..., int(out_index)] = slot_adc[..., int(rx_index)]
    return out


def simulate_simultaneous_tdm_adc(
    clip: BenchmarkClip,
    config: _configuration.BenchmarkRunConfig,
    *,
    solver: str,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Simulates selected TDM channels together at each TDM-cycle start.

    This intentionally drops the per-slot transmit-time offsets while retaining
    the interval between successive samples of a physical TX. The resulting ADC
    channels are reordered to match the source bundle's TDM channel axis. All
    selected TX/RX pairs share one simulator invocation per solver.
    """

    from sionna.rt import load_scene

    from mmWaveRadar.materials import human_skin_material
    from mmWaveRadar.targets import MeshTarget

    selection = _channels.subarray_selection(
        clip,
        config,
        full_channels=int(np.asarray(clip.real_adc).shape[3]),
    )
    sensor, output_order = simultaneous_tdm_sensor(clip, selection)
    scene_path = resolve_scene_path(clip, config)
    scene = load_bundle_scene(clip, config)
    motion = simulation_mesh_sequence(clip, config)
    target = MeshTarget(
        name="benchmark_human_tdm_simultaneous",
        mesh_sequence=BundleFrameMeshSequence(
            motion,
            frame_times_s=clip.motion_times_s,
            frame_period_s=float(clip.sensor_config.fmcw.frame_period),
        ),
        material=target_material(config),
    )
    simulator = validation_simulator(config, solver, cancel_check)
    cube = run_observed_simulator(
        simulator, scene, sensor, target, clip.num_frames, config, solver
    )
    adc = np.asarray(cube.adc)
    expected_shape = (
        clip.num_frames,
        int(clip.sensor_config.fmcw.num_chirps_per_frame),
        int(clip.sensor_config.fmcw.num_adc_samples),
        selection.channel_count,
    )
    if adc.shape != expected_shape:
        raise ValueError(
            "Simultaneous TDM simulation returned unexpected shape "
            f"{adc.shape}; expected {expected_shape}"
        )
    return adc[..., output_order]


def simultaneous_tdm_sensor(
    clip: BenchmarkClip,
    selection: _channels.SubarraySelection,
):
    """Builds selected simultaneous-TX hardware and its source-axis ordering."""

    from mmWaveRadar.radar import RadarSensor

    source = clip.sensor_config.to_sensor(full_array=True)
    tx_indices = (
        np.asarray(
            sorted(set(int(v) for v in selection.channel_tx_indices_1based)),
            dtype=np.int64,
        )
        - 1
    )
    rx_indices = (
        np.asarray(
            sorted(set(int(v) for v in selection.channel_rx_indices_1based)),
            dtype=np.int64,
        )
        - 1
    )
    hardware = source.hardware.subset_tx(tx_indices).subset_rx(rx_indices)

    # Each output chirp is one TDM loop. Preserve that loop-to-loop PRI while
    # removing only the offsets among TX slots inside the loop.
    fmcw = replace(
        clip.sensor_config.fmcw,
        chirp_repetition_time=clip.sensor_config.fmcw.slow_time_interval,
        num_tx=hardware.num_tx,
        tx_schedule="simultaneous",
    )
    sensor = RadarSensor(
        hardware=hardware,
        fmcw=fmcw,
        name=f"{source.name}-tdm-simultaneous",
        position=source.position,
        orientation=source.orientation,
        tx_power_dbm=source.tx_power_dbm,
    )

    simulated_pairs = list(
        zip(
            (tx_indices[hardware.virtual_tx_indices()] + 1).tolist(),
            (rx_indices[hardware.virtual_rx_indices()] + 1).tolist(),
        )
    )
    pair_to_index = {(int(tx), int(rx)): index for index, (tx, rx) in enumerate(simulated_pairs)}
    desired_pairs = zip(
        selection.channel_tx_indices_1based,
        selection.channel_rx_indices_1based,
    )
    output_order = np.asarray(
        [pair_to_index[(int(tx), int(rx))] for tx, rx in desired_pairs],
        dtype=np.int64,
    )
    return sensor, output_order


def tdm_slot_rx_outputs(
    selection: _channels.SubarraySelection,
    *,
    tx_order: np.ndarray,
    hardware,
) -> dict[int, list[tuple[int, int]]]:
    """Maps selected source channels to physical TDM slots and slot outputs."""

    slot_by_tx = {
        int(tx_index) + 1: slot for slot, tx_index in enumerate(np.asarray(tx_order).reshape(-1))
    }
    rx_output_by_tx: dict[int, dict[int, int]] = {}
    for tx_1based in slot_by_tx:
        slot_hardware = hardware.subset_tx([tx_1based - 1])
        rx_indices = slot_hardware.virtual_rx_indices() + 1
        rx_outputs = {
            int(rx_1based): output_index for output_index, rx_1based in enumerate(rx_indices)
        }
        if len(rx_outputs) != int(hardware.num_rx):
            raise ValueError(f"TDM Tx {tx_1based} does not map each receiver exactly once")
        rx_output_by_tx[tx_1based] = rx_outputs

    outputs: dict[int, list[tuple[int, int]]] = {}
    channel_pairs = zip(
        selection.channel_tx_indices_1based,
        selection.channel_rx_indices_1based,
    )
    for out_index, (tx_1based, rx_1based) in enumerate(channel_pairs):
        tx = int(tx_1based)
        rx = int(rx_1based)
        if tx not in slot_by_tx:
            raise ValueError(f"Selected TDM channel uses unscheduled Tx {tx}")
        if rx not in rx_output_by_tx[tx]:
            raise ValueError(f"Selected TDM channel uses unavailable Tx {tx}/Rx {rx} pair")
        slot = slot_by_tx[tx]
        rx_output = rx_output_by_tx[tx][rx]
        outputs.setdefault(slot, []).append((int(out_index), rx_output))
    return outputs


class TimedMeshSequence(MeshSequenceView):
    """Mesh sequence view sampled by benchmark-frame timestamps."""

    def __init__(self, base, times_s: np.ndarray):
        self._base = base
        self._times_s = np.asarray(times_s, dtype=float).reshape(-1)
        self.faces = np.asarray(base.faces, dtype=np.uint32)
        self.times = np.arange(self._times_s.size, dtype=float)

    def vertices_at(self, time: float) -> np.ndarray:
        """Returns vertices for the nearest benchmark-frame index."""

        index = int(round(float(time)))
        index = min(max(index, 0), self._times_s.size - 1)
        return self._base.vertices_at(float(self._times_s[index]))


class BundleFrameMeshSequence(MeshSequenceView):
    """Map simulator-local frame/chirp time to explicit bundle motion time."""

    def __init__(
        self,
        base,
        *,
        frame_times_s: np.ndarray,
        frame_period_s: float,
    ):
        self._base = base
        self._frame_times_s = np.asarray(frame_times_s, dtype=float).reshape(-1)
        self._frame_period_s = float(frame_period_s)
        if self._frame_times_s.size == 0 or not np.all(np.isfinite(self._frame_times_s)):
            raise ValueError("bundle frame motion times must be non-empty and finite")
        if not np.isfinite(self._frame_period_s) or self._frame_period_s <= 0.0:
            raise ValueError("bundle frame period must be positive and finite")
        self.faces = np.asarray(base.faces, dtype=np.uint32)
        self.times = np.arange(self._frame_times_s.size, dtype=float) * self._frame_period_s

    def _source_time(self, simulation_time_s: float) -> float:
        local_time = max(0.0, float(simulation_time_s))
        frame = min(
            int(np.floor(local_time / self._frame_period_s + 1.0e-12)),
            self._frame_times_s.size - 1,
        )
        frame_local_start = frame * self._frame_period_s
        return float(self._frame_times_s[frame] + local_time - frame_local_start)

    def vertices_at(self, time: float) -> np.ndarray:
        return np.asarray(
            self._base.vertices_at(self._source_time(time)),
            dtype=np.float32,
        )


def simulation_mesh_sequence(clip: BenchmarkClip, config: _configuration.BenchmarkRunConfig):
    from mmWaveRadar.targets import MeshSequence
    from bundle_prepare.contract import load_static_target_mesh

    if clip.resimulation_issues:
        raise ValueError("Cannot re-simulate: " + "; ".join(clip.resimulation_issues))

    if clip.motion_path is not None:
        motion = load_amass_motion_sequence(clip.motion_path, config)
    elif clip.evaluated_motion_path is not None:
        motion = MeshSequence.from_npz(clip.evaluated_motion_path)
    elif clip.target_mesh_path is not None:
        vertices, faces = load_static_target_mesh(clip.target_mesh_path)
        fmcw = clip.sensor_config.fmcw
        stop = float(np.max(clip.motion_times_s)) + fmcw.frame_period
        motion = MeshSequence(
            np.stack([vertices, vertices]),
            faces,
            np.array([float(np.min(clip.motion_times_s)), stop]),
        )
    elif clip.target_obj_path is not None:
        from mmWaveRadar.experiments.physics_microscope import load_human_mesh

        vertices, faces = load_human_mesh(clip.target_obj_path)
        stop = float(np.max(clip.motion_times_s)) + clip.sensor_config.fmcw.frame_period
        motion = MeshSequence(
            np.stack([vertices, vertices]),
            faces,
            np.array([float(np.min(clip.motion_times_s)), stop]),
        )
    else:
        raise ValueError(
            "Simulation requires motion.parameters, motion.evaluated_mesh or target geometry"
        )
    environment = clip.environment or {}
    if clip.motion_path is not None and "human_position_m" in environment:
        motion = PlacedBundleMotionSequence(
            motion,
            position=environment["human_position_m"],
            yaw_deg=environment.get("human_yaw_deg", 0.0),
        )
    validate_motion_acquisition_coverage(clip, motion)
    return motion


PlacedBundleMotionSequence = PlacedMeshSequence  # Keep native bundle timestamps.


def validate_motion_acquisition_coverage(clip: BenchmarkClip, motion) -> None:
    """Require motion coverage through the final physical chirp in each frame."""

    from bundle_prepare.contract import validate_acquisition_coverage

    validate_acquisition_coverage(clip.motion_times_s, motion.times, clip.sensor_config.fmcw)


def solver_cache_settings(
    config: _configuration.BenchmarkRunConfig,
    solver: str,
) -> dict[str, object]:
    """Settings that change one solver's ADC; others are not applicable."""

    runs_rt = solver in ("rt", "hybrid_po")
    return {
        "resolved_solver_settings": json.dumps(
            config.effective_solver_settings, sort_keys=True, separators=(",", ":")
        ),
        "target_material": json.dumps(config.target_material, sort_keys=True),
        "frozen_human_po_amplitude_gain": (
            "none"
            if config.frozen_human_po_amplitude_gain is None
            else config.frozen_human_po_amplitude_gain
        ),
        "rt_refresh_frames": (int(config.rt_refresh_frames) if solver == "rt" else -1),
        "rt_diffuse_reflection": runs_rt and bool(config.rt_diffuse_reflection),
        "po_refresh_on_phase_error_rad": (
            float(config.po_refresh_on_phase_error_rad) if solver != "rt" else "not_applicable"
        ),
        "rt_refraction": runs_rt and bool(config.rt_refraction),
        "rt_human_specular_reflection": runs_rt and bool(config.rt_human_specular_reflection),
        "hybrid_calibration": (
            str(config.hybrid_calibration) if solver == "hybrid_po" else "not_applicable"
        ),
    }


def validation_simulator(
    config: _configuration.BenchmarkRunConfig,
    solver: str,
    cancel_check: Callable[[], bool] | None = None,
):
    """Build the simulator one validation solver run uses."""

    from mmWaveRadar.simulation import HybridPOConfig, MmWaveRadarSimulator

    arguments = simulator_arguments(config.effective_solver_settings)
    if solver != "hybrid_po":
        arguments["hybrid_config"] = None
    return MmWaveRadarSimulator(
        solver=solver,
        **arguments,
        progress=config.progress,
        cancel_check=cancel_check,
    )


def target_material(config):
    """Build a declared target material without altering solver physics."""

    if not config.target_material:
        from mmWaveRadar.materials import human_skin_material

        return human_skin_material("benchmark-human-skin")
    from sionna.rt import RadioMaterial

    return RadioMaterial(name="benchmark-target-material", **config.target_material)


def benchmark_cube(cube, config, solver):
    """Observe native diagnostics and optionally scale only PO-derived terms."""

    gain = config.frozen_human_po_amplitude_gain
    if gain is not None and solver != "rt":
        if solver == "po":
            adc = cube.adc * gain
        else:
            components = cube.components or {}
            required = ("human_po", "human_env", "env_human")
            if not all(name in components for name in required):
                raise ValueError("Frozen Hybrid gain requires native human/coupling components")
            adc = cube.adc + (gain - 1.0) * sum(components[name] for name in required)
        cube = replace(cube, adc=adc.astype(cube.adc.dtype))
    if config.cube_observer is not None:
        config.cube_observer(cube)
    return cube


def run_observed_simulator(simulator, scene, sensor, target, num_frames, config, solver):
    """Time the native public entry point, excluding bundle/scene preparation."""

    if config.cube_observer is None:
        return benchmark_cube(
            simulator.run(scene, sensor, [target], num_frames=num_frames), config, solver
        )
    from .study_runtime import synchronize

    synchronize()
    start = time.perf_counter()
    cube = simulator.run(scene, sensor, [target], num_frames=num_frames)
    synchronize()
    elapsed = time.perf_counter() - start
    cube = replace(
        cube,
        metadata=replace(
            cube.metadata,
            runtime_profile_s={
                **(cube.metadata.runtime_profile_s or {}),
                "validation_simulator_run": elapsed,
            },
        ),
    )
    return benchmark_cube(cube, config, solver)


def load_amass_motion_sequence(path: Path, config: _configuration.BenchmarkRunConfig):
    from mmWaveRadar.targets import AMASSSMPLMotionSequence

    with np.load(path, allow_pickle=False) as data:
        required = ("poses", "trans", "betas", "times", "faces")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"{path} missing required SMPL arrays: {missing}")

        poses = np.array(data["poses"], dtype=np.float32, copy=True)
        betas = np.array(data["betas"], dtype=np.float32, copy=True)
        translations = np.array(data["trans"], dtype=np.float32, copy=True)
        faces = np.array(data["faces"], dtype=np.uint32, copy=True)
        timing_key = "bundle_times" if "bundle_times" in data.files else "times"
        times = np.array(data[timing_key], dtype=float, copy=True)
        npz_model_dir = optional_npz_string(data, "smpl_model_dir")
        gender = optional_npz_string(data, "gender") or "neutral"
        model_type = optional_npz_string(data, "model_type") or "smpl"
        axes = (
            np.array(data["output_axes"], dtype=np.int64, copy=True)
            if "output_axes" in data.files
            else None
        )
        signs = (
            np.array(data["output_signs"], dtype=np.float32, copy=True)
            if "output_signs" in data.files
            else None
        )

    model_dir = config.smpl_model_dir or npz_model_dir
    if not model_dir:
        raise ValueError(
            f"{path} requires smpl_model_dir; pass it in config or set " "MMWAVE_SMPL_MODEL_DIR"
        )
    base = AMASSSMPLMotionSequence(
        poses=poses,
        betas=betas,
        translations=translations,
        faces=faces,
        times=times,
        smpl_model_dir=str(model_dir),
        model_type=str(model_type),
        gender=str(gender),
    )
    return TransformedSMPLMotionSequence(base, axes=axes, signs=signs)


class TransformedSMPLMotionSequence(MeshSequenceView):
    """SMPL mesh sequence with optional axis permutation and sign flips."""

    def __init__(self, base, *, axes: np.ndarray | None, signs: np.ndarray | None):
        from mmWaveRadar.targets.mesh import ensure_outward_face_winding

        self._base = base
        self._axes = axes
        self._signs = signs
        self.times = np.asarray(base.times, dtype=float)
        faces = np.asarray(base.faces, dtype=np.uint32)
        if self.times.size and hasattr(base, "vertices_at"):
            faces = ensure_outward_face_winding(
                self.vertices_at(float(self.times[0])),
                faces,
            )
        self.faces = faces

    def vertices_at(self, time: float) -> np.ndarray:
        """Returns transformed vertices at ``time`` in seconds."""

        vertices = self._base.vertices_at(float(time))
        if self._axes is not None:
            vertices = vertices[..., self._axes]
        if self._signs is not None:
            vertices = vertices * self._signs
        return np.asarray(vertices, dtype=np.float32)


def optional_npz_string(data, key: str) -> str | None:
    if key not in data.files:
        return None
    value = data[key]
    if np.asarray(value).shape == ():
        return str(value.item())
    return str(value)
