# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Image-source adapter for the shared human PO incremental updater."""

from dataclasses import dataclass, field

import numpy as np

from .physical_optics import _phase_center_delays_for_centroids
from .po_incremental import POFacetState, update_po_facets, vertex_path_offsets


@dataclass
class CouplingIncrementalState:
    """Sequence-local raw surface fields per reflector and path direction.

    Array, hardware, and calibration gains stay outside the cached surface
    term, just as in direct human PO. Visibility remains the caller's job.
    """

    facets: dict[tuple[str, int], POFacetState] = field(default_factory=dict)
    signature: tuple | None = None
    chirp_index: int = 0
    force_refresh: bool = False
    recomputed_faces: int = 0
    phase_updated_faces: int = 0
    active_keys: set[tuple[str, int]] = field(default_factory=set)

    def begin_chirp(self, signature, full_refresh_chirps):
        """Invalidate fields when fixed solver inputs or the schedule change."""

        if self.signature != signature:
            self.facets.clear()
            self.signature = signature
            self.chirp_index = 0
        self.force_refresh = bool(
            full_refresh_chirps and self.chirp_index % full_refresh_chirps == 0
        )
        self.recomputed_faces = 0
        self.phase_updated_faces = 0
        self.active_keys.clear()
        self.chirp_index += 1

    def end_chirp(self):
        """Drop invisible reflector states so reappearing facets solve exactly."""

        self.facets = {key: value for key, value in self.facets.items() if key in self.active_keys}

    def evaluate(self, *, integrate, cache_key, **kwargs):
        """Supply image geometry and reflected fields to the human PO updater."""

        selected = np.asarray(kwargs["face_indices"], dtype=np.int64)
        self.active_keys.add(cache_key)
        centroids = kwargs["centroids"][selected]
        tx, rx = kwargs["tx_position"], kwargs["rx_position"]
        tx_dist = np.linalg.norm(centroids - tx, axis=1)
        rx_dist = np.linalg.norm(centroids - rx, axis=1)
        lengths = tx_dist + rx_dist
        spreads = np.maximum(tx_dist * rx_dist, 1e-18)
        response = kwargs.get("delay_response")
        path_indices = kwargs.get("path_offset", 0) + selected
        if response is not None:
            response.set_reference(path_indices, centroids, tx, rx)

        def exact(positions):
            values, _ = integrate(**{**kwargs, "face_indices": selected[positions]})
            return values[None], values

        config = kwargs["po_config"]
        state, recomputed = update_po_facets(
            previous=self.facets.get(cache_key),
            selected=selected,
            vertex_offsets=vertex_path_offsets(
                kwargs["vertices"][kwargs["faces"][selected]], tx, rx
            ),
            path_lengths=lengths[None],
            spreads=spreads[None],
            phase_lengths=lengths,
            phase_spreads=spreads,
            weights=np.ones(selected.size),
            frequency_hz=kwargs["frequency_hz"],
            phase_error_threshold_rad=config.po_refresh_on_phase_error_rad,
            area_fraction_threshold=config.po_refresh_on_area_fraction,
            normal_threshold_deg=config.po_refresh_on_normal_deg,
            delay_response=response,
            response_indices=path_indices,
            integrate=exact,
            force_recompute=self.force_refresh,
            areas=kwargs["areas"][selected],
            normals=kwargs["normals"][selected],
            dtype=np.complex64 if kwargs["precision"] == "float32" else np.complex128,
        )
        self.facets[cache_key] = state
        self.recomputed_faces += int(recomputed.size)
        self.phase_updated_faces += int(selected.size - recomputed.size)
        return state.phase_center_contributions, _phase_center_delays_for_centroids(
            centroids, tx, rx, kwargs["precision"]
        )
