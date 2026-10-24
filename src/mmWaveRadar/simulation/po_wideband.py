# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Compact delay-resolved surface integrals for wideband FMCW synthesis."""

from dataclasses import dataclass, replace, field
import math

import numpy as np
from scipy.constants import c
from scipy.special import gammaln
from ..radar.adc import max_if_delay_s, record_if_filter, adc_mode
from .compute import numpy_dtypes_for_precision, resolve_torch_runtime


def _clipped_delay_triangles(vertex_delays, limit):
    """Clip the reference simplex by affine delay; return fan triangles."""
    polygon = [
        (np.array([0.0, 0.0]), vertex_delays[0]),
        (np.array([1.0, 0.0]), vertex_delays[1]),
        (np.array([0.0, 1.0]), vertex_delays[2]),
    ]
    clipped = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        inside_a, inside_b = a[1] <= limit, b[1] <= limit
        if inside_a:
            clipped.append(a)
        if inside_a != inside_b:
            fraction = (limit - a[1]) / (b[1] - a[1])
            clipped.append((a[0] + fraction * (b[0] - a[0]), limit))
    triangles = []
    for i in range(1, len(clipped) - 1):
        a, b, d = clipped[0], clipped[i], clipped[i + 1]
        edge1, edge2 = b[0] - a[0], d[0] - a[0]
        determinant = abs(edge1[0] * edge2[1] - edge1[1] * edge2[0])
        triangles.append((determinant, a[1], b[1], d[1]))
    return triangles


@dataclass
class PODelayResponse:
    """Surface response relative to each parent path's geometric delay.

    Chebyshev nodes span the actual FMCW frequency offsets. Only the within-
    face delay response is interpolated; the large parent delay is factored
    out. Material/current amplitudes stay at the carrier, as in RT synthesis.
    Channel gains carry the existing synthetic-array projection and hardware.
    Delay bounds classify full/partial IF rejection per channel. Only parents
    near the receiver edge retain leaf geometry for analytic domain clipping.
    Without a receiver config, leaves are retained for any later cutoff.
    """

    frequencies_hz: np.ndarray
    values: np.ndarray
    reference_delays_s: np.ndarray
    channel_gains: np.ndarray | None = None
    min_delta_s: np.ndarray | None = None
    max_delta_s: np.ndarray | None = None
    retained_paths: np.ndarray | None = None
    leaf_batches: list = field(default_factory=list)
    delay_bound_s: float = 0.0
    receiver_fmcw: object = None
    channel_delay_margin_s: float = 0.0

    @classmethod
    def for_mesh(
        cls,
        vertices,
        faces,
        num_paths,
        max_frequency_offset_hz,
        *,
        receiver_fmcw=None,
        channel_delay_margin_s=0.0,
        previous=None,
        compute_precision="float64",
    ):
        """Choose nodes at a 1e-8 (float32) or 1e-14 (float64) tail tolerance."""
        float_dtype, _ = numpy_dtypes_for_precision(compute_precision)
        interpolation_tolerance = 1e-8 if float_dtype == np.float32 else 1e-14
        triangles = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int)]
        max_edge = max(
            (
                float(
                    np.max(np.linalg.norm(triangles[:, i] - triangles[:, j], axis=1), initial=0.0)
                )
                for i, j in ((0, 1), (1, 2), (2, 0))
            ),
            default=0.0,
        )
        # Exact sample paths differ by at most 2*max_edge. Allow 4*max_edge
        # also to bound the analytic mode's center shift plus its linear
        # phase within a subtriangle, for any real/image endpoints.
        delay_bound = 4.0 * max_edge / c
        bandwidth = abs(float(max_frequency_offset_hz))
        x = np.pi * bandwidth * delay_bound
        # Bound the Chebyshev tail of exp(-j*x*y), |y|<=1, using
        # |J_n(x)| <= (x/2)^n/n! * exp(x*x/(4*(n+1))). Keep the omitted
        # tail below the precision-specific tolerance times the sum of sample
        # magnitudes, before the modest interpolation Lebesgue factor.
        order = max(8, int(math.ceil(x)))
        if x > 0.0:
            while True:
                ratio = x / (2.0 * (order + 1.0))
                log_bound = (
                    math.log(2.0)
                    + order * math.log(x / 2.0)
                    - gammaln(order + 1.0)
                    + x * x / (4.0 * (order + 1.0))
                )
                if ratio < 1.0 and log_bound - math.log1p(-ratio) < math.log(
                    interpolation_tolerance
                ):
                    break
                order += 1
        # Retain an adequate grid and reserve headroom on growth. This avoids
        # invalidating every cached facet when the required order fluctuates.
        if (
            previous is not None
            and len(previous.frequencies_hz) - 1 >= order
            and previous.frequencies_hz[-1] == max_frequency_offset_hz
        ):
            order = len(previous.frequencies_hz) - 1
        else:
            order += 2
        nodes = 0.5 * bandwidth * (1.0 - np.cos(np.linspace(0.0, np.pi, order + 1)))
        if max_frequency_offset_hz < 0.0:
            nodes = -nodes
        return cls(
            nodes,
            np.zeros((len(nodes), num_paths), complex),
            np.zeros(num_paths, dtype=float),
            min_delta_s=np.full(num_paths, np.inf),
            max_delta_s=np.full(num_paths, -np.inf),
            retained_paths=np.ones(num_paths, dtype=bool),
            delay_bound_s=delay_bound,
            receiver_fmcw=receiver_fmcw,
            channel_delay_margin_s=channel_delay_margin_s,
        )

    def set_reference(self, indices, centroids, tx, rx):
        self.reference_delays_s[indices] = (
            np.linalg.norm(centroids - tx, axis=1) + np.linalg.norm(centroids - rx, axis=1)
        ) / c
        if self.receiver_fmcw is not None:
            margin = self.delay_bound_s + self.channel_delay_margin_s
            self.retained_paths[indices] = (
                abs(self.reference_delays_s[indices] - max_if_delay_s(self.receiver_fmcw))
                <= margin
            )

    def accumulate(
        self,
        coefficients,
        delays_s,
        path_indices,
        *,
        phase_integral=None,
        phase_integrals=None,
        vertex_delay_offsets_s=None,
        carrier_frequency_hz=0.0,
        backend="numpy",
        precision="float64",
        runtime=None,
    ):
        coefficients = np.asarray(coefficients)
        delta = np.asarray(delays_s) - self.reference_delays_s[path_indices]
        offsets = vertex_delay_offsets_s
        if (phase_integral is not None or phase_integrals is not None) and offsets is None:
            raise ValueError("Analytic delay responses require vertex delay offsets")
        low = delta if offsets is None else delta + np.min(offsets, axis=1)
        high = delta if offsets is None else delta + np.max(offsets, axis=1)
        np.minimum.at(self.min_delta_s, path_indices, low)
        np.maximum.at(self.max_delta_s, path_indices, high)
        keep = self.retained_paths[path_indices]
        if np.any(keep):
            self.leaf_batches.append(
                (
                    np.array(path_indices[keep]),
                    np.array(coefficients[keep]),
                    np.array(delta[keep]),
                    None if offsets is None else np.array(offsets[keep]),
                    float(carrier_frequency_hz),
                )
            )
        runtime = resolve_torch_runtime(backend, precision) if runtime is None else runtime
        if runtime is not None:
            self._accumulate_torch(
                coefficients, delta, path_indices, phase_integral, phase_integrals, runtime
            )
            return
        # Batch frequency nodes while bounding temporary [node,leaf] buffers.
        node_chunk = max(1, 1_000_000 // max(len(delta), 1))
        for start in range(0, len(self.frequencies_hz), node_chunk):
            frequencies = self.frequencies_hz[start : start + node_chunk]
            fields = coefficients[None] * np.exp(-2j * np.pi * frequencies[:, None] * delta)
            if phase_integrals is not None:
                fields *= phase_integrals[start : start + node_chunk]
            elif phase_integral is not None:
                fields *= np.asarray([phase_integral(frequency) for frequency in frequencies])
            for i, values in enumerate(fields, start):
                # bincount is substantially faster than a scatter for dense meshes.
                self.values[i] += np.bincount(
                    path_indices, weights=values.real, minlength=self.values.shape[1]
                ) + 1j * np.bincount(
                    path_indices, weights=values.imag, minlength=self.values.shape[1]
                )

    def _accumulate_torch(
        self, coefficients, delta, path_indices, phase_integral, phase_integrals, runtime
    ):
        """Integrate all nodes and reduce leaves on device before copying parents."""
        import torch

        parents, inverse = np.unique(path_indices, return_inverse=True)
        if parents.size == 0:
            return
        inverse_t = torch.as_tensor(inverse, dtype=torch.int64, device=runtime.device)
        coefficients_t = torch.as_tensor(
            coefficients, dtype=runtime.complex_dtype, device=runtime.device
        )
        delta_t = torch.as_tensor(delta, dtype=runtime.float_dtype, device=runtime.device)
        node_chunk = max(1, 1_000_000 // len(delta))
        for start in range(0, len(self.frequencies_hz), node_chunk):
            frequencies = self.frequencies_hz[start : start + node_chunk]
            frequencies_t = torch.as_tensor(
                frequencies, dtype=runtime.float_dtype, device=runtime.device
            )
            fields = coefficients_t[None] * torch.exp(
                -2j * np.pi * frequencies_t[:, None] * delta_t
            )
            if phase_integrals is not None:
                fields *= torch.as_tensor(
                    phase_integrals[start : start + node_chunk],
                    dtype=runtime.complex_dtype,
                    device=runtime.device,
                )
            elif phase_integral is not None:
                fields *= torch.as_tensor(
                    np.asarray([phase_integral(frequency) for frequency in frequencies]),
                    dtype=runtime.complex_dtype,
                    device=runtime.device,
                )
            # Real scatter-add also works on devices without complex atomics.
            real = torch.zeros(
                (len(frequencies), len(parents)), dtype=runtime.float_dtype, device=runtime.device
            )
            imag = torch.zeros_like(real)
            real.index_add_(1, inverse_t, fields.real)
            imag.index_add_(1, inverse_t, fields.imag)
            summed = torch.complex(real, imag).cpu().numpy()
            self.values[start : start + len(frequencies), parents] += summed

    @property
    def valid_paths(self):
        return np.isfinite(self.min_delta_s)

    def reset_paths(self, indices):
        self.values[:, indices] = 0.0
        self.min_delta_s[indices] = np.inf
        self.max_delta_s[indices] = -np.inf

    def _filtered_values(self, parents, tau, cutoff, *, runtime=None):
        """Reintegrate only parents intersecting the IF edge, before summation."""
        from .physical_optics import _standard_simplex_exp_integral

        if np.any(~self.retained_paths[parents]):
            raise ValueError("PO response must be reevaluated for the new IF cutoff")
        if runtime is not None:
            return self._filtered_values_torch(parents, tau, cutoff, runtime)
        result = np.zeros((len(self.frequencies_hz), len(parents)), complex)
        lookup = np.full(self.values.shape[1], -1, dtype=int)
        lookup[parents] = np.arange(len(parents))
        for ids, coefficients, deltas, offsets, carrier in self.leaf_batches:
            selected = np.flatnonzero(lookup[ids] >= 0)
            for leaf in selected:
                parent = ids[leaf]
                delay = tau[parent] + deltas[leaf]
                coefficient = coefficients[leaf]
                fields = coefficient * np.exp(-2j * np.pi * self.frequencies_hz * deltas[leaf])
                if offsets is None:
                    if delay <= cutoff:
                        result[:, lookup[parent]] += fields
                    continue
                vertex_delays = offsets[leaf]
                # Clip the reference simplex by its affine delay. This changes
                # the integration domain, not the analytic integration rule.
                limit = cutoff - delay
                integral = np.zeros(len(self.frequencies_hz), complex)
                phase_scale = 2.0 * np.pi * (carrier + self.frequencies_hz)
                for determinant, a, b, d in _clipped_delay_triangles(vertex_delays, limit):
                    integral += (
                        2.0
                        * determinant
                        * np.exp(-1j * phase_scale * a)
                        * _standard_simplex_exp_integral(
                            phase_scale * (b - a), phase_scale * (d - a)
                        )
                    )
                result[:, lookup[parent]] += fields * integral
        return result

    def _filtered_values_torch(self, parents, tau, cutoff, runtime):
        """Keep clipped-simplex evaluation and parent summation on device."""
        import torch
        from .physical_optics import _standard_simplex_exp_integral_torch

        lookup = np.full(self.values.shape[1], -1, dtype=int)
        lookup[parents] = np.arange(len(parents))
        leaves = []
        for ids, coefficients, deltas, offsets, carrier in self.leaf_batches:
            for leaf in np.flatnonzero(lookup[ids] >= 0):
                parent = ids[leaf]
                delay = tau[parent] + deltas[leaf]
                if offsets is None:
                    if delay <= cutoff:
                        # 2*simplex(0,0)=1 gives the point-quadrature response.
                        leaves.append(
                            (
                                lookup[parent],
                                2.0 * coefficients[leaf],
                                deltas[leaf],
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                            )
                        )
                else:
                    for determinant, a, b, d in _clipped_delay_triangles(
                        offsets[leaf], cutoff - delay
                    ):
                        leaves.append(
                            (
                                lookup[parent],
                                2.0 * determinant * coefficients[leaf],
                                deltas[leaf],
                                carrier,
                                a,
                                b - a,
                                d - a,
                            )
                        )
        shape = (len(self.frequencies_hz), len(parents))
        real = torch.zeros(shape, dtype=runtime.float_dtype, device=runtime.device)
        imag = torch.zeros_like(real)
        frequencies = torch.as_tensor(
            self.frequencies_hz, dtype=runtime.float_dtype, device=runtime.device
        )[:, None]
        chunk_size = max(1, 1_000_000 // len(self.frequencies_hz))
        for start in range(0, len(leaves), chunk_size):
            ids, coefficients, deltas, carriers, origins, alpha, beta = zip(
                *leaves[start : start + chunk_size]
            )
            ids_t = torch.as_tensor(ids, dtype=torch.int64, device=runtime.device)
            coefficients_t = torch.as_tensor(
                coefficients, dtype=runtime.complex_dtype, device=runtime.device
            )
            deltas, carriers, origins, alpha, beta = (
                torch.as_tensor(values, dtype=runtime.float_dtype, device=runtime.device)
                for values in (deltas, carriers, origins, alpha, beta)
            )
            phase_scale = 2.0 * np.pi * (carriers[None] + frequencies)
            fields = (
                coefficients_t[None]
                * torch.exp(-2j * np.pi * frequencies * deltas)
                * torch.exp(-1j * phase_scale * origins)
                * _standard_simplex_exp_integral_torch(phase_scale * alpha, phase_scale * beta)
            )
            real.index_add_(1, ids_t, fields.real)
            imag.index_add_(1, ids_t, fields.imag)
        return torch.complex(real, imag)

    def with_channel_gains(self, gains):
        return replace(self, channel_gains=np.asarray(gains))

    def synthesize_adc(
        self, delays_s, valid, fmcw, *, backend="numpy", precision="float64", diagnostics=None
    ):
        """Interpolate and mix on NumPy or the selected Torch device."""
        _, complex_dtype = numpy_dtypes_for_precision(precision)
        runtime = resolve_torch_runtime(backend, precision)
        frequencies = fmcw.slope * np.arange(fmcw.num_adc_samples) / fmcw.sampling_frequency
        nodes = self.frequencies_hz
        tolerance = 16.0 * np.finfo(float).eps * max(float(np.max(abs(nodes))), 1.0)
        if np.any(frequencies < min(nodes) - tolerance) or np.any(
            frequencies > max(nodes) + tolerance
        ):
            raise ValueError("PO delay response does not cover the FMCW bandwidth")
        bary_weights = (-1.0) ** np.arange(len(nodes))
        bary_weights[[0, -1]] *= 0.5
        differences = frequencies[:, None] - nodes
        exact = abs(differences) <= tolerance
        weights = np.divide(
            bary_weights, differences, out=np.zeros_like(differences), where=~exact
        )
        exact_rows = np.any(exact, axis=1)
        weights[exact_rows] = exact[exact_rows]
        weights /= weights.sum(axis=1, keepdims=True)
        gains = (
            np.ones((1, self.values.shape[1]), complex)
            if self.channel_gains is None
            else self.channel_gains
        )
        tau = np.broadcast_to(delays_s, gains.shape)
        mask = np.broadcast_to(valid, gains.shape)
        cutoff = max_if_delay_s(fmcw)
        rejected = mask & (tau + self.min_delta_s[None] > cutoff)
        partial = mask & ~rejected & (tau + self.max_delta_s[None] > cutoff)
        accepted = mask & ~rejected
        record_if_filter(
            diagnostics,
            fmcw,
            tested=np.count_nonzero(mask),
            rejected=np.count_nonzero(rejected),
            partial=np.count_nonzero(partial),
        )
        # Keep diagnostics in the original path domain. A fully cancelled
        # response can still contribute after partial IF clipping, so retain
        # those parents even when all their unfiltered nodes are zero.
        contributing = accepted & (gains != 0.0)
        path_ids = np.flatnonzero(
            np.any(contributing, axis=0)
            & (np.any(self.values != 0.0, axis=0) | np.any(partial & contributing, axis=0))
        )
        original_tau = tau
        values = self.values[:, path_ids]
        gains = gains[:, path_ids]
        tau = tau[:, path_ids]
        accepted = accepted[:, path_ids]
        partial = partial[:, path_ids]
        if runtime is not None:
            return self._synthesize_adc_torch(
                weights,
                frequencies,
                values,
                gains,
                tau,
                accepted,
                partial,
                cutoff,
                fmcw,
                path_ids=path_ids,
                original_tau=original_tau,
                runtime=runtime,
            )
        output = np.zeros((len(frequencies), gains.shape[0]), complex)
        # Keep [sample,path] buffers bounded independently of mesh size.
        chunk_size = max(1, 1_000_000 // max(len(frequencies), 1))
        for start in range(0, len(path_ids), chunk_size):
            sl = slice(start, start + chunk_size)
            integral = weights @ values[:, sl].conjugate()
            for channel in range(gains.shape[0]):
                channel_integral = integral
                if np.any(partial[channel, sl]):
                    channel_integral = integral.copy()
                    local = np.flatnonzero(partial[channel, sl])
                    parents = path_ids[local + start]
                    channel_integral[:, local] = (
                        weights
                        @ self._filtered_values(parents, original_tau[channel], cutoff).conjugate()
                    )
                beat = np.exp(2j * np.pi * frequencies[:, None] * tau[channel, sl])
                output[:, channel] += np.sum(
                    channel_integral
                    * beat
                    * (gains[channel, sl].conjugate() * accepted[channel, sl]),
                    axis=1,
                )
        if adc_mode(fmcw) == "real":
            output = output.real
        return output.astype(complex_dtype)

    def _synthesize_adc_torch(
        self,
        weights,
        frequencies,
        values,
        gains,
        tau,
        accepted,
        partial,
        cutoff,
        fmcw,
        *,
        path_ids,
        original_tau,
        runtime,
    ):
        """Keep interpolation, mixing, and path reduction on the Torch device."""
        import torch

        device, dtype, cdtype = runtime.device, runtime.float_dtype, runtime.complex_dtype
        complex_dtype = runtime.numpy_complex_dtype
        weights_t = torch.as_tensor(weights, dtype=cdtype, device=device)
        # Large beat phases lose accuracy if delays are rounded to float32 first.
        # MPS lacks float64; CUDA and CPU can form and reduce the phase in double.
        frequencies_t = torch.as_tensor(
            2.0 * np.pi * frequencies, dtype=runtime.phase_dtype, device=device
        )
        num_samples, num_channels = len(frequencies), gains.shape[0]
        output = torch.zeros((num_samples, num_channels), dtype=cdtype, device=device)
        # Bound all channel/sample/path buffers together, including dense MIMO arrays.
        chunk_size = max(1, 8_000_000 // max(num_samples * num_channels, 1))
        for start in range(0, len(path_ids), chunk_size):
            sl = slice(start, start + chunk_size)
            values_t = torch.as_tensor(values[:, sl].conjugate(), dtype=cdtype, device=device)
            integral = weights_t @ values_t
            channel_integrals = integral[None].expand(num_channels, -1, -1)
            if np.any(partial[:, sl]):
                channel_integrals = channel_integrals.clone()
                for channel in range(num_channels):
                    local = np.flatnonzero(partial[channel, sl])
                    if local.size:
                        # Only IF-edge parents need the retained leaf geometry.
                        filtered_t = self._filtered_values(
                            path_ids[local + start], original_tau[channel], cutoff, runtime=runtime
                        ).conj()
                        local_t = torch.as_tensor(local, device=device)
                        channel_integrals[channel, :, local_t] = weights_t @ filtered_t
            tau_t = torch.as_tensor(np.array(tau[:, sl]), dtype=runtime.phase_dtype, device=device)
            gains_t = torch.as_tensor(
                gains[:, sl].conjugate() * accepted[:, sl], dtype=cdtype, device=device
            )
            phase = frequencies_t[None, :, None] * tau_t[:, None, :]
            if dtype == torch.float32:
                phase = torch.remainder(phase, 2.0 * np.pi).to(dtype)
            beat = torch.exp(1j * phase)
            beat *= channel_integrals
            beat *= gains_t[:, None, :]
            output += beat.sum(dim=2).T
        if adc_mode(fmcw) == "real":
            output = output.real.to(cdtype)
        return output.cpu().numpy().astype(complex_dtype, copy=False)
