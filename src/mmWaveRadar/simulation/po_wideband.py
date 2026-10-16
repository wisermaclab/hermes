# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Compact delay-resolved surface integrals for wideband FMCW synthesis."""

from dataclasses import dataclass, replace, field
import math

import numpy as np
from scipy.constants import c
from scipy.special import gammaln
from ..radar.adc import max_if_delay_s, record_if_filter, adc_mode


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
    delay_bound_s: float = 0.
    receiver_fmcw: object = None
    channel_delay_margin_s: float = 0.

    @classmethod
    def for_mesh(cls, vertices, faces, num_paths, max_frequency_offset_hz,
                 *, receiver_fmcw=None, channel_delay_margin_s=0.):
        triangles = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int)]
        max_edge = max((float(np.max(np.linalg.norm(
            triangles[:, i] - triangles[:, j], axis=1), initial=0.))
            for i, j in ((0, 1), (1, 2), (2, 0))), default=0.)
        # Exact sample paths differ by at most 2*max_edge. Allow 4*max_edge
        # also to bound the analytic mode's center shift plus its linear
        # phase within a subtriangle, for any real/image endpoints.
        delay_bound = 4. * max_edge / c
        bandwidth = abs(float(max_frequency_offset_hz))
        x = np.pi * bandwidth * delay_bound
        # Bound the Chebyshev tail of exp(-j*x*y), |y|<=1, using
        # |J_n(x)| <= (x/2)^n/n! * exp(x*x/(4*(n+1))). Keep the omitted
        # tail below 1e-14 times the sum of sample magnitudes, before the
        # modest interpolation Lebesgue factor. No fixed-order truncation.
        order = max(8, int(math.ceil(x)))
        if x > 0.:
            while True:
                ratio = x / (2. * (order + 1.))
                log_bound = (math.log(2.) + order * math.log(x / 2.)
                             - gammaln(order + 1.) + x*x / (4.*(order + 1.)))
                if ratio < 1. and log_bound - math.log1p(-ratio) < math.log(1e-14):
                    break
                order += 1
        nodes = .5 * bandwidth * (1. - np.cos(np.linspace(0., np.pi, order + 1)))
        if max_frequency_offset_hz < 0.:
            nodes = -nodes
        return cls(nodes, np.zeros((len(nodes), num_paths), complex),
                   np.zeros(num_paths, dtype=float),
                   min_delta_s=np.full(num_paths, np.inf),
                   max_delta_s=np.full(num_paths, -np.inf),
                   retained_paths=np.ones(num_paths, dtype=bool),
                   delay_bound_s=delay_bound, receiver_fmcw=receiver_fmcw,
                   channel_delay_margin_s=channel_delay_margin_s)

    def set_reference(self, indices, centroids, tx, rx):
        self.reference_delays_s[indices] = (
            np.linalg.norm(centroids - tx, axis=1)
            + np.linalg.norm(centroids - rx, axis=1)
        ) / c
        if self.receiver_fmcw is not None:
            margin = self.delay_bound_s + self.channel_delay_margin_s
            self.retained_paths[indices] = abs(
                self.reference_delays_s[indices] - max_if_delay_s(self.receiver_fmcw)
            ) <= margin

    def accumulate(self, coefficients, delays_s, path_indices, *, phase_integral=None,
                   vertex_delay_offsets_s=None, carrier_frequency_hz=0.):
        delta = np.asarray(delays_s) - self.reference_delays_s[path_indices]
        offsets = vertex_delay_offsets_s
        if phase_integral is not None and offsets is None:
            raise ValueError("Analytic delay responses require vertex delay offsets")
        low = delta if offsets is None else delta + np.min(offsets, axis=1)
        high = delta if offsets is None else delta + np.max(offsets, axis=1)
        np.minimum.at(self.min_delta_s, path_indices, low)
        np.maximum.at(self.max_delta_s, path_indices, high)
        keep = self.retained_paths[path_indices]
        if np.any(keep):
            self.leaf_batches.append((
                np.array(path_indices[keep]), np.array(coefficients[keep]),
                np.array(delta[keep]), None if offsets is None else np.array(offsets[keep]),
                float(carrier_frequency_hz),
            ))
        for i, frequency in enumerate(self.frequencies_hz):
            fields = coefficients * np.exp(-2j * np.pi * frequency * delta)
            if phase_integral is not None:
                fields = fields * phase_integral(frequency)
            # bincount is substantially faster than a scatter for dense meshes.
            self.values[i] += (
                np.bincount(path_indices, weights=fields.real, minlength=self.values.shape[1])
                + 1j * np.bincount(path_indices, weights=fields.imag, minlength=self.values.shape[1])
            )

    @property
    def valid_paths(self):
        return np.isfinite(self.min_delta_s)

    def reset_paths(self, indices):
        self.values[:, indices] = 0.
        self.min_delta_s[indices] = np.inf
        self.max_delta_s[indices] = -np.inf

    def _filtered_values(self, parents, tau, cutoff):
        """Reintegrate only parents intersecting the IF edge, before summation."""
        from .physical_optics import _standard_simplex_exp_integral

        if np.any(~self.retained_paths[parents]):
            raise ValueError("PO response must be reevaluated for the new IF cutoff")
        result = np.zeros((len(self.frequencies_hz), len(parents)), complex)
        lookup = np.full(self.values.shape[1], -1, dtype=int)
        lookup[parents] = np.arange(len(parents))
        for ids, coefficients, deltas, offsets, carrier in self.leaf_batches:
            selected = np.flatnonzero(lookup[ids] >= 0)
            for leaf in selected:
                parent = ids[leaf]
                delay = tau[parent] + deltas[leaf]
                coefficient = coefficients[leaf]
                fields = coefficient * np.exp(-2j*np.pi*self.frequencies_hz*deltas[leaf])
                if offsets is None:
                    if delay <= cutoff:
                        result[:, lookup[parent]] += fields
                    continue
                vertex_delays = offsets[leaf]
                # Clip the reference simplex by its affine delay. This changes
                # the integration domain, not the analytic integration rule.
                polygon = [(np.array([0., 0.]), vertex_delays[0]),
                           (np.array([1., 0.]), vertex_delays[1]),
                           (np.array([0., 1.]), vertex_delays[2])]
                clipped = []
                limit = cutoff - delay
                for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                    inside_a, inside_b = a[1] <= limit, b[1] <= limit
                    if inside_a:
                        clipped.append(a)
                    if inside_a != inside_b:
                        fraction = (limit - a[1]) / (b[1] - a[1])
                        clipped.append((a[0] + fraction*(b[0]-a[0]), limit))
                integral = np.zeros(len(self.frequencies_hz), complex)
                phase_scale = 2.*np.pi*(carrier + self.frequencies_hz)
                for i in range(1, len(clipped)-1):
                    a, b, d = clipped[0], clipped[i], clipped[i+1]
                    edge1, edge2 = b[0]-a[0], d[0]-a[0]
                    determinant = abs(edge1[0]*edge2[1] - edge1[1]*edge2[0])
                    integral += (2.*determinant * np.exp(-1j*phase_scale*a[1])
                                 * _standard_simplex_exp_integral(
                                     phase_scale*(b[1]-a[1]), phase_scale*(d[1]-a[1])))
                result[:, lookup[parent]] += fields * integral
        return result

    def with_channel_gains(self, gains):
        return replace(self, channel_gains=np.asarray(gains))

    def synthesize_adc(self, delays_s, valid, fmcw, *, precision="float64", diagnostics=None):
        """Interpolate the surface integral and apply the receiver mixer once."""
        frequencies = fmcw.slope * np.arange(fmcw.num_adc_samples) / fmcw.sampling_frequency
        nodes = self.frequencies_hz
        tolerance = 16. * np.finfo(float).eps * max(float(np.max(abs(nodes))), 1.)
        if np.any(frequencies < min(nodes) - tolerance) or np.any(frequencies > max(nodes) + tolerance):
            raise ValueError("PO delay response does not cover the FMCW bandwidth")
        bary_weights = (-1.) ** np.arange(len(nodes))
        bary_weights[[0, -1]] *= .5
        differences = frequencies[:, None] - nodes
        exact = abs(differences) <= tolerance
        weights = np.divide(bary_weights, differences,
                            out=np.zeros_like(differences), where=~exact)
        exact_rows = np.any(exact, axis=1)
        weights[exact_rows] = exact[exact_rows]
        weights /= weights.sum(axis=1, keepdims=True)
        gains = (np.ones((1, self.values.shape[1]), complex)
                 if self.channel_gains is None else self.channel_gains)
        tau = np.broadcast_to(delays_s, gains.shape)
        mask = np.broadcast_to(valid, gains.shape)
        cutoff = max_if_delay_s(fmcw)
        rejected = mask & (tau + self.min_delta_s[None] > cutoff)
        partial = (mask & ~rejected & (tau + self.max_delta_s[None] > cutoff))
        accepted = mask & ~rejected
        record_if_filter(diagnostics, fmcw, tested=np.count_nonzero(mask),
                         rejected=np.count_nonzero(rejected), partial=np.count_nonzero(partial))
        output = np.zeros((len(frequencies), gains.shape[0]), complex)
        # Keep [sample,path] buffers bounded independently of mesh size.
        chunk_size = max(1, 1_000_000 // max(len(frequencies), 1))
        for start in range(0, self.values.shape[1], chunk_size):
            sl = slice(start, start + chunk_size)
            integral = weights @ self.values[:, sl].conjugate()
            for channel in range(gains.shape[0]):
                channel_integral = integral
                if np.any(partial[channel, sl]):
                    channel_integral = integral.copy()
                    local = np.flatnonzero(partial[channel, sl])
                    parents = local + start
                    channel_integral[:, local] = weights @ self._filtered_values(
                        parents, tau[channel], cutoff).conjugate()
                beat = np.exp(2j*np.pi*frequencies[:, None]*tau[channel, sl])
                output[:, channel] += np.sum(
                    channel_integral * beat * (gains[channel, sl].conjugate()
                                              * accepted[channel, sl]), axis=1,
                )
        if adc_mode(fmcw) == "real":
            output = output.real
        return output.astype(np.complex64 if precision == "float32" else np.complex128)
