# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Independent endpoint fields through the full PO integration entry point."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import c

from mmWaveRadar.polarization import transverse_linear_basis
from mmWaveRadar.simulation.options import (
    POArraySettings,
    POChannelSettings,
    POIntegrationSettings,
    POVisibilitySettings,
)
from mmWaveRadar.simulation.physical_optics import (
    PhysicalOpticsMaterial,
    evaluate_po_channel,
    po_channel,
    vector_facet_face_contributions,
)

MODES = [
    "face_centroid",
    "parent_face_quadrature",
    "parent_face_adaptive_analytic",
    "parent_face_far_field_analytic",
]


def settings(mode, callback=None):
    return POChannelSettings(
        array=POArraySettings(
            tx_positions=np.array([[0.0, -0.1, 0.2]]),
            rx_positions=np.array([[0.0, 0.2, -0.1]]),
            frequency_hz=28e9,
            material=PhysicalOpticsMaterial(16.5, 25.8, model="interface"),
            polarization_callback=callback,
            max_frequency_offset_hz=100e6,
        ),
        visibility=POVisibilitySettings(visibility="none"),
        integration=POIntegrationSettings(
            po_integration_mode=mode, po_quadrature_max_refinement_depth=3
        ),
    )


VERTICES = np.array([[2.0, -0.02, -0.02], [2.0, -0.02, 0.02], [2.0, 0.02, -0.02]])
FACES = np.array([[0, 1, 2]], dtype=np.uint32)


def callback(points):
    tx_v, _ = transverse_linear_basis(points - [0.0, -0.1, 0.2])
    rx_v, _ = transverse_linear_basis([0.0, 0.2, -0.1] - points)
    return tx_v, rx_v


@pytest.mark.parametrize("mode", MODES)
def test_explicit_vv_fields_preserve_default_and_grouped_keyword_results(mode):
    original = evaluate_po_channel(VERTICES, FACES, settings=settings(mode))
    explicit_settings = settings(mode, callback)
    explicit = evaluate_po_channel(VERTICES, FACES, settings=explicit_settings)
    keyword = po_channel(VERTICES, FACES, **explicit_settings.as_keywords())
    assert np.max(abs(original.coefficients)) > 0
    np.testing.assert_allclose(
        explicit.coefficients, original.coefficients, rtol=1e-12, atol=1e-20
    )
    np.testing.assert_array_equal(explicit.delays_s, original.delays_s)
    np.testing.assert_allclose(explicit.coefficients, keyword.coefficients, rtol=1e-12)
    if original.delay_response is not None:
        np.testing.assert_allclose(
            explicit.delay_response.values,
            original.delay_response.values,
            rtol=1e-12,
            atol=1e-20,
        )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_callback_preserves_complex_tx_linearity_and_hermitian_rx_projection(
    mode, backend, monkeypatch
):
    if backend == "torch":
        pytest.importorskip("torch")
        monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")
    base_settings = settings(mode, callback)
    base_settings = replace(
        base_settings, compute=replace(base_settings.compute, compute_backend=backend)
    )
    base = evaluate_po_channel(VERTICES, FACES, settings=base_settings)

    def scaled(points):
        tx, rx = callback(points)
        return 2 * tx, 3j * rx

    actual = evaluate_po_channel(
        VERTICES,
        FACES,
        settings=replace(
            base_settings,
            array=replace(base_settings.array, polarization_callback=scaled),
        ),
    )
    np.testing.assert_allclose(
        actual.coefficients, -6j * base.coefficients, rtol=1e-12, atol=1e-20
    )
    if base.delay_response is not None:
        np.testing.assert_allclose(
            actual.delay_response.values,
            -6j * base.delay_response.values,
            rtol=1e-12,
            atol=1e-20,
        )


def test_callback_with_independent_rx_polarization_reaches_the_scattering_kernel():
    def rx_h(points):
        tx, _ = callback(points)
        _, rx = transverse_linear_basis([0.0, 0.2, -0.1] - points)
        return tx, rx

    base = evaluate_po_channel(
        VERTICES, FACES, settings=settings("parent_face_adaptive_analytic", callback)
    )
    cross = evaluate_po_channel(
        VERTICES, FACES, settings=settings("parent_face_adaptive_analytic", rx_h)
    )
    assert np.max(abs(cross.coefficients)) > 0
    assert not np.allclose(cross.coefficients, base.coefficients, rtol=1e-3, atol=1e-20)


def test_non_callable_callback_is_rejected_before_solving():
    invalid = settings("face_centroid")
    invalid = replace(invalid, array=replace(invalid.array, polarization_callback="V"))
    with pytest.raises(ValueError, match="polarization_callback"):
        evaluate_po_channel(VERTICES, FACES, settings=invalid)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_nonfinite_callback_fields_are_rejected(mode, backend, monkeypatch):
    if backend == "torch":
        pytest.importorskip("torch")
        monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")

    def invalid(points):
        tx, rx = callback(points)
        return tx * np.nan, rx

    config = settings(mode, invalid)
    config = replace(config, compute=replace(config.compute, compute_backend=backend))
    with pytest.raises(ValueError, match="fields must be finite"):
        evaluate_po_channel(VERTICES, FACES, settings=config)


GEOMETRIES = [
    ([0, 0, 0], [0, 0, 0]),
    ([0, -0.1, 0.2], [0, 0.2, -0.1]),
    ([0, -1.5, 0.8], [0.5, 1.2, -0.7]),
    ([1, -4, 2], [1, 4, -1.5]),
]


def endpoint_fields(tx_position, rx_position, tx_mix=(1, 0), rx_mix=(1, 0)):
    def fields(points):
        tx_v, tx_h = transverse_linear_basis(points - tx_position)
        rx_v, rx_h = transverse_linear_basis(rx_position - points)
        return tx_mix[0] * tx_v + tx_mix[1] * tx_h, rx_mix[0] * rx_v + rx_mix[1] * rx_h

    return fields


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_callback_is_reciprocal_when_tx_rx_and_complex_fields_are_exchanged(mode, geometry):
    tx, rx = np.asarray(geometry, dtype=float)
    fields = endpoint_fields(tx, rx, tx_mix=(1, 0.4j), rx_mix=(0.7j, 1))
    original = settings(mode, fields)
    original = replace(
        original, array=replace(original.array, tx_positions=tx[None], rx_positions=rx[None])
    )
    forward = evaluate_po_channel(VERTICES, FACES, settings=original)

    def reciprocal_fields(points):
        incident, receive = fields(points)
        return receive.conj(), incident.conj()

    swapped = replace(
        original,
        array=replace(
            original.array,
            tx_positions=rx[None],
            rx_positions=tx[None],
            polarization_callback=reciprocal_fields,
        ),
    )
    reverse = evaluate_po_channel(VERTICES, FACES, settings=swapped)
    assert np.max(abs(forward.coefficients)) > 1e-15
    np.testing.assert_allclose(reverse.coefficients, forward.coefficients, rtol=1e-11, atol=1e-20)
    np.testing.assert_allclose(reverse.delays_s, forward.delays_s, rtol=1e-14)
    if forward.delay_response is not None:
        np.testing.assert_allclose(
            reverse.delay_response.values, forward.delay_response.values, rtol=1e-11, atol=1e-20
        )


def _reference_fields(direction):
    """Closed-form projected-Z basis; H's sign follows physical propagation."""
    k = direction / np.linalg.norm(direction)
    transverse = np.hypot(k[0], k[1])
    return (
        np.array([-k[2] * k[0], -k[2] * k[1], transverse**2]) / transverse,
        np.array([k[1], -k[0], 0.0]) / transverse,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("channel", ["VV", "VH", "HV", "HH"])
def test_signed_complex_polarization_response_matches_closed_form_pec_dyadic(
    mode, geometry, channel
):
    # An electrically small facet makes centroid and integrated solutions
    # agree while retaining the complex phase and the sign of both H fields.
    center = np.array([2.0, 0.1, -0.2])
    vertices = center + 1e-6 * np.array([[0, -1, -1], [0, -1, 2], [0, 2, -1]])
    tx, rx = np.asarray(geometry, dtype=float)
    indices = [0 if pol == "V" else 1 for pol in channel]
    fields = endpoint_fields(
        tx,
        rx,
        tx_mix=(1, 0) if indices[0] == 0 else (0, 1),
        rx_mix=(1, 0) if indices[1] == 0 else (0, 1),
    )
    config = settings(mode, fields)
    config = replace(
        config,
        array=replace(
            config.array,
            tx_positions=tx[None],
            rx_positions=rx[None],
            material=PhysicalOpticsMaterial(1, 0, model="pec"),
        ),
    )
    actual = evaluate_po_channel(vertices, FACES, settings=config)
    tx_distance, rx_distance = np.linalg.norm(tx - center), np.linalg.norm(rx - center)
    source, observer = (tx - center) / tx_distance, (rx - center) / rx_distance
    normal = np.array([-1.0, 0.0, 0.0])
    incident = _reference_fields(-source)[indices[0]]
    receive = _reference_fields(observer)[indices[1]]
    # PEC tangent-plane dyadic after contraction with transverse endpoint
    # fields, independent of the solver's TE/TM decomposition and kernel.
    projected = (
        np.vdot(receive, source) * (normal @ incident)
        + np.vdot(receive, normal) * (observer @ incident)
        - (normal @ (source + observer)) * np.vdot(receive, incident)
    )
    area = 4.5e-12
    k0 = 2 * np.pi * config.array.frequency_hz / c
    expected = (
        1j
        * area
        * projected
        * np.exp(-1j * k0 * (tx_distance + rx_distance))
        / (8 * np.pi * tx_distance * rx_distance)
    )
    np.testing.assert_allclose(actual.coefficients, expected, rtol=2e-7, atol=1e-24)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("discard", ["back_facing", "zero_weight", "all"])
def test_nonfinite_fields_on_discarded_facets_do_not_abort_the_solve(
    backend, discard, monkeypatch
):
    if backend == "torch":
        pytest.importorskip("torch")
        monkeypatch.setenv("MMWAVE_TORCH_DEVICE", "cpu")
    centroids = np.array([[2.0, 0, 0], [2.0, 0.1, 0]])
    normals = np.array([[-1.0, 0, 0], [-1.0, 0, 0]])
    weights = np.ones(2)
    if discard in {"back_facing", "all"}:
        normals[1] *= -1
    if discard in {"zero_weight", "all"}:
        weights[1] = 0
    if discard == "all":
        weights[0] = 0

    def fields(points):
        tx = np.tile([0, 0, 1.0], (len(points), 1))
        tx[points[:, 1] > 0.05] = np.nan
        return tx, tx.copy()

    kwargs = dict(
        face_indices=np.arange(2),
        centroids=centroids,
        normals=normals,
        areas=np.ones(2),
        tx_position=np.zeros(3),
        rx_position=np.zeros(3),
        frequency_hz=28e9,
        material=PhysicalOpticsMaterial(1, 0, model="pec"),
        weights=weights,
        backend=backend,
    )
    expected, delays = vector_facet_face_contributions(**kwargs)
    actual, actual_delays = vector_facet_face_contributions(**kwargs, polarization_callback=fields)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-20)
    np.testing.assert_array_equal(actual_delays, delays)
